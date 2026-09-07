import abc
import atexit
import csv
import functools
import json
import os
import os.path as osp
import random
import re
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Generic, Mapping, NamedTuple, Sequence, Tuple, TypeVar, Union

import flax
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
from brax.envs.wrappers import training as brax_training
from distutils.util import strtobool
from flax import nnx, struct
from jax import flatten_util
from mujoco_playground import registry, wrapper
from tensorboardX import SummaryWriter

from utils.acting import actor_step, actor_step_rep, wrap_env_for_training

# [NSTEP] the n-step helpers are the same ones dhpg.py uses, so the two
# baselines cannot drift on how returns are computed.
from utils.buffer import (
    RunningMeanStd,
    RunningStatistics,
    UniformSamplingQueue,
    nstep_aggregate,  # NEW
    nstep_fifo_init,  # NEW
    nstep_fifo_push,  # NEW
    nstep_template_from_dims,  # NEW
)
from utils.logger import EpochLogger
from utils.models import (
    EnsembleCritic,
    SACGaussianActor,
    Scalar,
    get_tree_norm,
)
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(5e4),
    "eval_episode_freq": 5,
    "hidden_size": 256,
    "lr": 3e-4,
    "max_grad_norm": 10,
    "gamma": 0.99,
    "update_tau": 0.005,
    "train_per_step": 1,
    "episode_length": 1000,
    "warmup_samples": int(5e3),
    "max_replay_size": int(1e5),
    "batch_size": int(256),
    "total_env_steps": int(1e6),
    "init_temperature": 0.1,
    # ---- [NSTEP] n-step returns, same keys/values as dhpg.py ----
    "nstep": 3,  # NEW; 1 reproduces the old 1-step behaviour EXACTLY
    "bootstrap_on_truncation": True,  # NEW; see the note at the bottom of main()
}


def polyak_update(target_model, curr_model, tau: float):

    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


def sac_train_step(
    actor: SACGaussianActor,
    actor_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    data: Transition,
    config,
    key: jnp.ndarray,
):
    obs = data.observation
    act = data.action
    reward = data.reward
    discount = data.discount
    next_obs = data.next_observation
    truncation = data.extras["state_extras"]["truncation"]
    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
    alpha = jnp.exp(log_alpha())

    def alpha_loss_fn(log_alpha):
        _, log_prob = actor(obs, key_alpha)
        a = jnp.exp(log_alpha())
        loss = jnp.mean(a * jax.lax.stop_gradient(-log_prob - config.target_entropy))
        return loss

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)

    def critic_loss_fn(critic):
        next_act, next_log_prob = actor(next_obs, key_critic)
        q1_t, q2_t = target_critic(jnp.concatenate([next_obs, next_act], axis=-1))
        next_v = jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
        # [NSTEP] CHANGED: `discount` now comes from nstep_aggregate and ALREADY
        # CONTAINS gamma^n (and the per-step termination factors), so the old
        # `config.gamma * discount` would apply gamma^(n+1). For nstep=1,
        # discount == gamma * (1 - done), i.e. identical to the old line.
        target_q = jax.lax.stop_gradient(
            reward * config.reward_scaling + discount * next_v
        )

        q1, q2 = critic(jnp.concatenate([obs, act], axis=-1))

        q_error = jnp.stack([q1, q2], axis=-1) - target_q[..., None]
        # [NSTEP] `truncation` is now a WINDOW-level flag: 1.0 iff this n-step
        # window ended at a time limit. For nstep=1 it reduces exactly to the
        # raw per-step flag, so this line is unchanged.
        q_error = q_error * (1.0 - truncation)[..., None]
        loss = 0.5 * jnp.mean(jnp.square(q_error))
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True
    )(critic)

    def actor_loss_fn(actor):
        pi, log_pi = actor(obs, key_actor)
        q1, q2 = critic(jnp.concatenate([obs, pi], axis=-1))
        loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return loss, jnp.mean(log_pi)

    (actor_loss, log_pi_mean), actor_grads = nnx.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor)

    alpha_opt.update(log_alpha, alpha_grads)
    critic_opt.update(critic, critic_grads)
    actor_opt.update(actor, actor_grads)

    polyak_update(target_critic, critic, config.update_tau)
    alpha_post = jnp.exp(log_alpha())

    return (
        critic_loss,
        actor_loss,
        alpha_loss,
        alpha_post,
        log_pi_mean,
        q1_mean,
        q2_mean,
    )


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    actor,
    actor_opt,
    critic,
    critic_opt,
    target_critic,
    log_alpha,
    alpha_opt,
    config,
    nstep_fifo,  # NEW
    nstep_count,  # NEW
    key,
):
    num_steps = config.log_freq

    def body_fun(i, carry):
        (
            key,
            env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            nstep_fifo,  # NEW
            nstep_count,  # NEW
            models,
            val,
        ) = carry
        (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt) = (
            models
        )

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env,
            env_state,
            actor,
            obs_normalizer,
            env_key,
            extra_fields=("truncation",),
        )
        # [NSTEP] the window is built on the ROLLOUT side: push the raw 1-step
        # transition, insert the aggregated n-step one. The buffer itself is
        # untouched.
        nstep_fifo = nstep_fifo_push(nstep_fifo, transition)  # NEW
        nstep_count = jnp.minimum(nstep_count + 1, config.nstep)  # NEW
        nstep_transition = nstep_aggregate(  # NEW
            nstep_fifo,
            nstep_count,
            config.gamma,
            config.nstep,
            config.bootstrap_on_truncation,
        )
        buffer_state = buffer.insert(buffer_state, nstep_transition)  # CHANGED
        # the normalizer still sees the RAW 1-step observation, not the window
        obs_normalizer = obs_normalizer.update(transition.observation)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        def do_train(j, carry):
            key, env_state, buffer_state, obs_normalizer, models, _ = carry
            (
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(  # normalize at point of use, not storage
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = sac_train_step(
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                batch,
                config,
                train_key,
            )
            models = (
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            )
            return (key, env_state, buffer_state, obs_normalizer, models, val)

        init_val = (jnp.zeros((), jnp.float32),) * 7
        models = (
            actor,
            actor_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
        )
        key, _, buffer_state, obs_normalizer, models, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, n_env_state, buffer_state, obs_normalizer, models, init_val),
        )
        (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt) = (
            models
        )
        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            nstep_fifo,  # NEW
            nstep_count,  # NEW
            (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt),
            val,
        )

    init_val = (jnp.zeros((), jnp.float32),) * 7
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        nstep_fifo,  # NEW
        nstep_count,  # NEW
        (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt),
        init_val,
    )

    (
        _,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        nstep_fifo,  # NEW
        nstep_count,  # NEW
        models,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)
    (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt) = models

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        nstep_fifo,  # NEW
        nstep_count,  # NEW
        num_steps * config.num_envs,
    )


@functools.partial(
    nnx.jit, static_argnames=("env", "episode_length", "num_eval_envs", "deterministic")
)
def evaluate(
    env,
    actor,
    obs_normalizer,
    key,
    episode_length,
    num_eval_envs,
    deterministic: bool = False,
):
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, num_eval_envs))

    def body(carry, _):
        state, ret, alive, k = carry
        k, act_key = jax.random.split(k)
        norm_obs = obs_normalizer.normalize(state.obs)
        if deterministic:
            action = actor.mean_action(norm_obs)
        else:
            action, _ = actor.sample(norm_obs, act_key)
        nstate = env.step(state, action)
        ret = ret + nstate.reward * alive  # count the terminating step
        alive = alive * (1.0 - nstate.done)  # then stop counting
        return (nstate, ret, alive, k), ()

    (_, ret, _, _), _ = jax.lax.scan(
        body,
        (state, jnp.zeros(num_eval_envs), jnp.ones(num_eval_envs), key),
        (),
        length=episode_length,
    )
    # brax reports both mean and std across the 128 eval envs; report both.
    return jnp.mean(ret), jnp.std(ret)


def prefill_buffer(
    key,
    env,
    env_state,
    buffer_state,
    policy,
    buffer,
    obs_normalizer,
    config,  # NEW — needs nstep / gamma / bootstrap_on_truncation
    num_itr: int,
    nstep_fifo,  # NEW
    nstep_count,  # NEW
):
    def body(carry, _):
        key, env_state, buffer_state, obs_normalizer, fifo, count = carry  # CHANGED
        key, subkey = jax.random.split(key)
        n_state, transition = actor_step(
            env=env,
            env_state=env_state,
            policy=policy,
            obs_normalizer=obs_normalizer,
            key=subkey,
            extra_fields=("truncation",),
        )
        # [NSTEP] same FIFO as the main loop, so the seed phase and the training
        # phase put transitions of the SAME kind into the buffer.
        fifo = nstep_fifo_push(fifo, transition)  # NEW
        count = jnp.minimum(count + 1, config.nstep)  # NEW
        buffer_state = buffer.insert(  # CHANGED
            buffer_state,
            nstep_aggregate(
                fifo,
                count,
                config.gamma,
                config.nstep,
                config.bootstrap_on_truncation,
            ),
        )
        obs_normalizer = obs_normalizer.update(transition.observation)
        return (key, n_state, buffer_state, obs_normalizer, fifo, count), ()

    jitted_body = jax.jit(body)
    (
        (_, env_state, buffer_state, obs_normalizer, nstep_fifo, nstep_count),
        (),
    ) = jax.lax.scan(
        jitted_body,
        (key, env_state, buffer_state, obs_normalizer, nstep_fifo, nstep_count),
        (),
        length=num_itr,
    )
    return env_state, buffer_state, obs_normalizer, nstep_fifo, nstep_count


# ===========================================================================
# Section 10 – MAIN
# ===========================================================================


def main(args, cfg_env=None):
    # ── reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    # nnx.Rngs manages separate PRNG streams for parameter init, dropout, etc.
    # Each stream needs its own seed so initialisation is fully reproducible.
    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
    )

    # ── device ────────────────────────────────────────────────────────────
    # GPU is strongly recommended: MuJoCo Playground environments are
    # JAX-native (physics runs on GPU). CPU training is 10–50× slower.
    jax.default_device = jax.devices(args.device)[args.device_id]

    # ── build config ──────────────────────────────────────────────────────
    config = dict(default_cfg)
    config.update(
        {
            "gamma": args.gamma,
            "update_tau": args.update_tau,
            "init_temperature": args.init_temperature,
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
            "hidden_size": args.hidden_size,
            "train_per_step": args.train_per_step,
            "warmup_samples": args.warmup_samples,
            "max_replay_size": args.max_replay_size,
            "total_env_steps": args.total_env_steps,
            "log_freq": args.log_freq,
            "save_freq": args.save_freq,
            "episode_length": args.episode_length,
            "eval_episode_freq": args.eval_episode_freq,
            "batch_size": args.batch_size,
            "num_envs": args.num_envs,
            "num_eval_envs": args.num_eval_envs,
            "reward_scaling": args.reward_scaling,
        }
    )

    # [NSTEP] per-task override, copied from dhpg.py so the two baselines use
    # the SAME return length on every task: the author's cfgs/task/*.yaml put
    # the whole walker domain on nstep=1. Must run BEFORE config_data is frozen.
    if args.task.lower().startswith("walker"):
        config["nstep"] = 1

    # ── environment ───────────────────────────────────────────────────────
    prng_key, env_key = jax.random.split(prng_key)
    env_key = jax.random.split(env_key, config["num_envs"])  # batch_size = 1 env

    env = wrap_env_for_training(
        registry.load(args.task, config_overrides={"impl": "jax"}),
        episode_length=config["episode_length"],
        full_reset=False,
    )
    env_state = env.reset(env_key)
    obs_dim = env.observation_size
    act_dim = env.action_size
    obs_normalizer = RunningMeanStd.init((obs_dim,))

    # Standard SAC target entropy: −|A|
    # Targets roughly uniform distribution over actions at start.
    config["target_entropy"] = float(act_dim) * -0.5

    # Freeze config into an immutable Flax struct (required for nnx.jit stability)
    config_data = make_static_config_from_dict("SACConfig", config)()

    # ── networks ──────────────────────────────────────────────────────────
    actor = SACGaussianActor(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )
    actor_opt = nnx.Optimizer(
        model=actor,
        tx=optax.adam(learning_rate=config["lr"]),
        # tx=optax.chain(
        #     optax.clip_by_global_norm(config["max_grad_norm"]),
        #     optax.adam(learning_rate=config["lr"]),
        #     # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        # ),
        wrt=nnx.Param,
    )

    critic = EnsembleCritic(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )
    critic_opt = nnx.Optimizer(
        model=critic,
        tx=optax.adam(learning_rate=config["lr"]),
        # tx=optax.chain(
        #     optax.clip_by_global_norm(config["max_grad_norm"]),
        #     optax.adam(learning_rate=config["lr"]),
        #     # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        # ),
        wrt=nnx.Param,
    )
    target_critic = deepcopy(critic)  # separate copy for Polyak updates

    log_alpha = Scalar(float(jnp.log(config["init_temperature"])))
    alpha_opt = nnx.Optimizer(
        model=log_alpha, tx=optax.adam(learning_rate=3e-4), wrt=nnx.Param
    )

    # ── replay buffer ─────────────────────────────────────────────────────
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, act_dim))
    dummy_zero = jnp.zeros((1,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_act,
        reward=dummy_zero,
        discount=dummy_zero,
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": dummy_zero}},
    )

    buffer = UniformSamplingQueue(
        max_replay_size=config["max_replay_size"],
        dummy_data_sample=dummy_transition,
        sample_batch_size=config["batch_size"],
    )
    prng_key, buffer_key = jax.random.split(prng_key)
    buffer_state = buffer.init(buffer_key)

    # ── [NSTEP] n-step FIFO state (leading dim nstep, then num_envs) ───────
    # Identical construction to dhpg.py. `nstep_count` is the number of valid
    # entries so far, clipped to nstep, so early windows are simply SHORTER
    # than n rather than contaminated by the zero-fill.
    nstep_fifo = nstep_fifo_init(  # NEW
        nstep_template_from_dims(config["num_envs"], obs_dim, act_dim),
        config["nstep"],
    )
    nstep_count = jnp.array(0, dtype=jnp.int32)  # NEW

    # ── running reward statistics ─────────────────────────────────────────
    prng_key, running_key = jax.random.split(prng_key)
    running_state = RunningStatistics.init(
        (config["eval_episode_freq"] * config["episode_length"],),
        running_key,
    )

    # ── logger ────────────────────────────────────────────────────────────
    dict_args = dict(config)
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)

    # ── warmup ────────────────────────────────────────────────────────────
    logger.log("Start prefilling replay buffer")
    prng_key, buffer_key = jax.random.split(prng_key)
    # env_state, buffer_state = prefill_buffer(
    #     key=buffer_key,
    #     env=env,
    #     env_state=env_state,
    #     buffer_state=buffer_state,
    #     policy=actor,
    #     buffer=buffer,
    #     num_itr=config["warmup_samples"],
    # )
    warmup_iters = max(1, config["warmup_samples"] // config["num_envs"])
    (
        env_state,
        buffer_state,
        obs_normalizer,
        nstep_fifo,  # NEW
        nstep_count,  # NEW
    ) = prefill_buffer(  # CHANGED unpack
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        policy=actor,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        config=config_data,  # NEW arg
        num_itr=warmup_iters,
        nstep_fifo=nstep_fifo,  # NEW arg
        nstep_count=nstep_count,  # NEW arg
    )
    # prng_key, base_key = jax.random.split(prng_key)
    # base_return, base_std = evaluate(
    #     env=env,
    #     actor=actor,
    #     obs_normalizer=obs_normalizer,
    #     key=base_key,
    #     episode_length=config["episode_length"],
    #     num_eval_envs=config["num_eval_envs"],
    #     deterministic=True,
    # )
    # logger.log(
    #     f"[baseline] untrained deterministic return = {float(base_return):.1f} "
    #     f"+/- {float(base_std):.1f}  (over {config['num_eval_envs']} episodes, "
    #     f"0 gradient steps)"
    # )

    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start SAC training")
    logger.log(f"{config}")
    logger.log(
        f"[nstep] nstep={config['nstep']} "
        f"bootstrap_on_truncation={config['bootstrap_on_truncation']} "
        f"(discount from the buffer already contains gamma^nstep)"
    )
    steps = buffer.size(buffer_state)
    steps = int(buffer.size(buffer_state))
    # eval_interval = config["total_env_steps"] // 10  # num_evals = 10
    # next_eval = steps + eval_interval
    next_save = steps + config["save_freq"]
    # last_eval_return = 0.0

    # while steps < config["total_env_steps"]:
    #     prng_key, subkey = jax.random.split(prng_key)

    #     # train_n_steps compiles on first call (~60 s), then runs at GPU speed
    #     val = train_n_steps(
    #         env=env,
    #         env_state=env_state,
    #         buffer_state=buffer_state,
    #         buffer=buffer,
    #         running_state=running_state,
    #         actor=actor,
    #         actor_opt=actor_opt,
    #         critic=critic,
    #         critic_opt=critic_opt,
    #         target_critic=target_critic,
    #         log_alpha=log_alpha,
    #         alpha_opt=alpha_opt,
    #         config=config_data,
    #         key=subkey,
    #     )

    #     (
    #         critic_loss,
    #         actor_loss,
    #         alpha_loss,
    #         alpha,
    #         log_pi_mean,
    #         q1_mean,
    #         q2_mean,
    #         env_state,
    #         running_state,
    #         buffer_state,
    #         num_steps,
    #     ) = val
    while steps < config["total_env_steps"]:
        prng_key, subkey = jax.random.split(prng_key)

        val = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
            obs_normalizer=obs_normalizer,
            actor=actor,
            actor_opt=actor_opt,
            critic=critic,
            critic_opt=critic_opt,
            target_critic=target_critic,
            log_alpha=log_alpha,
            alpha_opt=alpha_opt,
            config=config_data,
            nstep_fifo=nstep_fifo,  # NEW arg
            nstep_count=nstep_count,  # NEW arg
            key=subkey,
        )

        (
            critic_loss,
            actor_loss,
            alpha_loss,
            alpha,
            log_pi_mean,
            q1_mean,
            q2_mean,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            nstep_fifo,  # NEW
            nstep_count,  # NEW
            num_steps,  # CHANGED unpack
        ) = val
        steps += num_steps
        logger.logged = False

        # ── logging (mirrors gpe key naming exactly) ──────────────────────
        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_critic", critic_loss.item())
        logger.log_tabular("Loss/Loss_actor", actor_loss.item())
        logger.log_tabular("Loss/Loss_alpha", alpha_loss.item())

        logger.log_tabular("SAC/Alpha", alpha.item())
        logger.log_tabular("SAC/LogPi_mean", log_pi_mean.item())
        logger.log_tabular("SAC/Q1_mean", q1_mean.item())
        logger.log_tabular("SAC/Q2_mean", q2_mean.item())

        logger.log_tabular(
            "Norm/actor_model",
            get_tree_norm(nnx.state(actor, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/critic_model",
            get_tree_norm(nnx.state(critic, nnx.Param)),
        )

        # logger.log_tabular(
        #     "Eval/Return",
        #     running_state.reward_state.data.sum() / config["eval_episode_freq"],
        # )

        prng_key, eval_key = jax.random.split(prng_key)
        eval_return, eval_std = evaluate(
            env=env,
            actor=actor,
            obs_normalizer=obs_normalizer,
            key=eval_key,
            episode_length=config["episode_length"],
            num_eval_envs=config["num_eval_envs"],
            deterministic=True,
        )

        logger.log_tabular("Eval/Return", float(eval_return))

        logger.dump_tabular()

        # # ── periodic checkpoint ───────────────────────────────────────────
        # if (steps - config["warmup_samples"] * config["num_envs"]) % config[
        #     "save_freq"
        # ] == 0:
        #     logger.nn_model_save(
        #         itr=steps, nn_model_saver_element=actor, prefix="actor"
        #     )
        #     logger.nn_model_save(
        #         itr=steps, nn_model_saver_element=critic, prefix="critic"
        #     )

        # if steps >= config["total_env_steps"]:
        #     break

        if steps >= next_save:
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=actor, prefix="actor"
            )
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=critic, prefix="critic"
            )
            while next_save <= steps:
                next_save += config["save_freq"]

    # ── final save ────────────────────────────────────────────────────────
    logger.nn_model_save(itr=steps, nn_model_saver_element=actor, prefix="actor")
    logger.nn_model_save(itr=steps, nn_model_saver_element=critic, prefix="critic")
    logger.close()


if __name__ == "__main__":
    args, cfg_env = sac_args()

    # Log path:  runs/<experiment>/<task>/sac/seed-000-YYYY-MM-DD-HH-MM-SS/
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "seed-" + str(args.seed).zfill(3)
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]  # "sac_single"
    args.log_dir = os.path.join(args.log_dir, args.task, algo, relpath)

    if not args.write_terminal:
        os.makedirs(args.log_dir, exist_ok=True)
        t_log = f"seed{args.seed}_terminal.log"
        e_log = f"seed{args.seed}_error.log"
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        with open(osp.join(args.log_dir, t_log), "w", encoding="utf-8") as f_out:
            sys.stdout = f_out
            with open(osp.join(args.log_dir, e_log), "w", encoding="utf-8") as f_err:
                sys.stderr = f_err
                main(args, cfg_env)
    else:
        main(args, cfg_env)