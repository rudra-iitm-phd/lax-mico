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
from distutils.util import strtobool
from typing import Any, Generic, Mapping, NamedTuple, Sequence, Tuple, TypeVar, Union

import flax
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
from brax.envs.wrappers import training as brax_training
from flax import nnx, struct
from jax import flatten_util
from mujoco_playground import registry, wrapper
from tensorboardX import SummaryWriter

from utils.acting import actor_step, actor_step_rep, wrap_env_for_training
from utils.buffer import RunningMeanStd, RunningStatistics, UniformSamplingQueue
from utils.logger import EpochLogger

# from utils.rep_models import (
#     EnsembleStateActionMetric,
#     EnsembleStateMetric,
#     MinStateActiontoStateMetric,
#     StateActionDiffuseMetric,
#     StateAsymmetricMetric,
# )
# use utils.metric_models to not get nan
from utils.metric_models_prev import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
from utils.models import (
    EnsembleCritic,
    EnsembleCriticRep,
    RepNet,
    SACGaussianActor,
    SACGaussianActorRep,
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
    state_metric: EnsembleStateMetric,
    state_metric_opt: nnx.Optimizer,
    state_action_metric: EnsembleStateActionMetric,
    state_action_metric_opt: nnx.Optimizer,
    min_state_action_to_state_metric: MinStateActiontoStateMetric,
    min_state_action_to_state_metric_opt: nnx.Optimizer,
    target_state_metric: EnsembleStateMetric,
    target_state_action_to_state_metric: MinStateActiontoStateMetric,
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
    alpha = jnp.exp(log_alpha())
    beta = 0.1
    grad_steps = config.grad_steps

    key, next_key = jax.random.split(key)
    next_act, next_log_prob = actor(next_obs, next_key)

    def critic_loss_fn(critic):
        q1_t, q2_t = target_critic(jnp.concatenate([next_obs, next_act], axis=-1))
        backup = reward + config.gamma * discount * (
            jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
        )
        backup = jax.lax.stop_gradient(backup)
        q1, q2 = critic(jnp.concatenate([obs, act], axis=-1))
        loss = jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2)
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True
    )(critic)
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.clip(g, -1.0, 1.0), critic_grads
    )
    critic_opt.update(critic, critic_grads)

    key, act_key = jax.random.split(key)

    def actor_loss_fn(actor):
        pi, log_pi = actor(obs, act_key)
        q1, q2 = critic(jnp.concatenate([obs, pi], axis=-1))
        loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return loss, jnp.mean(log_pi)

    (actor_loss, log_pi_mean), actor_grads = nnx.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor)
    # actor_opt.update(actor, actor_grads)

    # B X (obs_dim , act_dim, 1, obs_dim) | I am assuming reward is of shape (B, ), so I am expanding it.

    s, a, r, s_next = obs, act, reward[:, None], next_obs
    batch = jnp.concatenate([s, a, r, s_next], axis=-1)
    key, perm_key = jax.random.split(key)
    batch = jax.random.permutation(perm_key, batch)
    batch = batch[:, -1]

    obs_dim, act_dim = obs.shape[-1], act.shape[-1]
    # B X 1 X (obs_dim, act_dim, None, obs_dim)
    x, b, y, x_next = (
        batch[:, :obs_dim],
        batch[:, obs_dim : obs_dim + act_dim],
        batch[:, obs_dim + act_dim],
        batch[:, obs_dim + act_dim + 1 :],
    )

    r = r[:, -1]  ## shaping reward from (B, 1, 1) --> (B, 1)

    x, b, y, x_next = x[:, None, :], b[:, None, :], y[:, None], x_next[:, None, :]
    g_sx_next, g_xs_next = target_state_metric(s_next, x_next)
    u_target = jnp.maximum(g_sx_next, g_xs_next)
    lambda_target = jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

    def state_action_metric_loss_fn(state_action_metric: EnsembleStateActionMetric):
        d_sa_xb, d_xb_sa = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_current = jnp.maximum(d_sa_xb, d_xb_sa)
        loss = jnp.mean((lambda_current - lambda_target) ** 2)

        return loss

    lambda_loss, lambda_grads = nnx.value_and_grad(state_action_metric_loss_fn)(
        state_action_metric
    )
    state_action_metric_opt.update(state_action_metric, lambda_grads)

    def min_state_action_to_state_metric_loss_fn(
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
    ):

        d_sa_xb, d_xb_sa = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )

        # lambda_current = jax.lax.stop_gradient(jnp.maximum(d_sa_xb, d_xb_sa))

        h_sax, h_xbs = (
            min_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), x),
            min_state_action_to_state_metric(jnp.concatenate([x, b], axis=-1), s),
        )
        d_sa_xb, d_xb_sa = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )

        score_p1, score_p2 = (
            (h_sax - jax.lax.stop_gradient(d_sa_xb)) / beta,
            (h_xbs - jax.lax.stop_gradient(d_xb_sa)) / beta,
        )
        max_score = jax.lax.stop_gradient(jnp.maximum(score_p1.max(), score_p2.max()))
        p1 = (
            jnp.exp(score_p1 - max_score)
            - score_p1 * jnp.exp(-max_score)
            - jnp.exp(-max_score)
        )
        p2 = (
            jnp.exp(score_p2 - max_score)
            - score_p2 * jnp.exp(-max_score)
            - jnp.exp(-max_score)
        )
        loss = jnp.mean(p1) + jnp.mean(p2)
        return loss

    h_loss, h_grads = nnx.value_and_grad(min_state_action_to_state_metric_loss_fn)(
        min_state_action_to_state_metric
    )
    min_state_action_to_state_metric_opt.update(
        min_state_action_to_state_metric, h_grads
    )

    def state_metric_loss_fn(state_metric: EnsembleStateMetric):

        h_sax, h_xbs = (
            target_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), x),
            target_state_action_to_state_metric(jnp.concatenate([x, b], axis=-1), s),
        )

        h_sax, h_xbs = jax.lax.stop_gradient(h_sax), jax.lax.stop_gradient(h_xbs)
        g_sx, g_xs = state_metric(s, x)
        score_p1, score_p2 = (h_sax - g_sx) / beta, (h_xbs - g_xs) / beta
        max_score = jax.lax.stop_gradient(jnp.maximum(score_p1.max(), score_p2.max()))

        p1 = (
            jnp.exp(score_p1 - max_score)
            - score_p1 * jnp.exp(-max_score)
            - jnp.exp(-max_score)
        )

        p2 = (
            jnp.exp(score_p2 - max_score)
            - score_p2 * jnp.exp(-max_score)
            - jnp.exp(-max_score)
        )

        loss = jnp.mean(p1) + jnp.mean(p2)

        return loss

    g_loss, g_grads = nnx.value_and_grad(state_metric_loss_fn)(state_metric)
    state_metric_opt.update(state_metric, g_grads)

    def compute_state_diff(
        s: jnp.ndarray,
        x: jnp.ndarray,
        state_metric: EnsembleStateMetric,
    ):
        g_sx, g_xs = state_metric(s, x)
        u = jnp.maximum(g_sx, g_xs)
        return jnp.mean(u)

    def compute_state_action_diff(
        s: jnp.ndarray,
        pi: jnp.ndarray,
        x: jnp.ndarray,
        b: jnp.ndarray,
        state_action_metric: EnsembleStateActionMetric,
    ):

        d_spi_xb, d_xb_spi = state_action_metric(
            jnp.concatenate([s, pi], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        return jnp.mean(jnp.maximum(d_spi_xb, d_xb_spi))

    def state_grad_steps(i, carry):
        (x, state_metric, s_eq) = carry
        opt = optax.adam(config.lr)
        opt_state = opt.init(s_eq)
        grad_s = jax.grad(lambda s: compute_state_diff(s, x, state_metric))(s_eq)
        updates, opt_state = opt.update(grad_s, opt_state)
        s_eq = optax.apply_updates(s_eq, updates)
        return (x, state_metric, s_eq)

    # given a state s find the equivalent state
    (s, state_metric, s_eq) = nnx.fori_loop(
        0,
        grad_steps,
        state_grad_steps,
        (s, state_metric, jnp.zeros_like(s)),
    )

    def action_grad_steps(i, carry):
        (s, pi, x, state_action_metric, b_eq) = carry
        opt = optax.adam(config.lr)
        opt_state = opt.init(b_eq)
        grads_b = jax.grad(
            lambda b: jnp.mean(
                compute_state_action_diff(s, pi, x, b, state_action_metric)
            )
        )(b_eq)
        updates, opt_state = opt.update(grads_b, opt_state)
        b_eq = optax.apply_updates(b_eq, updates)
        return (s, pi, x, state_action_metric, b_eq)

    # key, act_key = jax.random.split(key)

    pi, _ = actor(s, act_key)

    # given state and it's action pair, and the equivalent state, find the equivalent action
    (s, pi, s_eq, state_action_metric, pi_eq) = nnx.fori_loop(
        0,
        grad_steps,
        action_grad_steps,
        (s, pi, s_eq, state_action_metric, jnp.zeros_like(pi)),
    )

    def act_match_loss_fn(actor: SACGaussianActorRep):
        actions, _ = actor(s_eq, act_key)
        return jnp.mean(
            optax.huber_loss(actions, jax.lax.stop_gradient(pi_eq), delta=1.0)
        )

    matching_loss, actor_matching_loss = nnx.value_and_grad(act_match_loss_fn)(actor)

    actor_grads_total = jax.tree_util.tree_map(
        lambda a, b: a + b, actor_grads, actor_matching_loss
    )
    actor_grads_total = jax.tree_util.tree_map(
        lambda g: jnp.clip(g, -1.0, 1.0), actor_grads_total
    )
    actor_opt.update(actor, actor_grads_total)

    def alpha_loss_fn(log_alpha):
        a = jnp.exp(log_alpha())
        loss = jnp.mean(
            -a * (jax.lax.stop_gradient(log_pi_mean) + config.target_entropy)
        )
        return loss

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_opt.update(log_alpha, alpha_grads)

    polyak_update(target_critic, critic, config.update_tau)
    polyak_update(target_state_metric, state_metric, config.update_tau)
    polyak_update(
        target_state_action_to_state_metric,
        min_state_action_to_state_metric,
        config.update_tau,
    )

    return (
        critic_loss,
        actor_loss,
        alpha_loss,
        alpha,
        log_pi_mean,
        q1_mean,
        q2_mean,
        lambda_loss,
        g_loss,
        h_loss,
        matching_loss,
    )


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    state_metric: EnsembleStateMetric,
    state_metric_opt: nnx.Optimizer,
    state_action_metric: EnsembleStateActionMetric,
    state_action_metric_opt: nnx.Optimizer,
    min_state_action_to_state_metric: MinStateActiontoStateMetric,
    min_state_action_to_state_metric_opt: nnx.Optimizer,
    target_state_metric: EnsembleStateMetric,
    target_state_action_to_state_metric: MinStateActiontoStateMetric,
    actor: SACGaussianActor,
    actor_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    config,
    key: jnp.ndarray,
):

    num_steps = config.log_freq

    def body_fun(i, carry):
        key, env_state, buffer_state, running_state, obs_normalizer, models, val = carry
        (
            state_metric,
            state_metric_opt,
            state_action_metric,
            state_action_metric_opt,
            min_state_action_to_state_metric,
            min_state_action_to_state_metric_opt,
            target_state_metric,
            target_state_action_to_state_metric,
            actor,
            actor_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
        ) = models

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env, env_state, actor, obs_normalizer, env_key, extra_fields=("truncation",)
        )
        buffer_state = buffer.insert(buffer_state, transition)
        obs_normalizer = obs_normalizer.update(transition.observation)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        def do_train(j, carry):
            key, env_state, buffer_state, obs_normalizer, models, _ = carry
            (
                state_metric,
                state_metric_opt,
                state_action_metric,
                state_action_metric_opt,
                min_state_action_to_state_metric,
                min_state_action_to_state_metric_opt,
                target_state_metric,
                target_state_action_to_state_metric,
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = sac_train_step(
                state_metric,
                state_metric_opt,
                state_action_metric,
                state_action_metric_opt,
                min_state_action_to_state_metric,
                min_state_action_to_state_metric_opt,
                target_state_metric,
                target_state_action_to_state_metric,
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
                state_metric,
                state_metric_opt,
                state_action_metric,
                state_action_metric_opt,
                min_state_action_to_state_metric,
                min_state_action_to_state_metric_opt,
                target_state_metric,
                target_state_action_to_state_metric,
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            )

            return (key, env_state, buffer_state, obs_normalizer, models, val)

        init_val = (jnp.zeros((), jnp.float32),) * 11
        models = (
            state_metric,
            state_metric_opt,
            state_action_metric,
            state_action_metric_opt,
            min_state_action_to_state_metric,
            min_state_action_to_state_metric_opt,
            target_state_metric,
            target_state_action_to_state_metric,
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
        (
            state_metric,
            state_metric_opt,
            state_action_metric,
            state_action_metric_opt,
            min_state_action_to_state_metric,
            min_state_action_to_state_metric_opt,
            target_state_metric,
            target_state_action_to_state_metric,
            actor,
            actor_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
        ) = models
        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            (
                state_metric,
                state_metric_opt,
                state_action_metric,
                state_action_metric_opt,
                min_state_action_to_state_metric,
                min_state_action_to_state_metric_opt,
                target_state_metric,
                target_state_action_to_state_metric,
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            ),
            val,
        )

    init_val = (jnp.zeros((), jnp.float32),) * 11
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        (
            state_metric,
            state_metric_opt,
            state_action_metric,
            state_action_metric_opt,
            min_state_action_to_state_metric,
            min_state_action_to_state_metric_opt,
            target_state_metric,
            target_state_action_to_state_metric,
            actor,
            actor_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
        ),
        init_val,
    )

    (_, env_state, buffer_state, running_state, obs_normalizer, models, val) = (
        nnx.fori_loop(0, num_steps, body_fun, init_carry)
    )

    (
        state_metric,
        state_metric_opt,
        state_action_metric,
        state_action_metric_opt,
        min_state_action_to_state_metric,
        min_state_action_to_state_metric_opt,
        target_state_metric,
        target_state_action_to_state_metric,
        actor,
        actor_opt,
        critic,
        critic_opt,
        target_critic,
        log_alpha,
        alpha_opt,
    ) = models

    return *val, env_state, running_state, obs_normalizer, buffer_state, num_steps


""" Copied from claude """


def prefill_buffer(
    key, env, env_state, buffer_state, policy, buffer, obs_normalizer, num_itr: int
):
    """
    Collect `num_itr` transitions before training begins.

    Uses jax.lax.scan (not a Python loop) so the warmup is JIT-compiled.
    The policy has random initial weights so actions are effectively random.
    """

    def body(carry, _):
        key, env_state, buffer_state, obs_normalizer = carry
        key, subkey = jax.random.split(key)
        n_state, transition = actor_step(
            env=env,
            env_state=env_state,
            policy=policy,
            obs_normalizer=obs_normalizer,
            key=subkey,
            extra_fields=("truncation",),
        )
        buffer_state = buffer.insert(buffer_state, transition)
        obs_normalizer = obs_normalizer.update(transition.observation)
        return (key, n_state, buffer_state, obs_normalizer), ()

    jitted_body = jax.jit(body)
    (_, env_state, buffer_state, obs_normalizer), () = jax.lax.scan(
        jitted_body,
        (key, env_state, buffer_state, obs_normalizer),
        (),
        length=num_itr,
    )
    return env_state, buffer_state, obs_normalizer


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
            "vis_feq": args.vis_freq,
            "n_vis_frames": args.n_vis_frames,
            "num_envs": args.num_envs,
            "grad_steps": args.grad_steps,
        }
    )

    # ── environment ───────────────────────────────────────────────────────
    prng_key, env_key = jax.random.split(prng_key)
    env_key = jax.random.split(env_key, config["num_envs"])

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
    config["target_entropy"] = float(-act_dim)

    # Freeze config into an immutable Flax struct (required for nnx.jit stability)
    config_data = make_static_config_from_dict("SACConfig", config)()

    # # ── networks ──────────────────────────────────────────────────────────
    # actor = SACGaussianActor(
    #     rngs=rngs,
    #     obs_dim=obs_dim,
    #     act_dim=act_dim,
    #     hidden_size=config["hidden_size"],
    # )
    # actor_opt = nnx.Optimizer(
    #     model=actor,
    #     tx=optax.chain(
    #         optax.clip_by_global_norm(config["max_grad_norm"]),
    #         optax.adam(learning_rate=config["lr"]),
    #     ),
    #     wrt = nnx.Param
    # )

    # critic = EnsembleCritic(
    #     rngs=rngs,
    #     obs_dim=obs_dim,
    #     act_dim=act_dim,
    #     hidden_size=config["hidden_size"],
    # )
    # critic_opt = nnx.Optimizer(
    #     model=critic,
    #     tx=optax.chain(
    #         optax.clip_by_global_norm(config["max_grad_norm"]),
    #         optax.adam(learning_rate=config["lr"]),
    #     ),
    #     wrt = nnx.Param
    # )
    # target_critic = deepcopy(critic)   # separate copy for Polyak updates

    state_metric = EnsembleStateMetric(
        rngs=rngs, obs_dim=obs_dim, hidden_size=config["hidden_size"]
    )

    state_metric_opt = nnx.Optimizer(
        model=state_metric,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
            # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        ),
        wrt=nnx.Param,
    )

    state_action_metric = EnsembleStateActionMetric(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )

    state_action_metric_opt = nnx.Optimizer(
        model=state_action_metric,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
            # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        ),
        wrt=nnx.Param,
    )

    min_state_action_to_state_metric = MinStateActiontoStateMetric(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )

    min_state_action_to_state_metric_opt = nnx.Optimizer(
        model=min_state_action_to_state_metric,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
            # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        ),
        wrt=nnx.Param,
    )

    target_state_metric = deepcopy(state_metric)

    target_state_action_to_state_metric = deepcopy(min_state_action_to_state_metric)

    actor = SACGaussianActor(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )

    actor_opt = nnx.Optimizer(
        model=actor,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
            # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        ),
        wrt=nnx.Param,
    )

    critic = EnsembleCritic(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )

    critic_opt = nnx.Optimizer(
        model=critic,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
            # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        ),
        wrt=nnx.Param,
    )

    target_critic = deepcopy(critic)

    log_alpha = Scalar(float(jnp.log(config["init_temperature"])))
    alpha_opt = nnx.Optimizer(
        model=log_alpha,
        tx=optax.adam(learning_rate=config["lr"]),
        wrt=nnx.Param,
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

    env_state, buffer_state, obs_normalizer = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        policy=actor,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        num_itr=config["warmup_samples"],
    )
    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start SAC training")
    steps = buffer.size(buffer_state)

    while steps < config["total_env_steps"]:
        prng_key, subkey = jax.random.split(prng_key)

        # train_n_steps compiles on first call (~60 s), then runs at GPU speed
        # val = train_n_steps(
        #     env=env,
        #     env_state=env_state,
        #     buffer_state=buffer_state,
        #     buffer=buffer,
        #     running_state=running_state,
        #     actor=actor,            actor_opt=actor_opt,
        #     critic=critic,          critic_opt=critic_opt,
        #     target_critic=target_critic,
        #     log_alpha=log_alpha,    alpha_opt=alpha_opt,
        #     config=config_data,
        #     key=subkey,
        # )

        val = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
            obs_normalizer=obs_normalizer,
            state_metric=state_metric,
            state_metric_opt=state_metric_opt,
            state_action_metric=state_action_metric,
            state_action_metric_opt=state_action_metric_opt,
            min_state_action_to_state_metric=min_state_action_to_state_metric,
            min_state_action_to_state_metric_opt=min_state_action_to_state_metric_opt,
            target_state_metric=target_state_metric,
            target_state_action_to_state_metric=target_state_action_to_state_metric,
            actor=actor,
            actor_opt=actor_opt,
            critic=critic,
            critic_opt=critic_opt,
            target_critic=target_critic,
            log_alpha=log_alpha,
            alpha_opt=alpha_opt,
            config=config_data,
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
            lambda_loss,
            g_loss,
            h_loss,
            matching_loss,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            num_steps,
        ) = val

        steps += num_steps
        logger.logged = False

        # ── logging (mirrors gpe key naming exactly) ──────────────────────
        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_critic", critic_loss.item())
        logger.log_tabular("Loss/Loss_actor", actor_loss.item())
        logger.log_tabular("Loss/Loss_alpha", alpha_loss.item())
        logger.log_tabular("Loss/Loss_state_action_metric", lambda_loss.item())
        logger.log_tabular("Loss/Loss_state_metric", g_loss.item())
        logger.log_tabular("Loss/Loss_state_action_to_state", h_loss.item())
        logger.log_tabular("Loss/Matching_loss", matching_loss.item())

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
        logger.log_tabular(
            "Norm/state_action_metric_model",
            get_tree_norm(nnx.state(state_action_metric, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/state_metric_model", get_tree_norm(nnx.state(state_metric, nnx.Param))
        )
        logger.log_tabular(
            "Norm/state_action_state_metric_model",
            get_tree_norm(nnx.state(min_state_action_to_state_metric, nnx.Param)),
        )

        logger.log_tabular(
            "Eval/Return",
            running_state.reward_state.data.sum() / config["eval_episode_freq"],
        )

        logger.dump_tabular()

        # ── periodic checkpoint ───────────────────────────────────────────
        if (steps - config["warmup_samples"]) % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=actor, prefix="actor"
            )
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=critic, prefix="critic"
            )

        if steps >= config["total_env_steps"]:
            break

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
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)

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
