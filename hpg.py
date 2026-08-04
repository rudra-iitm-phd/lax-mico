import argparse
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
from utils.buffer import RunningMeanStd, RunningStatistics, UniformSamplingQueue
from utils.logger import EpochLogger
from utils.models import (
    EnsembleCritic,
    SACGaussianActor,
    Scalar,
    get_tree_norm,
)
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args

# ===========================================================================
# Section 0 - config
# ===========================================================================
# This extends the base SAC config with the hyper-parameters used by the
# Deep Homomorphic Policy Gradient (DHPG) paper / repo
# (https://github.com/sahandrez/homomorphic_policy_gradient) for its
# *stochastic* variant on *state* observations (`agent=stochastichpg`,
# `pixel_obs=false`).

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
    # ---- DHPG-specific -----------------------------------------------
    "feature_dim": 50,  # abstract state dim, used unless matching_dims
    "matching_dims": False,  # if True, abstract state dim == obs dim
    "homomorphic_coef": 1.0,  # weight of the lax-bisimulation loss
    "lifting_weight": 100.0,  # weight of the actor <-> abstract-actor lifting loss
    "lifting_repeat_obs": 100,  # number of action samples used to estimate lifting loss
    "alpha_lr": None,  # falls back to `lr` if None
    "hpg_critic_target_tau": 0.01,  # separate (paper uses 0.01, not 0.005) tau for HPG branch
    "update_every_steps": 2,  # delay actor / abstract-actor / target updates, as in the reference
}


def add_hpg_args(args):
    """Adds the DHPG-specific CLI flags on top of whatever `sac_args()` parsed.

    Kept separate from `utils.utils.sac_args` (whose source isn't touched
    here) so this file is self-contained. Any flag not passed on the CLI
    falls back to the defaults in `default_cfg`.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feature_dim", type=int, default=default_cfg["feature_dim"])
    parser.add_argument(
        "--matching_dims",
        type=lambda x: bool(strtobool(x)),
        default=default_cfg["matching_dims"],
    )
    parser.add_argument(
        "--homomorphic_coef", type=float, default=default_cfg["homomorphic_coef"]
    )
    parser.add_argument(
        "--lifting_weight", type=float, default=default_cfg["lifting_weight"]
    )
    parser.add_argument(
        "--lifting_repeat_obs", type=int, default=default_cfg["lifting_repeat_obs"]
    )
    parser.add_argument("--alpha_lr", type=float, default=None)
    parser.add_argument(
        "--hpg_critic_target_tau",
        type=float,
        default=default_cfg["hpg_critic_target_tau"],
    )
    parser.add_argument(
        "--update_every_steps", type=int, default=default_cfg["update_every_steps"]
    )
    known, _ = parser.parse_known_args()
    for k, v in vars(known).items():
        setattr(args, k, v)
    return args


# ===========================================================================
# Section 1 - new nnx modules: state/action encoders, transition & reward
# models. `SACGaussianActor` and `EnsembleCritic` are reused as-is (from
# utils.models) for the abstract actor / abstract critic, just instantiated
# with the abstract state/action dims instead of the real obs/act dims.
# ===========================================================================


class StateEncoder(nnx.Module):
    """phi: S -> Z. Maps the (already obs-normalized) state to the abstract
    (lax-bisimulation) state used by the MDP homomorphism."""

    def __init__(
        self, obs_dim: int, abstract_state_dim: int, hidden_size: int, *, rngs: nnx.Rngs
    ):
        self.l1 = nnx.Linear(obs_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.l3 = nnx.Linear(hidden_size, abstract_state_dim, rngs=rngs)

    def __call__(self, obs):
        x = jax.nn.relu(self.l1(obs))
        x = jax.nn.relu(self.l2(x))
        return self.l3(x)


class ActionEncoder(nnx.Module):
    """psi: S x A -> Z_A. State-dependent action embedding into the
    abstract action space (tanh-bounded, as in the reference implementation)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        abstract_action_dim: int,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.l1 = nnx.Linear(obs_dim + act_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.l3 = nnx.Linear(hidden_size, abstract_action_dim, rngs=rngs)

    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = jax.nn.relu(self.l1(x))
        x = jax.nn.relu(self.l2(x))
        return jnp.tanh(self.l3(x))


class ProbabilisticTransitionModel(nnx.Module):
    """Abstract transition model T: Z x Z_A -> N(mu, sigma). Used both for
    the lax-bisimulation loss and as the (probabilistic) reward-model input."""

    def __init__(
        self,
        abstract_state_dim: int,
        abstract_action_dim: int,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
        min_sigma: float = 1e-4,
        max_sigma: float = 1e1,
    ):
        self.l1 = nnx.Linear(
            abstract_state_dim + abstract_action_dim, hidden_size, rngs=rngs
        )
        self.l2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.mu = nnx.Linear(hidden_size, abstract_state_dim, rngs=rngs)
        self.sigma = nnx.Linear(hidden_size, abstract_state_dim, rngs=rngs)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, z, abstract_action):
        x = jnp.concatenate([z, abstract_action], axis=-1)
        x = jax.nn.relu(self.l1(x))
        x = jax.nn.relu(self.l2(x))
        mu = self.mu(x)
        sigma = jax.nn.sigmoid(self.sigma(x))
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        return mu, sigma

    def sample_prediction(self, z, abstract_action, key):
        mu, sigma = self(z, abstract_action)
        eps = jax.random.normal(key, mu.shape)
        return mu + sigma * eps


class RewardPredictor(nnx.Module):
    """R: Z -> reward. Predicts the reward from the predicted next
    abstract state (as in the reference implementation)."""

    def __init__(self, abstract_state_dim: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(abstract_state_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.l3 = nnx.Linear(hidden_size, 1, rngs=rngs)

    def __call__(self, z):
        x = jax.nn.relu(self.l1(z))
        x = jax.nn.relu(self.l2(x))
        return self.l3(x)


# ===========================================================================
# Section 2 - polyak update + lax-bisimulation / transition / reward losses
# ===========================================================================


def polyak_update(target_model, curr_model, tau: float):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


def _huber(x, delta: float = 1.0):
    """Elementwise smooth-L1 / huber loss with beta=delta=1, matching
    torch.nn.functional.smooth_l1_loss's default."""
    abs_x = jnp.abs(x)
    quad = jnp.minimum(abs_x, delta)
    lin = abs_x - quad
    return 0.5 * quad**2 + delta * lin


def get_lax_bisim(transition_model, z, abstract_action, reward, discount, key):
    """Computes the lax-bisimulation consistency loss:
        (||z_i - z_j|| - (|r_i - r_j| + gamma * ||T(z_i,a_i) - T(z_j,a_j)||))^2
    following Rezaei-Shoshtari et al., 2022 (Eq. for the lax bisimulation
    metric), using a random permutation of the batch as the second sample.
    """
    batch_size = z.shape[0]
    perm = jax.random.permutation(key, batch_size)

    z2 = z[perm]
    reward2 = reward[perm]

    # Stop gradient through the transition model here: this term only
    # trains the encoders to be consistent with a *frozen* transition model;
    # the transition model itself is trained via `get_transition_reward_loss`.
    mu1, sigma1 = transition_model(z, abstract_action)
    mu1 = jax.lax.stop_gradient(mu1)
    sigma1 = jax.lax.stop_gradient(sigma1)
    mu2 = mu1[perm]
    sigma2 = sigma1[perm]

    z_dist = jnp.mean(_huber(z - z2), axis=-1)
    r_dist = _huber(reward - reward2)
    transition_dist = jnp.mean(
        jnp.sqrt((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2 + 1e-8), axis=-1
    )

    lax_bisimilarity = r_dist + discount * transition_dist
    lax_bisim_loss = jnp.mean((z_dist - lax_bisimilarity) ** 2)
    return lax_bisim_loss


def get_transition_reward_loss(
    transition_model, reward_predictor, z, abstract_action, reward, next_z, key
):
    mu, sigma = transition_model(z, abstract_action)
    diff = (mu - jax.lax.stop_gradient(next_z)) / sigma
    transition_loss = jnp.mean(0.5 * diff**2 + jnp.log(sigma))

    pred_next_latent = transition_model.sample_prediction(z, abstract_action, key)
    pred_reward = reward_predictor(pred_next_latent)
    reward_loss = jnp.mean((pred_reward.squeeze(-1) - reward) ** 2)

    return transition_loss, reward_loss


# ===========================================================================
# Section 3 - HPG train step (replaces `sac_train_step`)
# ===========================================================================


def hpg_train_step(
    # real (grounded) MDP actor-critic
    actor: SACGaussianActor,
    actor_opt: nnx.Optimizer,
    actor_target: SACGaussianActor,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    # MDP homomorphism map
    state_encoder: StateEncoder,
    state_encoder_opt: nnx.Optimizer,
    action_encoder: ActionEncoder,
    action_encoder_opt: nnx.Optimizer,
    transition_model: ProbabilisticTransitionModel,
    transition_opt: nnx.Optimizer,
    reward_predictor: RewardPredictor,
    reward_opt: nnx.Optimizer,
    # abstract (homomorphic image) actor-critic
    abstract_actor: SACGaussianActor,
    abstract_actor_opt: nnx.Optimizer,
    abstract_actor_target: SACGaussianActor,
    abstract_critic: EnsembleCritic,
    abstract_critic_opt: nnx.Optimizer,
    abstract_critic_target: EnsembleCritic,
    data: Transition,
    config,
    key: jnp.ndarray,
    step_idx: jnp.ndarray,
):
    # 1.0 on steps where the actor / abstract actor / target networks should
    # update, 0.0 otherwise - mirrors `if step % update_every_steps == 0` in
    # the reference (TD3-style delayed policy update).
    should_update = (jnp.mod(step_idx, config.update_every_steps) == 0).astype(
        jnp.float32
    )

    obs = data.observation
    act = data.action
    reward = data.reward
    discount = data.discount
    next_obs = data.next_observation
    alpha = jnp.exp(log_alpha())

    (
        key,
        actor_key,
        next_actor_key,
        bisim_key,
        tr_key,
        abs_next_key,
        abs_actor_key,
        lift_key1,
        lift_key2,
    ) = jax.random.split(key, 9)

    # -----------------------------------------------------------------
    # 1) Joint update of: real critic, state encoder, action encoder,
    #    transition model, reward predictor. This mirrors `update_critic`
    #    in the reference `stochastic_hpg.py`, where a single backward pass
    #    updates the critic + the homomorphism map together.
    # -----------------------------------------------------------------
    def joint_loss_fn(models):
        (
            critic_m,
            state_encoder_m,
            action_encoder_m,
            transition_model_m,
            reward_predictor_m,
        ) = models

        z = state_encoder_m(obs)
        next_z = jax.lax.stop_gradient(state_encoder_m(next_obs))
        abstract_action = action_encoder_m(obs, act)

        lax_bisim_loss = get_lax_bisim(
            transition_model_m, z, abstract_action, reward, discount, bisim_key
        )
        transition_loss, reward_loss = get_transition_reward_loss(
            transition_model_m,
            reward_predictor_m,
            z,
            abstract_action,
            reward,
            next_z,
            tr_key,
        )

        next_act, next_log_prob = actor_target(next_obs, next_actor_key)
        q1_t, q2_t = target_critic(jnp.concatenate([next_obs, next_act], axis=-1))
        backup = reward + config.gamma * discount * (
            jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
        )
        backup = jax.lax.stop_gradient(backup)
        q1, q2 = critic_m(jnp.concatenate([obs, act], axis=-1))
        critic_loss = jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2)

        total = (
            critic_loss
            + config.homomorphic_coef * lax_bisim_loss
            + transition_loss
            + reward_loss
        )
        aux = (
            critic_loss,
            lax_bisim_loss,
            transition_loss,
            reward_loss,
            jnp.mean(q1),
            jnp.mean(q2),
        )
        return total, aux

    models_in = (
        critic,
        state_encoder,
        action_encoder,
        transition_model,
        reward_predictor,
    )
    (joint_loss, aux), grads = nnx.value_and_grad(joint_loss_fn, has_aux=True)(
        models_in
    )
    critic_grads, se_grads, ae_grads, tm_grads, rp_grads = grads
    critic_opt.update(critic, critic_grads)
    state_encoder_opt.update(state_encoder, se_grads)
    action_encoder_opt.update(action_encoder, ae_grads)
    transition_opt.update(transition_model, tm_grads)
    reward_opt.update(reward_predictor, rp_grads)

    critic_loss, lax_bisim_loss, transition_loss, reward_loss, q1_mean, q2_mean = aux

    # -----------------------------------------------------------------
    # 2) Abstract critic update. Uses frozen (stop-gradient) encoder /
    #    action-encoder outputs, and the *target* abstract actor for the
    #    bootstrapped next abstract action - mirrors `update_abstract_critic`.
    # -----------------------------------------------------------------
    z_sg = jax.lax.stop_gradient(state_encoder(obs))
    next_z_sg = jax.lax.stop_gradient(state_encoder(next_obs))
    abstract_action_sg = jax.lax.stop_gradient(action_encoder(obs, act))

    def abstract_critic_loss_fn(abstract_critic_m):
        next_abs_act, next_abs_log_prob = abstract_actor_target(next_z_sg, abs_next_key)
        q1_t, q2_t = abstract_critic_target(
            jnp.concatenate([next_z_sg, next_abs_act], axis=-1)
        )
        backup = reward + config.gamma * discount * (
            jnp.minimum(q1_t, q2_t) - alpha * next_abs_log_prob
        )
        backup = jax.lax.stop_gradient(backup)
        q1, q2 = abstract_critic_m(jnp.concatenate([z_sg, abstract_action_sg], axis=-1))
        loss = jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2)
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (abs_critic_loss, (abs_q1_mean, abs_q2_mean)), abs_critic_grads = (
        nnx.value_and_grad(abstract_critic_loss_fn, has_aux=True)(abstract_critic)
    )
    abstract_critic_opt.update(abstract_critic, abs_critic_grads)

    # diagnostic: value equivalence between the grounded and abstract critics
    q1_c, q2_c = critic(jnp.concatenate([obs, act], axis=-1))
    q1_a, q2_a = abstract_critic(jnp.concatenate([z_sg, abstract_action_sg], axis=-1))
    value_equivalence = jnp.mean(
        jnp.abs(jnp.minimum(q1_c, q2_c) - jnp.minimum(q1_a, q2_a))
    )

    # -----------------------------------------------------------------
    # 3) Real actor + alpha update (standard SAC actor step).
    # -----------------------------------------------------------------
    def actor_loss_fn(actor_m):
        pi, log_pi = actor_m(obs, actor_key)
        q1, q2 = critic(jnp.concatenate([obs, pi], axis=-1))
        loss = should_update * jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return loss, jnp.mean(log_pi)

    (actor_loss, log_pi_mean), actor_grads = nnx.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor)
    actor_opt.update(actor, actor_grads)

    def alpha_loss_fn(log_alpha_m):
        a = jnp.exp(log_alpha_m())
        loss = should_update * jnp.mean(
            -a * (jax.lax.stop_gradient(log_pi_mean) + config.target_entropy)
        )
        return loss

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_opt.update(log_alpha, alpha_grads)

    # -----------------------------------------------------------------
    # 4) Abstract actor update via the Homomorphic Policy Gradient theorem,
    #    plus the policy-lifting loss that pulls the abstract actor towards
    #    the (target) grounded actor pushed through the action encoder -
    #    mirrors `update_abstract_actor(..., lift_towards_actor=True)` under
    #    `hpg_update_type="double"` (the variant used for state observations).
    # -----------------------------------------------------------------
    z_for_actor = jax.lax.stop_gradient(state_encoder(obs))
    batch_size = obs.shape[0]
    repeat = config.lifting_repeat_obs

    obs_rep = jnp.repeat(obs[:, None, :], repeat, axis=1).reshape(
        batch_size * repeat, -1
    )
    z_rep = jax.lax.stop_gradient(state_encoder(obs_rep))
    pi_target, _ = actor_target(obs_rep, lift_key1)
    pi_transformed = jax.lax.stop_gradient(action_encoder(obs_rep, pi_target))

    def abstract_actor_loss_fn(abstract_actor_m):
        abs_pi, abs_log_pi = abstract_actor_m(z_for_actor, abs_actor_key)
        q1a, q2a = abstract_critic(jnp.concatenate([z_for_actor, abs_pi], axis=-1))
        qa = jnp.minimum(q1a, q2a)
        abstract_actor_loss = jnp.mean(jax.lax.stop_gradient(alpha) * abs_log_pi - qa)

        abs_pi_rep, _ = abstract_actor_m(z_rep, lift_key2)
        pi_t_r = pi_transformed.reshape(batch_size, repeat, -1)
        abs_pi_r = abs_pi_rep.reshape(batch_size, repeat, -1)
        lifting_loss = jnp.mean((pi_t_r.mean(axis=1) - abs_pi_r.mean(axis=1)) ** 2)
        lifting_loss += jnp.mean((pi_t_r.std(axis=1) - abs_pi_r.std(axis=1)) ** 2)

        total = should_update * (
            abstract_actor_loss + config.lifting_weight * lifting_loss
        )
        return total, (abstract_actor_loss, lifting_loss)

    (abs_actor_total, (abs_actor_loss, lifting_loss)), abs_actor_grads = (
        nnx.value_and_grad(abstract_actor_loss_fn, has_aux=True)(abstract_actor)
    )
    abstract_actor_opt.update(abstract_actor, abs_actor_grads)

    # -----------------------------------------------------------------
    # 5) Target network updates (critic, actor, abstract critic, abstract actor)
    # -----------------------------------------------------------------
    tau = should_update * config.hpg_critic_target_tau
    polyak_update(target_critic, critic, tau)
    polyak_update(actor_target, actor, tau)
    polyak_update(abstract_critic_target, abstract_critic, tau)
    polyak_update(abstract_actor_target, abstract_actor, tau)

    return (
        critic_loss,
        actor_loss,
        alpha_loss,
        alpha,
        log_pi_mean,
        q1_mean,
        q2_mean,
        lax_bisim_loss,
        transition_loss,
        reward_loss,
        abs_critic_loss,
        abs_actor_loss,
        lifting_loss,
        value_equivalence,
        abs_q1_mean,
        abs_q2_mean,
    )


# ===========================================================================
# Section 4 - rollout + train loop (mirrors `train_n_steps`)
# ===========================================================================


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
    actor_target,
    critic,
    critic_opt,
    target_critic,
    log_alpha,
    alpha_opt,
    state_encoder,
    state_encoder_opt,
    action_encoder,
    action_encoder_opt,
    transition_model,
    transition_opt,
    reward_predictor,
    reward_opt,
    abstract_actor,
    abstract_actor_opt,
    abstract_actor_target,
    abstract_critic,
    abstract_critic_opt,
    abstract_critic_target,
    config,
    key,
):
    num_steps = config.log_freq
    n_val_fields = 16  # must match the tuple length returned by hpg_train_step

    def body_fun(i, carry):
        (
            key,
            env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            models,
            val,
        ) = carry
        (
            actor,
            actor_opt,
            actor_target,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            transition_model,
            transition_opt,
            reward_predictor,
            reward_opt,
            abstract_actor,
            abstract_actor_opt,
            abstract_actor_target,
            abstract_critic,
            abstract_critic_opt,
            abstract_critic_target,
        ) = models

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env,
            env_state,
            actor,
            obs_normalizer,
            env_key,
            extra_fields=("truncation",),
        )
        buffer_state = buffer.insert(buffer_state, transition)
        obs_normalizer = obs_normalizer.update(transition.observation)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        def do_train(j, carry):
            key, env_state, buffer_state, obs_normalizer, models, _ = carry
            (
                actor,
                actor_opt,
                actor_target,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                transition_model,
                transition_opt,
                reward_predictor,
                reward_opt,
                abstract_actor,
                abstract_actor_opt,
                abstract_actor_target,
                abstract_critic,
                abstract_critic_opt,
                abstract_critic_target,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = hpg_train_step(
                actor,
                actor_opt,
                actor_target,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                transition_model,
                transition_opt,
                reward_predictor,
                reward_opt,
                abstract_actor,
                abstract_actor_opt,
                abstract_actor_target,
                abstract_critic,
                abstract_critic_opt,
                abstract_critic_target,
                batch,
                config,
                train_key,
                i,  # outer env-step index, used to gate delayed actor/target updates
            )
            models = (
                actor,
                actor_opt,
                actor_target,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                transition_model,
                transition_opt,
                reward_predictor,
                reward_opt,
                abstract_actor,
                abstract_actor_opt,
                abstract_actor_target,
                abstract_critic,
                abstract_critic_opt,
                abstract_critic_target,
            )
            return (key, env_state, buffer_state, obs_normalizer, models, val)

        init_val = (jnp.zeros((), jnp.float32),) * n_val_fields
        models = (
            actor,
            actor_opt,
            actor_target,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            transition_model,
            transition_opt,
            reward_predictor,
            reward_opt,
            abstract_actor,
            abstract_actor_opt,
            abstract_actor_target,
            abstract_critic,
            abstract_critic_opt,
            abstract_critic_target,
        )
        key, _, buffer_state, obs_normalizer, models, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, n_env_state, buffer_state, obs_normalizer, models, init_val),
        )
        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            models,
            val,
        )

    init_val = (jnp.zeros((), jnp.float32),) * n_val_fields
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        (
            actor,
            actor_opt,
            actor_target,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            transition_model,
            transition_opt,
            reward_predictor,
            reward_opt,
            abstract_actor,
            abstract_actor_opt,
            abstract_actor_target,
            abstract_critic,
            abstract_critic_opt,
            abstract_critic_target,
        ),
        init_val,
    )

    (_, env_state, buffer_state, running_state, obs_normalizer, models, val) = (
        nnx.fori_loop(0, num_steps, body_fun, init_carry)
    )

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        num_steps,
    )


""" Copied from claude """


def prefill_buffer(
    key, env, env_state, buffer_state, policy, buffer, obs_normalizer, num_itr: int
):
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
# Section 10 - MAIN
# ===========================================================================


def main(args, cfg_env=None):
    # ── reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
    )

    # ── device ────────────────────────────────────────────────────────────
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
            # DHPG-specific (this run is always state-based, i.e. pixel_obs=false)
            "feature_dim": args.feature_dim,
            "matching_dims": args.matching_dims,
            "homomorphic_coef": args.homomorphic_coef,
            "lifting_weight": args.lifting_weight,
            "lifting_repeat_obs": args.lifting_repeat_obs,
            "hpg_critic_target_tau": args.hpg_critic_target_tau,
        }
    )

    # ── environment (state / vector observations, NOT pixels) ──────────────
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

    config["target_entropy"] = float(-act_dim)

    # abstract (homomorphic) state/action dims - matching_dims=True keeps the
    # abstract state space the same size as the real (vector) obs space,
    # exactly as in the reference repo for state observations.
    abstract_state_dim = obs_dim if config["matching_dims"] else config["feature_dim"]
    abstract_action_dim = act_dim  # stochastic DHPG always matches action dims
    alpha_lr = config["alpha_lr"] if config["alpha_lr"] is not None else config["lr"]
    config["alpha_lr"] = alpha_lr

    config_data = make_static_config_from_dict("HPGConfig", config)()

    # ── real (grounded) networks ────────────────────────────────────────────
    actor = SACGaussianActor(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )
    actor_target = deepcopy(actor)
    actor_opt = nnx.Optimizer(
        model=actor,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
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
        ),
        wrt=nnx.Param,
    )
    target_critic = deepcopy(critic)

    log_alpha = Scalar(float(jnp.log(config["init_temperature"])))
    alpha_opt = nnx.Optimizer(
        model=log_alpha, tx=optax.adam(learning_rate=alpha_lr), wrt=nnx.Param
    )

    # ── MDP homomorphism map: state/action encoders, transition, reward ────
    state_encoder = StateEncoder(
        obs_dim=obs_dim,
        abstract_state_dim=abstract_state_dim,
        hidden_size=config["hidden_size"],
        rngs=rngs,
    )
    state_encoder_opt = nnx.Optimizer(
        model=state_encoder, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    action_encoder = ActionEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        abstract_action_dim=abstract_action_dim,
        hidden_size=config["hidden_size"],
        rngs=rngs,
    )
    action_encoder_opt = nnx.Optimizer(
        model=action_encoder, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    transition_model = ProbabilisticTransitionModel(
        abstract_state_dim=abstract_state_dim,
        abstract_action_dim=abstract_action_dim,
        hidden_size=config["hidden_size"],
        rngs=rngs,
    )
    transition_opt = nnx.Optimizer(
        model=transition_model, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    reward_predictor = RewardPredictor(
        abstract_state_dim=abstract_state_dim,
        hidden_size=config["hidden_size"],
        rngs=rngs,
    )
    reward_opt = nnx.Optimizer(
        model=reward_predictor, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    # ── abstract (homomorphic image) actor-critic ───────────────────────────
    abstract_actor = SACGaussianActor(
        rngs=rngs,
        obs_dim=abstract_state_dim,
        act_dim=abstract_action_dim,
        hidden_size=config["hidden_size"],
    )
    abstract_actor_target = deepcopy(abstract_actor)
    abstract_actor_opt = nnx.Optimizer(
        model=abstract_actor,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    abstract_critic = EnsembleCritic(
        rngs=rngs,
        obs_dim=abstract_state_dim,
        act_dim=abstract_action_dim,
        hidden_size=config["hidden_size"],
    )
    abstract_critic_target = deepcopy(abstract_critic)
    abstract_critic_opt = nnx.Optimizer(
        model=abstract_critic,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
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
    logger.log("Start Stochastic DHPG (state observations, lax bisimulation) training")
    steps = buffer.size(buffer_state)

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
            actor_target=actor_target,
            critic=critic,
            critic_opt=critic_opt,
            target_critic=target_critic,
            log_alpha=log_alpha,
            alpha_opt=alpha_opt,
            state_encoder=state_encoder,
            state_encoder_opt=state_encoder_opt,
            action_encoder=action_encoder,
            action_encoder_opt=action_encoder_opt,
            transition_model=transition_model,
            transition_opt=transition_opt,
            reward_predictor=reward_predictor,
            reward_opt=reward_opt,
            abstract_actor=abstract_actor,
            abstract_actor_opt=abstract_actor_opt,
            abstract_actor_target=abstract_actor_target,
            abstract_critic=abstract_critic,
            abstract_critic_opt=abstract_critic_opt,
            abstract_critic_target=abstract_critic_target,
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
            lax_bisim_loss,
            transition_loss,
            reward_loss,
            abs_critic_loss,
            abs_actor_loss,
            lifting_loss,
            value_equivalence,
            abs_q1_mean,
            abs_q2_mean,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            num_steps,
        ) = val
        steps += num_steps
        logger.logged = False

        # ── logging ─────────────────────────────────────────────────────
        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_critic", critic_loss.item())
        logger.log_tabular("Loss/Loss_actor", actor_loss.item())
        logger.log_tabular("Loss/Loss_alpha", alpha_loss.item())

        logger.log_tabular("SAC/Alpha", alpha.item())
        logger.log_tabular("SAC/LogPi_mean", log_pi_mean.item())
        logger.log_tabular("SAC/Q1_mean", q1_mean.item())
        logger.log_tabular("SAC/Q2_mean", q2_mean.item())

        logger.log_tabular("HPG/Lax_bisim_loss", lax_bisim_loss.item())
        logger.log_tabular("HPG/Transition_loss", transition_loss.item())
        logger.log_tabular("HPG/Reward_loss", reward_loss.item())
        logger.log_tabular("HPG/Abstract_critic_loss", abs_critic_loss.item())
        logger.log_tabular("HPG/Abstract_actor_loss", abs_actor_loss.item())
        logger.log_tabular("HPG/Lifting_loss", lifting_loss.item())
        logger.log_tabular("HPG/Value_equivalence", value_equivalence.item())
        logger.log_tabular("HPG/Abstract_Q1_mean", abs_q1_mean.item())
        logger.log_tabular("HPG/Abstract_Q2_mean", abs_q2_mean.item())

        logger.log_tabular(
            "Norm/actor_model",
            get_tree_norm(nnx.state(actor, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/critic_model",
            get_tree_norm(nnx.state(critic, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/state_encoder",
            get_tree_norm(nnx.state(state_encoder, nnx.Param)),
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
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=state_encoder, prefix="state_encoder"
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=action_encoder,
                prefix="action_encoder",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=transition_model,
                prefix="transition_model",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=reward_predictor,
                prefix="reward_predictor",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=abstract_actor,
                prefix="abstract_actor",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=abstract_critic,
                prefix="abstract_critic",
            )

        if steps >= config["total_env_steps"]:
            break

    # ── final save ────────────────────────────────────────────────────────
    logger.nn_model_save(itr=steps, nn_model_saver_element=actor, prefix="actor")
    logger.nn_model_save(itr=steps, nn_model_saver_element=critic, prefix="critic")
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=state_encoder, prefix="state_encoder"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=action_encoder, prefix="action_encoder"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=transition_model, prefix="transition_model"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=reward_predictor, prefix="reward_predictor"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=abstract_actor, prefix="abstract_actor"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=abstract_critic, prefix="abstract_critic"
    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = sac_args()
    args = add_hpg_args(args)

    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "seed-" + str(args.seed).zfill(3)
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]  # "hpg_single"
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
