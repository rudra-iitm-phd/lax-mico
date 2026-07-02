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

from utils.acting import (
    actor_step,
    actor_step_rep_source,
    actor_step_rep_target,
    wrap_env_for_training,
)
from utils.buffer import RunningMeanStd, RunningStatistics, UniformSamplingQueue
from utils.logger import EpochLogger
from utils.metric_models_transfer import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
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
}


def polyak_update(target_model, curr_model, tau: float):

    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


""" Representation Modules """


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
    actor_e2: SACGaussianActor,
    actor_e2_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    log_alpha_e2: Scalar,
    alpha_e2_opt: nnx.Optimizer,
    data: Transition,
    data_e2: Transition,
    config,
    key: jnp.ndarray,
):
    obs = data.observation
    act = data.action
    reward = data.reward
    discount = data.discount
    next_obs = data.next_observation

    obs_e2 = data_e2.observation
    act_e2 = data_e2.action
    reward_e2 = data_e2.reward
    discount_e2 = data_e2.discount
    next_obs_e2 = data_e2.next_observation

    alpha = jnp.exp(log_alpha())
    beta = 0.1
    grad_steps = config.grad_steps

    key, next_key = jax.random.split(key)
    next_act, next_log_prob = actor(next_obs, next_key)

    def critic_loss_fn(critic: EnsembleCritic):
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
    critic_opt.update(critic, critic_grads)

    key, act_key = jax.random.split(key)

    def actor_loss_fn(actor: SACGaussianActor):
        pi, log_pi = actor(obs, act_key)
        q1, q2 = critic(jnp.concatenate([obs, pi], axis=-1))
        loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return loss, jnp.mean(log_pi)

    (actor_loss, log_pi_mean), actor_grads = nnx.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor)

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

    ## shape mismatch error fixed

    ## source-source state distance

    g_sx_next, g_xs_next = target_state_metric.get_source_distance(s_next, x_next)
    u_target = jnp.maximum(g_sx_next, g_xs_next)
    lambda_target = jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

    ## env2 representations

    s2, a2, r2, s2_next = obs_e2, act_e2, reward_e2[:, None], next_obs_e2
    batch_e2 = jnp.concatenate([s2, a2, r2, s2_next], axis=-1)
    key, perm_2_key = jax.random.split(key)
    batch_e2 = jax.random.permutation(perm_2_key, batch_e2)
    batch_e2 = batch_e2[:, -1]

    obs_dim_e2, act_dim_e2 = obs_e2.shape[-1], act_e2.shape[-1]

    x2, b2, y2, x2_next = (
        batch_e2[:, :obs_dim_e2],
        batch_e2[:, obs_dim_e2 : obs_dim_e2 + act_dim_e2],
        batch_e2[:, obs_dim_e2 + act_dim_e2],
        batch_e2[:, obs_dim_e2 + act_dim_e2 + 1 :],
    )

    r2 = r2[:, -1]  ## shaping reward from (B, 1, 1) --> (B, 1)

    x2, b2, y2, x2_next = (
        x2[:, None, :],
        b2[:, None, :],
        y2[:, None],
        x2_next[:, None, :],
    )

    # cross state distance
    g_s_s2_next, g_s2_s_next = target_state_metric.get_cross_distance(s_next, s2_next)
    u_cross_target = jnp.maximum(g_s_s2_next, g_s2_s_next)
    lambda_cross_target = jax.lax.stop_gradient(
        jnp.abs(r - r2) + discount_e2 * discount * u_cross_target
    )

    # target-target state distance
    g_sx_2_next, g_xs_2_next = target_state_metric.get_target_distance(s2_next, x2_next)
    u2_target = jnp.maximum(g_sx_2_next, g_xs_2_next)
    lambda_2_target = jax.lax.stop_gradient(jnp.abs(r2 - y2) + discount_e2 * u2_target)

    ## naming_convention : write model when it's a model

    def state_action_metric_loss_fn(state_action_metric: EnsembleStateActionMetric):

        # source-source
        d_sa_xb, d_xb_sa = state_action_metric.get_source_distance(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_current = jnp.maximum(d_sa_xb, d_xb_sa)
        loss = jnp.mean((lambda_current - lambda_target) ** 2)

        # cross
        d_sa_s2a2, d_s2a2_sa = state_action_metric.get_cross_distance(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([s2, a2], axis=-1)
        )
        lambda_e2_current = jnp.maximum(d_sa_s2a2, d_s2a2_sa)
        loss_e2 = jnp.mean((lambda_e2_current - lambda_cross_target) ** 2)

        # target-target
        d_sa_xb_2, d_xb_sa_2 = state_action_metric.get_target_distance(
            jnp.concatenate([s2, a2], axis=-1), jnp.concatenate([x2, b2], axis=-1)
        )
        lambda_curr_2 = jnp.maximum(d_sa_xb_2, d_xb_sa_2)
        loss_2 = jnp.mean((lambda_curr_2 - lambda_2_target) ** 2)

        return loss + loss_e2 + loss_2

    lambda_loss, lambda_grads = nnx.value_and_grad(state_action_metric_loss_fn)(
        state_action_metric
    )
    state_action_metric_opt.update(state_action_metric, lambda_grads)

    def min_state_action_to_state_metric_loss_fn(
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
    ):

        # source-source

        h_sax, h_xbs = (
            min_state_action_to_state_metric.get_source_source_distance(
                jnp.concatenate([s, a], axis=-1), x
            ),
            min_state_action_to_state_metric.get_source_source_distance(
                jnp.concatenate([x, b], axis=-1), s
            ),
        )

        d_sa_xb, d_xb_sa = state_action_metric.get_source_distance(
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

        # cross
        h_sa_s2, h_s2a2_s = (
            min_state_action_to_state_metric.get_source_target_distance(
                jnp.concatenate([s, a], axis=-1), s2
            ),
            min_state_action_to_state_metric.get_target_source_distance(
                jnp.concatenate([s2, a2], axis=-1), s
            ),
        )
        d_sa_s2a2, d_s2a2_as = state_action_metric.get_cross_distance(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([s2, a2], axis=-1)
        )
        score_p1_e2, score_p2_e2 = (
            (h_sa_s2 - jax.lax.stop_gradient(d_sa_s2a2)) / beta,
            (h_s2a2_s - jax.lax.stop_gradient(d_s2a2_as)) / beta,
        )
        max_score_e2 = jax.lax.stop_gradient(
            jnp.maximum(score_p1_e2.max(), score_p2_e2.max())
        )
        p1_e2 = (
            jnp.exp(score_p1_e2 - max_score_e2)
            - score_p1_e2 * jnp.exp(-max_score_e2)
            - jnp.exp(-max_score_e2)
        )
        p2_e2 = (
            jnp.exp(score_p2_e2 - max_score_e2)
            - score_p2_e2 * jnp.exp(-max_score_e2)
            - jnp.exp(-max_score_e2)
        )
        loss_e2 = jnp.mean(p1_e2) + jnp.mean(p2_e2)

        # target-target

        h_sax_2, h_xbs_2 = (
            min_state_action_to_state_metric.get_target_target_distance(
                jnp.concatenate([s2, a2], axis=-1), x2
            ),
            min_state_action_to_state_metric.get_target_target_distance(
                jnp.concatenate([x2, b2], axis=-1), s2
            ),
        )
        d_sa2_xb2, d_xb2_sa2 = state_action_metric.get_target_distance(
            jnp.concatenate([s2, a2], axis=-1), jnp.concatenate([x2, b2], axis=-1)
        )
        score_p1_2, score_p2_2 = (
            (h_sax_2 - jax.lax.stop_gradient(d_sa2_xb2)) / beta,
            (h_xbs_2 - jax.lax.stop_gradient(d_xb2_sa2)) / beta,
        )
        max_score_2 = jax.lax.stop_gradient(
            jnp.maximum(score_p1_2.max(), score_p2_2.max())
        )
        p1_2 = (
            jnp.exp(score_p1_2 - max_score_2)
            - score_p1_2 * jnp.exp(-max_score_2)
            - jnp.exp(-max_score_2)
        )
        p2_2 = (
            jnp.exp(score_p2_2 - max_score_2)
            - score_p2_2 * jnp.exp(-max_score_2)
            - jnp.exp(-max_score_2)
        )
        loss_2 = jnp.mean(p1_2) + jnp.mean(p2_2)

        return loss + loss_e2 + loss_2

    h_loss, h_grads = nnx.value_and_grad(min_state_action_to_state_metric_loss_fn)(
        min_state_action_to_state_metric
    )
    min_state_action_to_state_metric_opt.update(
        min_state_action_to_state_metric, h_grads
    )

    def state_metric_loss_fn(state_metric: EnsembleStateMetric):

        # source-source
        h_sax, h_xbs = (
            target_state_action_to_state_metric.get_source_source_distance(
                jnp.concatenate([s, a], axis=-1), x
            ),
            target_state_action_to_state_metric.get_source_source_distance(
                jnp.concatenate([x, b], axis=-1), s
            ),
        )
        h_sax, h_xbs = jax.lax.stop_gradient(h_sax), jax.lax.stop_gradient(h_xbs)
        g_sx, g_xs = state_metric.get_source_distance(s, x)
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

        # cross
        h_sa_s2, h_s2a2_s = (
            min_state_action_to_state_metric.get_source_target_distance(
                jnp.concatenate([s, a], axis=-1), s2
            ),
            min_state_action_to_state_metric.get_target_source_distance(
                jnp.concatenate([s2, a2], axis=-1), s
            ),
        )
        h_sa_s2, h_s2a2_s = (
            jax.lax.stop_gradient(h_sa_s2),
            jax.lax.stop_gradient(h_s2a2_s),
        )
        g_s_s2, g_s2_s = state_metric.get_cross_distance(s, s2)

        score_p1_e2, score_p2_e2 = (
            (h_sa_s2 - g_s_s2) / beta,
            (h_s2a2_s - g_s2_s) / beta,
        )
        max_score_e2 = jax.lax.stop_gradient(
            jnp.maximum(score_p1_e2.max(), score_p2_e2.max())
        )
        p1_e2 = (
            jnp.exp(score_p1_e2 - max_score_e2)
            - score_p1_e2 * jnp.exp(-max_score_e2)
            - jnp.exp(-max_score_e2)
        )
        p2_e2 = (
            jnp.exp(score_p2_e2 - max_score_e2)
            - score_p2_e2 * jnp.exp(-max_score_e2)
            - jnp.exp(-max_score_e2)
        )
        loss_e2 = jnp.mean(p1_e2) + jnp.mean(p2_e2)

        # target-target

        h_sax_2, h_xbs_2 = (
            min_state_action_to_state_metric.get_target_target_distance(
                jnp.concatenate([s2, a2], axis=-1), x2
            ),
            min_state_action_to_state_metric.get_target_target_distance(
                jnp.concatenate([x2, b2], axis=-1), s2
            ),
        )
        h_sax_2, h_xbs_2 = (
            jax.lax.stop_gradient(h_sax_2),
            jax.lax.stop_gradient(h_xbs_2),
        )
        g_sx_2, g_xs_2 = state_metric.get_target_distance(s2, x2)

        score_p1_2, score_p2_2 = (
            (h_sax_2 - g_sx_2) / beta,
            (h_xbs_2 - g_xs_2) / beta,
        )
        max_score_2 = jax.lax.stop_gradient(
            jnp.maximum(score_p1_2.max(), score_p2_2.max())
        )
        p1_2 = (
            jnp.exp(score_p1_2 - max_score_2)
            - score_p1_2 * jnp.exp(-max_score_2)
            - jnp.exp(-max_score_2)
        )
        p2_2 = (
            jnp.exp(score_p2_2 - max_score_2)
            - score_p2_2 * jnp.exp(-max_score_2)
            - jnp.exp(-max_score_2)
        )
        loss_2 = jnp.mean(p1_2) + jnp.mean(p2_2)

        return loss + loss_e2 + loss_2

    g_loss, g_grads = nnx.value_and_grad(state_metric_loss_fn)(state_metric)
    state_metric_opt.update(state_metric, g_grads)

    key, act_e2_key = jax.random.split(key)

    def compute_source_diff(
        source: jnp.ndarray,
        source_prime: jnp.ndarray,
        state_metric: EnsembleStateMetric,
    ):
        g_s_sp, g_sp_s = state_metric.get_source_distance(source, source_prime)
        # return jnp.mean(jnp.maximum(g_s_sp, g_sp_s))
        return jnp.maximum(g_s_sp, g_sp_s)

    def compute_cross_diff(
        source: jnp.ndarray,
        target: jnp.ndarray,
        state_metric: EnsembleStateMetric,
    ):
        g_s_t, g_t_s = state_metric.get_cross_distance(source, target)
        # return jnp.mean(jnp.maximum(g_s_t, g_t_s))
        return jnp.maximum(g_s_t, g_t_s)

    def source_state_action_diff(
        source: jnp.ndarray,
        act: jnp.ndarray,
        source_prime: jnp.ndarray,
        act_prime: jnp.ndarray,
        state_action_metric: EnsembleStateActionMetric,
    ):
        d_sa_spap, d_spap_sa = state_action_metric.get_source_distance(
            jnp.concatenate([source, act], axis=-1),
            jnp.concatenate([source_prime, act_prime], axis=-1),
        )
        # return jnp.mean(jnp.maximum(d_sa_spap, d_spap_sa))
        return jnp.maximum(d_sa_spap, d_spap_sa)

    def cross_state_action_diff(
        source: jnp.ndarray,
        source_act: jnp.ndarray,
        target: jnp.ndarray,
        target_act: jnp.ndarray,
        state_action_metric: EnsembleStateActionMetric,
    ):
        d_sa_tb, d_tb_sa = state_action_metric.get_cross_distance(
            jnp.concatenate([source, source_act], axis=-1),
            jnp.concatenate([target, target_act], axis=-1),
        )
        # return jnp.mean(jnp.maximum(d_sa_tb, d_tb_sa))
        return jnp.maximum(d_sa_tb, d_tb_sa)

    def find_equivalent_states(i, carry):
        (states, metric, states_eq) = carry
        source_state, target_state = states
        s_s_eq, t_s_eq = states_eq
        # given : source, find : source_eq
        ss_opt = optax.adam(config.lr)
        ss_opt_state = ss_opt.init(s_s_eq)
        # grad_ss = jax.grad(
        #     lambda source_prime: compute_source_diff(source_state, source_prime, metric)
        # )(s_s_eq)
        grad_ss = jax.grad(
            lambda source_prime: jnp.sum(
                compute_source_diff(source_state, source_prime, metric)
            )
        )(s_s_eq)
        ss_updates, ss_opt_state = ss_opt.update(grad_ss, ss_opt_state)
        s_s_eq = optax.apply_updates(s_s_eq, ss_updates)

        # given : target, find : source_eq
        ts_opt = optax.adam(config.lr)
        ts_opt_state = ts_opt.init(t_s_eq)
        grad_ts = jax.grad(
            lambda source_eq: jnp.sum(
                compute_cross_diff(source_eq, target_state, metric)
            )
        )(t_s_eq)
        ts_updates, ts_opt_state = ts_opt.update(grad_ts, ts_opt_state)
        t_s_eq = optax.apply_updates(t_s_eq, ts_updates)

        states_eq = (s_s_eq, t_s_eq)
        return (states, metric, states_eq)

    ((s, s2), state_metric, (s_s_eq, t_s_eq)) = nnx.fori_loop(
        0,
        grad_steps,
        find_equivalent_states,
        (
            (s, s2),
            state_metric,
            (
                jnp.zeros_like(s),
                jnp.zeros_like(s),
            ),
        ),
    )

    def find_equivalent_actions(i, carry):
        (states, actions, states_eq, metric, action_eq) = carry
        source_state, target_state = states
        source_actions, target_actions = actions
        ss_eq, ts_eq = states_eq
        ss_eq_a, ts_eq_a = action_eq

        # given : (s, a), s', find : equivalent (s', a')
        ss_eq_a_opt = optax.adam(config.lr)
        ss_eq_a_opt_state = ss_eq_a_opt.init(ss_eq_a)
        grad_ss_eq_a = jax.grad(
            lambda act_eq: jnp.sum(
                source_state_action_diff(
                    source_state, source_actions, ss_eq, act_eq, metric
                )
            )
        )(ss_eq_a)
        ss_eq_a_updates, ss_eq_a_opt_state = ss_eq_a_opt.update(
            grad_ss_eq_a, ss_eq_a_opt_state
        )
        ss_eq_a = optax.apply_updates(ss_eq_a, ss_eq_a_updates)
        ss_eq_a = jnp.tanh(ss_eq_a)

        # given : (t, b), s', find : equivalent (s', a')
        ts_eq_a_opt = optax.adam(config.lr)
        ts_eq_a_opt_state = ts_eq_a_opt.init(ts_eq_a)
        ts_eq_a = jax.grad(
            lambda act_eq: jnp.sum(
                cross_state_action_diff(
                    ts_eq, act_eq, target_state, target_actions, metric
                )
            )
        )(ts_eq_a)
        ts_eq_a_updates, ts_eq_a_opt_state = ts_eq_a_opt.update(
            ts_eq_a, ts_eq_a_opt_state
        )
        ts_eq_a = optax.apply_updates(ts_eq_a, ts_eq_a_updates)
        ts_eq_a = jnp.tanh(ts_eq_a)

        action_eq = (ss_eq_a, ts_eq_a)
        return (states, actions, states_eq, metric, action_eq)

    (
        (s, s2),
        (a, a2),
        (s_s_eq, t_s_eq),
        state_action_metric,
        (ss_eq_a, ts_eq_a),
    ) = nnx.fori_loop(
        0,
        grad_steps,
        find_equivalent_actions,
        (
            (s, s2),
            (a, a2),
            (s_s_eq, t_s_eq),
            state_action_metric,
            (jnp.zeros_like(a), jnp.zeros_like(a)),
        ),
    )

    def source_actor_match_loss_fn(source_actor: SACGaussianActor):
        ss_eq_a_preds, _ = source_actor(s_s_eq, act_key)
        ts_eq_a_preds, _ = source_actor(t_s_eq, act_key)
        ss_eq_a_preds += 1
        ts_eq_a_preds += 1
        ss_eq_a_preds, ts_eq_a_preds = jnp.exp(ss_eq_a_preds), jnp.exp(ts_eq_a_preds)
        target1, target2 = jnp.exp(ss_eq_a + 1), jnp.exp(ts_eq_a + 1)
        loss = jnp.mean(
            ((ss_eq_a_preds - jax.lax.stop_gradient(target1)) ** 2).sum(axis=-1)
        ) + jnp.mean(
            ((ts_eq_a_preds - jax.lax.stop_gradient(target2)) ** 2).sum(axis=-1)
        )
        return loss

    source_match_loss, source_match_grads = nnx.value_and_grad(
        source_actor_match_loss_fn
    )(actor)

    actor_grads_total = jax.tree_util.tree_map(
        lambda a, b: a + b, actor_grads, source_match_grads
    )

    actor_opt.update(actor, actor_grads_total)

    ###########################################################################################
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
        source_match_loss,
    )


@functools.partial(nnx.jit, static_argnames=("env", "env_2", "buffer", "buffer_e2"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    env_2,
    env_state_2,
    buffer_state_e2,
    buffer_e2,
    running_state_e2,
    obs_normalizer_e2,
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
    actor_e2: SACGaussianActor,
    actor_e2_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    log_alpha_e2: Scalar,
    alpha_opt_e2: nnx.Optimizer,
    config,
    key: jnp.ndarray,
):

    num_steps = config.log_freq

    def body_fun(i, carry):
        (
            key,
            env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            env_state_2,
            buffer_state_e2,
            running_state_e2,
            obs_normalizer_e2,
            models,
            val,
        ) = carry
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        ) = models

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env, env_state, actor, obs_normalizer, env_key, extra_fields=("truncation",)
        )
        buffer_state = buffer.insert(buffer_state, transition)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        key, env_2_key = jax.random.split(key)
        n_env_2_state, transition_e2 = actor_step(
            env_2,
            env_state_2,
            actor_e2,
            obs_normalizer_e2,
            env_2_key,
            extra_fields=("truncation",),
        )
        buffer_state_e2 = buffer_e2.insert(buffer_state_e2, transition_e2)
        running_state_e2 = RunningStatistics.insert_reward(
            running_state_e2, n_env_2_state.reward
        )

        def do_train(j, carry):
            (
                key,
                env_state,
                buffer_state,
                buffer_state_e2,
                obs_normalizer,
                obs_normalizer_e2,
                models,
                _,
            ) = carry
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            buffer_state_e2, batch_e2 = buffer_e2.sample(buffer_state_e2)
            batch_e2 = batch_e2._replace(
                observation=obs_normalizer_e2.normalize(batch_e2.observation),
                next_observation=obs_normalizer_e2.normalize(batch_e2.next_observation),
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
                batch,
                batch_e2,
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
            )

            return (
                key,
                env_state,
                buffer_state,
                buffer_state_e2,
                obs_normalizer,
                obs_normalizer_e2,
                models,
                val,
            )

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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        )
        (
            key,
            _,
            buffer_state,
            buffer_state_e2,
            obs_normalizer,
            obs_normalizer_e2,
            models,
            val,
        ) = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (
                key,
                n_env_state,
                buffer_state,
                buffer_state_e2,
                obs_normalizer,
                obs_normalizer_e2,
                models,
                init_val,
            ),
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        ) = models
        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            n_env_2_state,
            buffer_state_e2,
            running_state_e2,
            obs_normalizer_e2,
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
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
        env_state_2,
        buffer_state_e2,
        running_state_e2,
        obs_normalizer_e2,
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        ),
        init_val,
    )

    (
        _,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        env_state_2,
        buffer_state_e2,
        running_state_e2,
        obs_normalizer_e2,
        models,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)

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
        actor_e2,
        actor_e2_opt,
        critic,
        critic_opt,
        target_critic,
        log_alpha,
        alpha_opt,
        log_alpha_e2,
        alpha_opt_e2,
    ) = models

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        env_state_2,
        running_state_e2,
        obs_normalizer_e2,
        buffer_state_e2,
        num_steps,
    )


def transfer_tuning(
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
    actor_e2: SACGaussianActor,
    actor_e2_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    log_alpha_e2: Scalar,
    alpha_e2_opt: nnx.Optimizer,
    data: Transition,
    data_e2: Transition,
    config,
    key: jnp.ndarray,
):
    obs = data.observation
    act = data.action
    reward = data.reward
    discount = data.discount
    next_obs = data.next_observation

    obs_e2 = data_e2.observation
    act_e2 = data_e2.action
    reward_e2 = data_e2.reward
    discount_e2 = data_e2.discount
    next_obs_e2 = data_e2.next_observation

    alpha = jnp.exp(log_alpha())
    beta = 0.1
    grad_steps = config.grad_steps

    key, next_key = jax.random.split(key)
    next_act, next_log_prob = actor(next_obs, next_key)

    key, act_e2_key = jax.random.split(key)

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

    ## shape mismatch error fixed

    ## source-source state distance

    g_sx_next, g_xs_next = target_state_metric.get_source_distance(s_next, x_next)
    u_target = jnp.maximum(g_sx_next, g_xs_next)
    lambda_target = jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

    ## env2 representations

    s2, a2, r2, s2_next = obs_e2, act_e2, reward_e2[:, None], next_obs_e2
    batch_e2 = jnp.concatenate([s2, a2, r2, s2_next], axis=-1)
    key, perm_2_key = jax.random.split(key)
    batch_e2 = jax.random.permutation(perm_2_key, batch_e2)
    batch_e2 = batch_e2[:, -1]

    obs_dim_e2, act_dim_e2 = obs_e2.shape[-1], act_e2.shape[-1]

    x2, b2, y2, x2_next = (
        batch_e2[:, :obs_dim_e2],
        batch_e2[:, obs_dim_e2 : obs_dim_e2 + act_dim_e2],
        batch_e2[:, obs_dim_e2 + act_dim_e2],
        batch_e2[:, obs_dim_e2 + act_dim_e2 + 1 :],
    )

    r2 = r2[:, -1]  ## shaping reward from (B, 1, 1) --> (B, 1)

    x2, b2, y2, x2_next = (
        x2[:, None, :],
        b2[:, None, :],
        y2[:, None],
        x2_next[:, None, :],
    )

    # cross state distance
    g_s_s2_next, g_s2_s_next = target_state_metric.get_cross_distance(s_next, s2_next)
    u_cross_target = jnp.maximum(g_s_s2_next, g_s2_s_next)
    lambda_cross_target = jax.lax.stop_gradient(
        jnp.abs(r - r2) + discount_e2 * discount * u_cross_target
    )

    # target-target state distance
    g_sx_2_next, g_xs_2_next = target_state_metric.get_target_distance(s2_next, x2_next)
    u2_target = jnp.maximum(g_sx_2_next, g_xs_2_next)
    lambda_2_target = jax.lax.stop_gradient(jnp.abs(r2 - y2) + discount_e2 * u2_target)

    ## naming_convention : write model when it's a model

    def compute_cross_diff(
        source: jnp.ndarray,
        target: jnp.ndarray,
        state_metric: EnsembleStateMetric,
    ):
        g_s_t, g_t_s = state_metric.get_cross_distance(source, target)
        # return jnp.mean(jnp.maximum(g_s_t, g_t_s))
        return jnp.maximum(g_s_t, g_t_s)

    def compute_target_diff(
        target: jnp.ndarray,
        target_prime: jnp.ndarray,
        state_metric: EnsembleStateMetric,
    ):
        g_t_tp, g_tp_t = state_metric.get_target_distance(target, target_prime)
        # return jnp.mean(jnp.maximum(g_t_tp, g_tp_t))
        return jnp.maximum(g_t_tp, g_tp_t)

    def cross_state_action_diff(
        source: jnp.ndarray,
        source_act: jnp.ndarray,
        target: jnp.ndarray,
        target_act: jnp.ndarray,
        state_action_metric: EnsembleStateActionMetric,
    ):
        d_sa_tb, d_tb_sa = state_action_metric.get_cross_distance(
            jnp.concatenate([source, source_act], axis=-1),
            jnp.concatenate([target, target_act], axis=-1),
        )
        # return jnp.mean(jnp.maximum(d_sa_tb, d_tb_sa))
        return jnp.maximum(d_sa_tb, d_tb_sa)

    def target_state_action_diff(
        target: jnp.ndarray,
        act: jnp.ndarray,
        target_prime: jnp.ndarray,
        act_prime: jnp.ndarray,
        state_action_metric: EnsembleStateActionMetric,
    ):
        d_ta_tpap, d_tpap_ta = state_action_metric.get_target_distance(
            jnp.concatenate([target, act], axis=-1),
            jnp.concatenate([target_prime, act_prime], axis=-1),
        )
        # return jnp.mean(jnp.maximum(d_ta_tpap, d_tpap_ta))
        return jnp.maximum(d_ta_tpap, d_tpap_ta)

    def find_equivalent_states(i, carry):
        (states, metric, states_eq) = carry
        source_state, target_state = states
        s_t_eq, t_t_eq = states_eq

        # given : source, find : target_eq
        st_opt = optax.adam(config.lr)
        st_opt_state = st_opt.init(s_t_eq)
        grad_st = jax.grad(
            lambda target: jnp.sum(compute_cross_diff(source_state, target, metric))
        )(s_t_eq)
        st_updates, st_opt_state = st_opt.update(grad_st, st_opt_state)
        s_t_eq = optax.apply_updates(s_t_eq, st_updates)

        # given : target, find : target_eq
        tt_opt = optax.adam(config.lr)
        tt_opt_state = tt_opt.init(t_t_eq)
        grad_tt = jax.grad(
            lambda target_eq: jnp.sum(
                compute_target_diff(target_state, target_eq, metric)
            )
        )(t_t_eq)
        tt_updates, tt_opt_state = tt_opt.update(grad_tt, tt_opt_state)
        t_t_eq = optax.apply_updates(t_t_eq, tt_updates)

        states_eq = (s_t_eq, t_t_eq)
        return (states, metric, states_eq)

    ((s, s2), state_metric, (s_t_eq, t_t_eq)) = nnx.fori_loop(
        0,
        grad_steps,
        find_equivalent_states,
        (
            (s, s2),
            state_metric,
            (
                jnp.zeros_like(s2),
                jnp.zeros_like(s2),
            ),
        ),
    )

    def find_equivalent_actions(i, carry):
        (states, actions, states_eq, metric, action_eq) = carry
        source_state, target_state = states
        source_actions, target_actions = actions
        st_eq, tt_eq = states_eq
        st_eq_b, tt_eq_b = action_eq

        # given : (s, a), t, find : equivalent (t, b)
        st_eq_b_opt = optax.adam(config.lr)
        st_eq_b_opt_state = st_eq_b_opt.init(st_eq_b)
        grad_st_eq_b = jax.grad(
            lambda act_eq: jnp.sum(
                cross_state_action_diff(
                    source_state, source_actions, st_eq, act_eq, metric
                )
            )
        )(st_eq_b)
        st_eq_b_updates, st_eq_b_opt_state = st_eq_b_opt.update(
            grad_st_eq_b, st_eq_b_opt_state
        )
        st_eq_b = optax.apply_updates(st_eq_b, st_eq_b_updates)
        st_eq_b = jnp.tanh(st_eq_b)

        # given : (t, b), t', find : equivalent (t', b')
        tt_eq_b_opt = optax.adam(config.lr)
        tt_eq_b_opt_state = tt_eq_b_opt.init(tt_eq_b)
        grad_tt_eq_b = jax.grad(
            lambda act_eq: jnp.sum(
                target_state_action_diff(
                    target_state, target_actions, tt_eq, act_eq, metric
                )
            )
        )(tt_eq_b)
        tt_eq_b_updates, tt_eq_b_opt_state = tt_eq_b_opt.update(
            grad_tt_eq_b, tt_eq_b_opt_state
        )
        tt_eq_b = optax.apply_updates(tt_eq_b, tt_eq_b_updates)
        tt_eq_b = jnp.tanh(tt_eq_b)

        action_eq = (st_eq_b, tt_eq_b)
        return (states, actions, states_eq, metric, action_eq)

    (
        (s, s2),
        (a, a2),
        (s_t_eq, t_t_eq),
        state_action_metric,
        (st_eq_b, tt_eq_b),
    ) = nnx.fori_loop(
        0,
        grad_steps,
        find_equivalent_actions,
        (
            (s, s2),
            (a, a2),
            (s_t_eq, t_t_eq),
            state_action_metric,
            (
                jnp.zeros_like(a2),
                jnp.zeros_like(a2),
            ),
        ),
    )

    def target_actor_match_loss_fn(target_actor: SACGaussianActor):
        st_eq_b_preds, log_pi2_mean1 = target_actor(s_t_eq, act_e2_key)
        tt_eq_b_preds, log_pi2_mean2 = target_actor(t_t_eq, act_e2_key)
        st_eq_b_preds += 1
        tt_eq_b_preds += 1
        st_eq_b_preds, tt_eq_b_preds = jnp.exp(st_eq_b_preds), jnp.exp(tt_eq_b_preds)
        target_1 = st_eq_b + 1
        target_2 = tt_eq_b + 1
        target_1, target_2 = jnp.exp(target_1), jnp.exp(target_2)

        loss = jnp.mean(
            ((st_eq_b_preds - jax.lax.stop_gradient(target_1)) ** 2).sum(axis=-1)
        ) + jnp.mean(
            ((tt_eq_b_preds - jax.lax.stop_gradient(target_2)) ** 2).sum(axis=-1)
        )
        return loss, jnp.mean(log_pi2_mean1 + log_pi2_mean2)

    (target_match_loss, log_pi2), target_match_grads = nnx.value_and_grad(
        target_actor_match_loss_fn, has_aux=True
    )(actor_e2)

    actor_e2_opt.update(actor_e2, target_match_grads)

    def alpha_loss_fn_e2(log_alpha_2):
        a = jnp.exp(log_alpha_2())
        loss = jnp.mean(
            -a * (jax.lax.stop_gradient(jnp.mean(log_pi2)) + float(-a2.shape[-1]))
        )
        return loss

    alpha_loss_e2, alpha_grads_e2 = nnx.value_and_grad(alpha_loss_fn_e2)(log_alpha_e2)
    alpha_e2_opt.update(log_alpha, alpha_grads_e2)
    return target_match_loss, alpha_loss_e2


@functools.partial(nnx.jit, static_argnames=("env", "env_2", "buffer", "buffer_e2"))
def transfer_train_n_step(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    env_2,
    env_state_2,
    buffer_state_e2,
    buffer_e2,
    running_state_e2,
    obs_normalizer_e2,
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
    actor_e2: SACGaussianActor,
    actor_e2_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    log_alpha_e2: Scalar,
    alpha_opt_e2: nnx.Optimizer,
    config,
    key: jnp.ndarray,
):

    num_steps = config.transfer_freq

    def body_fun(i, carry):
        (
            key,
            env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            env_state_2,
            buffer_state_e2,
            running_state_e2,
            obs_normalizer_e2,
            models,
            val,
        ) = carry
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        ) = models

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env, env_state, actor, obs_normalizer, env_key, extra_fields=("truncation",)
        )
        buffer_state = buffer.insert(buffer_state, transition)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        key, env_2_key = jax.random.split(key)
        n_env_2_state, transition_e2 = actor_step(
            env_2,
            env_state_2,
            actor_e2,
            obs_normalizer_e2,
            env_2_key,
            extra_fields=("truncation",),
        )
        buffer_state_e2 = buffer_e2.insert(buffer_state_e2, transition_e2)
        running_state_e2 = RunningStatistics.insert_reward(
            running_state_e2, n_env_2_state.reward
        )

        def do_train(j, carry):
            (
                key,
                env_state,
                buffer_state,
                buffer_state_e2,
                obs_normalizer,
                obs_normalizer_e2,
                models,
                _,
            ) = carry
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            buffer_state_e2, batch_e2 = buffer_e2.sample(buffer_state_e2)
            batch_e2 = batch_e2._replace(
                observation=obs_normalizer_e2.normalize(batch_e2.observation),
                next_observation=obs_normalizer_e2.normalize(batch_e2.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = transfer_tuning(
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
                batch,
                batch_e2,
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
            )

            return (
                key,
                env_state,
                buffer_state,
                buffer_state_e2,
                obs_normalizer,
                obs_normalizer_e2,
                models,
                val,
            )

        init_val = (jnp.zeros((), jnp.float32),) * 2
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        )
        (
            key,
            _,
            buffer_state,
            buffer_state_e2,
            obs_normalizer,
            obs_normalizer_e2,
            models,
            val,
        ) = nnx.fori_loop(
            0,
            config.transfer_steps,
            do_train,
            (
                key,
                n_env_state,
                buffer_state,
                buffer_state_e2,
                obs_normalizer,
                obs_normalizer_e2,
                models,
                init_val,
            ),
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        ) = models
        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            n_env_2_state,
            buffer_state_e2,
            running_state_e2,
            obs_normalizer_e2,
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
                actor_e2,
                actor_e2_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
                log_alpha_e2,
                alpha_opt_e2,
            ),
            val,
        )

    init_val = (jnp.zeros((), jnp.float32),) * 2
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        env_state_2,
        buffer_state_e2,
        running_state_e2,
        obs_normalizer_e2,
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
            actor_e2,
            actor_e2_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
            log_alpha_e2,
            alpha_opt_e2,
        ),
        init_val,
    )

    (
        _,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        env_state_2,
        buffer_state_e2,
        running_state_e2,
        obs_normalizer_e2,
        models,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)

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
        actor_e2,
        actor_e2_opt,
        critic,
        critic_opt,
        target_critic,
        log_alpha,
        alpha_opt,
        log_alpha_e2,
        alpha_opt_e2,
    ) = models

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        env_state_2,
        running_state_e2,
        obs_normalizer_e2,
        buffer_state_e2,
        num_steps,
    )


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


# def prefill_buffer_target(
#     key, env, env_state, buffer_state, rep_net, policy, buffer, num_itr: int
# ):
#     """
#     Collect `num_itr` transitions before training begins.

#     Uses jax.lax.scan (not a Python loop) so the warmup is JIT-compiled.
#     The policy has random initial weights so actions are effectively random.
#     """

#     def body(carry, _):
#         key, env_state, buffer_state = carry
#         key, subkey = jax.random.split(key)
#         n_state, transition = actor_step_rep_target(
#             env=env,
#             env_state=env_state,
#             repnet=rep_net,
#             policy=policy,
#             key=subkey,
#             extra_fields=("truncation",),
#         )
#         buffer_state = buffer.insert(buffer_state, transition)
#         return (key, n_state, buffer_state), ()

#     jitted_body = jax.jit(body)
#     (_, env_state, buffer_state), () = jax.lax.scan(
#         jitted_body,
#         (key, env_state, buffer_state),
#         (),
#         length=num_itr,
#     )
#     return env_state, buffer_state


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
            "transfer_freq": args.transfer_freq,
            "transfer_steps": args.transfer_steps,
            "env2_warmup": args.env2_warmup,
            "grad_steps": args.grad_steps,
            "num_envs": args.num_envs,
        }
    )

    # ── environment 1 ───────────────────────────────────────────────────────
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

    # Environment 2 (target)

    prng_key, env2_key = jax.random.split(prng_key)
    env2_key = jax.random.split(env2_key, config["num_envs"])
    env2 = wrap_env_for_training(
        registry.load(args.target_task, config_overrides={"impl": "jax"}),
        episode_length=config["episode_length"],
        full_reset=False,
    )
    env2_state = env2.reset(env2_key)
    obs_dim_e2 = env2.observation_size
    act_dim_e2 = env2.action_size
    obs_normalizer_e2 = RunningMeanStd.init((obs_dim_e2,))

    # Standard SAC target entropy: −|A|
    # Targets roughly uniform distribution over actions at start.
    config["target_entropy"] = float(-act_dim)

    # Freeze config into an immutable Flax struct (required for nnx.jit stability)
    config_data = make_static_config_from_dict("SACConfig", config)()

    state_metric = EnsembleStateMetric(
        rngs=rngs,
        source_obs_dim=obs_dim,
        target_obs_dim=obs_dim_e2,
        hidden_size=config["hidden_size"],
    )

    state_metric_opt = nnx.Optimizer(
        model=state_metric,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    state_action_metric = EnsembleStateActionMetric(
        rngs=rngs,
        source_obs_dim=obs_dim,
        source_act_dim=act_dim,
        target_obs_dim=obs_dim_e2,
        target_act_dim=act_dim_e2,
        hidden_size=config["hidden_size"],
    )

    state_action_metric_opt = nnx.Optimizer(
        model=state_action_metric,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    min_state_action_to_state_metric = MinStateActiontoStateMetric(
        rngs=rngs,
        source_obs_dim=obs_dim,
        source_act_dim=act_dim,
        target_obs_dim=obs_dim_e2,
        target_act_dim=act_dim_e2,
        hidden_size=config["hidden_size"],
    )

    min_state_action_to_state_metric_opt = nnx.Optimizer(
        model=min_state_action_to_state_metric,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    target_state_metric = deepcopy(state_metric)

    target_state_action_to_state_metric = deepcopy(min_state_action_to_state_metric)

    actor = SACGaussianActor(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )

    actor_opt = nnx.Optimizer(
        model=actor,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    actor_e2 = SACGaussianActor(
        rngs=rngs,
        obs_dim=obs_dim_e2,
        act_dim=act_dim_e2,
        hidden_size=config["hidden_size"],
    )
    actor_e2_opt = nnx.Optimizer(
        model=actor_e2,
        tx=optax.chain(
            optax.clip_by_global_norm(5.0),
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
        model=log_alpha, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )
    log_alpha_e2 = Scalar(float(jnp.log(config["init_temperature"])))
    alpha_opt_e2 = nnx.Optimizer(
        model=log_alpha_e2, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
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

    ## replay buffer for env 2
    dummy_obs_e2 = jnp.zeros((1, obs_dim_e2))
    dummy_act_e2 = jnp.zeros((1, act_dim_e2))
    dummy_zero_e2 = jnp.zeros((1,))

    dummy_transition_e2 = Transition(
        observation=dummy_obs_e2,
        action=dummy_act_e2,
        reward=dummy_zero_e2,
        discount=dummy_zero_e2,
        next_observation=dummy_obs_e2,
        extras={"state_extras": {"truncation": dummy_zero_e2}},
    )

    buffer_e2 = UniformSamplingQueue(
        max_replay_size=config["max_replay_size"],
        dummy_data_sample=dummy_transition_e2,
        sample_batch_size=config["batch_size"],
    )

    prng_key, buffer_e2_key = jax.random.split(prng_key)
    buffer_state_e2 = buffer_e2.init(buffer_e2_key)

    prng_key, running_key_e2 = jax.random.split(prng_key)
    running_state_e2 = RunningStatistics.init(
        (config["eval_episode_freq"] * config["episode_length"],), running_key_e2
    )

    # ── logger ────────────────────────────────────────────────────────────
    dict_args = dict(config)
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)

    # ── warmup ────────────────────────────────────────────────────────────
    logger.log("Start prefilling env 1 replay buffer")
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
    logger.log("Start prefilling env 2 replay buffer")
    prng_key, buffer_e2_key = jax.random.split(prng_key)
    env2_state, buffer_state_e2, obs_normalizer_e2 = prefill_buffer(
        key=buffer_e2_key,
        env=env2,
        env_state=env2_state,
        buffer_state=buffer_state_e2,
        policy=actor,  # changed
        buffer=buffer_e2,
        obs_normalizer=obs_normalizer_e2,
        num_itr=config["warmup_samples"],
    )

    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start SAC training")
    steps = buffer.size(buffer_state)

    while steps < config["total_env_steps"]:
        prng_key, subkey = jax.random.split(prng_key)

        val = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
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
            log_alpha_e2=log_alpha_e2,
            alpha_opt_e2=alpha_opt_e2,
            config=config_data,
            key=subkey,
            env_2=env2,
            env_state_2=env2_state,
            buffer_e2=buffer_e2,
            buffer_state_e2=buffer_state_e2,
            running_state_e2=running_state_e2,
            actor_e2=actor_e2,
            actor_e2_opt=actor_e2_opt,
            obs_normalizer=obs_normalizer,
            obs_normalizer_e2=obs_normalizer_e2,
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
            source_match_loss,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            env2_state,
            running_state_e2,
            obs_normalizer_e2,
            buffer_state_e2,
            num_steps,
        ) = val

        # transfer_val = transfer_train_n_step(
        #     env=env,
        #     env_state=env_state,
        #     buffer_state=buffer_state,
        #     buffer=buffer,
        #     running_state=running_state,
        #     state_metric=state_metric,
        #     state_metric_opt=state_metric_opt,
        #     state_action_metric=state_action_metric,
        #     state_action_metric_opt=state_action_metric_opt,
        #     min_state_action_to_state_metric=min_state_action_to_state_metric,
        #     min_state_action_to_state_metric_opt=min_state_action_to_state_metric_opt,
        #     target_state_metric=target_state_metric,
        #     target_state_action_to_state_metric=target_state_action_to_state_metric,
        #     actor=actor,
        #     actor_opt=actor_opt,
        #     critic=critic,
        #     critic_opt=critic_opt,
        #     target_critic=target_critic,
        #     log_alpha=log_alpha,
        #     alpha_opt=alpha_opt,
        #     log_alpha_e2=log_alpha_e2,
        #     alpha_opt_e2=alpha_opt_e2,
        #     config=config_data,
        #     key=subkey,
        #     env_2=env2,
        #     env_state_2=env2_state,
        #     buffer_e2=buffer_e2,
        #     buffer_state_e2=buffer_state_e2,
        #     running_state_e2=running_state_e2,
        #     actor_e2=actor_e2,
        #     actor_e2_opt=actor_e2_opt,
        #     obs_normalizer=obs_normalizer,
        #     obs_normalizer_e2=obs_normalizer_e2,
        # )

        # (
        #     target_match_loss,
        #     alpha_loss_e2,
        #     env_state,
        #     running_state,
        #     obs_normalizer,
        #     buffer_state,
        #     env2_state,
        #     running_state_e2,
        #     obs_normalizer_e2,
        #     buffer_state_e2,
        #     transfer_steps,
        # ) = transfer_val

        # transfer_loss, return from env 2

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
        logger.log_tabular("Loss/source_action_matching_loss", source_match_loss.item())
        # logger.log_tabular("Loss/target_action_matching_loss", target_match_loss.item())

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
            "Norm/actor_e2_model",
            get_tree_norm(nnx.state(actor_e2, nnx.Param)),
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
        logger.log_tabular(
            "Eval/Return_e2",
            running_state_e2.reward_state.data.sum() / config["eval_episode_freq"],
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
            # logger.nn_model_save(
            #     itr=steps, nn_model_saver_element=actor_e2, prefix="actor_e2"
            # )

        if steps >= config["total_env_steps"]:
            break

    # ── final save ────────────────────────────────────────────────────────
    logger.nn_model_save(itr=steps, nn_model_saver_element=actor, prefix="actor")
    logger.nn_model_save(itr=steps, nn_model_saver_element=critic, prefix="critic")
    # logger.nn_model_save(itr=steps, nn_model_saver_element=actor_e2, prefix="actor_e2")
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
