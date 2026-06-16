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
from utils.buffer import RunningStatistics, UniformSamplingQueue
from utils.logger import EpochLogger
from utils.models import (
    EnsembleCritic,
    EnsembleCriticRep,
    RepNet,
    SACGaussianActor,
    SACGaussianActorRep,
    Scalar,
    get_tree_norm,
)
from utils.rep_models import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
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


# def sac_train_step(
#         actor : SACGaussianActor,
#         actor_opt : nnx.Optimizer,
#         critic : EnsembleCritic,
#         critic_opt : nnx.Optimizer,
#         target_critic : EnsembleCritic,
#         log_alpha : Scalar,
#         alpha_opt : nnx.Optimizer,
#         data : Transition,
#         config,
#         key : jnp.ndarray
# ):
#     obs = data.observation
#     act = data.action
#     reward = data.reward
#     discount = data.discount
#     next_obs = data.next_observation
#     alpha = jnp.exp(log_alpha())

#     key, next_key = jax.random.split(key)
#     next_act, next_log_prob = actor(next_obs, next_key)

#     def critic_loss_fn(critic):
#         q1_t , q2_t = target_critic( jnp.concatenate([next_obs, next_act], axis = -1))
#         backup = reward + config.gamma * discount * (jnp.minimum(q1_t, q2_t) - alpha * next_log_prob)
#         backup = jax.lax.stop_gradient(backup)
#         q1, q2 = critic(jnp.concatenate([obs, act], axis = -1))
#         loss = jnp.mean((q1 - backup)**2) + jnp.mean((q2 - backup)**2)
#         return loss, (jnp.mean(q1), jnp.mean(q2))

#     (critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(critic_loss_fn, has_aux = True)(critic)
#     critic_opt.update(critic, critic_grads)

#     key, act_key = jax.random.split(key)

#     def actor_loss_fn(actor):
#         pi, log_pi = actor(obs, act_key)
#         q1, q2 = critic(jnp.concatenate([obs, pi], axis = -1))
#         loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
#         return loss, jnp.mean(log_pi)

#     (actor_loss, log_pi_mean), actor_grads = nnx.value_and_grad(actor_loss_fn, has_aux = True)(actor)
#     actor_opt.update(actor, actor_grads)

#     def alpha_loss_fn(log_alpha):
#         a = jnp.exp(log_alpha())
#         loss = jnp.mean(-a * (jax.lax.stop_gradient(log_pi_mean) + config.target_entropy))
#         return loss

#     alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)
#     alpha_opt.update(log_alpha, alpha_grads)

#     polyak_update(target_critic, critic, config.update_tau)

#     return critic_loss, actor_loss, alpha_loss, alpha, log_pi_mean, q1_mean, q2_mean

# @functools.partial(nnx.jit, static_argnames = ("env", "buffer"))
# def train_n_steps(env, env_state, buffer_state, buffer, running_state, actor : SACGaussianActor, actor_opt : nnx.Optimizer, critic : EnsembleCritic, critic_opt : nnx.Optimizer, target_critic : EnsembleCritic, log_alpha : Scalar, alpha_opt : nnx.Optimizer, config, key : jnp.ndarray):

#     num_steps = config.log_freq

#     def body_fun(i, carry):
#         key, env_state, buffer_state, running_state, models, val = carry
#         (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt) = models

#         key, env_key = jax.random.split(key)
#         n_env_state, transition = actor_step(env, env_state, actor, env_key, extra_fields = ("truncation", ))
#         buffer_state = buffer.insert(buffer_state, transition)
#         running_state = RunningStatistics.insert_reward(running_state, n_env_state.reward)

#         def do_train(j, carry):
#             key, env_state, buffer_state, models, _ = carry
#             (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt) = models

#             buffer_state, batch = buffer.sample(buffer_state)
#             key, train_key = jax.random.split(key)

#             val = sac_train_step(actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt, batch, config, train_key)

#             models = (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt)

#             return (key, env_state, buffer_state, models, val)

#         init_val = (jnp.zeros((), jnp.float32),)*7
#         models = (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt)
#         key, _, buffer_state, models, val = nnx.fori_loop(0, config.train_per_step, do_train, (key, n_env_state, buffer_state, models, init_val))
#         (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt) = models
#         return (key, n_env_state, buffer_state, running_state, (actor, actor_opt, critic, critic_opt, target_critic, log_alpha, alpha_opt), val)
#     init_val   = (jnp.zeros((), jnp.float32),) * 7
#     init_carry = (
#         key,
#         env_state,
#         buffer_state,
#         running_state,
#         (actor, actor_opt, critic, critic_opt,
#          target_critic, log_alpha, alpha_opt),
#         init_val,
#     )

#     (_, env_state, buffer_state, running_state, models, val) = nnx.fori_loop(
#         0, num_steps, body_fun, init_carry
#     )

#     (actor, actor_opt, critic, critic_opt,
#      target_critic, log_alpha, alpha_opt) = models

#     return *val, env_state, running_state, buffer_state, num_steps

""" Representation Modules """


def sac_train_step(
    rep_net: RepNet,
    rep_net_opt: nnx.Optimizer,
    state_metric: EnsembleStateMetric,
    state_metric_opt: nnx.Optimizer,
    state_action_metric: EnsembleStateActionMetric,
    state_action_metric_opt: nnx.Optimizer,
    min_state_action_to_state_metric: MinStateActiontoStateMetric,
    min_state_action_to_state_metric_opt: nnx.Optimizer,
    target_state_metric: EnsembleStateMetric,
    target_state_action_to_state_metric: MinStateActiontoStateMetric,
    actor: SACGaussianActorRep,
    actor_opt: nnx.Optimizer,
    critic: EnsembleCriticRep,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCriticRep,
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

    key, next_key = jax.random.split(key)

    next_obs_rep = rep_net.state_rep(next_obs)
    next_act, next_log_prob = actor(next_obs_rep, next_key)

    def critic_loss_fn(critic_and_rep):
        critic_, rep_ = critic_and_rep
        next_obs_act_rep = rep_.state_action_rep(next_obs, next_act)
        q1_t, q2_t = target_critic(jax.lax.stop_gradient(next_obs_act_rep))
        backup = reward + config.gamma * discount * (
            jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
        )
        backup = jax.lax.stop_gradient(backup)
        obs_act_rep = rep_.state_action_rep(obs, act)
        q1, q2 = critic_(obs_act_rep)
        loss = jnp.mean((q1 - backup) ** 2) + jnp.mean((q2 - backup) ** 2)
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (critic_loss, (q1_mean, q2_mean)), (critic_grads, rep_grads_from_critic) = (
        nnx.value_and_grad(critic_loss_fn, has_aux=True)((critic, rep_net))
    )
    critic_opt.update(critic, critic_grads)

    key, act_key = jax.random.split(key)

    def actor_loss_fn(actor_and_rep):
        actor_, rep_ = actor_and_rep
        obs_rep = rep_.state_rep(obs)
        pi, log_pi = actor_(obs_rep, act_key)
        obs_act_rep = rep_net.state_action_rep(obs, pi)
        q1, q2 = critic(obs_act_rep)
        loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return loss, jnp.mean(log_pi)

    (actor_loss, log_pi_mean), (actor_grads, rep_grads_from_actor) = nnx.value_and_grad(
        actor_loss_fn, has_aux=True
    )((actor, rep_net))
    actor_opt.update(actor, actor_grads)

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

    s_rep, x_rep = rep_net.state_rep(s), rep_net.state_rep(x)

    s_next_rep, x_next_rep = rep_net.state_rep(s_next), rep_net.state_rep(x_next)

    sa_rep, xb_rep = rep_net.state_action_rep(s, a), rep_net.state_action_rep(x, b)

    g_sx_next, g_xs_next = target_state_metric(s_next_rep, x_next_rep)

    u_target = jnp.maximum(g_sx_next, g_xs_next)

    lambda_target = jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

    ## naming_convention : write model when it's a model

    def state_action_metric_loss_fn(state_action_metric: EnsembleStateActionMetric):

        d_sa_xb, d_xb_sa = state_action_metric(sa_rep, xb_rep)

        lambda_current = jnp.maximum(d_sa_xb, d_xb_sa)

        loss = jnp.mean((lambda_current - lambda_target) ** 2)

        # jax.debug.print("d_min = {}", d_sa_xb.min())
        # jax.debug.print("d_mean = {}", d_sa_xb.mean())
        # jax.debug.print("d_max = {}", d_sa_xb.max())

        # jax.debug.print("g_next_min = {}", g_sx_next.min())
        # jax.debug.print("g_next_mean = {}", g_sx_next.mean())
        # jax.debug.print("g_next_max = {}", g_sx_next.max())

        return loss

    lambda_loss, lambda_grads = nnx.value_and_grad(state_action_metric_loss_fn)(
        state_action_metric
    )
    state_action_metric_opt.update(state_action_metric, lambda_grads)

    def min_state_action_to_state_metric_loss_fn(
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
    ):

        # d_sa_xb, d_xb_sa = state_action_metric(sa_rep, xb_rep)

        # lambda_current = jnp.maximum(d_sa_xb, d_xb_sa)

        h_sax, h_xbs = (
            min_state_action_to_state_metric(sa_rep, x_rep),
            min_state_action_to_state_metric(xb_rep, s_rep),
        )

        score_p1, score_p2 = (
            (h_sax - lambda_target) / beta,
            (h_xbs - lambda_target) / beta,
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
            target_state_action_to_state_metric(sa_rep, x_rep),
            target_state_action_to_state_metric(xb_rep, s_rep),
        )

        h_sax, h_xbs = jax.lax.stop_gradient(h_sax), jax.lax.stop_gradient(h_xbs)

        g_sx, g_xs = state_metric(s_rep, x_rep)

        # B
        score_p1, score_p2 = (h_sax - g_sx) / beta, (h_xbs - g_xs) / beta

        # print(score_p1.shape, score_p2.shape)

        ## ensure this returns a scalar

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

    def rep_loss_fn(rep_net: RepNet):

        s_rep, x_rep = rep_net.state_rep(s), rep_net.state_rep(x)

        g_sx, g_xs = target_state_metric(s_rep, x_rep)

        u_target = jax.lax.stop_gradient(jnp.maximum(g_sx, g_xs))

        s_diff = jnp.linalg.norm(s_rep - x_rep, axis=-1, ord=1)

        loss = jnp.mean(optax.huber_loss(s_diff, u_target, delta=1.0))

        return loss

    rep_loss, rep_grads = nnx.value_and_grad(rep_loss_fn)(rep_net)

    rep_grads_total = jax.tree_util.tree_map(
        lambda a, b, c: a + b + c,
        rep_grads_from_actor,
        rep_grads_from_critic,
        rep_grads,
    )

    rep_net_opt.update(rep_net, rep_grads_total)

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
        rep_loss,
    )


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    rep_net: RepNet,
    rep_net_opt: nnx.Optimizer,
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
        key, env_state, buffer_state, running_state, models, val = carry
        (
            rep_net,
            rep_net_opt,
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
        n_env_state, transition = actor_step_rep(
            env, env_state, rep_net, actor, env_key, extra_fields=("truncation",)
        )
        buffer_state = buffer.insert(buffer_state, transition)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        def do_train(j, carry):
            key, env_state, buffer_state, models, _ = carry
            (
                rep_net,
                rep_net_opt,
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
            key, train_key = jax.random.split(key)

            val = sac_train_step(
                rep_net,
                rep_net_opt,
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
                rep_net,
                rep_net_opt,
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

            return (key, env_state, buffer_state, models, val)

        init_val = (jnp.zeros((), jnp.float32),) * 11
        models = (
            rep_net,
            rep_net_opt,
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
        key, _, buffer_state, models, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, n_env_state, buffer_state, models, init_val),
        )
        (
            rep_net,
            rep_net_opt,
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
            (
                rep_net,
                rep_net_opt,
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
        (
            rep_net,
            rep_net_opt,
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

    (_, env_state, buffer_state, running_state, models, val) = nnx.fori_loop(
        0, num_steps, body_fun, init_carry
    )

    (
        rep_net,
        rep_net_opt,
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

    return *val, env_state, running_state, buffer_state, num_steps


""" Copied from claude """


def prefill_buffer(
    key, env, env_state, buffer_state, rep_net, policy, buffer, num_itr: int
):
    """
    Collect `num_itr` transitions before training begins.

    Uses jax.lax.scan (not a Python loop) so the warmup is JIT-compiled.
    The policy has random initial weights so actions are effectively random.
    """

    def body(carry, _):
        key, env_state, buffer_state = carry
        key, subkey = jax.random.split(key)
        n_state, transition = actor_step_rep(
            env=env,
            env_state=env_state,
            repnet=rep_net,
            policy=policy,
            key=subkey,
            extra_fields=("truncation",),
        )
        buffer_state = buffer.insert(buffer_state, transition)
        return (key, n_state, buffer_state), ()

    jitted_body = jax.jit(body)
    (_, env_state, buffer_state), () = jax.lax.scan(
        jitted_body,
        (key, env_state, buffer_state),
        (),
        length=num_itr,
    )
    return env_state, buffer_state


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
        }
    )

    # ── environment ───────────────────────────────────────────────────────
    prng_key, env_key = jax.random.split(prng_key)
    env_key = jax.random.split(env_key, 2)  # batch_size = 1 env

    env = wrap_env_for_training(
        registry.load(args.task, config_overrides={"impl": "jax"}),
        episode_length=config["episode_length"],
        full_reset=False,
    )
    env_state = env.reset(env_key)
    obs_dim = env.observation_size
    act_dim = env.action_size
    rep_dim = 16

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

    rep_net = RepNet(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        rep_dim=rep_dim,
        hidden_dim=config["hidden_size"],
    )

    rep_net_opt = nnx.Optimizer(
        model=rep_net,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    state_metric = EnsembleStateMetric(
        rngs=rngs, rep_dim=rep_dim, hidden_size=config["hidden_size"]
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
        rngs=rngs, rep_dim=rep_dim, hidden_size=config["hidden_size"]
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
        state_action_rep_dim=rep_dim,
        state_rep_dim=rep_dim,
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

    actor = SACGaussianActorRep(
        rngs=rngs, rep_dim=rep_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )

    actor_opt = nnx.Optimizer(
        model=actor,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        ),
        wrt=nnx.Param,
    )

    critic = EnsembleCriticRep(
        rngs=rngs, rep_dim=rep_dim, hidden_size=config["hidden_size"]
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

    env_state, buffer_state = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        rep_net=rep_net,
        policy=actor,
        buffer=buffer,
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
            rep_net=rep_net,
            rep_net_opt=rep_net_opt,
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
            rep_loss,
            env_state,
            running_state,
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
        logger.log_tabular("Loss/Loss_representation", rep_loss)

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
            "Norm/rep_model", get_tree_norm(nnx.state(rep_net, nnx.Param))
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
