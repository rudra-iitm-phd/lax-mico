import functools
import os
import os.path as osp
import random
import sys
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from mujoco_playground import registry

from utils.acting import actor_step, wrap_env_for_training
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
from utils.metric_models import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
from utils.algo_models import EnsembleCritic, SACGaussianActor, Scalar, get_tree_norm
from utils.parameterized_models import (
    AgentAux,
    MetricAux,
    Models,
    Optimizers,
    TrainingState,
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
    "rep_lr_scale":0.5
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
    state: TrainingState, data: Transition, config, key: jnp.ndarray
) -> tuple[AgentAux, MetricAux]:
    obs = data.observation
    act = data.action
    reward = data.reward
    discount = data.discount
    next_obs = data.next_observation
    truncation = data.extras["state_extras"]["truncation"]
    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
    alpha = jnp.exp(state.models.log_alpha())
    beta = 1.0

    def alpha_loss_fn(log_alpha):
        _, log_prob = state.models.actor(obs, key_alpha)
        a = jnp.exp(log_alpha())
        loss = jnp.mean(a * jax.lax.stop_gradient(-log_prob - config.target_entropy))
        return loss

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(state.models.log_alpha)

    def critic_loss_fn(critic, target_critic):
        next_act, next_log_prob = state.models.actor(next_obs, key_critic)
        q1_t, q2_t = target_critic(jnp.concatenate([next_obs, next_act], axis=-1))
        next_v = jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            reward * config.reward_scaling + discount * config.gamma * next_v
        )
        q1, q2 = critic(jnp.concatenate([obs, act], axis=-1))
        q_error = jnp.stack([q1, q2], axis=-1) - target_q[..., None]
        q_error = q_error * (1.0 - truncation)[..., None]
        loss = 0.5 * jnp.mean(jnp.square(q_error))
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True
    )(state.models.critic, state.models.target_critic)

    def actor_loss_fn(actor):
        pi, log_pi = actor(obs, key_actor)
        q1, q2 = state.models.critic(jnp.concatenate([obs, pi], axis=-1))
        loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return loss, jnp.mean(log_pi)

    (actor_loss, log_pi_mean), actor_grads = nnx.value_and_grad(
        actor_loss_fn, has_aux=True
    )(state.models.actor)

    state.optimizers.log_alpha.update(state.models.log_alpha, alpha_grads)
    state.optimizers.critic.update(state.models.critic, critic_grads)
    state.optimizers.actor.update(state.models.actor, actor_grads)

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
    g_sx_next, g_xs_next = state.models.target_state_metric(s_next, x_next)
    u_target = jnp.maximum(g_sx_next, g_xs_next)
    lambda_target = jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

    def state_action_metric_loss_fn(state_action_metric: EnsembleStateActionMetric):
        lambda_sa_xb, lambda_xb_sa = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_target_1 = jax.lax.stop_gradient(jnp.abs(r - y) + discount * g_sx_next)
        loss1 = jnp.mean((lambda_sa_xb - lambda_target_1) ** 2) + 0.1 * jnp.mean(
            (1 - lambda_sa_xb) ** 2
        )

        lambda_target_2 = jax.lax.stop_gradient(jnp.abs(r - y) + discount * g_xs_next)
        loss2 = jnp.mean((lambda_xb_sa - lambda_target_2) ** 2) + 0.1 * jnp.mean(
            (1 - lambda_xb_sa) ** 2
        )

        lambda_sa_sa1, lambda_sa_sa2 = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([s, a], axis=-1)
        )

        # lambda_curr = jnp.maximum(lambda_sa_xb, lambda_xb_sa)

        # loss = jnp.mean((lambda_curr - lambda_target) ** 2) + 0.1 * jnp.mean(
        #     (1 - lambda_curr) ** 2
        # )

        self_loss = jnp.mean(jnp.maximum(lambda_sa_sa1, lambda_sa_sa2) ** 2)

        return loss1 + loss2 + 0.2 * self_loss

    lambda_loss, lambda_grads = nnx.value_and_grad(state_action_metric_loss_fn)(
        state.models.state_action_metric
    )
    state.optimizers.state_action_metric.update(
        state.models.state_action_metric, lambda_grads
    )

    def min_state_action_to_state_metric_loss_fn(
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
    ):

        # d_sa_xb = jnp.abs(r - y) + discount * g_sx_next
        # d_xb_sa = jnp.abs(r - y) + discount * g_xs_next

        d_sa_xb, d_xb_sa = state.models.target_state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_target = jax.lax.stop_gradient(jnp.maximum(d_sa_xb, d_xb_sa))

        n_act_samples = 5

        act_aug = jax.random.uniform(
            perm_key,
            shape=(n_act_samples, b.shape[1], b.shape[2]),
            minval=-1.0,
            maxval=1.0,
        )
        s_repeat = jnp.repeat(s, act_aug.shape[0], axis=0)
        x_repeat = jnp.repeat(x, act_aug.shape[0], axis=0)

        act_aug = jnp.repeat(act_aug[None, :], s.shape[0], axis=0).reshape(
            -1, b.shape[1], b.shape[2]
        )

        d_sax_b_aug, d_xbs_a_aug = state.models.target_state_action_metric(
            jnp.concatenate([s_repeat, act_aug], axis=-1),
            jnp.concatenate([x_repeat, act_aug], axis=-1),
        )

        lambda_aug_target = jax.lax.stop_gradient(jnp.maximum(d_sax_b_aug, d_xbs_a_aug))

        h_sax, h_xbs = (
            min_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), x),
            min_state_action_to_state_metric(jnp.concatenate([x, b], axis=-1), s),
        )

        h_sax_repeat, h_xbs_repeat = (
            jnp.repeat(h_sax, n_act_samples, axis=0),
            jnp.repeat(h_xbs, n_act_samples, axis=0),
        )
        # d_sa_xb, d_xb_sa = state_action_metric(
        #     jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        # )

        # score_p1, score_p2 = (
        #     (h_sax - jax.lax.stop_gradient(d_sa_xb)) / beta,
        #     (h_xbs - jax.lax.stop_gradient(d_xb_sa)) / beta,
        # )
        score_p1, score_p2, score_p1_aug, score_p2_aug = (
            (h_sax - lambda_target) / beta,
            (h_xbs - lambda_target) / beta,
            (h_sax_repeat - lambda_aug_target) / beta,
            (h_xbs_repeat - lambda_aug_target) / beta,
        )
        max_score = jax.lax.stop_gradient(jnp.maximum(score_p1.max(), score_p2.max()))
        max_score_aug = jax.lax.stop_gradient(
            jnp.maximum(score_p1_aug.max(), score_p2_aug.max())
        )
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
        p1_aug = (
            jnp.exp(score_p1_aug - max_score_aug)
            - score_p1_aug * jnp.exp(-max_score_aug)
            - jnp.exp(-max_score_aug)
        )
        p2_aug = (
            jnp.exp(score_p2_aug - max_score_aug)
            - score_p2_aug * jnp.exp(-max_score_aug)
            - jnp.exp(-max_score_aug)
        )
        # loss = jnp.mean(p1) + jnp.mean(p2) + jnp.mean(p1_aug) + jnp.mean(p2_aug)
        h_sa_s = min_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), s)
        loss = jnp.mean(p1) + jnp.mean(p2) + 0.2 * jnp.mean(h_sa_s**2)
        return loss

    h_loss, h_grads = nnx.value_and_grad(min_state_action_to_state_metric_loss_fn)(
        state.models.min_state_action_to_state_metric
    )
    state.optimizers.min_state_action_to_state_metric.update(
        state.models.min_state_action_to_state_metric, h_grads
    )

    def state_metric_loss_fn(state_metric: EnsembleStateMetric):

        # h_sax, h_xbs = (
        #     target_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), x),
        #     target_state_action_to_state_metric(jnp.concatenate([x, b], axis=-1), s),
        # )
        h_sax, h_xbs = (
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([s, a], axis=-1), x
            ),
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([x, b], axis=-1), s
            ),
        )

        n_act_samples = 5
        act_aug = jax.random.uniform(
            perm_key,
            shape=(n_act_samples, b.shape[1], b.shape[2]),
            minval=-1.0,
            maxval=1.0,
        )
        s_repeat = jnp.repeat(s, act_aug.shape[0], axis=0)
        x_repeat = jnp.repeat(x, act_aug.shape[0], axis=0)
        act_aug = jnp.repeat(act_aug[None, :], s.shape[0], axis=0).reshape(
            -1, b.shape[1], b.shape[2]
        )
        h_sax_aug, h_xbs_aug = (
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([s_repeat, act_aug], axis=-1), x_repeat
            ),
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([x_repeat, act_aug], axis=-1), s_repeat
            ),
        )
        h_sax_aug, h_xbs_aug = (
            jax.lax.stop_gradient(h_sax_aug),
            jax.lax.stop_gradient(h_xbs_aug),
        )
        g_sx_repeat, g_xs_repeat = state_metric(s_repeat, x_repeat)

        h_sax, h_xbs = jax.lax.stop_gradient(h_sax), jax.lax.stop_gradient(h_xbs)
        g_sx, g_xs = state_metric(s, x)
        score_p1, score_p2, score_p1_aug, score_p2_aug = (
            (h_sax - g_sx) / beta,
            (h_xbs - g_xs) / beta,
            (h_sax_aug - g_sx_repeat) / beta,
            (h_xbs_aug - g_xs_repeat) / beta,
        )
        max_score = jax.lax.stop_gradient(jnp.maximum(score_p1.max(), score_p2.max()))
        max_score_aug = jax.lax.stop_gradient(
            jnp.maximum(score_p1_aug.max(), score_p2_aug.max())
        )

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

        p1_aug = (
            jnp.exp(score_p1_aug - max_score_aug)
            - score_p1_aug * jnp.exp(-max_score_aug)
            - jnp.exp(-max_score_aug)
        )

        p2_aug = (
            jnp.exp(score_p2_aug - max_score_aug)
            - score_p2_aug * jnp.exp(-max_score_aug)
            - jnp.exp(-max_score_aug)
        )

        # loss = jnp.mean(p1) + jnp.mean(p2) + jnp.mean(p1_aug) + jnp.mean(p2_aug)
        g_ss1, g_ss2 = state_metric(s, s)
        loss = (
            jnp.mean(p1)
            + jnp.mean(p2)
            + 0.1 * jnp.mean((1 - jnp.max(g_sx, -1)) ** 2)
            + 0.1 * jnp.mean((1 - jnp.max(g_xs, -1)) ** 2)
            + 0.2 * jnp.mean(jnp.maximum(g_ss1, g_ss2) ** 2)
        )

        return loss

    g_loss, g_grads = nnx.value_and_grad(state_metric_loss_fn)(
        state.models.state_metric
    )
    state.optimizers.state_metric.update(state.models.state_metric, g_grads)

    g_ss1, g_ss2 = state.models.state_metric(s, s)
    avg_self_state_asymmetry = jnp.mean(jnp.abs(g_ss1 - g_ss2))
    self_state_diff = jnp.mean(jnp.maximum(g_ss1, g_ss2))

    g_sx, g_xs = state.models.state_metric(s, x)
    cross_state_diff = jnp.mean(jnp.maximum(g_sx, g_xs))
    avg_cross_state_asymmetry = jnp.mean(jnp.abs(g_sx - g_xs))

    d_sa1, d_sa2 = state.models.state_action_metric(
        jnp.concatenate([s, a], axis=-1), jnp.concatenate([s, a], axis=-1)
    )
    avg_self_sa_asymmetry = jnp.mean(jnp.abs(d_sa1 - d_sa2))
    lambda_self = jnp.maximum(d_sa1, d_sa2)
    self_state_action_diff = jnp.mean(lambda_self)

    d_saxb, d_xbsa = state.models.state_action_metric(
        jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
    )
    avg_cross_sa_asymmetry = jnp.mean(jnp.abs(d_saxb - d_xbsa))
    lambda_cross = jnp.maximum(d_saxb, d_xbsa)

    cross_state_action_diff = jnp.mean(lambda_cross)

    self_state_action_to_state_distance = state.models.min_state_action_to_state_metric(
        jnp.concatenate([s, a], axis=-1), s
    )
    cross_state_action_to_state_distance = (
        state.models.min_state_action_to_state_metric(
            jnp.concatenate([s, a], axis=-1), x
        )
    )

    h_lambda_self = jnp.mean(jnp.abs(lambda_self - self_state_action_to_state_distance))
    h_lambda_cross = jnp.mean(
        jnp.abs(lambda_cross - cross_state_action_to_state_distance)
    )

    polyak_update(
        state.models.target_state_metric, state.models.state_metric, config.update_tau
    )
    polyak_update(
        state.models.target_state_action_to_state_metric,
        state.models.min_state_action_to_state_metric,
        config.update_tau,
    )
    polyak_update(
        state.models.target_state_action_metric,
        state.models.state_action_metric,
        config.update_tau,
    )
    polyak_update(state.models.target_critic, state.models.critic, config.update_tau)

    alpha = jnp.exp(state.models.log_alpha())

    agent_aux = AgentAux(
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        log_pi_mean=log_pi_mean,
        q1_mean=q1_mean,
        q2_mean=q2_mean,
        alpha=alpha,
    )

    metric_aux = MetricAux(
        state_metric_loss=g_loss,
        state_action_metric_loss=lambda_loss,
        state_action_to_state_metric_loss=h_loss,
        self_state_distance=self_state_diff,
        cross_state_distance=cross_state_diff,
        self_state_action_distance=self_state_action_diff,
        cross_state_action_distance=cross_state_action_diff,
        self_state_asymmetry_avg=avg_self_state_asymmetry,
        cross_state_asymmetry_avg=avg_cross_state_asymmetry,
        self_state_action_asymmetry_avg=avg_self_sa_asymmetry,
        cross_state_action_asymmetry_avg=avg_cross_sa_asymmetry,
        self_state_action_to_state_distance=jnp.mean(
            self_state_action_to_state_distance
        ),
        cross_state_action_to_state_distance=jnp.mean(
            cross_state_action_to_state_distance
        ),
        h_lambda_diff_self=h_lambda_self,
        h_lambda_diff_cross=h_lambda_cross,
    )

    return (agent_aux, metric_aux)


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    state: TrainingState,
    config,
    key: jnp.ndarray,
):

    num_steps = config.log_freq

    def body_fun(i, carry):
        key, env_state, buffer_state, running_state, obs_normalizer, state, val = carry

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env,
            env_state,
            state.models.actor,
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
            key, env_state, buffer_state, obs_normalizer, state, _ = carry

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = sac_train_step(
                state,
                batch,
                config,
                train_key,
            )

            return (key, env_state, buffer_state, obs_normalizer, state, val)

        init_val = (AgentAux(), MetricAux())

        key, _, buffer_state, obs_normalizer, state, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, n_env_state, buffer_state, obs_normalizer, state, init_val),
        )

        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            state,
            val,
        )

    init_val = (AgentAux(), MetricAux())
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        state,
        init_val,
    )

    (_, env_state, buffer_state, running_state, obs_normalizer, state, val) = (
        nnx.fori_loop(0, num_steps, body_fun, init_carry)
    )

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        num_steps * config.num_envs,
    )


def transfer_tuning(
    state: TrainingState,
    data: Transition,
    config,
    key: jnp.ndarray,
):
    obs = data.observation
    act = data.action
    reward = data.reward
    discount = data.discount
    next_obs = data.next_observation
    jnp.exp(state.models.log_alpha())

    key, next_key = jax.random.split(key)
    next_act, next_log_prob = state.models.actor(next_obs, next_key)

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
    g_sx_next, g_xs_next = state.models.target_state_metric(s_next, x_next)
    u_target = jnp.maximum(g_sx_next, g_xs_next)
    jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

    key, act_key = jax.random.split(key)

    def act_match_loss_fn(actor: SACGaussianActor):
        omega = 1.0
        action, _ = actor(s, act_key)
        # action_prime = b
        action_prime, _ = actor(x, act_key)

        g_sx, g_xs = state.models.state_metric(s, x)
        u = jnp.maximum(g_sx, g_xs)
        u = u / omega
        state_diff_weight = jnp.exp(jax.lax.stop_gradient(-u + u.min()))
        u_min = jax.lax.stop_gradient(-u.min())

        d_spi_xb, d_xb_spi = state.models.state_action_metric(
            jnp.concatenate([s, action], axis=-1),
            jnp.concatenate([x, action_prime], axis=-1),
        )
        d = jnp.maximum(d_spi_xb, d_xb_spi)
        d = d / omega
        state_action_diff_weight = jnp.exp(jax.lax.stop_gradient(-d + d.min()))
        d_min = jax.lax.stop_gradient(-d.min())

        # loss = jnp.mean(
        #     state_diff_weight
        #     * state_action_diff_weight
        #     * jnp.exp(u_min)
        #     * jnp.exp(d_min)
        #     * (action - jax.lax.stop_gradient(action_prime)) ** 2
        # )

        loss = jnp.mean(
            jax.lax.stop_gradient(jnp.abs(1 - u)) * d
            - jax.lax.stop_gradient(jnp.abs(u)) * d
        )

        return loss

    act_rep_loss, act_rep_grads = nnx.value_and_grad(act_match_loss_fn)(
        state.models.actor
    )
    state.optimizers.actor.update(state.models.actor, act_rep_grads)

    # def critic_rep_loss_fn(critic: EnsembleCritic):
    #     lambda_sa_xb, lambda_xb_sa = state.models.state_action_metric(
    #         jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
    #     )
    #     d_sa_xb = jnp.maximum(lambda_sa_xb, lambda_xb_sa)

    #     Q_sa1, Q_sa2 = critic(jnp.concatenate([s, a], axis=-1))
    #     Q_xb1, Q_xb2 = critic(jnp.concatenate([x, b], axis=-1))

    #     loss = jnp.mean(
    #         jax.nn.relu(jnp.abs(Q_sa1 - Q_xb1) - jax.lax.stop_gradient(d_sa_xb))
    #     ) + jnp.mean(
    #         jax.nn.relu(jnp.abs(Q_sa2 - Q_xb2) - jax.lax.stop_gradient(d_sa_xb))
    #     )
    #     return loss

    # critic_rep_loss, critic_rep_grads = nnx.value_and_grad(critic_rep_loss_fn)(
    #     state.models.critic
    # )
    # state.optimizers.critic.update(state.models.critic, critic_rep_grads)

    metric_aux = MetricAux(act_rep_loss=act_rep_loss)

    return metric_aux


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def tune_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    state: TrainingState,
    config,
    key: jnp.ndarray,
):

    num_steps = config.transfer_freq

    def body_fun(i, carry):

        key, buffer_state, running_state, obs_normalizer, state, val = carry

        def do_train(j, carry):
            key, buffer_state, obs_normalizer, state, _ = carry

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = transfer_tuning(
                state,
                batch,
                config,
                train_key,
            )

            return (key, buffer_state, obs_normalizer, state, val)

        init_val = MetricAux()

        key, buffer_state, obs_normalizer, state, val = nnx.fori_loop(
            0,
            config.transfer_steps,
            do_train,
            (key, buffer_state, obs_normalizer, state, init_val),
        )

        return (
            key,
            # n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            state,
            val,
        )

    init_val = MetricAux()
    init_carry = (
        key,
        # env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        state,
        init_val,
    )

    (_, buffer_state, running_state, obs_normalizer, state, val) = nnx.fori_loop(
        0, num_steps, body_fun, init_carry
    )

    return val, env_state, running_state, obs_normalizer, buffer_state, num_steps


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
    num_itr: int,
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


def main(args, cfg_env=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
    )

    jax.default_device = jax.devices(args.device)[args.device_id]

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
            "transfer_freq": args.transfer_freq,
            "transfer_steps": args.transfer_steps,
            "reward_scaling": args.reward_scaling,
            "num_eval_envs": args.num_eval_envs,
        }
    )

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

    config["target_entropy"] = float(act_dim) * -0.5

    config_data = make_static_config_from_dict("SACConfig", config)()

    state_metric = EnsembleStateMetric(
        rngs=rngs, obs_dim=obs_dim, hidden_size=config["hidden_size"]
    )

    state_metric_opt = nnx.Optimizer(
        model=state_metric,
        tx=optax.adam(learning_rate=config["lr"]),
        # tx=optax.chain(
        #     optax.clip_by_global_norm(config["max_grad_norm"]),
        #     optax.adam(learning_rate=config["lr"]),
        #     # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        # ),
        wrt=nnx.Param,
    )

    state_action_metric = EnsembleStateActionMetric(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )

    state_action_metric_opt = nnx.Optimizer(
        model=state_action_metric,
        tx=optax.adam(learning_rate=config["lr"]),
        # tx=optax.chain(
        #     optax.clip_by_global_norm(config["max_grad_norm"]),
        #     optax.adam(learning_rate=config["lr"]),
        #     # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        # ),
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
        tx=optax.adam(learning_rate=config["lr"]),
        # tx=optax.chain(
        #     optax.clip_by_global_norm(config["max_grad_norm"]),
        #     optax.adam(learning_rate=config["lr"]),
        #     # optax.adamw(learning_rate=config["lr"], weight_decay=0.01),
        # ),
        wrt=nnx.Param,
    )

    target_state_metric = deepcopy(state_metric)
    target_state_action_metric = deepcopy(state_action_metric)

    target_state_action_to_state_metric = deepcopy(min_state_action_to_state_metric)

    actor = SACGaussianActor(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
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
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
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

    target_critic = deepcopy(critic)

    log_alpha = Scalar(float(jnp.log(config["init_temperature"])))
    log_alpha_opt = nnx.Optimizer(
        model=log_alpha,
        tx=optax.adam(learning_rate=config["lr"]),
        wrt=nnx.Param,
    )

    

    models = Models(
        critic=critic,
        target_critic=target_critic,
        actor=actor,
        state_metric=state_metric,
        target_state_metric=target_state_metric,
        state_action_metric=state_action_metric,
        target_state_action_metric=target_state_action_metric,
        min_state_action_to_state_metric=min_state_action_to_state_metric,
        target_state_action_to_state_metric=target_state_action_to_state_metric,
        log_alpha=log_alpha,
    )

    optimizers = Optimizers(
        critic=critic_opt,
        actor=actor_opt,
        log_alpha=log_alpha_opt,
        state_metric=state_metric_opt,
        state_action_metric=state_action_metric_opt,
        min_state_action_to_state_metric=min_state_action_to_state_metric_opt,
        
    )

    state = TrainingState(models=models, optimizers=optimizers)

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

    warmup_iters = max(1, config["warmup_samples"] // config["num_envs"])
    env_state, buffer_state, obs_normalizer = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        policy=actor,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        num_itr=warmup_iters,
    )
    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start SAC training")
    logger.log(f"{config}")
    steps = buffer.size(buffer_state)
    steps = int(buffer.size(buffer_state))
    next_save = steps + config["save_freq"]

    while steps < config["total_env_steps"]:
        prng_key, subkey = jax.random.split(prng_key)

        val = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
            obs_normalizer=obs_normalizer,
            state=state,
            config=config_data,
            key=subkey,
        )

        (
            agent_aux,
            metric_aux,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            num_steps,
        ) = val

        tune_val = tune_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
            obs_normalizer=obs_normalizer,
            state=state,
            config=config_data,
            key=subkey,
        )

        (
            tune_aux,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            tune_steps,
        ) = tune_val

        steps += num_steps
        logger.logged = False

        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_critic", agent_aux.critic_loss.item())
        logger.log_tabular("Loss/Loss_actor", agent_aux.actor_loss.item())
        logger.log_tabular("Loss/Loss_alpha", agent_aux.alpha_loss.item())
        logger.log_tabular(
            "Loss/Loss_state_action_metric", metric_aux.state_action_metric_loss.item()
        )
        logger.log_tabular(
            "Loss/Loss_state_metric", metric_aux.state_metric_loss.item()
        )
        logger.log_tabular(
            "Loss/Loss_state_action_to_state",
            metric_aux.state_action_to_state_metric_loss.item(),
        )
        logger.log_tabular("Loss/Act_rep_loss", tune_aux.act_rep_loss.item())
        # logger.log_tabular("Loss/Critic_rep_loss", tune_aux.critic_rep_loss.item())
        # logger.log_tabular("Loss/Value_Matching_loss", val_match_loss.item())

        logger.log_tabular("SAC/Alpha", agent_aux.alpha.item())
        logger.log_tabular("SAC/LogPi_mean", agent_aux.log_pi_mean.item())
        logger.log_tabular("SAC/Q1_mean", agent_aux.q1_mean.item())
        logger.log_tabular("SAC/Q2_mean", agent_aux.q2_mean.item())

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
            "Metric/self_state_distance", metric_aux.self_state_distance.item()
        )
        logger.log_tabular(
            "Metric/cross_state_distance", metric_aux.cross_state_distance.item()
        )
        logger.log_tabular(
            "Metric/self_state_action_distance",
            metric_aux.self_state_action_distance.item(),
        )
        logger.log_tabular(
            "Metric/cross_state_action_distance",
            metric_aux.cross_state_action_distance.item(),
        )

        logger.log_tabular(
            "Metric/avg_self_state_asymmetry",
            metric_aux.self_state_asymmetry_avg.item(),
        )
        logger.log_tabular(
            "Metric/avg_cross_state_asymmetry",
            metric_aux.cross_state_asymmetry_avg.item(),
        )
        logger.log_tabular(
            "Metric/avg_self_sa_asymmetry",
            metric_aux.self_state_action_asymmetry_avg.item(),
        )
        logger.log_tabular(
            "Metric/avg_cross_sa_asymmetry",
            metric_aux.cross_state_action_asymmetry_avg.item(),
        )

        logger.log_tabular(
            "Metric/self_state_action_to_state_distance",
            metric_aux.self_state_action_to_state_distance.item(),
        )
        logger.log_tabular(
            "Metric/cross_state_action_to_state_distance",
            metric_aux.cross_state_action_to_state_distance.item(),
        )
        logger.log_tabular(
            "Metric/h_lambda_diff_self", metric_aux.h_lambda_diff_self.item()
        )
        logger.log_tabular(
            "Metric/h_lambda_diff_cross", metric_aux.h_lambda_diff_cross.item()
        )

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

        logger.log_tabular(
            "Eval/Return",
            float(eval_return),
        )

        logger.dump_tabular()

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
