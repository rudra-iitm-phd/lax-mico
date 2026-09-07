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
from utils.algo_metric import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
from utils.algo_models import EnsembleCritic, SACGaussianActor, Scalar, get_tree_norm

# [NSTEP] same helpers as sac_single.py / dhpg.py.
from utils.buffer import (
    RunningMeanStd,
    RunningStatistics,
    UniformSamplingQueue,
    nstep_aggregate,
    nstep_fifo_init,
    nstep_fifo_push,
    nstep_template_from_dims,
)
from utils.logger import EpochLogger
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
    # ---- [NSTEP] n-step returns, same keys/values as sac_single.py ----
    "nstep": 3,
    "bootstrap_on_truncation": True,
    "rep_lr_scale": 1.5,
    # ================= [QREG] metric constraints on the critic ===========
    # Both terms act on the SAME buffer pair the metric losses use:
    #   (s, a) from the batch, (x, b) from the shuffled batch.
    #
    # 1. BOUND  |Q(s,a) - Q(x,b)| <= d((s,a),(x,b))
    #    A one-sided hinge per direction. Inert until violated, so it can only
    #    ever remove Q-value spread that the metric says should not exist. This
    #    is the safe term; it is on by default.
    "critic_bound_scale": 0.1,
    #
    # 2. CONTRASTIVE  |Q(s,a) - Q(x,b)| -> u * d((s,a),(x,b))
    #    u = soft_gate(state distance). u -> 0 pulls the Q gap to 0, u -> 1
    #    pushes it up to the bound, graded in between. This asserts equality,
    #    not just an inequality, so it is opt-in. See the note in
    #    sac_train_step before turning it on.
    "critic_con_scale": 0.05,
    #
    # Both terms are off for the first `warmup` env steps, then ramped linearly
    # to full strength over `ramp`. Before that the metrics sit near their
    # random init and constraining Q to them constrains it to noise.
    "critic_reg_warmup_steps": int(1e5),
    "critic_reg_ramp_steps": int(1e5),
    # Converts the metric into Q units. d is fit against |r - y| + discount * g
    # on the RAW reward; Q is fit against reward * reward_scaling. If the metric
    # recursion were an exact fixed point this would be 1.0; the 0.1*(1-lambda)^2
    # pull in the metric losses compresses d, so > 1 loosens the budget.
    "critic_rep_metric_scale": 1.0,
    # Crossover of the soft gate: the distance at which u = 0.5.
    #   > 0.0 -> fixed constant, keeps an absolute meaning for "equivalent"
    #   = 0.0 -> batch mean of the distance, fully scale-free
    "metric_gate_scale": 0.0,
    # Replace the actor's clip(u, 0, 1) / relu(1 - d) with non-saturating
    # versions. See the comment in actor_loss_fn.
    "actor_soft_gate": True,
}


# =============================================================================
# helpers
# =============================================================================
def polyak_update(target_model, curr_model, tau: float):

    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


def soft_gate(dist, gate_scale, eps=1e-6):
    """Non-saturating stand-in for clip(dist, 0, 1).

        u = dist / (dist + c)

    - u in [0, 1), strictly monotone: dist = 2 and dist = 5 stay
      distinguishable instead of both collapsing onto 1.
    - u = 0 exactly at dist = 0, so "equivalent" still means equivalent.
    - u = 0.5 at dist = c, so c is the attract/repel crossover.
    - du/d(dist) = c / (dist + c)^2 > 0 everywhere: no dead zone.

    c = gate_scale if gate_scale > 0, else the batch mean of dist (which makes
    the gate relative: it grades pairs against each other and is immune to the
    metric's overall scale drifting, at the cost of the absolute meaning).
    """
    dist = jax.nn.relu(dist)  # metrics are non-negative by construction; guard
    c = jnp.where(gate_scale > 0.0, gate_scale, jnp.mean(dist)) + eps
    return jax.lax.stop_gradient(dist / (dist + c))


def reg_ramp_at(steps, cfg):
    """Warmup + linear ramp in [0, 1] for the critic regularizers. Plain Python:
    evaluated once per outer-loop block and passed in as a traced scalar, so it
    does not retrigger compilation."""
    w0 = cfg["critic_reg_warmup_steps"]
    w1 = max(int(cfg["critic_reg_ramp_steps"]), 1)
    if steps < w0:
        return 0.0
    return min(1.0, (steps - w0) / w1)


# =============================================================================
# train step
# =============================================================================
def sac_train_step(
    state: TrainingState,
    data: Transition,
    config,
    reg_ramp: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[AgentAux, MetricAux]:
    obs = data.observation
    act = data.action
    reward = data.reward
    # [NSTEP] `discount` comes from nstep_aggregate and ALREADY CONTAINS
    # gamma^n. Every consumer below uses it RAW.
    discount = data.discount
    next_obs = data.next_observation
    truncation = data.extras["state_extras"]["truncation"]
    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
    alpha = jnp.exp(state.models.log_alpha())
    beta = 1.0

    # ---------------------------------------------------------------------
    # [QREG] MOVED UP. This block used to sit between the critic loss and the
    # actor loss; the critic loss now needs the (x, b) pair batch. The RNG
    # stream is UNCHANGED -- perm_key is still the first split taken after the
    # 4-way split above, nothing in between consumes the key, and no new split
    # is introduced anywhere in this function. With critic_bound_scale and
    # critic_con_scale at 0 this file is bit-identical to the original.
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # [QREG] budgets for the critic, on the buffer pair (s,a) vs (x,b).
    #
    # TARGET nets rather than online ones: a constraint that moves on every
    # gradient step is a moving target for the critic.
    # stop_gradient everywhere: the critic must not be able to satisfy its own
    # constraint by inflating the metric that defines it.
    # Rescaled into Q units: d is fit against |r - y| + discount * g on the RAW
    # reward, Q is fit against reward * reward_scaling.
    # Nothing is clipped: d = 2 simply means a budget of 2.
    # ---------------------------------------------------------------------
    sa_pair = jnp.concatenate([s, a], axis=-1)
    xb_pair = jnp.concatenate([x, b], axis=-1)

    d_budget_scale = config.reward_scaling * config.critic_rep_metric_scale

    d_sa_xb_raw, d_xb_sa_raw = state.models.target_state_action_metric(
        sa_pair, xb_pair
    )
    # directional budgets, matching the two directional heads: lambda_sa_xb is
    # fit against g_sx_next, lambda_xb_sa against g_xs_next.
    d_sa_xb = jax.lax.stop_gradient(d_sa_xb_raw) * d_budget_scale
    d_xb_sa = jax.lax.stop_gradient(d_xb_sa_raw) * d_budget_scale
    # symmetric budget, matching the actor's max(d1, d2).
    d_pair = jnp.maximum(d_sa_xb, d_xb_sa)

    # gate on the state distance, soft so it never saturates.
    g_sx_u, g_xs_u = state.models.target_state_metric(s, x)
    u_state = soft_gate(jnp.maximum(g_sx_u, g_xs_u), config.metric_gate_scale)

    # graded target for |Q(s,a) - Q(x,b)|:
    #   u -> 0 : target -> 0        equivalent states
    #   u -> 1 : target -> d_pair   dissimilar, the gap reaches its bound
    #   between: proportional, so the metric's magnitude is actually used.
    # One live branch everywhere: no dead zone, unlike a pair of hinges.
    #
    # CAVEAT, which is why critic_con_scale defaults to 0: a and b are
    # unrelated buffer actions, so when u -> 0 this asks for
    # Q(s,a) = Q(x,b) ~ Q(s,a) = Q(s,b), i.e. it drives the ACTION GAP to zero
    # on near-equivalent states. The bound term below has no such problem
    # because it only ever constrains from above. If you enable this and the
    # policy stops improving while td_loss stays healthy, that is the cause.
    target_gap = jax.lax.stop_gradient(u_state * d_pair)

    def alpha_loss_fn(log_alpha):
        _, log_prob = state.models.actor(obs, key_alpha)
        a_ = jnp.exp(log_alpha())
        loss = jnp.mean(a_ * jax.lax.stop_gradient(-log_prob - config.target_entropy))
        return loss

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(state.models.log_alpha)

    def critic_loss_fn(critic, target_critic):
        next_act, next_log_prob = state.models.actor(next_obs, key_critic)
        q1_t, q2_t = target_critic(jnp.concatenate([next_obs, next_act], axis=-1))
        next_v = jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
        # [NSTEP] no `config.gamma *`: gamma^n lives inside `discount`.
        target_q = jax.lax.stop_gradient(
            reward * config.reward_scaling + discount * next_v
        )
        q1, q2 = critic(sa_pair)
        q_error = jnp.stack([q1, q2], axis=-1) - target_q[..., None]
        # [NSTEP] window-level truncation flag.
        q_error = q_error * (1.0 - truncation)[..., None]
        td_loss = 0.5 * jnp.mean(jnp.square(q_error))

        # ---- [QREG] both terms use the buffer pair -----------------------
        q1_xb, q2_xb = critic(xb_pair)

        # 1. BOUND: |Q(s,a) - Q(x,b)| <= d. One hinge per direction, so the
        #    quasi-metric's asymmetry is preserved; for a symmetric d the pair
        #    is exactly |dQ| <= d. Gradient flows through both Q's on purpose:
        #    a violation should pull them together, not drag one onto a frozen
        #    other.
        bound_loss = 0.5 * jnp.mean(
            jax.nn.relu(q1 - q1_xb - d_sa_xb)
            + jax.nn.relu(q1_xb - q1 - d_xb_sa)
            + jax.nn.relu(q2 - q2_xb - d_sa_xb)
            + jax.nn.relu(q2_xb - q2 - d_xb_sa)
        )

        # 2. CONTRASTIVE: drive the gap to u * d. L1, not squared: constant
        #    gradient magnitude, so it will not fight the TD term as the Q
        #    scale and the metric scale drift.
        gap1 = jnp.abs(q1 - q1_xb)
        gap2 = jnp.abs(q2 - q2_xb)
        con_loss = 0.5 * (
            jnp.mean(jnp.abs(gap1 - target_gap))
            + jnp.mean(jnp.abs(gap2 - target_gap))
        )

        loss = td_loss + reg_ramp * (
            config.critic_bound_scale * bound_loss
            + config.critic_con_scale * con_loss
        )
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True
    )(state.models.critic, state.models.target_critic)

    def actor_loss_fn(actor):
        pi, log_pi = actor(obs, key_actor)
        q1, q2 = state.models.critic(jnp.concatenate([obs, pi], axis=-1))
        sac_loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))

        pi_x, _ = actor(x, key_actor)
        pi_x = jax.lax.stop_gradient(pi_x)

        g_sx, g_xs = state.models.state_metric(s, x)
        d1, d2 = state.models.state_action_metric(
            jnp.concatenate([s, pi], axis=-1),
            jnp.concatenate([x, pi_x], axis=-1),
        )
        d = jnp.maximum(d1, d2)

        # Original: u = clip(max(g), 0, 1), margin = 1.0. That combination has
        # an exact dead zone -- once g >= 1 the attract half is zeroed by
        # (1 - u), and once d >= 1 the repel half is zeroed by the relu, so far
        # pairs contribute NO gradient at all. With metrics reaching 2 this can
        # cover most of the batch.
        # The soft version keeps the identical shape but takes both constants
        # out: the gate never saturates, and the margin tracks the current
        # scale of d instead of a fixed 1.0.
        u_hard = jax.lax.stop_gradient(jnp.clip(jnp.maximum(g_sx, g_xs), 0.0, 1.0))
        u_soft = soft_gate(jnp.maximum(g_sx, g_xs), config.metric_gate_scale)
        u = jnp.where(config.actor_soft_gate, u_soft, u_hard)
        margin = jnp.where(
            config.actor_soft_gate, jax.lax.stop_gradient(jnp.mean(d)), 1.0
        )
        rep_loss = jnp.mean((1.0 - u) * d + u * jax.nn.relu(margin - d))

        loss = sac_loss + config.rep_lr_scale * rep_loss
        return loss, (jnp.mean(log_pi), sac_loss, rep_loss)

    (actor_loss_tot, (log_pi_mean, actor_loss, act_rep_loss)), actor_grads = (
        nnx.value_and_grad(actor_loss_fn, has_aux=True)(state.models.actor)
    )

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

        self_loss = jnp.mean(jnp.maximum(lambda_sa_sa1, lambda_sa_sa2) ** 2)

        return loss1 + loss2 + 0.2 * self_loss

    lambda_loss, lambda_grads = nnx.value_and_grad(state_action_metric_loss_fn)(
        state.models.state_action_metric
    )

    def min_state_action_to_state_metric_loss_fn(
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
    ):

        d_sa_xb_h, d_xb_sa_h = state.models.target_state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_target_h = jax.lax.stop_gradient(jnp.maximum(d_sa_xb_h, d_xb_sa_h))

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

        score_p1, score_p2, score_p1_aug, score_p2_aug = (
            (h_sax - lambda_target_h) / beta,
            (h_xbs - lambda_target_h) / beta,
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
        h_sa_s = min_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), s)
        loss = jnp.mean(p1) + jnp.mean(p2) + 0.2 * jnp.mean(h_sa_s**2)
        return loss

    h_loss, h_grads = nnx.value_and_grad(min_state_action_to_state_metric_loss_fn)(
        state.models.min_state_action_to_state_metric
    )

    def state_metric_loss_fn(state_metric: EnsembleStateMetric):

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
        max_score = jax.lax.stop_gradient(
            jnp.maximum(jnp.maximum(score_p1.max(), score_p2.max()), 0.0)
        )
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

    state.optimizers.log_alpha.update(state.models.log_alpha, alpha_grads)
    state.optimizers.critic.update(state.models.critic, critic_grads)
    state.optimizers.actor.update(state.models.actor, actor_grads)

    state.optimizers.state_action_metric.update(
        state.models.state_action_metric, lambda_grads
    )
    state.optimizers.min_state_action_to_state_metric.update(
        state.models.min_state_action_to_state_metric, h_grads
    )
    state.optimizers.state_metric.update(state.models.state_metric, g_grads)

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
        act_rep_loss=act_rep_loss,
    )

    return (agent_aux, metric_aux)


# =============================================================================
# rollout + train loop
# =============================================================================
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
    nstep_fifo,
    nstep_count,
    reg_ramp,
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
            nstep_fifo,
            nstep_count,
            state,
            val,
        ) = carry

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env,
            env_state,
            state.models.actor,
            obs_normalizer,
            env_key,
            extra_fields=("truncation",),
        )
        # [NSTEP] the window is built on the ROLLOUT side: push the raw 1-step
        # transition, insert the aggregated n-step one.
        nstep_fifo = nstep_fifo_push(nstep_fifo, transition)
        nstep_count = jnp.minimum(nstep_count + 1, config.nstep)
        nstep_transition = nstep_aggregate(
            nstep_fifo,
            nstep_count,
            config.gamma,
            config.nstep,
            config.bootstrap_on_truncation,
        )
        buffer_state = buffer.insert(buffer_state, nstep_transition)
        # the normalizer still sees the RAW 1-step observation
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
                reg_ramp,
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
            nstep_fifo,
            nstep_count,
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
        nstep_fifo,
        nstep_count,
        state,
        init_val,
    )

    (
        _,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        nstep_fifo,
        nstep_count,
        state,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        nstep_fifo,
        nstep_count,
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
    # brax reports both mean and std across the eval envs; report both.
    return jnp.mean(ret), jnp.std(ret)


def prefill_buffer(
    key,
    env,
    env_state,
    buffer_state,
    policy,
    buffer,
    obs_normalizer,
    config,
    num_itr: int,
    nstep_fifo,
    nstep_count,
):
    """Collect `num_itr` transitions before training begins.

    [NSTEP] runs the same FIFO as the main loop, so the seed phase and the
    training phase put transitions of the SAME kind into the buffer.
    """

    def body(carry, _):
        key, env_state, buffer_state, obs_normalizer, fifo, count = carry
        key, subkey = jax.random.split(key)
        n_state, transition = actor_step(
            env=env,
            env_state=env_state,
            policy=policy,
            obs_normalizer=obs_normalizer,
            key=subkey,
            extra_fields=("truncation",),
        )
        fifo = nstep_fifo_push(fifo, transition)
        count = jnp.minimum(count + 1, config.nstep)
        buffer_state = buffer.insert(
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


# =============================================================================
# main
# =============================================================================
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
            "rep_lr_scale": args.rep_lr_scale,
        }
    )

    # [QREG] optional CLI overrides. Add the flags to sac_args() if you want to
    # sweep them; without the flags these keep their default_cfg values.
    for _k in (
        "critic_bound_scale",
        "critic_con_scale",
        "critic_reg_warmup_steps",
        "critic_reg_ramp_steps",
        "critic_rep_metric_scale",
        "metric_gate_scale",
        "actor_soft_gate",
    ):
        _v = getattr(args, _k, None)
        if _v is not None:
            config[_k] = _v

    # [NSTEP] per-task override, same as sac_single.py / dhpg.py.
    if args.task.lower().startswith("walker"):
        config["nstep"] = 1

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
        wrt=nnx.Param,
    )

    state_action_metric = EnsembleStateActionMetric(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )
    state_action_metric_opt = nnx.Optimizer(
        model=state_action_metric,
        tx=optax.adam(learning_rate=config["lr"]),
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
        wrt=nnx.Param,
    )

    critic = EnsembleCritic(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )
    critic_opt = nnx.Optimizer(
        model=critic,
        tx=optax.adam(learning_rate=config["lr"]),
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

    # ── [NSTEP] n-step FIFO state ─────────────────────────────────────────
    nstep_fifo = nstep_fifo_init(
        nstep_template_from_dims(config["num_envs"], obs_dim, act_dim),
        config["nstep"],
    )
    nstep_count = jnp.array(0, dtype=jnp.int32)

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
    (
        env_state,
        buffer_state,
        obs_normalizer,
        nstep_fifo,
        nstep_count,
    ) = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        policy=actor,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        config=config_data,
        num_itr=warmup_iters,
        nstep_fifo=nstep_fifo,
        nstep_count=nstep_count,
    )

    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start SAC training")
    logger.log(f"{config}")
    logger.log(
        f"[nstep] nstep={config['nstep']} "
        f"bootstrap_on_truncation={config['bootstrap_on_truncation']} "
        f"(discount from the buffer already contains gamma^nstep)"
    )
    logger.log(
        f"[qreg] bound_scale={config['critic_bound_scale']} "
        f"con_scale={config['critic_con_scale']} "
        f"warmup={config['critic_reg_warmup_steps']} "
        f"ramp={config['critic_reg_ramp_steps']} "
        f"metric_scale={config['critic_rep_metric_scale']} "
        f"gate_scale={config['metric_gate_scale']} "
        f"actor_soft_gate={config['actor_soft_gate']}"
    )

    steps = int(buffer.size(buffer_state))
    next_save = steps + config["save_freq"]

    while steps < config["total_env_steps"]:
        prng_key, subkey = jax.random.split(prng_key)

        # [QREG] warmup + ramp in [0, 1], recomputed once per block and passed
        # in as a traced scalar so it does not retrigger compilation.
        ramp_now = reg_ramp_at(steps, config)

        val = train_n_steps(
            env=env,
            env_state=env_state,
            buffer_state=buffer_state,
            buffer=buffer,
            running_state=running_state,
            obs_normalizer=obs_normalizer,
            state=state,
            config=config_data,
            nstep_fifo=nstep_fifo,
            nstep_count=nstep_count,
            reg_ramp=jnp.float32(ramp_now),
            key=subkey,
        )

        (
            agent_aux,
            metric_aux,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            nstep_fifo,
            nstep_count,
            num_steps,
        ) = val

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
        logger.log_tabular("Loss/Act_rep_loss", metric_aux.act_rep_loss.item())

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

        logger.log_tabular("Eval/Return", float(eval_return))

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
    algo = os.path.basename(__file__).split(".")[0]
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