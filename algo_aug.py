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

# from utils.rep_models import (
#     EnsembleStateActionMetric,
#     EnsembleStateMetric,
#     MinStateActiontoStateMetric,
#     StateActionDiffuseMetric,
#     StateAsymmetricMetric,
# )
# use utils.metric_models to not get nan
from utils.algo_metric import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
# from utils.algo_models import EnsembleCritic, SACGaussianActor, Scalar, get_tree_norm
from utils.models import EnsembleCritic, SACGaussianActor, Scalar, get_tree_norm
# [NSTEP] same helpers as sac_single.py / dhpg.py, so all three scripts share
# one source of truth for how returns are computed.
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
    "nstep": 3,  # NEW; 1 reproduces the old 1-step behaviour EXACTLY
    "bootstrap_on_truncation": True,  # NEW
    "rep_lr_scale": 1.5,
    # ---- [EPSBALL] epsilon-ball action augmentation ------------------------
    # The metric recursions involve a max over actions (for g) and a min over
    # actions (for h). Both are only well-posed on a COMPACT action set, which
    # is what the epsilon ball gives us: the extremum is attained, so the
    # sampled max/min is a valid (if conservative) estimator of it.
    "n_act_samples": 5,  # samples per ball, INCLUDING the data action
    "act_eps": 0.1,  # L-inf radius; useful range ~0.05-0.2 for tanh actions
    "aug_weight": 1.0,  # weight on every augmented term
}


def polyak_update(target_model, curr_model, tau: float):

    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


# ---------------------------------------------------------------------------
# [EPSBALL] helpers
# ---------------------------------------------------------------------------
def sample_action_ball(key, act, n_samples, eps, include_center=True):
    """Uniform samples from the L-inf epsilon ball around `act`.

    `act` has shape (B, 1, act_dim) (the replay batch keeps a singleton env
    axis). The return has shape (B * n_samples, 1, act_dim), laid out so that
    row ``i * n_samples + j`` is (state i, perturbation j). That is exactly the
    layout produced by ``jnp.repeat(s, n_samples, axis=0)``, so the state and
    action tensors line up without any further bookkeeping.

    Samples are clipped to [-1, 1] so they stay inside the tanh action range.

    When ``include_center`` is True, sample j = 0 is the unperturbed data
    action. That matters: it guarantees the sampled max is >= the data-action
    value and the sampled min is <= it, so the augmented estimate of the
    extremum is never LOOSER than the un-augmented one it replaces.
    """
    noise = jax.random.uniform(
        key, (act.shape[0], n_samples) + act.shape[1:], minval=-eps, maxval=eps
    )
    if include_center:
        noise = noise.at[:, 0].set(0.0)
    aug = jnp.clip(act[:, None, ...] + noise, -1.0, 1.0)
    return aug.reshape((act.shape[0] * n_samples,) + act.shape[1:])


def exp_hinge(scores):
    """NaN-safe ``e^z - z - 1`` evaluated for a list of score arrays.

    The factorisation ``e^{-m} (e^{z} - z - 1)`` with ``m = max(z)`` is only
    numerically safe when ``m >= 0``; otherwise ``exp(-m)`` overflows. Early in
    training every augmented score is strongly negative (h << lambda, h << g),
    which is precisely what produced the inf -> NaN in the original augmented
    blocks. Clamping the shift at 0 removes that failure mode.

    All arrays in `scores` share one shift, so their relative weighting is
    preserved.
    """
    m = jax.lax.stop_gradient(
        jnp.maximum(jnp.max(jnp.stack([jnp.max(s) for s in scores])), 0.0)
    )
    em = jnp.exp(-m)
    return [jnp.exp(s - m) - s * em - em for s in scores]


def sac_train_step(
    state: TrainingState, data: Transition, config, key: jnp.ndarray
) -> tuple[AgentAux, MetricAux]:
    obs = data.observation
    act = data.action
    reward = data.reward
    # [NSTEP] `discount` now comes from nstep_aggregate and ALREADY CONTAINS
    # gamma^n. Every consumer below uses it RAW — see the critic loss.
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
        # [NSTEP] CHANGED: dropped `config.gamma *` — gamma^n is inside
        # `discount`. For nstep=1, discount == gamma * (1 - done), i.e.
        # identical to the old line.
        target_q = jax.lax.stop_gradient(
            reward * config.reward_scaling + discount * next_v
        )
        q1, q2 = critic(jnp.concatenate([obs, act], axis=-1))
        q_error = jnp.stack([q1, q2], axis=-1) - target_q[..., None]
        # [NSTEP] `truncation` is now a WINDOW-level flag: 1.0 iff the n-step
        # window ended at a time limit. Reduces to the raw per-step flag at n=1.
        q_error = q_error * (1.0 - truncation)[..., None]
        loss = 0.5 * jnp.mean(jnp.square(q_error))
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True
    )(state.models.critic, state.models.target_critic)

    s, a, r, s_next = obs, act, reward[:, None], next_obs
    batch = jnp.concatenate([s, a, r, s_next], axis=-1)
    key, perm_key = jax.random.split(key)
    batch = jax.random.permutation(perm_key, batch)
    batch = batch[:, -1]

    # [EPSBALL] Dedicated keys. The original reused `perm_key` for the
    # permutation AND for both uniform draws, so the h-loss and the g-loss saw
    # byte-identical noise and the augmentation covered half as much of the
    # ball as it appeared to.
    key, ka_h, kb_h, ka_g, kb_g = jax.random.split(key, 5)

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

    # [EPSBALL] One set of perturbations, drawn ONCE and shared by the lambda
    # loss and the h loss. Sharing is deliberate: it means lambda is fit at
    # exactly the points where h queries it, and h is fit at exactly the points
    # where g queries it, so nothing downstream ever extrapolates off-support.
    K = config.n_act_samples
    a_aug = sample_action_ball(ka_h, a, K, config.act_eps)  # (B*K, 1, act_dim)
    b_aug = sample_action_ball(kb_h, b, K, config.act_eps)  # (B*K, 1, act_dim)
    s_rep = jnp.repeat(s, K, axis=0)  # (B*K, 1, obs_dim)
    x_rep = jnp.repeat(x, K, axis=0)  # (B*K, 1, obs_dim)
    sa_aug = jnp.concatenate([s_rep, a_aug], axis=-1)
    xb_aug = jnp.concatenate([x_rep, b_aug], axis=-1)

    def actor_loss_fn(actor):
        pi, log_pi = actor(obs, key_actor)
        q1, q2 = state.models.critic(jnp.concatenate([obs, pi], axis=-1))
        sac_loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))

        pi_x, _ = actor(x, key_actor)
        pi_x = jax.lax.stop_gradient(pi_x)

        g_sx, g_xs = state.models.state_metric(s, x)
        u = jax.lax.stop_gradient(jnp.clip(jnp.maximum(g_sx, g_xs), 0.0, 1.0))

        d1, d2 = state.models.state_action_metric(
            jnp.concatenate([s, pi], axis=-1),
            jnp.concatenate([x, pi_x], axis=-1),
        )
        d = jnp.maximum(d1, d2)
        rep_loss = jnp.mean((1.0 - u) * d + u * jax.nn.relu(1.0 - d))

        loss = sac_loss + config.rep_lr_scale * rep_loss
        return loss, (jnp.mean(log_pi), sac_loss, rep_loss)

    (actor_loss_tot, (log_pi_mean, actor_loss, act_rep_loss)), actor_grads = (
        nnx.value_and_grad(actor_loss_fn, has_aux=True)(state.models.actor)
    )

    def state_action_metric_loss_fn(state_action_metric: EnsembleStateActionMetric):
        sa = jnp.concatenate([s, a], axis=-1)
        xb = jnp.concatenate([x, b], axis=-1)

        lambda_sa_xb, lambda_xb_sa = state_action_metric(sa, xb)

        lambda_target_1 = jax.lax.stop_gradient(jnp.abs(r - y) + discount * g_sx_next)
        loss1 = jnp.mean((lambda_sa_xb - lambda_target_1) ** 2) + 0.1 * jnp.mean(
            (1 - lambda_sa_xb) ** 2
        )

        lambda_target_2 = jax.lax.stop_gradient(jnp.abs(r - y) + discount * g_xs_next)
        loss2 = jnp.mean((lambda_xb_sa - lambda_target_2) ** 2) + 0.1 * jnp.mean(
            (1 - lambda_xb_sa) ** 2
        )

        lambda_sa_sa1, lambda_sa_sa2 = state_action_metric(sa, sa)

        self_loss = jnp.mean(jnp.maximum(lambda_sa_sa1, lambda_sa_sa2) ** 2)

        # ---- [EPSBALL] the same three terms, on the ball --------------------
        # Both the h loss and the g loss evaluate the TARGET lambda network at
        # perturbed actions. Without this block lambda is only ever trained at
        # logged actions, so those queries are pure extrapolation. Fitting it
        # here closes the loop.
        #
        # Validity rests on eps being small: we reuse |r - y| and the
        # next-state metric from the logged transition, which assumes the
        # reward gap and the successor states barely move over the ball. Above
        # eps ~ 0.2 that assumption starts injecting bias.
        lam1_aug, lam2_aug = state_action_metric(sa_aug, xb_aug)
        rgap = jnp.repeat(jnp.abs(r - y), K, axis=0)
        disc = jnp.repeat(discount, K, axis=0)
        tgt1_aug = jax.lax.stop_gradient(
            rgap + disc * jnp.repeat(g_sx_next, K, axis=0)
        )
        tgt2_aug = jax.lax.stop_gradient(
            rgap + disc * jnp.repeat(g_xs_next, K, axis=0)
        )

        # regression + the SAME "bound the distance to 1" term as the data path
        loss1_aug = jnp.mean((lam1_aug - tgt1_aug) ** 2) + 0.1 * jnp.mean(
            (1 - lam1_aug) ** 2
        )
        loss2_aug = jnp.mean((lam2_aug - tgt2_aug) ** 2) + 0.1 * jnp.mean(
            (1 - lam2_aug) ** 2
        )

        # and the SAME "self distance is 0" term: (s, a~) is identical to
        # itself for every a~ in the ball, so lambda must vanish there too.
        lam_self1_aug, lam_self2_aug = state_action_metric(sa_aug, sa_aug)
        self_loss_aug = jnp.mean(jnp.maximum(lam_self1_aug, lam_self2_aug) ** 2)

        return (
            loss1
            + loss2
            + 0.2 * self_loss
            + config.aug_weight * (loss1_aug + loss2_aug + 0.2 * self_loss_aug)
        )

    lambda_loss, lambda_grads = nnx.value_and_grad(state_action_metric_loss_fn)(
        state.models.state_action_metric
    )

    def min_state_action_to_state_metric_loss_fn(
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
    ):
        sa = jnp.concatenate([s, a], axis=-1)
        xb = jnp.concatenate([x, b], axis=-1)

        d_sa_xb, d_xb_sa = state.models.target_state_action_metric(sa, xb)
        lambda_target = jax.lax.stop_gradient(jnp.maximum(d_sa_xb, d_xb_sa))

        h_sax = min_state_action_to_state_metric(sa, x)
        h_xbs = min_state_action_to_state_metric(xb, s)

        p1, p2 = exp_hinge(
            [(h_sax - lambda_target) / beta, (h_xbs - lambda_target) / beta]
        )

        # ---- [EPSBALL] ------------------------------------------------------
        # FIXED vs the original: the two sides are now perturbed
        # INDEPENDENTLY, and h is evaluated at the SAME perturbed action that
        # lambda is. The original fed one shared `act_aug` to both arguments of
        # lambda while comparing against h at the *data* action, so the left
        # arguments did not match and the b-direction that the min ranges over
        # was never explored at all.
        d1_aug, d2_aug = state.models.target_state_action_metric(sa_aug, xb_aug)
        lambda_aug_target = jax.lax.stop_gradient(jnp.maximum(d1_aug, d2_aug))

        h_sax_aug = min_state_action_to_state_metric(sa_aug, x_rep)
        h_xbs_aug = min_state_action_to_state_metric(xb_aug, s_rep)

        p1_aug, p2_aug = exp_hinge(
            [
                (h_sax_aug - lambda_aug_target) / beta,
                (h_xbs_aug - lambda_aug_target) / beta,
            ]
        )

        # "self distance is 0", on the data action and on the ball. h((s,a~),s)
        # is a min over b that includes b = a~, so it must vanish for every
        # perturbed action, not just the logged one.
        h_sa_s = min_state_action_to_state_metric(sa, s)
        h_sa_s_aug = min_state_action_to_state_metric(sa_aug, s_rep)

        # NOTE: h never had a "bound to 1" term in the original, so none is
        # invented here. If you want one for symmetry with lambda, add:
        #   + 0.1 * jnp.mean((1 - h_sax) ** 2)
        #   + config.aug_weight * 0.1 * jnp.mean((1 - h_sax_aug) ** 2)
        return (
            jnp.mean(p1)
            + jnp.mean(p2)
            + 0.2 * jnp.mean(h_sa_s**2)
            + config.aug_weight
            * (
                jnp.mean(p1_aug)
                + jnp.mean(p2_aug)
                + 0.2 * jnp.mean(h_sa_s_aug**2)
            )
        )

    h_loss, h_grads = nnx.value_and_grad(min_state_action_to_state_metric_loss_fn)(
        state.models.min_state_action_to_state_metric
    )

    def state_metric_loss_fn(state_metric: EnsembleStateMetric):
        tgt_h = state.models.target_state_action_to_state_metric
        sa = jnp.concatenate([s, a], axis=-1)
        xb = jnp.concatenate([x, b], axis=-1)

        h_sax = jax.lax.stop_gradient(tgt_h(sa, x))
        h_xbs = jax.lax.stop_gradient(tgt_h(xb, s))

        g_sx, g_xs = state_metric(s, x)

        p1, p2 = exp_hinge([(h_sax - g_sx) / beta, (h_xbs - g_xs) / beta])

        # ---- [EPSBALL] ------------------------------------------------------
        # g(s, x) = max_a h((s,a), x). This is the term that actually needs the
        # ball: with only the logged action the max is estimated from a single
        # point. Fresh draws here (ka_g / kb_g) rather than reusing the h
        # draws, so the two losses cover different parts of the ball.
        a_aug_g = sample_action_ball(ka_g, a, K, config.act_eps)
        b_aug_g = sample_action_ball(kb_g, b, K, config.act_eps)

        h_sax_aug = jax.lax.stop_gradient(
            tgt_h(jnp.concatenate([s_rep, a_aug_g], axis=-1), x_rep)
        )
        h_xbs_aug = jax.lax.stop_gradient(
            tgt_h(jnp.concatenate([x_rep, b_aug_g], axis=-1), s_rep)
        )

        # g takes no action argument, so its value is unchanged by the
        # perturbation — repeat instead of re-running the network. This is
        # mathematically identical to the original `state_metric(s_repeat,
        # x_repeat)` call and gradients flow through the repeat.
        g_sx_rep = jnp.repeat(g_sx, K, axis=0)
        g_xs_rep = jnp.repeat(g_xs, K, axis=0)

        p1_aug, p2_aug = exp_hinge(
            [(h_sax_aug - g_sx_rep) / beta, (h_xbs_aug - g_xs_rep) / beta]
        )

        g_ss1, g_ss2 = state_metric(s, s)

        # NOTE: the boundedness and self-distance terms below are NOT
        # duplicated on the ball. The perturbation lives in action space and g
        # is a function of states only, so (s, x) and (s, s) are literally the
        # same inputs for every sample in the ball — an augmented copy would be
        # the identical term counted K times, i.e. just a weight change.
        return (
            jnp.mean(p1)
            + jnp.mean(p2)
            + config.aug_weight * (jnp.mean(p1_aug) + jnp.mean(p2_aug))
            + 0.1 * jnp.mean((1 - jnp.max(g_sx, -1)) ** 2)
            + 0.1 * jnp.mean((1 - jnp.max(g_xs, -1)) ** 2)
            + 0.2 * jnp.mean(jnp.maximum(g_ss1, g_ss2) ** 2)
        )

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
    nstep_fifo,  # NEW
    nstep_count,  # NEW
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
            nstep_fifo,  # NEW
            nstep_count,  # NEW
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
        # transition, insert the aggregated n-step one. The buffer is untouched.
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
            nstep_fifo,  # NEW
            nstep_count,  # NEW
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
        nstep_fifo,  # NEW
        nstep_count,  # NEW
        state,
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
        state,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)

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


# def transfer_tuning(
#     state: TrainingState,
#     data: Transition,
#     config,
#     key: jnp.ndarray,
# ):
#     obs = data.observation
#     act = data.action
#     reward = data.reward
#     # [NSTEP] as above: already contains gamma^n, used raw.
#     discount = data.discount
#     next_obs = data.next_observation
#     jnp.exp(state.models.log_alpha())

#     key, next_key = jax.random.split(key)
#     next_act, next_log_prob = state.models.actor(next_obs, next_key)

#     s, a, r, s_next = obs, act, reward[:, None], next_obs
#     batch = jnp.concatenate([s, a, r, s_next], axis=-1)
#     key, perm_key = jax.random.split(key)
#     batch = jax.random.permutation(perm_key, batch)
#     batch = batch[:, -1]

#     obs_dim, act_dim = obs.shape[-1], act.shape[-1]
#     # B X 1 X (obs_dim, act_dim, None, obs_dim)
#     x, b, y, x_next = (
#         batch[:, :obs_dim],
#         batch[:, obs_dim : obs_dim + act_dim],
#         batch[:, obs_dim + act_dim],
#         batch[:, obs_dim + act_dim + 1 :],
#     )

#     r = r[:, -1]  ## shaping reward from (B, 1, 1) --> (B, 1)

#     x, b, y, x_next = x[:, None, :], b[:, None, :], y[:, None], x_next[:, None, :]
#     g_sx_next, g_xs_next = state.models.target_state_metric(s_next, x_next)
#     u_target = jnp.maximum(g_sx_next, g_xs_next)
#     jax.lax.stop_gradient(jnp.abs(r - y) + discount * u_target)

#     key, act_key = jax.random.split(key)

#     def act_match_loss_fn(actor: SACGaussianActor):
#         omega = 1.0
#         action, _ = actor(s, act_key)
#         # action_prime = b
#         action_prime, _ = actor(x, act_key)

#         g_sx, g_xs = state.models.state_metric(s, x)
#         u = jnp.maximum(g_sx, g_xs)
#         u = u / omega
#         state_diff_weight = jnp.exp(jax.lax.stop_gradient(-u + u.min()))
#         u_min = jax.lax.stop_gradient(-u.min())

#         d_spi_xb, d_xb_spi = state.models.state_action_metric(
#             jnp.concatenate([s, action], axis=-1),
#             jnp.concatenate([x, action_prime], axis=-1),
#         )
#         d = jnp.maximum(d_spi_xb, d_xb_spi)
#         d = d / omega
#         state_action_diff_weight = jnp.exp(jax.lax.stop_gradient(-d + d.min()))
#         d_min = jax.lax.stop_gradient(-d.min())

#         # loss = jnp.mean(
#         #     state_diff_weight
#         #     * state_action_diff_weight
#         #     * jnp.exp(u_min)
#         #     * jnp.exp(d_min)
#         #     * (action - jax.lax.stop_gradient(action_prime)) ** 2
#         # )

#         loss = jnp.mean(
#             jax.lax.stop_gradient(jnp.abs(1 - u)) * d
#             - jax.lax.stop_gradient(jnp.abs(u)) * d
#         )

#         return loss

#     act_rep_loss, act_rep_grads = nnx.value_and_grad(act_match_loss_fn)(
#         state.models.actor
#     )
#     state.optimizers.actor.update(state.models.actor, act_rep_grads)

#     # def critic_rep_loss_fn(critic: EnsembleCritic):
#     #     lambda_sa_xb, lambda_xb_sa = state.models.state_action_metric(
#     #         jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
#     #     )
#     #     d_sa_xb = jnp.maximum(lambda_sa_xb, lambda_xb_sa)

#     #     Q_sa1, Q_sa2 = critic(jnp.concatenate([s, a], axis=-1))
#     #     Q_xb1, Q_xb2 = critic(jnp.concatenate([x, b], axis=-1))

#     #     loss = jnp.mean(
#     #         jax.nn.relu(jnp.abs(Q_sa1 - Q_xb1) - jax.lax.stop_gradient(d_sa_xb))
#     #     ) + jnp.mean(
#     #         jax.nn.relu(jnp.abs(Q_sa2 - Q_xb2) - jax.lax.stop_gradient(d_sa_xb))
#     #     )
#     #     return loss

#     # critic_rep_loss, critic_rep_grads = nnx.value_and_grad(critic_rep_loss_fn)(
#     #     state.models.critic
#     # )
#     # state.optimizers.critic.update(state.models.critic, critic_rep_grads)

#     metric_aux = MetricAux(act_rep_loss=act_rep_loss)

#     return metric_aux


# @functools.partial(nnx.jit, static_argnames=("env", "buffer"))
# def tune_n_steps(
#     env,
#     env_state,
#     buffer_state,
#     buffer,
#     running_state,
#     obs_normalizer,
#     state: TrainingState,
#     config,
#     key: jnp.ndarray,
# ):

#     num_steps = config.transfer_freq

#     def body_fun(i, carry):

#         key, buffer_state, running_state, obs_normalizer, state, val = carry

#         def do_train(j, carry):
#             key, buffer_state, obs_normalizer, state, _ = carry

#             buffer_state, batch = buffer.sample(buffer_state)
#             batch = batch._replace(
#                 observation=obs_normalizer.normalize(batch.observation),
#                 next_observation=obs_normalizer.normalize(batch.next_observation),
#             )
#             key, train_key = jax.random.split(key)

#             val = transfer_tuning(
#                 state,
#                 batch,
#                 config,
#                 train_key,
#             )

#             return (key, buffer_state, obs_normalizer, state, val)

#         init_val = MetricAux()

#         key, buffer_state, obs_normalizer, state, val = nnx.fori_loop(
#             0,
#             config.transfer_steps,
#             do_train,
#             (key, buffer_state, obs_normalizer, state, init_val),
#         )

#         return (
#             key,
#             # n_env_state,
#             buffer_state,
#             running_state,
#             obs_normalizer,
#             state,
#             val,
#         )

#     init_val = MetricAux()
#     init_carry = (
#         key,
#         # env_state,
#         buffer_state,
#         running_state,
#         obs_normalizer,
#         state,
#         init_val,
#     )

#     (_, buffer_state, running_state, obs_normalizer, state, val) = nnx.fori_loop(
#         0, num_steps, body_fun, init_carry
#     )

#     return val, env_state, running_state, obs_normalizer, buffer_state, num_steps


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
    """
    Collect `num_itr` transitions before training begins.

    Uses jax.lax.scan (not a Python loop) so the warmup is JIT-compiled.
    The policy has random initial weights so actions are effectively random.

    [NSTEP] runs the same FIFO as the main loop, so the seed phase and the
    training phase put transitions of the SAME kind into the buffer.
    """

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

    # [EPSBALL] optional CLI overrides. getattr with a None default so this
    # still runs if sac_args() has not been extended with these flags yet.
    for _k in ("n_act_samples", "act_eps", "aug_weight"):
        _v = getattr(args, _k, None)
        if _v is not None:
            config[_k] = _v

    # [NSTEP] per-task override, same as sac_single.py / dhpg.py: the walker
    # domain uses nstep=1. Must run BEFORE config_data is frozen.
    if args.task.lower().startswith("walker"):
        config["nstep"] = 1
    # if args.task.lower() == "cartpoleswingupsparse":
    #     config["rep_lr_scale"] = 1.0

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

    # ── [NSTEP] n-step FIFO state (leading dim nstep, then num_envs) ───────
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
    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start SAC training")
    logger.log(f"{config}")
    logger.log(
        f"[nstep] nstep={config['nstep']} "
        f"bootstrap_on_truncation={config['bootstrap_on_truncation']} "
        f"(discount from the buffer already contains gamma^nstep)"
    )
    logger.log(
        f"[epsball] n_act_samples={config['n_act_samples']} "
        f"act_eps={config['act_eps']} aug_weight={config['aug_weight']}"
    )
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
            nstep_fifo=nstep_fifo,  # NEW arg
            nstep_count=nstep_count,  # NEW arg
            key=subkey,
        )

        (
            agent_aux,
            metric_aux,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            nstep_fifo,  # NEW
            nstep_count,  # NEW
            num_steps,
        ) = val

        # tune_val = tune_n_steps(
        #     env=env,
        #     env_state=env_state,
        #     buffer_state=buffer_state,
        #     buffer=buffer,
        #     running_state=running_state,
        #     obs_normalizer=obs_normalizer,
        #     state=state,
        #     config=config_data,
        #     key=subkey,
        # )

        # (
        #     tune_aux,
        #     env_state,
        #     running_state,
        #     obs_normalizer,
        #     buffer_state,
        #     tune_steps,
        # ) = tune_val

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
    algo = os.path.basename(__file__).split(".")[0]
    # algo = f"{algo}_{args.rep_lr_scale}"
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