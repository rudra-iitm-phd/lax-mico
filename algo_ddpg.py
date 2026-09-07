"""
Metric-learning agent, DDPG base.

Companion to dhpg.py. The RL core here is the SAME code as the DHPG script's
core - same DeterministicActor, same SingleQCritic, same TD3-style target
smoothing, same exploration schedule, same delayed actor/Polyak cadence, same
joint-backward-then-N-optimizer-steps structure - so that `ddpg_metric` vs
`dhpg` differs ONLY in the auxiliary representation losses (your metrics vs the
author's homomorphism). Anything else would confound the comparison.

=========================================================================
BASELINE PARITY CHECKLIST vs dhpg.py  (the point of this file)
=========================================================================
Same by construction:
  actor            DeterministicActor, orthogonal init, zero bias, 2x256, tanh
  critic           SingleQCritic, ONE head, no clipped double-Q, 2x256
  target nets      target_actor + target_critic (+ your 3 metric targets)
  TD target        r + gamma^n * Q_targ(s', clip(pi_targ(s') + clip(eps)))
  stddev_clip      0.3
  exploration      uniform while env_step < num_expl_steps (2000), then
                   pi(s) + N(0, sigma) UNCLIPPED, sigma = linear(1.0, 0.1, D)
  schedule origin  clock starts at end of the seed phase; D = 0.1 *
                   total_env_steps ([FIX A]/[FIX F] in dhpg.py)
  update_every     2 - gates the actor AND EVERY Polyak update
  optimizer        plain Adam, lr from the CLI, NO gradient clipping on the
                   base networks
  loss structure   ONE joint backward over {critic + auxiliary losses}, then
                   one .update() per optimizer, then the delayed actor step
  n-step           nstep=3 (walker_* -> 1), shared FIFO, `discount` already
                   contains gamma^n and is used RAW everywhere
  reward_scaling   applied ONCE at the top, so the auxiliary targets are
                   learned in the units the critic sees
  harness          same sac_args() parser, same unconditional config.update,
                   same warmup accounting, same evaluate(), same monotone
                   checkpointing, same obs normalisation

Deliberately different (this is your novelty, and nothing else):
  the three metric losses and their target networks, and the act-matching
  transfer phase.

Two asymmetries are LEFT IN but flagged - read these before you report numbers:
  [ASYM-1] clip_metric_grads=True keeps your optax.clip_by_global_norm(10) on
      the three metric optimizers. DHPG clips nothing. The base networks
      (actor/critic) are unclipped in BOTH files, so the comparison is clean;
      this only affects your own modules. Set False for total parity.
  [ASYM-2] tune_n_steps runs transfer_freq * transfer_steps EXTRA actor
      optimizer steps per log cycle, on the act-matching loss, and those are
      NOT subject to update_every_steps. DHPG has no analogue. If those
      counts are large the act-matching gradient dominates the DPG gradient
      and the agent will look like it "isn't learning" for reasons that have
      nothing to do with the metric. Check transfer_freq/transfer_steps
      against train_per_step before blaming the algorithm.

=========================================================================
UPDATE CADENCE  (grep "[CADENCE]")
=========================================================================
Previously the metric target networks tracked EVERY step while the actor and
the actor/critic targets moved every 2 - two different effective taus inside
one agent, and the metric targets moving twice as fast as the critic target
they are supposed to be consistent with. Now there is exactly ONE cadence,
matching the author's ("all three Polyak updates gated on the same
update_every_steps condition as the actor"):

  every gradient step        critic, state_action_metric,
                             min_state_action_to_state_metric, state_metric
                             -> one joint backward, four optimizer steps
  every update_every_steps   actor, and ALL FIVE Polyak updates
                             (target_actor, target_critic,
                              target_state_metric,
                              target_state_action_metric,
                              target_state_action_to_state_metric)

update_every_steps lives in default_cfg and is NOT exposed to the CLI, in
both files. If you change it, change it in both, or the baseline is void.

=========================================================================
JOINT LOSS  (grep "[JOINT]")
=========================================================================
dhpg.py builds ONE loss - critic + homomorphic_coef*lax_bisim + transition +
reward - takes ONE nnx.value_and_grad over five modules, and then calls five
separate optimizer.update()s. This file now has the same shape: one
joint_loss_fn over {critic, state_action_metric,
min_state_action_to_state_metric, state_metric}, one backward, four
optimizer steps, in a fixed order with the critic first.

Be clear about what this does and does not change:
  - GRADIENTS ARE IDENTICAL to the previous three-separate-backwards version.
    In DHPG the joint loss is load-bearing because state_encoder and
    action_encoder receive gradient from THREE terms at once, so the terms
    must accumulate before the step. Here the four losses touch four
    DISJOINT parameter sets (every cross-reference goes through a TARGET
    network), so summing them changes nothing numerically.
  - metric_coef is therefore very nearly INERT, unlike DHPG's
    homomorphic_coef: it scales the gradient of parameters that have their
    own Adam, and Adam is invariant to a constant gradient scale (up to
    eps=1e-8). Do not tune it expecting DHPG-like behaviour.
  - What it DOES buy you: identical control flow to the baseline, and
    correctness the moment you couple the modules - e.g. if you ever share an
    encoder between the metrics, or drop a target network in favour of the
    online one, the separate-backwards version would silently drop the
    cross-terms and this one will not.

=========================================================================
TRUNCATION MASKING  (grep "[MASK]")
=========================================================================
BraxAutoResetWrapper overwrites state.obs on done, so at a truncated step
`next_observation` is the FIRST obs of the NEW episode, and the window's
`reward` is only a partial k<n sum. nstep_aggregate emits a WINDOW-level
truncation flag (1.0 iff the window ended at a time limit); at nstep=3 one
truncation contaminates THREE consecutive windows.

Audited term by term:
  critic TD                    reads next_obs, reward   -> MASKED
  state_action_metric TD       reads g(s',x'), r, y     -> MASKED, PAIRWISE
  min_state_action_to_state    reads only (s,a),(x,b)   -> clean, no mask
  state_metric                 reads only (s,a),x       -> clean, no mask
  act-matching (transfer)      reads only s, x          -> clean, no mask

The state_action_metric mask is `valid * valid[perm]`, not `valid`: the
permutation couples sample i with sample perm[i], so a contaminated PARTNER
poisons an otherwise clean sample's target through `y` and `g_sx_next`. Same
argument as the lax-bisimulation mask in dhpg.py. The shaping and
self-consistency terms - 0.1*(1-lambda)^2, self_loss, h_sa_s^2, g_ss - are
NOT masked: they depend only on the window's START state/action, which is
genuine even when the window is truncated. Masked terms use a plain masked
mean rather than renormalising by the valid count (~0.3% scale shift), as in
SAC and DHPG.

=========================================================================
NOT BIT-COMPARABLE TO THE SAC VERSION
=========================================================================
`jax.random.permutation(perm_key, batch)` became
`batch[jax.random.permutation(perm_key, B)]` so the pairwise mask can be
built; semantically identical, but the realised permutation for a given key
may differ. reward_scaling is now applied once at the top instead of only in
the TD target. The action augmentation shared by the h- and g-losses is
hoisted out of both closures (it was already drawn from the same key in both,
so the values still match; now that is structural rather than incidental).
"""

import functools
import os
import os.path as osp
import random
import sys
import time
from copy import deepcopy
from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, struct
from mujoco_playground import registry

from utils.acting import actor_step, wrap_env_for_training
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
from utils.metric_models import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
from utils.models import get_tree_norm
from utils.parameterized_models import MetricAux
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args

default_cfg = {
    # ---- harness; every one of these is overwritten by sac_args() ----
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
    # ---- n-step returns, shared with sac_single.py / dhpg.py ----
    "nstep": 3,
    "bootstrap_on_truncation": True,
    # ---- DDPG base; author-fixed, NOT exposed to the CLI. These MUST match
    #      dhpg.py's default_cfg value for value, or the baseline is void. ----
    "update_every_steps": 2,  # [CADENCE] gates the actor AND all 5 Polyaks
    "num_expl_steps": 2000,
    "stddev_clip": 0.3,
    "explore_stddev_start": 1.0,
    "explore_stddev_end": 0.1,
    "explore_stddev_decay_steps": int(1e5),
    "explore_stddev_decay_frac": 0.1,
    "explore_schedule_from_training_start": True,
    "scale_explore_to_budget": True,
    "explore_origin": 0,  # filled in by main()
    # ---- your part ----
    # [JOINT] mirrors dhpg.py's homomorphic_coef, but see the header: with
    # disjoint parameter sets and per-module Adam it is nearly inert.
    "metric_coef": 1.0,
    # [ASYM-1] your optax.clip_by_global_norm(max_grad_norm) on the three
    # metric optimizers. DHPG clips nothing anywhere. Base nets are unclipped
    # in both files either way; set False for total parity.
    "clip_metric_grads": True,
}


# --------------------------------------------------------------------------- #
# Base networks - identical to dhpg.py.
# --------------------------------------------------------------------------- #
def orthogonal_linear(rngs, in_dim, out_dim):
    """utils.utils.weight_init: orthogonal on the weight, zeros on the bias."""
    return nnx.Linear(
        in_dim,
        out_dim,
        kernel_init=jax.nn.initializers.orthogonal(),
        bias_init=nnx.initializers.zeros,
        rngs=rngs,
    )


class DeterministicActor(nnx.Module):
    """pi_theta(s) -> a in [-1, 1]^act_dim. 2 hidden layers, no LayerNorm."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, act_dim)

    def __call__(self, s, key=None):
        # `key` only for drop-in compatibility with actor_step's (obs, key)
        # signature; unused (deterministic policy).
        x = nnx.relu(self.l1(s))
        x = nnx.relu(self.l2(x))
        return jnp.tanh(self.l3(x))

    # evaluate() calls mean_action / sample; for a deterministic policy they
    # coincide - exploration noise is added by ExploratoryActor at rollout time.
    def mean_action(self, s):
        return self(s)

    def sample(self, s, key=None):
        return self(s), jnp.zeros(s.shape[:-1])


class SingleQCritic(nnx.Module):
    """ONE Q-head. No twin, no clipped double-Q. Matches DHPG's DDPGCritic."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act):
        x = nnx.relu(self.l1(obs_act))
        x = nnx.relu(self.l2(x))
        return jnp.squeeze(self.l3(x), axis=-1)


class ExploratoryActor(nnx.Module):
    """HPGAgent.act(): uniform-random while env_step < num_expl_steps, else
    pi(s) + N(0, stddev), UNCLIPPED."""

    def __init__(self, actor: DeterministicActor, stddev, use_random_action):
        self.actor = actor
        self.stddev = stddev
        self.use_random_action = use_random_action

    def __call__(self, s, key):
        key, noise_key, rand_key = jax.random.split(key, 3)
        mean_act = self.actor(s)
        noisy_act = mean_act + self.stddev * jax.random.normal(
            noise_key, mean_act.shape
        )
        random_act = jax.random.uniform(
            rand_key, mean_act.shape, minval=-1.0, maxval=1.0
        )
        return jnp.where(self.use_random_action, random_act, noisy_act), jnp.zeros(
            mean_act.shape[:-1]
        )


def explore_stddev(env_step, cfg):
    """linear(start, end, duration), clock shifted by cfg.explore_origin."""
    step = jnp.maximum(env_step - cfg.explore_origin, 0)
    frac = jnp.clip(step / cfg.explore_stddev_decay_steps, 0.0, 1.0)
    return cfg.explore_stddev_start + frac * (
        cfg.explore_stddev_end - cfg.explore_stddev_start
    )


def polyak_update(target_model, curr_model, tau: float):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


# --------------------------------------------------------------------------- #
# Containers. Local, because the base needs target_actor and has no log_alpha.
# MetricAux is imported unchanged.
# --------------------------------------------------------------------------- #
class Models(nnx.Module):
    def __init__(
        self,
        *,
        actor,
        target_actor,
        critic,
        target_critic,
        state_metric,
        target_state_metric,
        state_action_metric,
        target_state_action_metric,
        min_state_action_to_state_metric,
        target_state_action_to_state_metric,
    ):
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.state_metric = state_metric
        self.target_state_metric = target_state_metric
        self.state_action_metric = state_action_metric
        self.target_state_action_metric = target_state_action_metric
        self.min_state_action_to_state_metric = min_state_action_to_state_metric
        self.target_state_action_to_state_metric = target_state_action_to_state_metric


class Optimizers(nnx.Module):
    def __init__(
        self,
        *,
        actor,
        critic,
        state_metric,
        state_action_metric,
        min_state_action_to_state_metric,
    ):
        self.actor = actor
        self.critic = critic
        self.state_metric = state_metric
        self.state_action_metric = state_action_metric
        self.min_state_action_to_state_metric = min_state_action_to_state_metric


class TrainingState(nnx.Module):
    def __init__(self, *, models: Models, optimizers: Optimizers):
        self.models = models
        self.optimizers = optimizers


def _zero():
    return jnp.zeros((), jnp.float32)


@struct.dataclass
class DDPGAgentAux:
    critic_loss: jnp.ndarray = field(default_factory=_zero)
    actor_loss: jnp.ndarray = field(default_factory=_zero)
    joint_loss: jnp.ndarray = field(default_factory=_zero)
    q_mean: jnp.ndarray = field(default_factory=_zero)
    explore_stddev: jnp.ndarray = field(default_factory=_zero)


# --------------------------------------------------------------------------- #
# train step
# --------------------------------------------------------------------------- #
def ddpg_train_step(
    state: TrainingState,
    data: Transition,
    config,
    key: jnp.ndarray,
    grad_step: jnp.ndarray,
    env_step: jnp.ndarray,
) -> tuple[DDPGAgentAux, MetricAux]:
    obs = data.observation
    act = data.action
    # reward_scaling applied ONCE, so the metric targets are learned in the
    # same units the critic sees (dhpg.py [S2]).
    reward = data.reward * config.reward_scaling
    # `discount` comes from nstep_aggregate and ALREADY CONTAINS gamma^n.
    # Every consumer below uses it RAW.
    discount = data.discount
    next_obs = data.next_observation
    # [MASK] window-level flag: 1.0 iff the n-step window ended at a time limit.
    truncation = data.extras["state_extras"]["truncation"]
    valid = 1.0 - truncation

    key, noise_key, perm_key, act_aug_key = jax.random.split(key, 4)
    stddev = explore_stddev(env_step, config)
    beta = 1.0

    # ------------------------------------------------------------------ #
    # pair construction. Hoisted above the joint loss because three of the
    # four terms need it.
    # ------------------------------------------------------------------ #
    s, a, r, s_next = obs, act, reward[:, None], next_obs
    batch = jnp.concatenate([s, a, r, s_next], axis=-1)
    # [MASK] keep the permutation INDICES so the pairwise mask can be built.
    perm = jax.random.permutation(perm_key, obs.shape[0])
    batch = batch[perm]
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

    # [MASK] a contaminated PARTNER poisons an otherwise clean sample through
    # y and g_sx_next, so gate on BOTH ends of the pair. Shape (B, 1), same as
    # the metric outputs.
    pair_valid = valid * valid[perm]

    # ------------------------------------------------------------------ #
    # frozen targets, computed once outside the joint loss
    # ------------------------------------------------------------------ #
    next_act_mean = state.models.target_actor(next_obs)
    noise = jnp.clip(
        stddev * jax.random.normal(noise_key, next_act_mean.shape),
        -config.stddev_clip,
        config.stddev_clip,
    )
    next_action = jnp.clip(next_act_mean + noise, -1.0, 1.0)
    target_q = jax.lax.stop_gradient(
        reward
        + discount
        * state.models.target_critic(
            jnp.concatenate([next_obs, next_action], axis=-1)
        )
    )

    g_sx_next, g_xs_next = state.models.target_state_metric(s_next, x_next)

    # action augmentation, shared by the h- and g-losses (the SAC version drew
    # it from the same key in both closures; hoisting makes that structural)
    n_act_samples = 5
    act_aug = jax.random.uniform(
        act_aug_key,
        shape=(n_act_samples, b.shape[1], b.shape[2]),
        minval=-1.0,
        maxval=1.0,
    )
    s_repeat = jnp.repeat(s, n_act_samples, axis=0)
    x_repeat = jnp.repeat(x, n_act_samples, axis=0)
    act_aug = jnp.repeat(act_aug[None, :], s.shape[0], axis=0).reshape(
        -1, b.shape[1], b.shape[2]
    )

    # ------------------------------------------------------------------ #
    # [JOINT] ONE loss over {critic, state_action_metric,
    # min_state_action_to_state_metric, state_metric}, ONE backward, then one
    # .update() per optimizer - the same shape as dhpg.py's update_critic.
    # The four terms touch DISJOINT parameters (every cross-reference goes
    # through a target network), so the gradients equal the old
    # three-separate-backwards version exactly. See the header.
    # ------------------------------------------------------------------ #
    def joint_loss_fn(
        critic,
        state_action_metric: EnsembleStateActionMetric,
        min_state_action_to_state_metric: MinStateActiontoStateMetric,
        state_metric: EnsembleStateMetric,
    ):
        # ---- critic TD ------------------------------------------------ #
        q = critic(jnp.concatenate([obs, act], axis=-1))
        # [MASK] as in SAC. No 0.5 factor - matches dhpg.py.
        q_error = (q - target_q) * valid
        critic_loss = jnp.mean(q_error**2)

        # ---- state-action metric -------------------------------------- #
        lambda_sa_xb, lambda_xb_sa = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_target_1 = jax.lax.stop_gradient(jnp.abs(r - y) + discount * g_sx_next)
        # [MASK] TD-like term only; the (1 - lambda) shaping term depends on
        # the window's START (s,a)/(x,b), which is genuine under truncation.
        loss1 = jnp.mean(
            pair_valid * (lambda_sa_xb - lambda_target_1) ** 2
        ) + 0.1 * jnp.mean((1 - lambda_sa_xb) ** 2)

        lambda_target_2 = jax.lax.stop_gradient(jnp.abs(r - y) + discount * g_xs_next)
        loss2 = jnp.mean(
            pair_valid * (lambda_xb_sa - lambda_target_2) ** 2
        ) + 0.1 * jnp.mean((1 - lambda_xb_sa) ** 2)

        lambda_sa_sa1, lambda_sa_sa2 = state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([s, a], axis=-1)
        )
        self_loss = jnp.mean(jnp.maximum(lambda_sa_sa1, lambda_sa_sa2) ** 2)

        lambda_loss = loss1 + loss2 + 0.2 * self_loss

        # ---- min state-action-to-state metric ------------------------- #
        # [MASK] no mask needed: every input here is a CURRENT state/action.
        d_sa_xb, d_xb_sa = state.models.target_state_action_metric(
            jnp.concatenate([s, a], axis=-1), jnp.concatenate([x, b], axis=-1)
        )
        lambda_target = jax.lax.stop_gradient(jnp.maximum(d_sa_xb, d_xb_sa))

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
        # the _aug terms are computed but not summed into the loss, exactly as
        # in the SAC version (the line that used them is commented out there)
        _ = (score_p1_aug, score_p2_aug, max_score_aug)

        h_sa_s = min_state_action_to_state_metric(jnp.concatenate([s, a], axis=-1), s)
        h_loss = jnp.mean(p1) + jnp.mean(p2) + 0.2 * jnp.mean(h_sa_s**2)

        # ---- state metric --------------------------------------------- #
        # [MASK] no mask needed: current states/actions only.
        gh_sax, gh_xbs = (
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([s, a], axis=-1), x
            ),
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([x, b], axis=-1), s
            ),
        )
        gh_sax_aug, gh_xbs_aug = (
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([s_repeat, act_aug], axis=-1), x_repeat
            ),
            state.models.target_state_action_to_state_metric(
                jnp.concatenate([x_repeat, act_aug], axis=-1), s_repeat
            ),
        )
        gh_sax_aug, gh_xbs_aug = (
            jax.lax.stop_gradient(gh_sax_aug),
            jax.lax.stop_gradient(gh_xbs_aug),
        )
        gh_sax, gh_xbs = (
            jax.lax.stop_gradient(gh_sax),
            jax.lax.stop_gradient(gh_xbs),
        )

        g_sx_repeat, g_xs_repeat = state_metric(s_repeat, x_repeat)
        g_sx, g_xs = state_metric(s, x)

        g_score_p1, g_score_p2, g_score_p1_aug, g_score_p2_aug = (
            (gh_sax - g_sx) / beta,
            (gh_xbs - g_xs) / beta,
            (gh_sax_aug - g_sx_repeat) / beta,
            (gh_xbs_aug - g_xs_repeat) / beta,
        )
        g_max_score = jax.lax.stop_gradient(
            jnp.maximum(g_score_p1.max(), g_score_p2.max())
        )
        g_max_score_aug = jax.lax.stop_gradient(
            jnp.maximum(g_score_p1_aug.max(), g_score_p2_aug.max())
        )
        gp1 = (
            jnp.exp(g_score_p1 - g_max_score)
            - g_score_p1 * jnp.exp(-g_max_score)
            - jnp.exp(-g_max_score)
        )
        gp2 = (
            jnp.exp(g_score_p2 - g_max_score)
            - g_score_p2 * jnp.exp(-g_max_score)
            - jnp.exp(-g_max_score)
        )
        # as above: computed, not summed, matching the SAC version
        _ = (g_score_p1_aug, g_score_p2_aug, g_max_score_aug)

        g_ss1, g_ss2 = state_metric(s, s)
        g_loss = (
            jnp.mean(gp1)
            + jnp.mean(gp2)
            + 0.1 * jnp.mean((1 - jnp.max(g_sx, -1)) ** 2)
            + 0.1 * jnp.mean((1 - jnp.max(g_xs, -1)) ** 2)
            + 0.2 * jnp.mean(jnp.maximum(g_ss1, g_ss2) ** 2)
        )

        total_loss = critic_loss + config.metric_coef * (
            lambda_loss + h_loss + g_loss
        )
        return total_loss, (critic_loss, lambda_loss, h_loss, g_loss, jnp.mean(q))

    (
        (joint_loss, (critic_loss, lambda_loss, h_loss, g_loss, q_mean)),
        (critic_grads, lambda_grads, h_grads, g_grads),
    ) = nnx.value_and_grad(joint_loss_fn, argnums=(0, 1, 2, 3), has_aux=True)(
        state.models.critic,
        state.models.state_action_metric,
        state.models.min_state_action_to_state_metric,
        state.models.state_metric,
    )

    # critic first, so the delayed actor step below sees it updated - dhpg.py's
    # sequential ordering.
    state.optimizers.critic.update(state.models.critic, critic_grads)
    state.optimizers.state_action_metric.update(
        state.models.state_action_metric, lambda_grads
    )
    state.optimizers.min_state_action_to_state_metric.update(
        state.models.min_state_action_to_state_metric, h_grads
    )
    state.optimizers.state_metric.update(state.models.state_metric, g_grads)

    # ------------------------------------------------------------------ #
    # [CADENCE] actor + ALL FIVE Polyak updates, on one schedule.
    # ------------------------------------------------------------------ #
    def do_actor_update(state):
        def actor_loss_fn(actor):
            act_pi = actor(obs)
            q = state.models.critic(jnp.concatenate([obs, act_pi], axis=-1))
            return -jnp.mean(q)  # plain DPG; no alpha * log_pi

        loss, grads = nnx.value_and_grad(actor_loss_fn)(state.models.actor)
        state.optimizers.actor.update(state.models.actor, grads)

        polyak_update(
            state.models.target_critic, state.models.critic, config.update_tau
        )
        polyak_update(state.models.target_actor, state.models.actor, config.update_tau)
        polyak_update(
            state.models.target_state_metric,
            state.models.state_metric,
            config.update_tau,
        )
        polyak_update(
            state.models.target_state_action_metric,
            state.models.state_action_metric,
            config.update_tau,
        )
        polyak_update(
            state.models.target_state_action_to_state_metric,
            state.models.min_state_action_to_state_metric,
            config.update_tau,
        )
        return loss

    def skip_actor_update(state):
        return jnp.zeros((), jnp.float32)

    actor_loss = nnx.cond(
        grad_step % config.update_every_steps == 0,
        do_actor_update,
        skip_actor_update,
        state,
    )

    # ------------------------------------------------------------------ #
    # diagnostics (unchanged)
    # ------------------------------------------------------------------ #
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

    agent_aux = DDPGAgentAux(
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        joint_loss=joint_loss,
        q_mean=q_mean,
        explore_stddev=stddev,
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
    env_step,  # real-transition counter, drives the explore schedule
    grad_step,  # gradient-update counter, drives update_every_steps
    nstep_fifo,
    nstep_count,
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
            env_step,
            grad_step,
            nstep_fifo,
            nstep_count,
            state,
            val,
        ) = carry

        key, env_key = jax.random.split(key)
        # exploration lives in the rollout policy, not in the actor
        stddev = explore_stddev(env_step, config)
        use_random_action = env_step < config.num_expl_steps
        noisy_policy = ExploratoryActor(state.models.actor, stddev, use_random_action)

        n_env_state, transition = actor_step(
            env,
            env_state,
            noisy_policy,
            obs_normalizer,
            env_key,
            extra_fields=("truncation",),
        )
        # the window is built on the ROLLOUT side: push the raw 1-step
        # transition, insert the aggregated n-step one. The buffer is untouched.
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
        # the normalizer still sees the RAW 1-step observation, not the window
        obs_normalizer = obs_normalizer.update(transition.observation)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )
        env_step = env_step + config.num_envs

        def do_train(j, carry):
            key, buffer_state, obs_normalizer, grad_step, state, prev_val = carry

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            agent_aux, metric_aux = ddpg_train_step(
                state,
                batch,
                config,
                train_key,
                grad_step,
                env_step,
            )
            # [LOGFIX] actor_loss is 0.0 on skipped steps. Only the LAST inner
            # iteration's value is returned, and with train_per_step a multiple
            # of update_every_steps that last step is ALWAYS a skip - so the
            # logged actor loss would read exactly 0 forever even though the
            # actor is updating fine. Carry the most recent REAL value through.
            is_actor_update = (grad_step % config.update_every_steps) == 0
            agent_aux = agent_aux.replace(
                actor_loss=jnp.where(
                    is_actor_update, agent_aux.actor_loss, prev_val[0].actor_loss
                )
            )
            grad_step = grad_step + 1

            return (
                key,
                buffer_state,
                obs_normalizer,
                grad_step,
                state,
                (agent_aux, metric_aux),
            )

        init_val = (DDPGAgentAux(), MetricAux())

        key, buffer_state, obs_normalizer, grad_step, state, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, buffer_state, obs_normalizer, grad_step, state, init_val),
        )

        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            env_step,
            grad_step,
            nstep_fifo,
            nstep_count,
            state,
            val,
        )

    init_val = (DDPGAgentAux(), MetricAux())
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        env_step,
        grad_step,
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
        env_step,
        grad_step,
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
        env_step,
        grad_step,
        nstep_fifo,
        nstep_count,
        num_steps * config.num_envs,
    )


def transfer_tuning(
    state: TrainingState,
    data: Transition,
    config,
    key: jnp.ndarray,
):
    """Act-matching. Consumes only obs (and the permuted obs), so no truncation
    mask is needed.

    [ASYM-2] These actor steps are NOT gated by update_every_steps and have no
    analogue in dhpg.py. transfer_freq * transfer_steps of them run per log
    cycle, against roughly log_freq * train_per_step / update_every_steps DPG
    steps. If the former is not small relative to the latter, this loss - not
    the critic - is what is training the policy."""
    obs = data.observation

    s = obs
    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, obs.shape[0])
    x = s[perm]

    def act_match_loss_fn(actor: DeterministicActor):
        omega = 1.0
        action = actor(s)
        action_prime = actor(x)

        g_sx, g_xs = state.models.state_metric(s, x)
        u = jnp.maximum(g_sx, g_xs)
        u = u / omega

        d_spi_xb, d_xb_spi = state.models.state_action_metric(
            jnp.concatenate([s, action], axis=-1),
            jnp.concatenate([x, action_prime], axis=-1),
        )
        d = jnp.maximum(d_spi_xb, d_xb_spi)
        d = d / omega

        loss = jnp.mean(
            jax.lax.stop_gradient(jnp.abs(1 - u)) * d
            - jax.lax.stop_gradient(jnp.abs(u)) * d
        )

        return 0.1 * loss

    act_rep_loss, act_rep_grads = nnx.value_and_grad(act_match_loss_fn)(
        state.models.actor
    )
    state.optimizers.actor.update(state.models.actor, act_rep_grads)

    return MetricAux(act_rep_loss=act_rep_loss)


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
            buffer_state,
            running_state,
            obs_normalizer,
            state,
            val,
        )

    init_val = MetricAux()
    init_carry = (
        key,
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
    """Identical to dhpg.py's and sac_single.py's, so all baselines are scored
    by the same code."""
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
    return jnp.mean(ret), jnp.std(ret)


def prefill_buffer(
    key,
    env,
    env_state,
    buffer_state,
    actor,
    buffer,
    obs_normalizer,
    config,
    num_itr: int,
    env_step,
    nstep_fifo,
    nstep_count,
):
    """Seed phase. A deterministic actor with random weights is NOT a random
    policy, so the rollout uses ExploratoryActor with the real cutoff:
    uniform-random while env_step < num_expl_steps, then pi + N(0, sigma).
    Runs the same n-step FIFO as the main loop."""

    def body(carry, _):
        key, env_state, buffer_state, obs_normalizer, env_step, fifo, count = carry
        key, subkey = jax.random.split(key)

        stddev = explore_stddev(env_step, config)
        use_random_action = env_step < config.num_expl_steps
        policy = ExploratoryActor(actor, stddev, use_random_action)

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
        env_step = env_step + config.num_envs
        return (key, n_state, buffer_state, obs_normalizer, env_step, fifo, count), ()

    jitted_body = jax.jit(body)
    (
        (_, env_state, buffer_state, obs_normalizer, env_step, nstep_fifo, nstep_count),
        (),
    ) = jax.lax.scan(
        jitted_body,
        (
            key,
            env_state,
            buffer_state,
            obs_normalizer,
            env_step,
            nstep_fifo,
            nstep_count,
        ),
        (),
        length=num_itr,
    )
    return env_state, buffer_state, obs_normalizer, env_step, nstep_fifo, nstep_count


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

    # UNCONDITIONAL, identical to sac_single.py and dhpg.py: same parser, same
    # keys, same resolution order. Run all three with the same flags and they
    # see the same regime.
    config = dict(default_cfg)
    config.update(
        {
            "gamma": args.gamma,
            "update_tau": args.update_tau,
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

    # per-task override, same as sac_single.py / dhpg.py
    if args.task.lower().startswith("walker"):
        config["nstep"] = 1

    num_envs = config["num_envs"]
    warmup_iters = max(1, config["warmup_samples"] // num_envs)
    seed_transitions = warmup_iters * num_envs

    # resolve the exploration schedule for THIS budget, exactly as dhpg.py does
    if config["scale_explore_to_budget"]:
        config["explore_stddev_decay_steps"] = max(
            1, int(config["explore_stddev_decay_frac"] * config["total_env_steps"])
        )
    if config["explore_schedule_from_training_start"]:
        config["explore_origin"] = int(seed_transitions)

    prng_key, env_key = jax.random.split(prng_key)
    env_key = jax.random.split(env_key, num_envs)

    env = wrap_env_for_training(
        registry.load(args.task, config_overrides={"impl": "jax"}),
        episode_length=config["episode_length"],
        full_reset=False,
    )
    env_state = env.reset(env_key)
    obs_dim = env.observation_size
    act_dim = env.action_size
    obs_normalizer = RunningMeanStd.init((obs_dim,))

    config_data = make_static_config_from_dict("DDPGConfig", config)()

    # ── metric models ─────────────────────────────────────────────────────
    # [ASYM-1] clip_metric_grads keeps your clip_by_global_norm on these three.
    # The base networks below use plain Adam, as in dhpg.py.
    def make_metric_opt(model):
        if config["clip_metric_grads"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["lr"]),
            )
        else:
            tx = optax.adam(learning_rate=config["lr"])
        return nnx.Optimizer(model=model, tx=tx, wrt=nnx.Param)

    state_metric = EnsembleStateMetric(
        rngs=rngs, obs_dim=obs_dim, hidden_size=config["hidden_size"]
    )
    state_metric_opt = make_metric_opt(state_metric)

    state_action_metric = EnsembleStateActionMetric(
        rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )
    state_action_metric_opt = make_metric_opt(state_action_metric)

    min_state_action_to_state_metric = MinStateActiontoStateMetric(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )
    min_state_action_to_state_metric_opt = make_metric_opt(
        min_state_action_to_state_metric
    )

    target_state_metric = deepcopy(state_metric)
    target_state_action_metric = deepcopy(state_action_metric)
    target_state_action_to_state_metric = deepcopy(min_state_action_to_state_metric)

    # ── base networks: plain Adam, no clipping (dhpg.py parity) ───────────
    actor = DeterministicActor(rngs, obs_dim, act_dim, config["hidden_size"])
    actor_opt = nnx.Optimizer(
        model=actor, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )
    target_actor = deepcopy(actor)  # DDPG needs a target policy; SAC did not

    critic = SingleQCritic(rngs, obs_dim, act_dim, config["hidden_size"])
    critic_opt = nnx.Optimizer(
        model=critic, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )
    target_critic = deepcopy(critic)

    models = Models(
        actor=actor,
        target_actor=target_actor,
        critic=critic,
        target_critic=target_critic,
        state_metric=state_metric,
        target_state_metric=target_state_metric,
        state_action_metric=state_action_metric,
        target_state_action_metric=target_state_action_metric,
        min_state_action_to_state_metric=min_state_action_to_state_metric,
        target_state_action_to_state_metric=target_state_action_to_state_metric,
    )

    optimizers = Optimizers(
        actor=actor_opt,
        critic=critic_opt,
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

    # ── n-step FIFO state (leading dim nstep, then num_envs) ──────────────
    nstep_fifo = nstep_fifo_init(
        nstep_template_from_dims(num_envs, obs_dim, act_dim),
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

    # make the harness/algorithm split auditable, in the SAME format dhpg.py
    # prints, so the two run logs can be diffed line for line
    logger.log(
        f"[harness] num_envs={num_envs} total_env_steps={config['total_env_steps']} "
        f"batch={config['batch_size']} train_per_step={config['train_per_step']} "
        f"UTD={config['train_per_step'] / num_envs:.4f} "
        f"seed={seed_transitions} transitions "
        f"log_every={config['log_freq'] * num_envs} transitions"
    )
    logger.log(
        f"[algorithm] base=DDPG lr={config['lr']} tau={config['update_tau']} "
        f"nstep={config['nstep']} update_every={config['update_every_steps']} "
        f"stddev=linear({config['explore_stddev_start']},"
        f"{config['explore_stddev_end']},{config['explore_stddev_decay_steps']}) "
        f"origin={config['explore_origin']} stddev_clip={config['stddev_clip']} "
        f"bootstrap_on_truncation={config['bootstrap_on_truncation']} "
        f"(discount from the buffer already contains gamma^nstep)"
    )
    dpg_steps_per_cycle = (
        config["log_freq"] * config["train_per_step"] / config["update_every_steps"]
    )
    tune_steps_per_cycle = config["transfer_freq"] * config["transfer_steps"]
    logger.log(
        f"[metric] metric_coef={config['metric_coef']} "
        f"clip_metric_grads={config['clip_metric_grads']} "
        f"actor steps per log cycle: DPG={dpg_steps_per_cycle:.0f} "
        f"act_match={tune_steps_per_cycle} "
        f"(ratio={tune_steps_per_cycle / max(dpg_steps_per_cycle, 1):.2f}; "
        f"dhpg.py has no act_match phase)"
    )

    # ── warmup ────────────────────────────────────────────────────────────
    logger.log("Start prefilling replay buffer")
    env_step = jnp.array(0, dtype=jnp.int32)
    prng_key, buffer_key = jax.random.split(prng_key)

    (
        env_state,
        buffer_state,
        obs_normalizer,
        env_step,
        nstep_fifo,
        nstep_count,
    ) = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        actor=actor,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        config=config_data,
        num_itr=warmup_iters,
        env_step=env_step,
        nstep_fifo=nstep_fifo,
        nstep_count=nstep_count,
    )

    # ── main training loop ────────────────────────────────────────────────
    logger.log("Start DDPG (metric) training")
    logger.log(f"{config}")
    steps = int(buffer.size(buffer_state))
    grad_step = jnp.array(0, dtype=jnp.int32)
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
            env_step=env_step,
            grad_step=grad_step,
            nstep_fifo=nstep_fifo,
            nstep_count=nstep_count,
            key=subkey,
        )

        (
            agent_aux,
            metric_aux,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            env_step,
            grad_step,
            nstep_fifo,
            nstep_count,
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

        logger.log_tabular("Loss/Loss_joint", agent_aux.joint_loss.item())
        logger.log_tabular("Loss/Loss_critic", agent_aux.critic_loss.item())
        logger.log_tabular("Loss/Loss_actor", agent_aux.actor_loss.item())
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

        logger.log_tabular("DDPG/Q_mean", agent_aux.q_mean.item())
        logger.log_tabular("DDPG/Explore_stddev", agent_aux.explore_stddev.item())
        logger.log_tabular("DDPG/Grad_steps", int(grad_step))

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

    # Log path: runs/<experiment>/<task>/ddpg_metric/seed-000-YYYY-MM-DD-HH-MM-SS/
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "seed-" + str(args.seed).zfill(3)
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]  # "ddpg_metric"
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