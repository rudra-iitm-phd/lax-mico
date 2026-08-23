"""
Deep Homomorphic Policy Gradient (DHPG) - state-observation, deterministic
variant ('hpg_update_type=double_add' in the author's config).

This version is built from a direct read of the author's actual PyTorch
source (agents/hpg.py, models/core.py, models/transition_model.py,
utils/utils.py, cfgs/agent/hpg.yaml, cfgs/config.yaml) rather than from the
paper's prose/pseudocode alone. Several things the paper's Eq. 9-13 and
Appendix E.1 describe slightly differently from what the code actually does;
where they disagree, this file follows the code, since the goal is exact
reproduction of the author's implementation.

KEY CORRECTIONS RELATIVE TO PRIOR VERSIONS OF THIS FILE
--------------------------------------------------------
1. SINGLE-HEAD CRITICS, NO CLIPPED DOUBLE-Q.
   `HPGAgent` uses `DDPGCritic` (one Q output) for BOTH the actual and
   abstract critic - never the twin-Q `Critic` class also defined in
   models/core.py. No `min(q1,q2)` anywhere. This matches the paper's own
   text: "the only difference between our DDPG and TD3 is the clipped
   double Q-learning present in TD3, which appears to be hurting the
   performance in some tasks of DMC." -> `SingleQCritic` below.

2. NO LayerNorm ANYWHERE. Plain MLPs, orthogonal weight init + zero bias
   init (`utils.weight_init`), matching every network in models/core.py.

3. StateEncoder / ActionEncoder / RewardPredictor are 2-hidden-layer MLPs
   (not 1).

4. TransitionModel (ProbabilisticTransitionModel) predicts sigma directly
   via a sigmoid-scaled head (min_sigma=1e-4, max_sigma=10), not log_std.

5. Lax bisimulation loss (get_lax_bisim):
   - Huber ("smooth_l1", beta=1) distance for both z_dist and r_dist, not
     L1/abs.
   - The transition-model forward pass used for the bisim TARGET is
     computed with a fully detached (stop_gradient) input/output - this
     means action_encoder (eta) gets ZERO gradient from the lax loss.
     eta's only gradient source is the transition/reward-consistency loss.
   - The weight on the transition term is the actual per-transition
     `discount` (here: gamma * (1-done)), not a fixed hyperparameter.
   - The distance itself is sqrt((mu1-mu2)^2+(sigma1-sigma2)^2) computed
     PER-DIMENSION THEN AVERAGED, not a joint L2 norm over the full vector.

6. Reward-consistency loss (get_transition_reward_loss) predicts reward
   from the SAMPLED NEXT abstract state (`reward_predictor(sample)`), not
   from the current abstract state f(s).

7. Transition consistency loss is a real per-dimension Gaussian NLL:
   0.5*((mu-target)/sigma)^2 + log(sigma).

8. TD3 target-policy-smoothing noise is TWO INDEPENDENT draws - one for the
   actual critic's bootstrap, a separate one added directly in
   abstract-action space for the abstract critic's bootstrap. The noise
   SCALE is the same decaying stddev_schedule used for exploration
   (linear(1.0, 0.1, T)), clipped to +/-stddev_clip=0.3 - stddev_clip is a
   clip bound, not a fixed scale.

9. A separate pure-random-action exploration phase, `num_expl_steps=2000`,
   distinct from replay-buffer warmup (`num_seed_frames=4000`). Also:
   exploration noise added in `act()` is NOT clipped to [-1,1].

10. NO gradient clipping anywhere in the official optimizer setup (plain
    torch.optim.Adam, no clip_by_global_norm equivalent). Dropped here to
    match; if you see instability, re-adding a loose global-norm clip is a
    reasonable, paper-non-contradicting safety net (the paper doesn't
    specify either way).

11. `update_every_steps` (yaml) is ONE cadence gating BOTH the actor update
    and all three target-network Polyak updates - not two separate
    frequencies.

12. `critic` and `abstract_critic` are each optimized with their OWN
    backward pass:
      - critic + state_encoder + action_encoder + reward_predictor +
        transition_model: ONE joint loss / ONE backward
        (critic_loss + homomorphic_coef*lax_bisim_loss + transition_loss +
        reward_loss), matching `update_critic`.
      - abstract_critic: its OWN separate backward, with z/a_bar/next_z/
        next_a_bar all computed under stop_gradient - matching
        `update_abstract_critic`'s `with torch.no_grad():` block. This
        reverts an earlier ("Fix 1") change of mine that wired the abstract
        critic's gradient into the encoders - that was wrong; the original
        stop_gradient in this codebase's very first draft was correct.

OPEN ASSUMPTION - PLEASE CONFIRM
---------------------------------
`matching_dims` defaults to False in cfgs/config.yaml, which would make the
abstract state/action dims a fixed `feature_dim` (50) instead of
obs_dim/act_dim. Appendix E.2 says abstract dims == actual dims for state
observations, and your existing scaffolding already assumes this. This file
assumes `matching_dims=True` was set via a task-level or CLI override for
the state-observation experiments. If you find the actual override (a task
yaml, or your launch command), confirm/correct this.

NOT YET MATCHED (known remaining gaps, out of scope for this pass)
--------------------------------------------------------------------
- n-step returns (nstep=3 in cfgs/config.yaml); this buffer only does n=1.
- num_envs>1 / train_per_step>1 have NO equivalent in the author's strictly
  single-env, one-gradient-step-per-env-step training loop. Set num_envs
  as low as your infra allows and --train-per-step 1 for closest fidelity.
- The exact `stddev_schedule` string wasn't in the files you sent (only
  referenced as `${stddev_schedule}`). Assumed Table 1's
  "linear(1.0, 0.1, 1e6)" here - correct via CLI/default_cfg if different.
"""

import functools
import os
import os.path as osp
import random
import sys
import time
from copy import deepcopy
from dataclasses import field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, struct
from mujoco_playground import registry

from utils.acting import actor_step, wrap_env_for_training
from utils.buffer import RunningMeanStd, RunningStatistics, UniformSamplingQueue
from utils.logger import EpochLogger
from utils.models import get_tree_norm
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args  # reuse the CLI parser

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(5e4),
    "eval_episode_freq": 5,
    "hidden_size": 256,
    "lr": 1e-4,
    "gamma": 0.99,
    "update_tau": 0.01,  # critic_target_tau in cfgs/agent/hpg.yaml
    "train_per_step": 1,  # no equivalent in author code; keep at 1 for fidelity
    "episode_length": 1000,
    "warmup_samples": int(4e3),  # num_seed_frames
    "max_replay_size": int(1e6),
    "batch_size": int(256),
    "total_env_steps": int(1e6),
    # DHPG-specific, from cfgs/agent/hpg.yaml:
    "update_every_steps": 2,  # gates BOTH actor update and target Polyak updates
    "num_expl_steps": 2000,  # pure-random-action phase, separate from warmup_samples
    "stddev_clip": 0.3,
    "explore_stddev_start": 1.0,
    "explore_stddev_end": 0.1,
    "explore_stddev_decay_steps": int(1e6),  # ASSUMED - confirm actual stddev_schedule
    "homomorphic_coef": 1.0,  # single coefficient on lax_bisim_loss only
    "min_sigma": 1e-4,
    "max_sigma": 1e1,
}


def orthogonal_linear(rngs, in_dim, out_dim):
    """Matches utils.utils.weight_init: nn.init.orthogonal_ on the weight,
    zero-fill on the bias, applied to every nn.Linear in the author's code."""
    return nnx.Linear(
        in_dim,
        out_dim,
        kernel_init=jax.nn.initializers.orthogonal(),
        bias_init=nnx.initializers.zeros,
        rngs=rngs,
    )


def smooth_l1(a, b, beta: float = 1.0):
    """torch.nn.functional.smooth_l1_loss(reduction='none'), beta=1.0
    (the default PyTorch uses): 0.5*x^2/beta if |x|<beta else |x|-0.5*beta."""
    diff = jnp.abs(a - b)
    return jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)


# --------------------------------------------------------------------------- #
# Homomorphism-map components. Feature dims == obs/act dims (matching_dims);
# see "OPEN ASSUMPTION" above.
# --------------------------------------------------------------------------- #
class StateEncoder(nnx.Module):
    """f_phi(s) -> s_bar. models.core.StateEncoder: 2 hidden layers."""

    def __init__(
        self, rngs: nnx.Rngs, obs_dim: int, abstract_state_dim: int, hidden_size: int
    ):
        self.l1 = orthogonal_linear(rngs, obs_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, abstract_state_dim)

    def __call__(self, s):
        x = nnx.relu(self.l1(s))
        x = nnx.relu(self.l2(x))
        return self.l3(x)


class ActionEncoder(nnx.Module):
    """g_eta(s, a) -> a_bar, tanh-bounded. models.core.ActionEncoder: 2 hidden layers."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        abstract_action_dim: int,
        hidden_size: int,
    ):
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, abstract_action_dim)

    def __call__(self, s, a):
        x = nnx.relu(self.l1(jnp.concatenate([s, a], axis=-1)))
        x = nnx.relu(self.l2(x))
        return jnp.tanh(self.l3(x))


class RewardPredictor(nnx.Module):
    """R_bar_rho(s_bar) -> scalar reward. models.core.RewardPredictor: 2 hidden layers."""

    def __init__(self, rngs: nnx.Rngs, abstract_state_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, abstract_state_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, s_bar):
        x = nnx.relu(self.l1(s_bar))
        x = nnx.relu(self.l2(x))
        return jnp.squeeze(self.l3(x), axis=-1)


class TransitionModel(nnx.Module):
    """tau_nu(s_bar' | s_bar, a_bar) -> diagonal Gaussian (mean, sigma).
    Matches models.transition_model.ProbabilisticTransitionModel: sigma via
    a sigmoid-scaled head, NOT log_std."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        abstract_state_dim: int,
        abstract_action_dim: int,
        hidden_size: int,
        min_sigma: float = 1e-4,
        max_sigma: float = 1e1,
    ):
        self.l1 = orthogonal_linear(
            rngs, abstract_state_dim + abstract_action_dim, hidden_size
        )
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.fc_mu = orthogonal_linear(rngs, hidden_size, abstract_state_dim)
        self.fc_sigma = orthogonal_linear(rngs, hidden_size, abstract_state_dim)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, s_bar, a_bar):
        x = nnx.relu(self.l1(jnp.concatenate([s_bar, a_bar], axis=-1)))
        x = nnx.relu(self.l2(x))
        mu = self.fc_mu(x)
        sigma = nnx.sigmoid(self.fc_sigma(x))
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        return mu, sigma

    def sample(self, s_bar, a_bar, key):
        mu, sigma = self(s_bar, a_bar)
        return mu + sigma * jax.random.normal(key, mu.shape)


class DeterministicActor(nnx.Module):
    """pi_theta(s) -> a in [-1, 1]^act_dim. models.core.DeterministicActor
    with linear_approx=False: plain 2-hidden-layer MLP, NO LayerNorm."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, act_dim)

    def __call__(self, s, key=None):
        # `key` kept only for drop-in compatibility with utils.acting.actor_step's
        # (obs, key) -> action signature; unused (deterministic policy).
        x = nnx.relu(self.l1(s))
        x = nnx.relu(self.l2(x))
        return jnp.tanh(self.l3(x))


class SingleQCritic(nnx.Module):
    """models.core.DDPGCritic: ONE Q-head, no twin/ensemble, no clipped
    double-Q. Used for BOTH the actual and abstract critic - DHPG's own
    ablation text confirms it deliberately omits TD3's clipped double-Q."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act):
        x = nnx.relu(self.l1(obs_act))
        x = nnx.relu(self.l2(x))
        return jnp.squeeze(self.l3(x), axis=-1)


class ExploratoryActor(nnx.Module):
    """Matches HPGAgent.act(): for step < num_expl_steps, pure uniform
    random action; otherwise pi_theta(s) + N(0, stddev), UNCLIPPED (the
    author's act() does not clip the noisy action to [-1,1])."""

    def __init__(self, actor: DeterministicActor, stddev, use_random_action):
        self.actor = actor
        self.stddev = stddev
        self.use_random_action = use_random_action  # scalar bool/array

    def __call__(self, s, key):
        key, noise_key, rand_key = jax.random.split(key, 3)
        mean_act = self.actor(s)
        noisy_act = mean_act + self.stddev * jax.random.normal(
            noise_key, mean_act.shape
        )
        random_act = jax.random.uniform(
            rand_key, mean_act.shape, minval=-1.0, maxval=1.0
        )
        act = jnp.where(self.use_random_action, random_act, noisy_act)
        return act, jnp.zeros(act.shape[:-1])


def explore_stddev(step, cfg):
    """utils.utils.schedule(stddev_schedule, step) for a linear(init,final,duration) schedule."""
    frac = jnp.clip(step / cfg.explore_stddev_decay_steps, 0.0, 1.0)
    return cfg.explore_stddev_start + frac * (
        cfg.explore_stddev_end - cfg.explore_stddev_start
    )


# --------------------------------------------------------------------------- #
# Containers.
# --------------------------------------------------------------------------- #
class HomomorphismMap(nnx.Module):
    def __init__(
        self,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder,
        reward_predictor: RewardPredictor,
        transition_model: TransitionModel,
    ):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.reward_predictor = reward_predictor
        self.transition_model = transition_model


class CriticHomomorphismBundle(nnx.Module):
    """critic + homomorphism map, jointly optimized with ONE backward pass,
    matching `update_critic`'s single `loss.backward()` over
    critic_loss + homomorphic_coef*lax_bisim_loss + transition_loss +
    reward_loss. The abstract critic is DELIBERATELY NOT part of this
    bundle - see `abstract_critic_step`."""

    def __init__(self, critic: SingleQCritic, homomorphism: HomomorphismMap):
        self.critic = critic
        self.homomorphism = homomorphism


class DHPGModels(nnx.Module):
    def __init__(
        self,
        actor: DeterministicActor,
        target_actor: DeterministicActor,
        target_critic: SingleQCritic,
        abstract_critic: SingleQCritic,
        target_abstract_critic: SingleQCritic,
        bundle: CriticHomomorphismBundle,
    ):
        self.actor = actor
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.abstract_critic = abstract_critic
        self.target_abstract_critic = target_abstract_critic
        self.bundle = bundle
        # No target network for f_phi/g_eta - matches Alg.1's target init
        # (only psi, psi_bar, theta get targets).


class DHPGOptimizers(nnx.Module):
    def __init__(
        self,
        actor: nnx.Optimizer,
        critic_homomorphism: nnx.Optimizer,
        abstract_critic: nnx.Optimizer,
    ):
        self.actor = actor
        self.critic_homomorphism = critic_homomorphism
        self.abstract_critic = abstract_critic


class DHPGState(nnx.Module):
    def __init__(self, models: DHPGModels, optimizers: DHPGOptimizers):
        self.models = models
        self.optimizers = optimizers


@struct.dataclass
class DHPGAux:
    critic_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    abstract_critic_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    lax_bisim_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    transition_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    reward_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    actor_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    q_mean: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    q_bar_mean: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    value_equivalence: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))


def polyak_update(target_model, curr_model, tau: float):
    """utils.utils.soft_update_params: target <- tau*curr + (1-tau)*target."""
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


# --------------------------------------------------------------------------- #
# update_critic: critic + homomorphism map, ONE joint loss / ONE backward.
# --------------------------------------------------------------------------- #
def critic_and_homomorphism_step(state: DHPGState, data: Transition, config, step, key):
    obs, act, reward, done_discount, next_obs = (
        data.observation,
        data.action,
        data.reward,
        data.discount,  # raw (1-done); scale by gamma below to match author's stored `discount` semantics
        data.next_observation,
    )
    discount = config.gamma * done_discount
    models = state.models

    key, noise_key, perm_key, sample_key = jax.random.split(key, 4)

    stddev = explore_stddev(step, config)

    # ---- Target Q for the ACTUAL critic (TD3 smoothing noise #1) --------- #
    next_act_mean = models.target_actor(next_obs)
    noise = jnp.clip(
        stddev * jax.random.normal(noise_key, next_act_mean.shape),
        -config.stddev_clip,
        config.stddev_clip,
    )
    next_action = jnp.clip(next_act_mean + noise, -1.0, 1.0)
    target_Q = models.target_critic(jnp.concatenate([next_obs, next_action], axis=-1))
    target_Q = jax.lax.stop_gradient(reward + discount * target_Q)

    def joint_loss_fn(bundle: CriticHomomorphismBundle):
        critic = bundle.critic
        state_encoder = bundle.homomorphism.state_encoder
        action_encoder = bundle.homomorphism.action_encoder
        reward_predictor = bundle.homomorphism.reward_predictor
        transition_model = bundle.homomorphism.transition_model

        # --- actual critic loss (single Q, MSE) --- #
        current_Q = critic(jnp.concatenate([obs, act], axis=-1))
        critic_loss = jnp.mean((current_Q - target_Q) ** 2)

        # --- abstract representations (differentiable) --- #
        s_bar = state_encoder(obs)
        a_bar = action_encoder(obs, act)

        # --- lax bisimulation loss (get_lax_bisim) --- #
        # Transition-model call used ONLY for the bisim target is fully
        # detached: this is what makes action_encoder (eta) receive zero
        # gradient from this loss term (matches the `with torch.no_grad()`
        # block around this call in the author's code).
        mean1, sigma1 = transition_model(s_bar, a_bar)
        mean1 = jax.lax.stop_gradient(mean1)
        sigma1 = jax.lax.stop_gradient(sigma1)

        perm = jax.random.permutation(perm_key, obs.shape[0])
        s_bar_2 = s_bar[perm]
        reward_2 = reward[perm]
        mean2, sigma2 = mean1[perm], sigma1[perm]

        z_dist = jnp.mean(smooth_l1(s_bar, s_bar_2), axis=-1)
        r_dist = smooth_l1(reward, reward_2)
        transition_dist = jnp.mean(
            jnp.sqrt((mean1 - mean2) ** 2 + (sigma1 - sigma2) ** 2), axis=-1
        )
        lax_bisimilarity = jax.lax.stop_gradient(r_dist + discount * transition_dist)
        lax_bisim_loss = jnp.mean((z_dist - lax_bisimilarity) ** 2)

        # --- transition + reward consistency loss (get_transition_reward_loss) --- #
        next_s_bar_target = jax.lax.stop_gradient(state_encoder(next_obs))
        mean_pred, sigma_pred = transition_model(s_bar, a_bar)  # fresh, WITH grad
        diff = (mean_pred - next_s_bar_target) / sigma_pred
        transition_loss = jnp.mean(0.5 * diff**2 + jnp.log(sigma_pred))

        # sample_prediction() in the original code is a further independent
        # forward pass through the (deterministic-given-params) network;
        # reusing mean_pred/sigma_pred here is numerically equivalent and
        # avoids a redundant third forward pass.
        sampled_next = mean_pred + sigma_pred * jax.random.normal(
            sample_key, mean_pred.shape
        )
        pred_next_reward = reward_predictor(sampled_next)
        reward_loss = jnp.mean((pred_next_reward - reward) ** 2)

        total_loss = (
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
            jnp.mean(current_Q),
        )
        return total_loss, aux

    (_, aux), grads = nnx.value_and_grad(joint_loss_fn, has_aux=True)(models.bundle)
    state.optimizers.critic_homomorphism.update(models.bundle, grads)

    critic_loss, lax_bisim_loss, transition_loss, reward_loss, q_mean = aux
    return (
        critic_loss,
        lax_bisim_loss,
        transition_loss,
        reward_loss,
        q_mean,
        next_action,
        stddev,
    )


# --------------------------------------------------------------------------- #
# update_abstract_critic: entirely separate optimizer, everything stop-
# gradiented - eta/phi get NO gradient from this loss.
# --------------------------------------------------------------------------- #
def abstract_critic_step(state: DHPGState, data: Transition, config, step, key, stddev):
    obs, act, reward, done_discount, next_obs = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.next_observation,
    )
    discount = config.gamma * done_discount
    models = state.models
    key, noise_key = jax.random.split(key)

    z = jax.lax.stop_gradient(models.bundle.homomorphism.state_encoder(obs))
    next_z = jax.lax.stop_gradient(models.bundle.homomorphism.state_encoder(next_obs))
    a_bar = jax.lax.stop_gradient(models.bundle.homomorphism.action_encoder(obs, act))

    # TD3 smoothing noise #2: independent draw, added directly in
    # abstract-action space (matches `update_abstract_critic` exactly).
    next_act_actual = models.target_actor(next_obs)
    next_a_bar_clean = models.bundle.homomorphism.action_encoder(
        next_obs, next_act_actual
    )
    noise = jnp.clip(
        stddev * jax.random.normal(noise_key, next_a_bar_clean.shape),
        -config.stddev_clip,
        config.stddev_clip,
    )
    next_a_bar = jax.lax.stop_gradient(jnp.clip(next_a_bar_clean + noise, -1.0, 1.0))

    target_Q_bar = models.target_abstract_critic(
        jnp.concatenate([next_z, next_a_bar], axis=-1)
    )
    target_Q_bar = jax.lax.stop_gradient(reward + discount * target_Q_bar)

    def loss_fn(abstract_critic):
        current_Q_bar = abstract_critic(jnp.concatenate([z, a_bar], axis=-1))
        return jnp.mean((current_Q_bar - target_Q_bar) ** 2), current_Q_bar

    (loss, current_Q_bar), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        models.abstract_critic
    )
    state.optimizers.abstract_critic.update(models.abstract_critic, grads)

    # Value-equivalence diagnostic (paper's own Fig. 15 tool): |Q - Q_bar|,
    # computed with no gradient, purely for logging.
    Q = models.bundle.critic(jnp.concatenate([obs, act], axis=-1))
    value_equivalence = jnp.mean(jnp.abs(jax.lax.stop_gradient(Q) - current_Q_bar))

    return loss, jnp.mean(current_Q_bar), value_equivalence


# --------------------------------------------------------------------------- #
# update_actor / update_abstract_actor with hpg_update_type='double_add':
# actor_loss = DPG(-Q(s,pi(s))) + HPG(-Q_bar(f(s), g(s,pi(s)))), summed and
# backpropagated in one call. Also runs the target Polyak updates, gated by
# the SAME `update_every_steps` cadence as the actor update itself.
# --------------------------------------------------------------------------- #
def actor_and_target_step(state: DHPGState, data: Transition, config):
    obs = data.observation
    models = state.models

    def actor_loss_fn(actor):
        act = actor(obs)
        q = models.bundle.critic(jnp.concatenate([obs, act], axis=-1))
        dpg_loss = -jnp.mean(q)

        s_bar = jax.lax.stop_gradient(models.bundle.homomorphism.state_encoder(obs))
        a_bar = models.bundle.homomorphism.action_encoder(
            obs, act
        )  # HPG flows through g
        q_bar = models.abstract_critic(jnp.concatenate([s_bar, a_bar], axis=-1))
        hpg_loss = -jnp.mean(q_bar)

        return dpg_loss + hpg_loss

    actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(models.actor)
    state.optimizers.actor.update(models.actor, actor_grads)

    polyak_update(models.target_critic, models.bundle.critic, config.update_tau)
    polyak_update(
        models.target_abstract_critic, models.abstract_critic, config.update_tau
    )
    polyak_update(models.target_actor, models.actor, config.update_tau)

    return actor_loss


def dhpg_train_step(state: DHPGState, data: Transition, config, key, step):
    key, k1, k2 = jax.random.split(key, 3)

    critic_loss, lax_bisim_loss, transition_loss, reward_loss, q_mean, _, stddev = (
        critic_and_homomorphism_step(state, data, config, step, k1)
    )
    abstract_critic_loss, q_bar_mean, value_equivalence = abstract_critic_step(
        state, data, config, step, k2, stddev
    )

    def do_actor_update(state, data):
        return actor_and_target_step(state, data, config)

    def skip_actor_update(state, data):
        del state, data
        return jnp.array(0.0)

    actor_loss = nnx.cond(
        step % config.update_every_steps == 0,
        do_actor_update,
        skip_actor_update,
        state,
        data,
    )

    return DHPGAux(
        critic_loss=critic_loss,
        abstract_critic_loss=abstract_critic_loss,
        lax_bisim_loss=lax_bisim_loss,
        transition_loss=transition_loss,
        reward_loss=reward_loss,
        actor_loss=actor_loss,
        q_mean=q_mean,
        q_bar_mean=q_bar_mean,
        value_equivalence=value_equivalence,
    )


# --------------------------------------------------------------------------- #
# Rollout + train loop.
# --------------------------------------------------------------------------- #
@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    state: DHPGState,
    config,
    global_step,
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
            state,
            gstep,
            val,
        ) = carry

        key, env_key, act_key = jax.random.split(key, 3)
        stddev = explore_stddev(gstep, config)
        use_random_action = gstep < config.num_expl_steps
        noisy_policy = ExploratoryActor(state.models.actor, stddev, use_random_action)

        n_env_state, transition = actor_step(
            env,
            env_state,
            noisy_policy,
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
            key, buffer_state, obs_normalizer, state, _ = carry
            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)
            val = dhpg_train_step(state, batch, config, train_key, gstep + j)
            return (key, buffer_state, obs_normalizer, state, val)

        init_val = DHPGAux()
        key, buffer_state, obs_normalizer, state, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, buffer_state, obs_normalizer, state, init_val),
        )

        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            state,
            gstep + 1,
            val,
        )

    init_val = DHPGAux()
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        state,
        global_step,
        init_val,
    )

    (
        _,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        state,
        global_step,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)

    return (
        val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        global_step,
        num_steps,
    )


def prefill_buffer(
    key, env, env_state, buffer_state, policy, buffer, obs_normalizer, num_itr
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
        jitted_body, (key, env_state, buffer_state, obs_normalizer), (), length=num_itr
    )
    return env_state, buffer_state, obs_normalizer


def main(args, cfg_env=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    rngs = nnx.Rngs(default=args.seed, params=args.seed + 3, dropout=args.seed + 5)
    jax.default_device = jax.devices(args.device)[args.device_id]

    config = dict(default_cfg)
    # NOTE: only override with CLI args the user actually intended to change.
    # Blindly taking every args.* value (as a prior version did) silently
    # replaces the Table-1-matching defaults above with whatever sac_args()'s
    # own defaults are, which are tuned for a different (SAC) training
    # recipe. Pass explicit --lr/--batch-size/etc. on the CLI if you want to
    # deviate from default_cfg above.
    explicit_overrides = {
        k: v for k, v in vars(args).items() if v is not None and k in default_cfg
    }
    config.update(explicit_overrides)
    config["episode_length"] = args.episode_length
    config["eval_episode_freq"] = args.eval_episode_freq
    config["log_freq"] = args.log_freq
    config["save_freq"] = args.save_freq
    config["num_envs"] = args.num_envs

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

    config_data = make_static_config_from_dict("DHPGConfig", config)()

    def make_opt(model):
        return nnx.Optimizer(
            model=model, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
        )

    # matching_dims=True assumed - see "OPEN ASSUMPTION" in the module docstring.
    abstract_state_dim = obs_dim
    abstract_action_dim = act_dim

    actor = DeterministicActor(rngs, obs_dim, act_dim, config["hidden_size"])
    target_actor = deepcopy(actor)
    critic = SingleQCritic(rngs, obs_dim, act_dim, config["hidden_size"])
    target_critic = deepcopy(critic)
    abstract_critic = SingleQCritic(
        rngs, abstract_state_dim, abstract_action_dim, config["hidden_size"]
    )
    target_abstract_critic = deepcopy(abstract_critic)

    state_encoder = StateEncoder(
        rngs, obs_dim, abstract_state_dim, config["hidden_size"]
    )
    action_encoder = ActionEncoder(
        rngs, obs_dim, act_dim, abstract_action_dim, config["hidden_size"]
    )
    reward_predictor = RewardPredictor(rngs, abstract_state_dim, config["hidden_size"])
    transition_model = TransitionModel(
        rngs,
        abstract_state_dim,
        abstract_action_dim,
        config["hidden_size"],
        min_sigma=config["min_sigma"],
        max_sigma=config["max_sigma"],
    )
    homomorphism = HomomorphismMap(
        state_encoder, action_encoder, reward_predictor, transition_model
    )

    bundle = CriticHomomorphismBundle(critic=critic, homomorphism=homomorphism)

    models = DHPGModels(
        actor=actor,
        target_actor=target_actor,
        target_critic=target_critic,
        abstract_critic=abstract_critic,
        target_abstract_critic=target_abstract_critic,
        bundle=bundle,
    )
    optimizers = DHPGOptimizers(
        actor=make_opt(actor),
        critic_homomorphism=make_opt(bundle),
        abstract_critic=make_opt(abstract_critic),
    )
    state = DHPGState(models=models, optimizers=optimizers)

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

    prng_key, running_key = jax.random.split(prng_key)
    running_state = RunningStatistics.init(
        (config["eval_episode_freq"] * config["episode_length"],), running_key
    )

    dict_args = dict(config)
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)

    logger.log("Start prefilling replay buffer")
    prng_key, buffer_key = jax.random.split(prng_key)
    env_state, buffer_state, obs_normalizer = prefill_buffer(
        key=buffer_key,
        env=env,
        env_state=env_state,
        buffer_state=buffer_state,
        policy=ExploratoryActor(actor, config["explore_stddev_start"], jnp.array(True)),
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        num_itr=config["warmup_samples"],
    )

    logger.log("Start DHPG training")
    steps = buffer.size(buffer_state)
    global_step = jnp.array(0, dtype=jnp.int32)

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
            global_step=global_step,
            key=subkey,
        )
        (
            aux,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            global_step,
            num_steps,
        ) = val

        steps += num_steps
        logger.logged = False
        logger.log_tabular("Train/Steps", steps)
        logger.log_tabular("Loss/Critic", aux.critic_loss.item())
        logger.log_tabular("Loss/Abstract_critic", aux.abstract_critic_loss.item())
        logger.log_tabular("Loss/Lax_bisimulation", aux.lax_bisim_loss.item())
        logger.log_tabular("Loss/Transition", aux.transition_loss.item())
        logger.log_tabular("Loss/Reward", aux.reward_loss.item())
        logger.log_tabular("Loss/Actor", aux.actor_loss.item())
        logger.log_tabular("Q/Actual_Q_mean", aux.q_mean.item())
        logger.log_tabular("Q/Abstract_Q_mean", aux.q_bar_mean.item())
        logger.log_tabular("Diag/Value_equivalence", aux.value_equivalence.item())
        logger.log_tabular("Norm/actor", get_tree_norm(nnx.state(actor, nnx.Param)))
        logger.log_tabular("Norm/critic", get_tree_norm(nnx.state(critic, nnx.Param)))
        logger.log_tabular(
            "Norm/abstract_critic", get_tree_norm(nnx.state(abstract_critic, nnx.Param))
        )
        logger.log_tabular(
            "Norm/state_encoder_f", get_tree_norm(nnx.state(state_encoder, nnx.Param))
        )
        logger.log_tabular(
            "Norm/action_encoder_g", get_tree_norm(nnx.state(action_encoder, nnx.Param))
        )
        logger.log_tabular(
            "Eval/Return",
            running_state.reward_state.data.sum() / config["eval_episode_freq"],
        )
        logger.dump_tabular()

        if (steps - config["warmup_samples"]) % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=actor, prefix="actor"
            )
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=critic, prefix="critic"
            )

        if steps >= config["total_env_steps"]:
            break

    logger.nn_model_save(itr=steps, nn_model_saver_element=actor, prefix="actor")
    logger.nn_model_save(itr=steps, nn_model_saver_element=critic, prefix="critic")
    logger.close()


if __name__ == "__main__":
    args, cfg_env = sac_args()

    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "seed-" + str(args.seed).zfill(3)
    relpath = "-".join([subfolder, relpath])
    algo = "dhpg"
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
