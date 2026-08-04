"""
Deep Homomorphic Policy Gradient (DHPG) - state-observation, deterministic variant.

This mirrors "Continuous MDP Homomorphisms and Homomorphic Policy Gradient"
(Rezaei-Shoshtari et al., NeurIPS 2022), Algorithm 1 in Appendix E.1, with the
pixel-only lines (8-11: image augmentation + CNN encoding) removed, exactly as
the paper instructs for state observations.

WHAT CHANGED RELATIVE TO YOUR sac_single.py, AND WHY
------------------------------------------------------
1. Actor is now DETERMINISTIC (DDPG/TD3-style), not SACGaussianActor.
   The HPG theorem (Thm 4/5 in the paper) is derived for deterministic
   policies + a bijective action map g_s. There is no principled "stochastic
   DHPG" in the NeurIPS repo path you linked (that variant is in the JMLR
   follow-up, a different algorithm/repo path: `stochastichpg`). So to be
   "consistent with their implementation" the actor has to drop SAC's
   entropy term and become deterministic, with exploration noise injected
   externally (Alg. 1 line 5) and target policy smoothing (line 14, from TD3).

2. There is no log_alpha / entropy machinery at all in DHPG.

3. The "metric" networks (EnsembleStateMetric / EnsembleStateActionMetric /
   MinStateActiontoStateMetric) you had are NOT what DHPG uses. DHPG's
   homomorphism map h = (f, g) is just two small encoders:
       f_phi(s)      : S -> S_bar        (state encoder)
       g_eta(s, a)   : S x A -> A_bar    (action encoder, state-conditioned)
   trained with exactly two losses (Eq. 12-13 in the paper):
       L_lax  = lax bisimulation loss (pairwise, permuted batch)
       L_h    = transition-consistency + reward-consistency loss
   These replace your state_metric / state_action_metric / min_state_action
   networks and their asymmetric losses entirely.

4. DHPG additionally needs, and your code did not have:
       - an ABSTRACT critic Q_bar(s_bar, a_bar)         (Eq. 10)
       - a reward predictor R_bar(s_bar)                (used in Eq. 13)
       - a probabilistic transition model tau_nu(s_bar' | s_bar, a_bar)
         outputting a diagonal Gaussian                 (used in Eq. 12-13)

5. Target networks: ONLY psi (actual critic), psi_bar (abstract critic), and
   theta (actor) get Polyak-updated targets (Alg. 1 line 3). f, g, the reward
   predictor, and the transition model have no target copies - remove the
   deepcopy'd targets you had for the metric nets.

6. Critic loss is the standard (n-step, here 1-step for simplicity) TD error
   with a MIN over twin critics for the actual critic, and separately for the
   abstract critic using s_bar = f(s), a_bar = g(s, pi(s)) computed through
   the (non-target) homomorphism map but a TARGET actor for the bootstrap
   action, with clipped Gaussian noise (TD3 target policy smoothing).

7. Actor loss is Eq. (11): -(Q_actual(s, pi(s)) + Q_abstract(f(s), g(s, pi(s)))),
   i.e. DPG and HPG gradients are literally summed and backpropagated once,
   exactly as the paper's default `hpg` variant does (not `hpg_ind`).

8. Delayed actor + target updates (Alg. 1 line 19, "if t mod d"): the critic
   and homomorphism map are updated every step; actor and target nets are
   updated every `actor_update_freq` steps.

WHAT I DID NOT CHANGE / LEFT FOR YOU
-------------------------------------
- I kept n-step return at n=1 for simplicity, matching your buffer
  (UniformSamplingQueue with dummy 1-step transitions). The paper uses n=3.
  If you want n=3, you need an n-step buffer wrapper; happy to add if wanted.
- I reused your EnsembleCritic as-is for BOTH the actual and abstract critic
  (their input/output dims match exactly in the state-observation case, per
  Appendix E.2: "the abstract MDP has the same state and action dimensions
  as the actual MDP").
- I left prefill_buffer, RunningMeanStd/RunningStatistics, UniformSamplingQueue,
  EpochLogger, wrap_env_for_training untouched - only the agent/model side and
  the train step change.
- StateEncoder / ActionEncoder / RewardPredictor / TransitionModel are defined
  locally below using the same nnx.Module + `rngs=` convention your other
  models use, so you can freely move them into utils/models.py.
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
from utils.models import EnsembleCritic, get_tree_norm
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args  # reuse the CLI parser

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(5e4),
    "eval_episode_freq": 5,
    "hidden_size": 256,
    "lr": 1e-4,
    "max_grad_norm": 10,
    "gamma": 0.99,
    "update_tau": 0.01,  # Table 1: target soft-update tau
    "train_per_step": 1,
    "episode_length": 1000,
    "warmup_samples": int(4e3),  # Table 1: seed frames
    "max_replay_size": int(1e6),
    "batch_size": int(256),
    "total_env_steps": int(1e6),
    # DHPG-specific:
    "actor_update_freq": 2,  # Alg.1: delayed actor update d
    "target_update_freq": 2,  # Table 1: target network update frequency
    "stddev_clip": 0.3,  # TD3 target-policy-smoothing clip c
    "explore_stddev_start": 1.0,
    "explore_stddev_end": 0.1,
    "explore_stddev_decay_steps": int(1e6),
    "lax_reward_coef": 1.0,  # c_r in Eq. (5)/(12)
    "lax_transition_coef": 1.0,  # alpha (weight on the W2 term) in Eq. (12)
}


# --------------------------------------------------------------------------- #
# Homomorphism-map components (Eq. 12-13). Feature dims == obs/act dims for
# state observations (Appendix E.2).
# --------------------------------------------------------------------------- #
class StateEncoder(nnx.Module):
    """f_phi(s) -> s_bar. Maps actual states to abstract states."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, hidden_size: int):
        self.l1 = nnx.Linear(obs_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, obs_dim, rngs=rngs)

    def __call__(self, s):
        x = nnx.relu(self.l1(s))
        return self.l2(x)


class ActionEncoder(nnx.Module):
    """g_eta(s, a) -> a_bar. State-conditioned action encoder (tanh-bounded
    to stay in the same range as actions, consistent with the paper treating
    A_bar as a subset of R^n like A)."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = nnx.Linear(obs_dim + act_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, act_dim, rngs=rngs)

    def __call__(self, s, a):
        x = nnx.relu(self.l1(jnp.concatenate([s, a], axis=-1)))
        return jnp.tanh(self.l2(x))


class RewardPredictor(nnx.Module):
    """R_bar_rho(s_bar) -> scalar reward, used in Eq. (13)."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, hidden_size: int):
        self.l1 = nnx.Linear(obs_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, 1, rngs=rngs)

    def __call__(self, s_bar):
        x = nnx.relu(self.l1(s_bar))
        return self.l2(x)[..., 0]


class TransitionModel(nnx.Module):
    """tau_nu(s_bar' | s_bar, a_bar) -> diagonal Gaussian (mean, log_std),
    used in Eq. (12) (W2 distance) and Eq. (13) (next-state prediction)."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = nnx.Linear(obs_dim + act_dim, hidden_size, rngs=rngs)
        self.mean = nnx.Linear(hidden_size, obs_dim, rngs=rngs)
        self.log_std = nnx.Linear(hidden_size, obs_dim, rngs=rngs)

    def __call__(self, s_bar, a_bar):
        x = nnx.relu(self.l1(jnp.concatenate([s_bar, a_bar], axis=-1)))
        mean = self.mean(x)
        log_std = jnp.clip(self.log_std(x), -5.0, 2.0)
        return mean, log_std

    def sample(self, s_bar, a_bar, key):
        mean, log_std = self(s_bar, a_bar)
        return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


class DeterministicActor(nnx.Module):
    """pi_theta(s) -> a in [-1, 1]^act_dim. Replaces SACGaussianActor: DHPG's
    HPG derivation requires a deterministic policy."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = nnx.Linear(obs_dim, hidden_size, rngs=rngs)
        self.l2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.l3 = nnx.Linear(hidden_size, act_dim, rngs=rngs)

    def __call__(self, s, key=None):
        # `key` kept only so this class is drop-in compatible with the
        # (obs, key) call signature that utils.acting.actor_step expects for
        # a stochastic policy; it is unused here (deterministic policy).
        x = nnx.relu(self.l1(s))
        x = nnx.relu(self.l2(x))
        return jnp.tanh(self.l3(x))


class ExploratoryActor(nnx.Module):
    """Wraps a DeterministicActor with linearly-decayed Gaussian exploration
    noise (Alg. 1 line 5: a ~ pi_theta(s) + eps, eps ~ N(0, sigma)), while
    keeping the (obs, key) -> (action, log_prob) signature actor_step needs.
    log_prob is returned as zeros (unused, deterministic policy)."""

    def __init__(self, actor: DeterministicActor, stddev: float):
        self.actor = actor
        self.stddev = stddev

    def __call__(self, s, key):
        mean_act = self.actor(s)
        noise = self.stddev * jax.random.normal(key, mean_act.shape)
        act = jnp.clip(mean_act + noise, -1.0, 1.0)
        return act, jnp.zeros(act.shape[:-1])


def explore_stddev(step, cfg):
    frac = jnp.clip(step / cfg.explore_stddev_decay_steps, 0.0, 1.0)
    return cfg.explore_stddev_start + frac * (
        cfg.explore_stddev_end - cfg.explore_stddev_start
    )


# --------------------------------------------------------------------------- #
# Containers. These MUST be real nnx graph nodes (nnx.Module), not plain
# @dataclass objects, or nnx.jit has no idea how to split/trace them (that
# was the cause of the "Error interpreting argument ... as an abstract array"
# TypeError). nnx.Module.__setattr__ automatically registers nnx.Module /
# nnx.Optimizer attributes as sub-nodes of the graph.
# --------------------------------------------------------------------------- #
class HomomorphismMap(nnx.Module):
    """Bundles f_phi, g_eta, R_bar_rho, tau_nu into a single module so they
    can share one optimizer / one gradient call (Eq. 12 + 13 combined)."""

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


class DHPGModels(nnx.Module):
    def __init__(
        self,
        actor: DeterministicActor,
        target_actor: DeterministicActor,
        critic: EnsembleCritic,
        target_critic: EnsembleCritic,
        abstract_critic: EnsembleCritic,
        target_abstract_critic: EnsembleCritic,
        homomorphism: HomomorphismMap,
    ):
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.abstract_critic = abstract_critic
        self.target_abstract_critic = target_abstract_critic
        self.homomorphism = homomorphism


class DHPGOptimizers(nnx.Module):
    def __init__(
        self,
        actor: nnx.Optimizer,
        critic: nnx.Optimizer,
        abstract_critic: nnx.Optimizer,
        homomorphism: nnx.Optimizer,
    ):
        self.actor = actor
        self.critic = critic
        self.abstract_critic = abstract_critic
        self.homomorphism = homomorphism


class DHPGState(nnx.Module):
    def __init__(self, models: DHPGModels, optimizers: DHPGOptimizers):
        self.models = models
        self.optimizers = optimizers


@struct.dataclass
class DHPGAux:
    """A flax.struct.dataclass (NOT a plain @dataclass) so it's registered as
    a pytree: this is required because it flows through nnx.fori_loop as a
    carry value, and jax needs to know how to flatten/unflatten it. It's also
    immutable (frozen), so update fields with `.replace(...)`, never `x.f = v`."""

    actual_critic_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    abstract_critic_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    lax_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    homomorphism_consistency_loss: jnp.ndarray = field(
        default_factory=lambda: jnp.array(0.0)
    )
    actor_loss: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    q1_mean: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    q2_mean: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    q_bar_mean: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))


def polyak_update(target_model, curr_model, tau: float):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


def w2_gaussian(mean1, log_std1, mean2, log_std2):
    """Closed-form W2 distance between two diagonal Gaussians (Sec. 6, the
    paper's substitute for the Kantorovich metric, following Zhang et al.)."""
    mean_term = jnp.sum((mean1 - mean2) ** 2, axis=-1)
    std_term = jnp.sum((jnp.exp(log_std1) - jnp.exp(log_std2)) ** 2, axis=-1)
    return jnp.sqrt(jnp.clip(mean_term + std_term, 1e-8, None))


# --------------------------------------------------------------------------- #
# Critic + homomorphism-map update (Eq. 9, 10, 12, 13). Runs every step.
# --------------------------------------------------------------------------- #
def critic_and_homomorphism_step(state: DHPGState, data: Transition, config, key):
    obs, act, reward, discount, next_obs = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.next_observation,
    )
    models = state.models

    key, noise_key, perm_key, trans_key_i, trans_key_j = jax.random.split(key, 5)

    # Target policy smoothing (TD3, Alg.1 line 14).
    next_act_mean = models.target_actor(next_obs)
    smoothing_noise = jnp.clip(
        config.stddev_clip * jax.random.normal(noise_key, next_act_mean.shape),
        -2 * config.stddev_clip,
        2 * config.stddev_clip,
    )
    next_act = jnp.clip(next_act_mean + smoothing_noise, -1.0, 1.0)

    # ---- Eq. (9): actual critic loss ---------------------------------- #
    def actual_critic_loss_fn(critic):
        q1_t, q2_t = models.target_critic(
            jnp.concatenate([next_obs, next_act], axis=-1)
        )
        target_q = reward + config.gamma * discount * jnp.minimum(q1_t, q2_t)
        target_q = jax.lax.stop_gradient(target_q)
        q1, q2 = critic(jnp.concatenate([obs, act], axis=-1))
        loss = jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
        return loss, (jnp.mean(q1), jnp.mean(q2))

    (actual_critic_loss, (q1_mean, q2_mean)), critic_grads = nnx.value_and_grad(
        actual_critic_loss_fn, has_aux=True
    )(models.critic)
    state.optimizers.critic.update(models.critic, critic_grads)

    # ---- Homomorphism map + Eq. (12) lax bisimulation + Eq. (13) ------ #
    def homomorphism_loss_fn(homo: HomomorphismMap):
        state_encoder = homo.state_encoder
        action_encoder = homo.action_encoder
        reward_predictor = homo.reward_predictor
        transition_model = homo.transition_model

        s_bar = state_encoder(obs)
        a_bar = action_encoder(obs, act)
        next_s_bar_target = state_encoder(next_obs)  # for Eq. (13) target

        # Eq. (13): transition + reward consistency of the homomorphism map.
        sampled_next_s_bar = transition_model.sample(s_bar, a_bar, trans_key_i)
        trans_consistency = jnp.mean(
            (sampled_next_s_bar - jax.lax.stop_gradient(next_s_bar_target)) ** 2
        )
        reward_consistency = jnp.mean((reward - reward_predictor(s_bar)) ** 2)
        l_h = trans_consistency + reward_consistency

        # Eq. (12): lax bisimulation loss over a permuted (shuffled) batch.
        perm = jax.random.permutation(perm_key, obs.shape[0])
        s_bar_j = s_bar[perm]
        a_bar_j = a_bar[perm]
        reward_j = reward[perm]

        mean_i, log_std_i = transition_model(s_bar, a_bar)
        mean_j, log_std_j = transition_model(s_bar_j, a_bar_j)
        w2_dist = w2_gaussian(mean_i, log_std_i, mean_j, log_std_j)

        state_dist = jnp.sum(jnp.abs(s_bar - s_bar_j), axis=-1)
        reward_dist = config.lax_reward_coef * jnp.abs(reward - reward_j)
        target_dist = jax.lax.stop_gradient(
            reward_dist + config.lax_transition_coef * w2_dist
        )
        l_lax = jnp.mean((state_dist - target_dist) ** 2)

        return l_lax + l_h, (l_lax, l_h)

    (homo_loss, (lax_loss, l_h)), homo_grads = nnx.value_and_grad(
        homomorphism_loss_fn, has_aux=True
    )(models.homomorphism)
    state.optimizers.homomorphism.update(models.homomorphism, homo_grads)

    # ---- Eq. (10): abstract critic loss, using the (just-updated) map -- #
    def abstract_critic_loss_fn(abstract_critic):
        s_bar = jax.lax.stop_gradient(models.homomorphism.state_encoder(obs))
        a_bar = jax.lax.stop_gradient(models.homomorphism.action_encoder(obs, act))
        next_s_bar = jax.lax.stop_gradient(models.homomorphism.state_encoder(next_obs))
        next_a_bar = jax.lax.stop_gradient(
            models.homomorphism.action_encoder(next_obs, next_act)
        )

        q1_bar_t, q2_bar_t = models.target_abstract_critic(
            jnp.concatenate([next_s_bar, next_a_bar], axis=-1)
        )
        target_q_bar = reward + config.gamma * discount * jnp.minimum(
            q1_bar_t, q2_bar_t
        )
        target_q_bar = jax.lax.stop_gradient(target_q_bar)
        q1_bar, q2_bar = abstract_critic(jnp.concatenate([s_bar, a_bar], axis=-1))
        loss = jnp.mean((q1_bar - target_q_bar) ** 2) + jnp.mean(
            (q2_bar - target_q_bar) ** 2
        )
        return loss, jnp.mean(q1_bar)

    (abstract_critic_loss, q_bar_mean), abs_critic_grads = nnx.value_and_grad(
        abstract_critic_loss_fn, has_aux=True
    )(models.abstract_critic)
    state.optimizers.abstract_critic.update(models.abstract_critic, abs_critic_grads)

    return DHPGAux(
        actual_critic_loss=actual_critic_loss,
        abstract_critic_loss=abstract_critic_loss,
        lax_loss=lax_loss,
        homomorphism_consistency_loss=l_h,
        q1_mean=q1_mean,
        q2_mean=q2_mean,
        q_bar_mean=q_bar_mean,
    )


# --------------------------------------------------------------------------- #
# Actor + target-network update (Eq. 11). Runs every `actor_update_freq` steps.
# --------------------------------------------------------------------------- #
def actor_and_target_step(state: DHPGState, data: Transition, config):
    obs = data.observation
    models = state.models

    def actor_loss_fn(actor):
        act = actor(obs)
        q1, q2 = models.critic(jnp.concatenate([obs, act], axis=-1))
        q_actual = jnp.minimum(q1, q2)

        s_bar = jax.lax.stop_gradient(models.homomorphism.state_encoder(obs))
        a_bar = models.homomorphism.action_encoder(
            obs, act
        )  # NOT stop-gradient: HPG flows through g
        q1_bar, q2_bar = models.abstract_critic(
            jnp.concatenate([s_bar, a_bar], axis=-1)
        )
        q_abstract = jnp.minimum(q1_bar, q2_bar)

        # Eq. (11): DPG + HPG gradients summed into a single actor update.
        loss = -jnp.mean(q_actual + q_abstract)
        return loss

    actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(models.actor)
    state.optimizers.actor.update(models.actor, actor_grads)

    polyak_update(models.target_critic, models.critic, config.update_tau)
    polyak_update(
        models.target_abstract_critic, models.abstract_critic, config.update_tau
    )
    polyak_update(models.target_actor, models.actor, config.update_tau)

    return actor_loss


def dhpg_train_step(state: DHPGState, data: Transition, config, key, step_in_epoch):
    critic_aux = critic_and_homomorphism_step(state, data, config, key)

    def do_actor_update(state, data):
        return actor_and_target_step(state, data, config)

    def skip_actor_update(state, data):
        del state, data
        return jnp.array(0.0)

    # NOTE: was `jax.lax.cond`. Plain `jax.lax.cond` traces its branches at a
    # fresh JAX trace level and has no idea how to split/merge NNX graph
    # state (Params, Optimizer state, etc). Since `do_actor_update` calls
    # `nnx.value_and_grad` and mutates optimizers/targets in place, the
    # params captured by the *outer* trace (from `nnx.fori_loop`) collide
    # with the *inner* trace created by `jax.lax.cond`, which is exactly the
    # "Cannot extract graph node from different trace level" error. `nnx.cond`
    # is a drop-in replacement that wraps `jax.lax.cond` while correctly
    # threading NNX module/optimizer state through both branches.
    actor_loss = nnx.cond(
        step_in_epoch % config.actor_update_freq == 0,
        do_actor_update,
        skip_actor_update,
        state,
        data,
    )
    return critic_aux.replace(actor_loss=actor_loss)


# --------------------------------------------------------------------------- #
# Rollout + train loop (structurally the same shape as your train_n_steps,
# but the policy used for acting is the noisy deterministic actor and the
# per-step train call is dhpg_train_step instead of sac_train_step).
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
        noisy_policy = ExploratoryActor(state.models.actor, stddev)

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
            "num_envs": args.num_envs,
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

    config_data = make_static_config_from_dict("DHPGConfig", config)()

    def make_opt(model):
        return nnx.Optimizer(
            model=model,
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["lr"]),
            ),
            wrt=nnx.Param,
        )

    actor = DeterministicActor(rngs, obs_dim, act_dim, config["hidden_size"])
    target_actor = deepcopy(actor)
    critic = EnsembleCritic(
        rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )
    target_critic = deepcopy(critic)
    # Abstract critic operates on (s_bar, a_bar), which have the same dims as
    # (s, a) for state observations (Appendix E.2) -> same EnsembleCritic shape.
    abstract_critic = EnsembleCritic(
        rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_size"]
    )
    target_abstract_critic = deepcopy(abstract_critic)

    state_encoder = StateEncoder(rngs, obs_dim, config["hidden_size"])
    action_encoder = ActionEncoder(rngs, obs_dim, act_dim, config["hidden_size"])
    reward_predictor = RewardPredictor(rngs, obs_dim, config["hidden_size"])
    transition_model = TransitionModel(rngs, obs_dim, act_dim, config["hidden_size"])
    homomorphism = HomomorphismMap(
        state_encoder, action_encoder, reward_predictor, transition_model
    )

    models = DHPGModels(
        actor=actor,
        target_actor=target_actor,
        critic=critic,
        target_critic=target_critic,
        abstract_critic=abstract_critic,
        target_abstract_critic=target_abstract_critic,
        homomorphism=homomorphism,
    )
    optimizers = DHPGOptimizers(
        actor=make_opt(actor),
        critic=make_opt(critic),
        abstract_critic=make_opt(abstract_critic),
        homomorphism=make_opt(homomorphism),
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
        policy=ExploratoryActor(actor, config["explore_stddev_start"]),
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
        logger.log_tabular("Loss/Actual_critic", aux.actual_critic_loss.item())
        logger.log_tabular("Loss/Abstract_critic", aux.abstract_critic_loss.item())
        logger.log_tabular("Loss/Lax_bisimulation", aux.lax_loss.item())
        logger.log_tabular(
            "Loss/Homomorphism_consistency", aux.homomorphism_consistency_loss.item()
        )
        logger.log_tabular("Loss/Actor", aux.actor_loss.item())
        logger.log_tabular("Q/Actual_Q1_mean", aux.q1_mean.item())
        logger.log_tabular("Q/Actual_Q2_mean", aux.q2_mean.item())
        logger.log_tabular("Q/Abstract_Q_mean", aux.q_bar_mean.item())
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
