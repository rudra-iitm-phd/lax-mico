"""
Deep Homomorphic Policy Gradient (DHPG) - state-observation, deterministic
variant ('hpg_update_type=double_add'), restructured to be a STRUCTURAL
MIRROR of sac_single.py so the same conventions (and the same places to fix
things) apply to both scripts.

WHAT CHANGED IN THIS PASS (structure only - the algorithm itself is
unchanged from the previous, author-source-verified version)
----------------------------------------------------------------------
1. NO CONTAINER CLASSES. sac_single.py never wraps models/optimizers in an
   nnx.Module container - it threads flat tuples of raw nnx objects through
   nnx.fori_loop / nnx.jit, e.g.
     models = (actor, actor_opt, critic, critic_opt, target_critic,
               log_alpha, alpha_opt)
   DHPGState / DHPGModels / DHPGOptimizers / DHPGAux / HomomorphismMap /
   CriticHomomorphismBundle are all GONE. Same flat-tuple-of-raw-nnx-objects
   convention now, just with DHPG's (many more) components.

2. ONE OPTIMIZER PER MODEL, not bundled. sac_single.py gives actor, critic,
   and log_alpha each their own nnx.Optimizer. This also happens to be
   exactly what the actual author PyTorch code does (7 separate
   torch.optim.Adam instances: critic, abstract_critic, action_encoder,
   state_encoder, reward_predictor, transition_model, actor - each getting
   its own .step() after ONE shared .backward() for the critic/homomorphism
   group). So critic, state_encoder, action_encoder, reward_predictor, and
   transition_model each get their own optimizer here; gradients from the
   ONE joint loss are computed once (nnx.value_and_grad with argnums=(...))
   and then applied via 5 separate .update() calls - functionally identical
   to a single bundled optimizer (Adam has no cross-parameter coupling) but
   structurally matching both SAC's convention and the author's source.

3. dhpg_train_step IS ONE FLAT FUNCTION, mirroring sac_train_step exactly:
   takes every model/optimizer as a flat positional arg, returns a flat
   tuple of scalars (critic_loss, abstract_critic_loss, lax_bisim_loss,
   transition_loss, reward_loss, actor_loss, q_mean, q_bar_mean,
   value_equivalence) - 9 scalars, mirroring sac_train_step's 7-scalar
   return and SAC's `init_val = (jnp.zeros((), jnp.float32),) * N` pattern.

4. STEP COUNTING matches SAC's real convention exactly:
   `train_n_steps` returns `num_steps * config.num_envs` as the transition
   count added to `steps` in main() - SAC already does this correctly, so
   DHPG now does too. In addition, DHPG carries TWO separate counters that
   SAC has no equivalent for (SAC's entropy-based exploration needs no
   step-dependent schedule at all):
     - `env_step`: real environment transitions collected, incremented by
       `config.num_envs` once per OUTER (rollout) iteration. Drives the
       exploration stddev schedule and `num_expl_steps` cutoff, since
       Table 1's "Exploration steps: 2000" / "linear(1.0,0.1,1e6)" are
       specified in real-environment-step units, not iteration counts.
     - `grad_step`: a true gradient-update counter, incremented by 1 once
       per call to dhpg_train_step (i.e. `train_per_step` times per outer
       iteration). Drives `update_every_steps`'s delayed actor/target
       cadence, since Algorithm 1's "if t mod d" refers to the training
       iteration index t, which is a gradient-step concept, not a
       real-env-transition concept - conflating the two would make the
       actor-update cadence silently depend on `num_envs`.
   Both are threaded through train_n_steps' carry/return exactly like
   SAC threads (nothing extra) plus these two new fields.

5. CONFIG BUILDING matches SAC's exactly: unconditional
   `config.update({...args...})`, no allowlist/filtering logic. This is a
   deliberate reversal of an earlier version of this file that tried to
   protect Table-1 values from being overwritten by sac_args()' CLI
   defaults - given the current goal is a fair, matched comparison against
   your own algorithm under a shared parallelized regime (not literal
   Table-1 paper reproduction), SAC and DHPG should - and now do - resolve
   `num_envs`/`train_per_step`/`lr`/etc. identically from the same CLI
   parser, the same way.

6. LOGGING / CHECKPOINT naming mirrors SAC's style: `Loss/Loss_*`,
   `Norm/*_model`, and the `(steps - warmup_samples * num_envs) % save_freq`
   checkpoint condition (SAC's `warmup_samples` is an ITERATION count fed to
   `prefill_buffer`'s `num_itr`, so it must be multiplied by `num_envs` to
   compare against `steps`, which is a real-transition count - DHPG's
   `warmup_samples` is used identically, so the same correction applies).

WHAT DID NOT CHANGE (still author-source-verified, see prior turns)
----------------------------------------------------------------------
- Single-head critics (no clipped double-Q), no LayerNorm, orthogonal init,
  2-hidden-layer encoders, sigmoid-scaled TransitionModel sigma, Huber lax
  bisimulation distance with the transition-model target fully detached
  (zero gradient to action_encoder from lax loss), reward-consistency loss
  predicting from the sampled NEXT abstract state, per-transition
  `discount` weighting the lax target, TWO independent TD3-smoothing-noise
  draws, `num_expl_steps=2000` pure-random phase, no gradient clipping.
- matching_dims=True (abstract dims = obs_dim/act_dim) - confirmed by
  Appendix E.2 ("In the case of state observations, the abstract MDP has
  the same state and action dimensions as the actual MDP").
- Known gap, still unaddressed: n-step returns (paper Table 1: n=3; this
  buffer only supports n=1).
"""

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
    "train_per_step": 1,
    "episode_length": 1000,
    "warmup_samples": int(
        4e3
    ),  # num_seed_frames; ITERATION count fed to prefill_buffer's num_itr
    "max_replay_size": int(1e6),
    "batch_size": int(256),
    "total_env_steps": int(1e6),
    # DHPG-specific, from cfgs/agent/hpg.yaml / Table 1:
    "update_every_steps": 2,  # gates BOTH actor update and target Polyak updates (grad_step units)
    "num_expl_steps": 2000,  # pure-random-action phase (env_step units)
    "stddev_clip": 0.3,
    "explore_stddev_start": 1.0,
    "explore_stddev_end": 0.1,
    "explore_stddev_decay_steps": int(
        1e6
    ),  # env_step units; Table 1: linear(1.0, 0.1, 1e6)
    "homomorphic_coef": 1.0,
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
    """torch.nn.functional.smooth_l1_loss(reduction='none'), beta=1.0."""
    diff = jnp.abs(a - b)
    return jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)


# --------------------------------------------------------------------------- #
# Models. matching_dims=True: abstract state/action dims == obs_dim/act_dim.
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


class DeterministicActor(nnx.Module):
    """pi_theta(s) -> a in [-1, 1]^act_dim. Plain 2-hidden-layer MLP, NO LayerNorm."""

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
    double-Q. Used for BOTH the actual and abstract critic."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act):
        x = nnx.relu(self.l1(obs_act))
        x = nnx.relu(self.l2(x))
        return jnp.squeeze(self.l3(x), axis=-1)


class ExploratoryActor(nnx.Module):
    """Matches HPGAgent.act(): for env_step < num_expl_steps, pure uniform
    random action; otherwise pi_theta(s) + N(0, stddev), UNCLIPPED."""

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


def explore_stddev(env_step, cfg):
    """utils.utils.schedule(stddev_schedule, step) for linear(init,final,duration)."""
    frac = jnp.clip(env_step / cfg.explore_stddev_decay_steps, 0.0, 1.0)
    return cfg.explore_stddev_start + frac * (
        cfg.explore_stddev_end - cfg.explore_stddev_start
    )


def polyak_update(target_model, curr_model, tau: float):
    """Identical to sac_single.py's polyak_update."""
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


# --------------------------------------------------------------------------- #
# dhpg_train_step: ONE flat function, mirrors sac_train_step's shape exactly.
# --------------------------------------------------------------------------- #
def dhpg_train_step(
    actor,
    actor_opt,
    target_actor,
    critic,
    critic_opt,
    target_critic,
    abstract_critic,
    abstract_critic_opt,
    target_abstract_critic,
    state_encoder,
    state_encoder_opt,
    action_encoder,
    action_encoder_opt,
    reward_predictor,
    reward_predictor_opt,
    transition_model,
    transition_model_opt,
    data: Transition,
    config,
    key: jnp.ndarray,
    grad_step: jnp.ndarray,
    env_step: jnp.ndarray,
):
    obs = data.observation
    act = data.action
    reward = data.reward
    discount = config.gamma * data.discount  # raw (1-done) -> gamma*(1-done)
    next_obs = data.next_observation

    stddev = explore_stddev(env_step, config)
    key, noise_key1, noise_key2, perm_key, sample_key = jax.random.split(key, 5)

    # ------------------------------------------------------------------ #
    # update_critic: critic + homomorphism map. ONE joint loss / ONE
    # backward (nnx.value_and_grad over 5 args), then 5 SEPARATE
    # optimizer.update() calls - matches agents.hpg.HPGAgent.update_critic
    # (one loss.backward(), then critic/transition/reward/action_encoder/
    # state_encoder optimizers each .step() once).
    # ------------------------------------------------------------------ #
    next_act_mean = target_actor(next_obs)
    noise1 = jnp.clip(
        stddev * jax.random.normal(noise_key1, next_act_mean.shape),
        -config.stddev_clip,
        config.stddev_clip,
    )
    next_action = jnp.clip(next_act_mean + noise1, -1.0, 1.0)
    target_Q = target_critic(jnp.concatenate([next_obs, next_action], axis=-1))
    target_Q = jax.lax.stop_gradient(reward + discount * target_Q)

    def joint_loss_fn(
        critic, state_encoder, action_encoder, reward_predictor, transition_model
    ):
        current_Q = critic(jnp.concatenate([obs, act], axis=-1))
        critic_loss = jnp.mean((current_Q - target_Q) ** 2)

        s_bar = state_encoder(obs)
        a_bar = action_encoder(obs, act)

        # lax bisimulation (Huber; transition-model target fully detached,
        # so action_encoder gets zero gradient from this term)
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

        # transition + reward consistency (predicts reward from the
        # SAMPLED NEXT abstract state)
        next_s_bar_target = jax.lax.stop_gradient(state_encoder(next_obs))
        mean_pred, sigma_pred = transition_model(s_bar, a_bar)  # fresh, WITH grad
        diff = (mean_pred - next_s_bar_target) / sigma_pred
        transition_loss = jnp.mean(0.5 * diff**2 + jnp.log(sigma_pred))
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
        return total_loss, (
            critic_loss,
            lax_bisim_loss,
            transition_loss,
            reward_loss,
            jnp.mean(current_Q),
        )

    (
        (_, (critic_loss, lax_bisim_loss, transition_loss, reward_loss, q_mean)),
        (
            critic_grads,
            state_encoder_grads,
            action_encoder_grads,
            reward_predictor_grads,
            transition_model_grads,
        ),
    ) = nnx.value_and_grad(joint_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
        critic, state_encoder, action_encoder, reward_predictor, transition_model
    )
    critic_opt.update(critic, critic_grads)
    transition_model_opt.update(transition_model, transition_model_grads)
    reward_predictor_opt.update(reward_predictor, reward_predictor_grads)
    action_encoder_opt.update(action_encoder, action_encoder_grads)
    state_encoder_opt.update(state_encoder, state_encoder_grads)

    # ------------------------------------------------------------------ #
    # update_abstract_critic: own optimizer, everything stop-gradiented -
    # eta/phi get NO gradient from this loss.
    # ------------------------------------------------------------------ #
    z = jax.lax.stop_gradient(state_encoder(obs))
    next_z = jax.lax.stop_gradient(state_encoder(next_obs))
    a_bar = jax.lax.stop_gradient(action_encoder(obs, act))

    next_act_actual = target_actor(next_obs)
    next_a_bar_clean = action_encoder(next_obs, next_act_actual)
    noise2 = jnp.clip(
        stddev * jax.random.normal(noise_key2, next_a_bar_clean.shape),
        -config.stddev_clip,
        config.stddev_clip,
    )
    next_a_bar = jax.lax.stop_gradient(jnp.clip(next_a_bar_clean + noise2, -1.0, 1.0))

    target_Q_bar = target_abstract_critic(
        jnp.concatenate([next_z, next_a_bar], axis=-1)
    )
    target_Q_bar = jax.lax.stop_gradient(reward + discount * target_Q_bar)

    def abstract_critic_loss_fn(abstract_critic):
        current_Q_bar = abstract_critic(jnp.concatenate([z, a_bar], axis=-1))
        return jnp.mean((current_Q_bar - target_Q_bar) ** 2), current_Q_bar

    (abstract_critic_loss, current_Q_bar), abstract_critic_grads = nnx.value_and_grad(
        abstract_critic_loss_fn, has_aux=True
    )(abstract_critic)
    abstract_critic_opt.update(abstract_critic, abstract_critic_grads)

    # value-equivalence diagnostic (paper's own Fig. 15 tool), no gradient
    Q_diag = critic(jnp.concatenate([obs, act], axis=-1))
    value_equivalence = jnp.mean(jnp.abs(jax.lax.stop_gradient(Q_diag) - current_Q_bar))

    # ------------------------------------------------------------------ #
    # update_actor (+ update_abstract_actor, hpg_update_type=double_add)
    # and target Polyak updates, delayed by update_every_steps. Gated on
    # grad_step (a true gradient-step counter - Alg.1's "t"), NOT env_step
    # (which is num_envs-scaled and would make this cadence depend on
    # num_envs if used here).
    # ------------------------------------------------------------------ #
    def do_actor_update(
        actor,
        actor_opt,
        target_actor,
        critic,
        target_critic,
        abstract_critic,
        target_abstract_critic,
        state_encoder,
        action_encoder,
    ):
        def actor_loss_fn(actor):
            act_pi = actor(obs)
            q = critic(jnp.concatenate([obs, act_pi], axis=-1))
            dpg_loss = -jnp.mean(q)

            s_bar_a = jax.lax.stop_gradient(state_encoder(obs))
            a_bar_a = action_encoder(obs, act_pi)  # HPG flows through g
            q_bar = abstract_critic(jnp.concatenate([s_bar_a, a_bar_a], axis=-1))
            hpg_loss = -jnp.mean(q_bar)
            return dpg_loss + hpg_loss

        loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_opt.update(actor, grads)

        polyak_update(target_critic, critic, config.update_tau)
        polyak_update(target_abstract_critic, abstract_critic, config.update_tau)
        polyak_update(target_actor, actor, config.update_tau)
        return loss

    def skip_actor_update(
        actor,
        actor_opt,
        target_actor,
        critic,
        target_critic,
        abstract_critic,
        target_abstract_critic,
        state_encoder,
        action_encoder,
    ):
        del (
            actor_opt,
            target_actor,
            critic,
            target_critic,
            abstract_critic,
            target_abstract_critic,
            state_encoder,
            action_encoder,
        )
        return jnp.array(0.0)

    actor_loss = nnx.cond(
        grad_step % config.update_every_steps == 0,
        do_actor_update,
        skip_actor_update,
        actor,
        actor_opt,
        target_actor,
        critic,
        target_critic,
        abstract_critic,
        target_abstract_critic,
        state_encoder,
        action_encoder,
    )

    return (
        critic_loss,
        abstract_critic_loss,
        lax_bisim_loss,
        transition_loss,
        reward_loss,
        actor_loss,
        q_mean,
        jnp.mean(current_Q_bar),
        value_equivalence,
    )


# --------------------------------------------------------------------------- #
# train_n_steps: structural mirror of sac_single.py's train_n_steps.
# --------------------------------------------------------------------------- #
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
    target_actor,
    critic,
    critic_opt,
    target_critic,
    abstract_critic,
    abstract_critic_opt,
    target_abstract_critic,
    state_encoder,
    state_encoder_opt,
    action_encoder,
    action_encoder_opt,
    reward_predictor,
    reward_predictor_opt,
    transition_model,
    transition_model_opt,
    config,
    env_step,  # NEW vs SAC: real-transition counter, num_envs-scaled
    grad_step,  # NEW vs SAC: gradient-update counter
    key,
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
            models,
            val,
        ) = carry
        (
            actor,
            actor_opt,
            target_actor,
            critic,
            critic_opt,
            target_critic,
            abstract_critic,
            abstract_critic_opt,
            target_abstract_critic,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            reward_predictor,
            reward_predictor_opt,
            transition_model,
            transition_model_opt,
        ) = models

        key, env_key = jax.random.split(key)
        stddev = explore_stddev(env_step, config)
        use_random_action = env_step < config.num_expl_steps
        noisy_policy = ExploratoryActor(actor, stddev, use_random_action)

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
        env_step = (
            env_step + config.num_envs
        )  # real transitions collected this iteration

        def do_train(j, carry):
            key, buffer_state, obs_normalizer, grad_step, models, _ = carry
            (
                actor,
                actor_opt,
                target_actor,
                critic,
                critic_opt,
                target_critic,
                abstract_critic,
                abstract_critic_opt,
                target_abstract_critic,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                reward_predictor,
                reward_predictor_opt,
                transition_model,
                transition_model_opt,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            batch = batch._replace(
                observation=obs_normalizer.normalize(batch.observation),
                next_observation=obs_normalizer.normalize(batch.next_observation),
            )
            key, train_key = jax.random.split(key)

            val = dhpg_train_step(
                actor,
                actor_opt,
                target_actor,
                critic,
                critic_opt,
                target_critic,
                abstract_critic,
                abstract_critic_opt,
                target_abstract_critic,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                reward_predictor,
                reward_predictor_opt,
                transition_model,
                transition_model_opt,
                batch,
                config,
                train_key,
                grad_step,
                env_step,
            )
            grad_step = grad_step + 1
            models = (
                actor,
                actor_opt,
                target_actor,
                critic,
                critic_opt,
                target_critic,
                abstract_critic,
                abstract_critic_opt,
                target_abstract_critic,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                reward_predictor,
                reward_predictor_opt,
                transition_model,
                transition_model_opt,
            )
            return (key, buffer_state, obs_normalizer, grad_step, models, val)

        init_val = (jnp.zeros((), jnp.float32),) * 9
        models = (
            actor,
            actor_opt,
            target_actor,
            critic,
            critic_opt,
            target_critic,
            abstract_critic,
            abstract_critic_opt,
            target_abstract_critic,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            reward_predictor,
            reward_predictor_opt,
            transition_model,
            transition_model_opt,
        )
        key, buffer_state, obs_normalizer, grad_step, models, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, buffer_state, obs_normalizer, grad_step, models, init_val),
        )
        (
            actor,
            actor_opt,
            target_actor,
            critic,
            critic_opt,
            target_critic,
            abstract_critic,
            abstract_critic_opt,
            target_abstract_critic,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            reward_predictor,
            reward_predictor_opt,
            transition_model,
            transition_model_opt,
        ) = models

        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            env_step,
            grad_step,
            (
                actor,
                actor_opt,
                target_actor,
                critic,
                critic_opt,
                target_critic,
                abstract_critic,
                abstract_critic_opt,
                target_abstract_critic,
                state_encoder,
                state_encoder_opt,
                action_encoder,
                action_encoder_opt,
                reward_predictor,
                reward_predictor_opt,
                transition_model,
                transition_model_opt,
            ),
            val,
        )

    init_val = (jnp.zeros((), jnp.float32),) * 9
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        env_step,
        grad_step,
        (
            actor,
            actor_opt,
            target_actor,
            critic,
            critic_opt,
            target_critic,
            abstract_critic,
            abstract_critic_opt,
            target_abstract_critic,
            state_encoder,
            state_encoder_opt,
            action_encoder,
            action_encoder_opt,
            reward_predictor,
            reward_predictor_opt,
            transition_model,
            transition_model_opt,
        ),
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
        models,
        val,
    ) = nnx.fori_loop(0, num_steps, body_fun, init_carry)

    (
        actor,
        actor_opt,
        target_actor,
        critic,
        critic_opt,
        target_critic,
        abstract_critic,
        abstract_critic_opt,
        target_abstract_critic,
        state_encoder,
        state_encoder_opt,
        action_encoder,
        action_encoder_opt,
        reward_predictor,
        reward_predictor_opt,
        transition_model,
        transition_model_opt,
    ) = models

    return (
        *val,
        env_state,
        running_state,
        obs_normalizer,
        buffer_state,
        env_step,
        grad_step,
        num_steps * config.num_envs,  # matches SAC's exact convention
    )


def prefill_buffer(
    key, env, env_state, buffer_state, policy, buffer, obs_normalizer, num_itr: int
):
    """Identical to sac_single.py's prefill_buffer - no changes needed."""

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
    # ── reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    rngs = nnx.Rngs(default=args.seed, params=args.seed + 3, dropout=args.seed + 5)

    # ── device ────────────────────────────────────────────────────────────
    jax.default_device = jax.devices(args.device)[args.device_id]

    # ── build config (unconditional override, matches sac_single.py) ──────
    config = dict(default_cfg)
    config.update(
        {
            "gamma": args.gamma,
            "update_tau": args.update_tau,
            "lr": args.lr,
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

    # matching_dims=True (Appendix E.2): abstract dims == actual dims.
    abstract_state_dim = obs_dim
    abstract_action_dim = act_dim

    config_data = make_static_config_from_dict("DHPGConfig", config)()

    # ── networks + one optimizer per model ──────────────────────────────── #
    def make_opt(model):
        return nnx.Optimizer(
            model=model, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
        )

    actor = DeterministicActor(rngs, obs_dim, act_dim, config["hidden_size"])
    actor_opt = make_opt(actor)
    target_actor = deepcopy(actor)

    critic = SingleQCritic(rngs, obs_dim, act_dim, config["hidden_size"])
    critic_opt = make_opt(critic)
    target_critic = deepcopy(critic)

    abstract_critic = SingleQCritic(
        rngs, abstract_state_dim, abstract_action_dim, config["hidden_size"]
    )
    abstract_critic_opt = make_opt(abstract_critic)
    target_abstract_critic = deepcopy(abstract_critic)

    state_encoder = StateEncoder(
        rngs, obs_dim, abstract_state_dim, config["hidden_size"]
    )
    state_encoder_opt = make_opt(state_encoder)

    action_encoder = ActionEncoder(
        rngs, obs_dim, act_dim, abstract_action_dim, config["hidden_size"]
    )
    action_encoder_opt = make_opt(action_encoder)

    reward_predictor = RewardPredictor(rngs, abstract_state_dim, config["hidden_size"])
    reward_predictor_opt = make_opt(reward_predictor)

    transition_model = TransitionModel(
        rngs,
        abstract_state_dim,
        abstract_action_dim,
        config["hidden_size"],
        min_sigma=config["min_sigma"],
        max_sigma=config["max_sigma"],
    )
    transition_model_opt = make_opt(transition_model)

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

    # ── running reward statistics ───────────────────────────────────────── #
    prng_key, running_key = jax.random.split(prng_key)
    running_state = RunningStatistics.init(
        (config["eval_episode_freq"] * config["episode_length"],), running_key
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
        policy=ExploratoryActor(actor, config["explore_stddev_start"], jnp.array(True)),
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        num_itr=config["warmup_samples"],
    )

    # ── main training loop ───────────────────────────────────────────────
    logger.log("Start DHPG training")
    steps = buffer.size(buffer_state)  # real transitions, = warmup_samples * num_envs
    env_step = jnp.array(
        steps, dtype=jnp.int32
    )  # start the exploration schedule from here
    grad_step = jnp.array(0, dtype=jnp.int32)

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
            target_actor=target_actor,
            critic=critic,
            critic_opt=critic_opt,
            target_critic=target_critic,
            abstract_critic=abstract_critic,
            abstract_critic_opt=abstract_critic_opt,
            target_abstract_critic=target_abstract_critic,
            state_encoder=state_encoder,
            state_encoder_opt=state_encoder_opt,
            action_encoder=action_encoder,
            action_encoder_opt=action_encoder_opt,
            reward_predictor=reward_predictor,
            reward_predictor_opt=reward_predictor_opt,
            transition_model=transition_model,
            transition_model_opt=transition_model_opt,
            config=config_data,
            env_step=env_step,
            grad_step=grad_step,
            key=subkey,
        )

        (
            critic_loss,
            abstract_critic_loss,
            lax_bisim_loss,
            transition_loss,
            reward_loss,
            actor_loss,
            q_mean,
            q_bar_mean,
            value_equivalence,
            env_state,
            running_state,
            obs_normalizer,
            buffer_state,
            env_step,
            grad_step,
            num_steps,
        ) = val
        steps += num_steps
        logger.logged = False

        # ── logging (mirrors sac_single.py's key naming exactly) ──────────
        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_critic", critic_loss.item())
        logger.log_tabular("Loss/Loss_abstract_critic", abstract_critic_loss.item())
        logger.log_tabular("Loss/Loss_lax_bisimulation", lax_bisim_loss.item())
        logger.log_tabular("Loss/Loss_transition", transition_loss.item())
        logger.log_tabular("Loss/Loss_reward", reward_loss.item())
        logger.log_tabular("Loss/Loss_actor", actor_loss.item())

        logger.log_tabular("DHPG/Q_actual_mean", q_mean.item())
        logger.log_tabular("DHPG/Q_abstract_mean", q_bar_mean.item())
        logger.log_tabular("DHPG/Value_equivalence", value_equivalence.item())

        logger.log_tabular(
            "Norm/actor_model", get_tree_norm(nnx.state(actor, nnx.Param))
        )
        logger.log_tabular(
            "Norm/critic_model", get_tree_norm(nnx.state(critic, nnx.Param))
        )
        logger.log_tabular(
            "Norm/abstract_critic_model",
            get_tree_norm(nnx.state(abstract_critic, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/state_encoder_model",
            get_tree_norm(nnx.state(state_encoder, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/action_encoder_model",
            get_tree_norm(nnx.state(action_encoder, nnx.Param)),
        )

        logger.log_tabular(
            "Eval/Return",
            running_state.reward_state.data.sum() / config["eval_episode_freq"],
        )

        logger.dump_tabular()

        # ── periodic checkpoint (matches sac_single.py: warmup_samples is an
        # ITERATION count, multiply by num_envs to compare against `steps`) ──
        if (steps - config["warmup_samples"] * config["num_envs"]) % config[
            "save_freq"
        ] == 0:
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

    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "seed-" + str(args.seed).zfill(3)
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]  # "dhpg"
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
