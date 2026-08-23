"""
Deep Homomorphic Policy Gradient (DHPG) - state-observation, deterministic
variant ('hpg_update_type=double_add').

Verified line-for-line against sahandrez/homomorphic_policy_gradient @ master
(agents/hpg.py, models/core.py, models/transition_model.py, utils/utils.py,
utils/replay_buffer.py, utils/dmc.py, train.py, cfgs/*).

=========================================================================
THE DIVIDING LINE
=========================================================================
HARNESS = sac_single.py.  Everything about the experimental regime is
resolved identically to your SAC script, from the same sac_args() parser,
with the same unconditional `config.update({...})`. num_envs, the stopping
criterion (total_env_steps), the seed phase, log/save cadence, replay
capacity, batch size, and the update-to-data ratio all come from the CLI
exactly as they do for SAC. Run both scripts with the same flags and they
see the same regime.

ALGORITHM = the author.  Everything inside the DHPG update - losses,
gradient routing, network shapes, n-step returns, target smoothing,
schedules, delayed updates - follows the PyTorch source and is NOT exposed
to the CLI, so a SAC-shaped flag can never silently change it.

Only two keys sit on the boundary, because sac_args() supplies them and
the author's values differ:
    lr          sac_args 1e-3   author 1e-4
    update_tau  sac_args 5e-3   author 1e-2
These are algorithm hyperparameters, not regime, so `use_author_algo_hparams`
(default True) restores them AFTER the unconditional update. Set it False to
let DHPG inherit SAC's. That single block is the one place harness and
algorithm touch - see [BOUNDARY] in main().
(gamma 0.99 and hidden_size 256 agree between the two already.)

=========================================================================
ALGORITHM FIXES vs THE PREVIOUS VERSION  (grep "[FIX")
=========================================================================
[FIX A] EXPLORATION STDDEV DURATION IS 1e5, NOT 1e6. Table 1 says
    linear(1.0, 0.1, 1e6), but every task config that was actually run
    (cfgs/task/easy.yaml and medium.yaml, inherited by all 18 tasks) sets
    'linear(1.0,0.1,100000)'. The code is ground truth.

[FIX B] N-STEP RETURNS (n=3). utils/replay_buffer.py::_sample does
        reward = 0; discount = 1
        for i in range(nstep):
            reward   += discount * reward[idx+i]
            discount *= discount[idx+i] * gamma
    so the STORED discount already contains gamma^n. Implemented as an
    n-step FIFO on the ROLLOUT side (nstep_* helpers), leaving
    UniformSamplingQueue untouched. dhpg_train_step therefore uses
    `discount` RAW - no extra * gamma - and the lax-bisimulation target
    inherits the same gamma^n, matching get_lax_bisim(..., discount).
    n=1 reproduces the old behaviour exactly.
    Per-task: the entire walker domain uses nstep=1 (batch_size stays on
    the harness, since batch size is regime, not algorithm).

[FIX C] BOOTSTRAP THROUGH TIME-LIMIT TRUNCATION. actor_step sets
    discount = 1 - n_state.done, and brax's EpisodeWrapper raises done=1 at
    the 1000-step limit as well as at true termination - so every 1000 steps
    the old code emitted a discount=0 sample, cutting the bootstrap on a
    purely artificial boundary. The author's DMC envs never terminate
    (time_step.discount is always 1.0), so the target is r + gamma*Q ALWAYS.
    `truncation` is in state_extras precisely to undo this: the per-step
    bootstrap factor is 1 - done*(1 - truncation), zero only on TRUE
    termination. The n-step window still STOPS at a truncation (the author's
    episode-indexed buffer can never sample across a reset); it just keeps
    the bootstrap.
    >>> YOUR sac_single.py HAS THE SAME BIAS. For a matched comparison apply
    >>> it there too - in sac_train_step, replace
    >>>     discount = data.discount
    >>> with
    >>>     trunc = data.extras["state_extras"]["truncation"]
    >>>     discount = data.discount + (1.0 - data.discount) * trunc
    >>> Until you do, SAC is the one being penalised, not DHPG.
    Toggle: bootstrap_on_truncation (False = old behaviour = current SAC).

[FIX D] WARMUP EXPLORATION POLICY. HPGAgent.act() is uniform-random only
    while step < num_expl_steps (2000); after that it already uses
    actor + N(0, sigma). prefill_buffer now threads env_step and applies the
    same cutoff instead of being uniform-random throughout.

[FIX E] VALUE-EQUIVALENCE DIAGNOSTIC uses the POST-update abstract critic,
    matching update_abstract_critic(), which recomputes
    abstract_critic(z, abstract_action) after its optimizer .step().

[FIX F] EXPLORATION SCHEDULE ORIGIN. The schedule is in env-transition
    units. In the author's run the seed phase is 4000 of 1e6 transitions
    (0.4%) - negligible. Under the SAC harness with num_envs=128 and
    warmup_samples=5000 the seed phase is 5000*128 = 640,000 transitions,
    which ALREADY EXCEEDS the whole 1e5 decay window: DHPG would begin
    training with sigma pinned at the 0.1 floor and never explore. Two
    knobs, both on by default, preserve the author's INTENT under a
    different budget:
      explore_schedule_from_training_start - the schedule clock starts when
          gradient updates start, not when the seed phase does (in the
          author's regime this shifts things by 0.4% - a no-op).
      scale_explore_to_budget - decay over explore_stddev_decay_frac of
          total_env_steps (0.1, i.e. the author's 1e5 / 1e6) instead of a
          literal 1e5.
    Set BOTH False for the literal Table-1-code value. The resolved
    schedule is printed to the run log either way.

=========================================================================
DELIBERATE DEVIATION
=========================================================================
`matching_dims` defaults True (abstract_state_dim = obs_dim,
abstract_action_dim = act_dim) - the only reading consistent with Appendix
E.2 ("In the case of state observations, the abstract MDP has the same state
and action dimensions as the actual MDP") and Table 2 ("Feature dim: same as
the state dim of the task"). The repo DEFAULTS disagree: cfgs/config.yaml
has matching_dims: false and feature_dim: 50, and the README's state-obs
command overrides neither, so running it literally gives abstract dims 50/50.
Set matching_dims=False for that.

=========================================================================
HARNESS NOTES (identical to sac_single.py, listed so nothing is a surprise)
=========================================================================
- warmup_samples and log_freq are ITERATION counts, so the seed phase is
  warmup_samples * num_envs transitions and a log line covers
  log_freq * num_envs transitions - SAC's convention exactly.
- UTD = train_per_step / num_envs, SAC's convention. `match_author_utd`
  (default False) is available if you ever want the author's UTD=1, but
  leave it False for a matched comparison.
- Observations are normalized by the shared RunningMeanStd, as in SAC.
  The author uses raw observations; `normalize_obs=False` swaps in
  IdentityNormalizer, which is a TRUE pass-through - note that an
  un-updated RunningMeanStd is NOT, because .normalize() clips to +-10
  unconditionally and actor_step always calls it (measured: 12.5 -> 10.0,
  -25.0 -> -10.0, saturating cheetah/quadruped/walker velocities).
- `Eval/Return` is SAC's exact statistic (behaviour-policy running return)
  so the two logs line up key-for-key. `Eval/Return_deterministic` is added
  alongside: a noise-free rollout on a separate eval env, matching
  train.py::Workspace.eval(). Prefer the latter when plotting - DHPG's
  behaviour policy carries additive N(0, sigma) noise with sigma up to 1.0,
  which depresses the SAC-style statistic far more than SAC's entropy-tuned
  exploration does. Adding the same deterministic eval to sac_single.py
  would make the comparison exact.
- The checkpoint condition is SAC's expression verbatim. Note it can miss
  entirely when `steps` advances by log_freq*num_envs per iteration; the
  final save still fires. Fix both scripts together if it matters.

=========================================================================
VERIFIED CORRECT, UNCHANGED
=========================================================================
Single-head critics (no clipped double-Q), no LayerNorm, orthogonal init +
zero bias, 2-hidden-layer 256-unit MLPs, sigmoid-scaled TransitionModel
sigma in [1e-4, 1e1], Huber lax bisimulation with the transition-model
prediction fully detached, reward consistency predicted from the SAMPLED
next abstract state, TWO independent TD3-smoothing draws, unclipped
exploration noise in act(), no gradient clipping, one joint backward for the
critic/homomorphism group followed by five separate optimizer steps,
abstract critic fully stop-gradiented, actor loss = DPG + HPG with f
detached and gradient flowing through g, all three Polyak updates gated on
the same update_every_steps condition as the actor.
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
)
from utils.logger import EpochLogger
from utils.models import get_tree_norm
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args  # reuse the CLI parser

default_cfg = {
    # ---- harness defaults; every one of these is overwritten by sac_args() ----
    "log_freq": int(1e4),  # ITERATIONS, SAC convention
    "save_freq": int(5e4),
    "eval_episode_freq": 5,
    "hidden_size": 256,
    "lr": 1e-4,  # author value; see [BOUNDARY]
    "gamma": 0.99,
    "update_tau": 0.01,  # author value; see [BOUNDARY]
    "train_per_step": 1,
    "episode_length": 1000,
    "warmup_samples": int(4e3),  # ITERATIONS, SAC convention
    "max_replay_size": int(1e6),
    "batch_size": int(256),
    "total_env_steps": int(1e6),
    # ---- algorithm; author-fixed, NOT exposed to the CLI ----
    "update_every_steps": 2,  # gates actor AND all three Polyak updates
    "num_expl_steps": 2000,  # pure-random phase, env-transition units
    "stddev_clip": 0.3,
    "explore_stddev_start": 1.0,
    "explore_stddev_end": 0.1,
    "explore_stddev_decay_steps": int(1e6),  # [FIX A]
    "explore_stddev_decay_frac": 0.1,  # [FIX F] author 1e5 / 1e6
    "explore_schedule_from_training_start": True,  # [FIX F]
    "scale_explore_to_budget": True,  # [FIX F]
    "explore_origin": 0,  # filled in by main(); see [FIX F]
    "homomorphic_coef": 1.0,
    "min_sigma": 1e-4,
    "max_sigma": 1e1,
    "nstep": 3,  # [FIX B]; walker_* -> 1
    "bootstrap_on_truncation": True,  # [FIX C]
    "matching_dims": True,
    "feature_dim": 50,  # only used when matching_dims=False
    # ---- regime switches ----
    "use_author_algo_hparams": False,  # [BOUNDARY] restore lr / update_tau
    "match_author_utd": False,  # keep False for a matched comparison
    "normalize_obs": True,  # True = SAC parity; author uses raw obs
    "num_eval_episodes": 1,  # episodes PER PARALLEL ENV
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
# True pass-through normalizer, for normalize_obs=False (author behaviour).
#
# Not merely "an un-updated RunningMeanStd": actor_step ALWAYS calls
# obs_normalizer.normalize(obs) before the policy, and RunningMeanStd.normalize
# clips to +-10 even at init (mean=0, var=1), so it saturates any |obs| > 10 -
# which DMC velocity components routinely exceed. Zero fields => empty pytree
# => jit-safe, and it drops into every place RunningMeanStd was used.
# --------------------------------------------------------------------------- #
@struct.dataclass
class IdentityNormalizer:
    def update(self, x: jnp.ndarray) -> "IdentityNormalizer":
        return self

    def normalize(self, x: jnp.ndarray, clip: float = 10.0) -> jnp.ndarray:
        return x


# --------------------------------------------------------------------------- #
# [FIX B] + [FIX C] n-step return accumulation.
#
# FIFO index 0 = oldest, index nstep-1 = newest. `count` is the number of valid
# entries so far (clipped to nstep), so the oldest valid index is nstep - count.
# Early in a run the window is simply SHORTER than nstep, never malformed.
#
# Per step:  d_i = discount = 1 - done       (from actor_step)
#            t_i = truncation                (from state_extras)
#   bootstrap factor : d_i + (1 - d_i)*t_i   -> 0 only on TRUE termination
#   window continues : d_i * (1 - t_i)       -> stops on done of any kind
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Models. Abstract dims per `matching_dims`; see header.
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
    Matches models.transition_model.ProbabilisticTransitionModel: sigma via a
    sigmoid-scaled head, NOT log_std."""

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
    """models.core.DDPGCritic: ONE Q-head, no twin/ensemble, no clipped double-Q.
    Used for BOTH the actual and the abstract critic."""

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act):
        x = nnx.relu(self.l1(obs_act))
        x = nnx.relu(self.l2(x))
        return jnp.squeeze(self.l3(x), axis=-1)


class ExploratoryActor(nnx.Module):
    """Matches HPGAgent.act(): for env_step < num_expl_steps, pure uniform random
    action; otherwise pi_theta(s) + N(0, stddev), UNCLIPPED. With stddev=0 and
    use_random_action=False this is eval_mode=True."""

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
    """utils.utils.schedule(stddev_schedule, step) for linear(init,final,duration).
    [FIX F] cfg.explore_origin shifts the clock to the start of training when
    explore_schedule_from_training_start is set (0 otherwise)."""
    step = jnp.maximum(env_step - cfg.explore_origin, 0)
    frac = jnp.clip(step / cfg.explore_stddev_decay_steps, 0.0, 1.0)
    return cfg.explore_stddev_start + frac * (
        cfg.explore_stddev_end - cfg.explore_stddev_start
    )


def polyak_update(target_model, curr_model, tau: float):
    """Identical to sac_single.py's polyak_update / utils.utils.soft_update_params."""
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
    # [FIX B] `discount` is the buffer's n-step discount and ALREADY contains
    # gamma^n (see nstep_aggregate / utils/replay_buffer.py). Do NOT multiply
    # by gamma again here. (sac_train_step still does `config.gamma * discount`
    # because its buffer is 1-step.)
    discount = data.discount
    next_obs = data.next_observation

    stddev = explore_stddev(env_step, config)
    key, noise_key1, noise_key2, perm_key, sample_key = jax.random.split(key, 5)

    # ------------------------------------------------------------------ #
    # update_critic: critic + homomorphism map. ONE joint loss / ONE backward
    # (nnx.value_and_grad over 5 args), then 5 SEPARATE optimizer.update()
    # calls - matches agents.hpg.HPGAgent.update_critic (one loss.backward(),
    # then critic/transition/reward/action_encoder/state_encoder optimizers
    # each .step() once).
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

        # lax bisimulation (Huber; transition-model prediction fully detached,
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

        # transition + reward consistency (reward predicted from the SAMPLED
        # next abstract state)
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
    # update_abstract_critic: own optimizer, everything stop-gradiented - eta
    # and phi get NO gradient from this loss. Uses the POST-update encoders,
    # exactly as the author's sequential update() does.
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
        return jnp.mean((current_Q_bar - target_Q_bar) ** 2)

    abstract_critic_loss, abstract_critic_grads = nnx.value_and_grad(
        abstract_critic_loss_fn
    )(abstract_critic)
    abstract_critic_opt.update(abstract_critic, abstract_critic_grads)

    # [FIX E] value-equivalence diagnostic (paper Fig. 15) from the POST-update
    # abstract critic, matching update_abstract_critic().
    Q_diag = jax.lax.stop_gradient(critic(jnp.concatenate([obs, act], axis=-1)))
    Q_bar_diag = jax.lax.stop_gradient(
        abstract_critic(jnp.concatenate([z, a_bar], axis=-1))
    )
    value_equivalence = jnp.mean(jnp.abs(Q_diag - Q_bar_diag))

    # ------------------------------------------------------------------ #
    # update_actor (+ update_abstract_actor, hpg_update_type=double_add) and
    # the target Polyak updates, delayed by update_every_steps. Gated on
    # grad_step - Algorithm 1's "t" is a training-iteration index, so this
    # cadence must not depend on num_envs.
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
        jnp.mean(Q_bar_diag),
        value_equivalence,
    )


# --------------------------------------------------------------------------- #
# train_n_steps: structural mirror of sac_single.py's train_n_steps.
# Extra carry vs SAC: env_step, grad_step, nstep_fifo, nstep_count.
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
    env_step,  # real-transition counter, num_envs-scaled
    grad_step,  # gradient-update counter
    nstep_fifo,  # [FIX B]
    nstep_count,  # [FIX B]
    key,
):
    num_steps = config.log_freq  # ITERATIONS, SAC convention

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

        # [FIX B] push the 1-step transition, insert the aggregated n-step one
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

        if config.normalize_obs:  # SAC parity; author uses raw observations
            obs_normalizer = obs_normalizer.update(transition.observation)

        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )
        env_step = env_step + config.num_envs  # real transitions this iteration

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
            if config.normalize_obs:
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
        # UTD = train_per_step / num_envs, exactly as in sac_single.py.
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
            nstep_fifo,
            nstep_count,
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
        nstep_fifo,
        nstep_count,
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
        nstep_fifo,
        nstep_count,
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
        nstep_fifo,
        nstep_count,
        num_steps * config.num_envs,  # matches SAC's exact convention
    )


# --------------------------------------------------------------------------- #
# Deterministic evaluation, mirroring train.py::Workspace.eval() (separate eval
# env, eval_mode=True -> no exploration noise). Logged ALONGSIDE SAC's
# behaviour-policy statistic, not instead of it, so the two logs stay
# comparable key-for-key.
# --------------------------------------------------------------------------- #
@functools.partial(nnx.jit, static_argnames=("env", "num_steps"))
def eval_n_steps(env, env_state, actor, obs_normalizer, key, num_steps: int):
    policy = ExploratoryActor(actor, 0.0, jnp.array(False))

    def body(carry, _):
        key, env_state, total = carry
        key, subkey = jax.random.split(key)
        n_state, _ = actor_step(
            env=env,
            env_state=env_state,
            policy=policy,
            obs_normalizer=obs_normalizer,
            key=subkey,
            extra_fields=("truncation",),
        )
        return (key, n_state, total + jnp.sum(n_state.reward)), ()

    (_, env_state, total), () = jax.lax.scan(
        body, (key, env_state, jnp.zeros(())), (), length=num_steps
    )
    return total


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
    """Seeding phase. [FIX D] applies HPGAgent.act()'s real cutoff: uniform
    random only while env_step < num_expl_steps (2000), then actor + N(0, sigma).
    [FIX B] runs the same n-step FIFO. `num_itr` is an ITERATION count, so this
    collects num_itr * num_envs transitions - SAC's convention."""

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

        if config.normalize_obs:
            obs_normalizer = obs_normalizer.update(transition.observation)

        env_step = env_step + config.num_envs
        return (key, n_state, buffer_state, obs_normalizer, env_step, fifo, count), ()

    jitted_body = jax.jit(body)
    (
        (
            _,
            env_state,
            buffer_state,
            obs_normalizer,
            env_step,
            nstep_fifo,
            nstep_count,
        ),
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
    # ── reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    rngs = nnx.Rngs(default=args.seed, params=args.seed + 3, dropout=args.seed + 5)

    # ── device ────────────────────────────────────────────────────────────
    jax.default_device = jax.devices(args.device)[args.device_id]

    # ── build config: UNCONDITIONAL, identical to sac_single.py ───────────
    # Same parser, same keys, same resolution order. Run both scripts with the
    # same flags and they see the same regime.
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

    # ── [BOUNDARY] the only place harness and algorithm touch ─────────────
    # sac_args() supplies lr and update_tau, but those are DHPG algorithm
    # hyperparameters, not regime: the author uses 1e-4 and 0.01 where
    # sac_args defaults to 1e-3 and 0.005. Everything else in the update
    # above is regime, and SAC's value rightly wins.
    # Delete this block (or set use_author_algo_hparams=False) to let DHPG
    # inherit SAC's optimizer settings instead.
    if config["use_author_algo_hparams"]:
        config["lr"] = default_cfg["lr"]  # 1e-4
        config["update_tau"] = default_cfg["update_tau"]  # 0.01

    # ── per-task algorithm override from the author's cfgs/task/*.yaml ────
    # The walker domain uses nstep=1. (The author also sets batch_size=512
    # there, but batch size is regime, so it stays on the harness.)
    if args.task.lower().startswith("walker"):
        config["nstep"] = 1

    num_envs = config["num_envs"]
    seed_transitions = config["warmup_samples"] * num_envs

    # ── [FIX F] resolve the exploration schedule for this budget ─────────
    if config["scale_explore_to_budget"]:
        config["explore_stddev_decay_steps"] = max(
            1, int(config["explore_stddev_decay_frac"] * config["total_env_steps"])
        )
    if config["explore_schedule_from_training_start"]:
        config["explore_origin"] = int(seed_transitions)

    # ── environment ───────────────────────────────────────────────────────
    prng_key, env_key = jax.random.split(prng_key)
    env_keys = jax.random.split(env_key, num_envs)

    env = wrap_env_for_training(
        registry.load(args.task, config_overrides={"impl": "jax"}),
        episode_length=config["episode_length"],
        full_reset=False,
    )
    env_state = env.reset(env_keys)
    obs_dim = env.observation_size
    act_dim = env.action_size

    # SAC parity by default; IdentityNormalizer is a TRUE pass-through, which
    # an un-updated RunningMeanStd is not (it clips to +-10).
    obs_normalizer = (
        RunningMeanStd.init((obs_dim,))
        if config["normalize_obs"]
        else IdentityNormalizer()
    )

    # separate eval env, as in train.py (self.train_env / self.eval_env)
    eval_env = wrap_env_for_training(
        registry.load(args.task, config_overrides={"impl": "jax"}),
        episode_length=config["episode_length"],
        full_reset=False,
    )

    # Appendix E.2 / Table 2; see header for the repo-default caveat.
    if config["matching_dims"]:
        abstract_state_dim = obs_dim
        abstract_action_dim = act_dim
    else:
        abstract_state_dim = config["feature_dim"]
        abstract_action_dim = config["feature_dim"]

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

    # ── [FIX B] n-step FIFO state (leading dim nstep, then num_envs) ──────
    nstep_template = Transition(
        observation=jnp.zeros((num_envs, obs_dim), jnp.float32),
        action=jnp.zeros((num_envs, act_dim), jnp.float32),
        reward=jnp.zeros((num_envs,), jnp.float32),
        discount=jnp.zeros((num_envs,), jnp.float32),
        next_observation=jnp.zeros((num_envs, obs_dim), jnp.float32),
        extras={"state_extras": {"truncation": jnp.zeros((num_envs,), jnp.float32)}},
    )
    nstep_fifo = nstep_fifo_init(nstep_template, config["nstep"])
    nstep_count = jnp.array(0, dtype=jnp.int32)

    # ── running reward statistics ─────────────────────────────────────────
    prng_key, running_key = jax.random.split(prng_key)
    running_state = RunningStatistics.init(
        (config["eval_episode_freq"] * config["episode_length"],), running_key
    )

    # ── logger ────────────────────────────────────────────────────────────
    dict_args = dict(config)
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)

    # make the harness/algorithm split auditable in the run log
    logger.log(
        f"[harness] num_envs={num_envs} total_env_steps={config['total_env_steps']} "
        f"batch={config['batch_size']} train_per_step={config['train_per_step']} "
        f"UTD={config['train_per_step'] / num_envs:.4f} "
        f"seed={seed_transitions} transitions "
        f"log_every={config['log_freq'] * num_envs} transitions "
        f"normalize_obs={config['normalize_obs']}"
    )
    logger.log(
        f"[algorithm] lr={config['lr']} tau={config['update_tau']} "
        f"nstep={config['nstep']} update_every={config['update_every_steps']} "
        f"stddev=linear({config['explore_stddev_start']},"
        f"{config['explore_stddev_end']},{config['explore_stddev_decay_steps']}) "
        f"origin={config['explore_origin']} "
        f"bootstrap_on_truncation={config['bootstrap_on_truncation']} "
        f"matching_dims={config['matching_dims']}"
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
        num_itr=config["warmup_samples"],
        env_step=env_step,
        nstep_fifo=nstep_fifo,
        nstep_count=nstep_count,
    )

    # ── main training loop ───────────────────────────────────────────────
    logger.log("Start DHPG training")
    steps = buffer.size(buffer_state)
    grad_step = jnp.array(0, dtype=jnp.int32)
    eval_steps = config["episode_length"] * config["num_eval_episodes"]

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
            nstep_fifo=nstep_fifo,
            nstep_count=nstep_count,
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
            nstep_fifo,
            nstep_count,
            num_steps,
        ) = val
        steps += num_steps
        logger.logged = False

        # ── deterministic evaluation on a fresh eval env ─────────────────
        prng_key, eval_key, eval_reset_key = jax.random.split(prng_key, 3)
        eval_state = eval_env.reset(jax.random.split(eval_reset_key, num_envs))
        eval_total = eval_n_steps(
            env=eval_env,
            env_state=eval_state,
            actor=actor,
            obs_normalizer=obs_normalizer,
            key=eval_key,
            num_steps=eval_steps,
        )
        eval_return = eval_total / (num_envs * config["num_eval_episodes"])

        # ── logging (mirrors sac_single.py's key naming) ──────────────────
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

        # SAC's exact statistic, under SAC's exact key, so the logs align
        logger.log_tabular(
            "Eval/Return",
            running_state.reward_state.data.sum() / config["eval_episode_freq"],
        )
        # noise-free rollout - prefer this when plotting; see header
        logger.log_tabular("Eval/Return_deterministic", eval_return.item())

        logger.dump_tabular()

        # ── periodic checkpoint (sac_single.py's expression verbatim; note it
        # can miss entirely since `steps` jumps by log_freq*num_envs) ──────
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
