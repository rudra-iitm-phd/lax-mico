"""
MICo (Matching under Independent Couplings) - state-observation variant.

Castro, Kastner, Panangaden, Rowland, "MICo: Improved representations via
sampling-based state similarity for Markov decision processes", NeurIPS 2021.

Verified line-for-line against
  google-research/google-research @ master ::
      mico/atari/metric_utils.py           (the metric itself)
      mico/dm_control/metric_sac_agent.py  (the CONTINUOUS-CONTROL agent)
      mico/dm_control/configs/mico.gin     (the continuous-control hyperparams)
  google/dopamine @ master ::
      dopamine/jax/losses.py                                (huber_loss)
      dopamine/labs/sac_from_pixels/continuous_networks.py   (the encoder)
and against YOUR sac_single.py (n-step version) for everything else.

=========================================================================
WHAT MICo IS, AND THEREFORE WHAT THIS FILE IS
=========================================================================
MICo is not an algorithm, it is an AUXILIARY LOSS on the representation. The
author adds it to an existing agent and changes nothing else: on Atari to the
Dopamine value-based agents, and on DM-Control (paper Fig. 1 right, Sec. 6) to
SAC. So the honest port of "MICo for state vectors" is

        YOUR SAC + an encoder + the MICo loss on that encoder's output,

which is exactly what metric_sac_agent.py is, minus the convolutions.

Concretely, this file is sac_single.py with FOUR changes and nothing else:
    1. a StateEncoder phi between the observation and the critic, so that
       Q(x, a) = psi(phi(x), a). MICo needs a state-only representation.
    2. an ActorHead reading a DETACHED copy of phi's trunk, so the policy
       loss never shapes phi (the author's / SAC-AE's convention).
    3. the MICo loss, summed into the critic loss in one backward.
    4. target_encoder, Polyak-updated alongside target_critic.
Everything else - the alpha loss, the critic loss algebra, the actor loss,
the update ORDER, the optimizers, the buffer, the n-step FIFO, evaluate(),
the logger keys, the checkpointing - is copied from your file verbatim.
SACGaussianActor, EnsembleCritic and Scalar are IMPORTED, not reimplemented,
so the actor and critic are bit-identical to your SAC's.

THE CONTROL YOU SHOULD RUN. Set mico_weight = 0.0 and this file becomes SAC
with the same encoder and the same everything else, so a MICo-vs-control
comparison isolates the loss and not the architecture. Your sac_single.py
stays as the plain-SAC baseline. That is the comparison the paper makes
(Fig. 1 right: SAC vs MICo, same backbone).

=========================================================================
THE DIVIDING LINE  (same convention as dhpg.py)
=========================================================================
HARNESS = sac_single.py. Same sac_args() parser, same unconditional
`config.update({...})`, same num_envs / total_env_steps / seed phase / log and
save cadence / replay capacity / batch size / UTD. Run all three scripts with
the same flags and they see the same regime.

ALGORITHM = the author. mico_weight, beta, the distance function, the Huber
loss, which network supplies which representation, and the encoder shape are
NOT exposed to the CLI.

Boundary keys - keys sac_args() supplies where the author's value differs:
    lr            sac_args 3e-4   author 1e-3 (mico.gin), 3e-4 (mico_saclr)
    update_tau    sac_args 5e-3   author 5e-3                  AGREE
    gamma         sac_args 0.99   author 0.99                  AGREE
    batch_size    sac_args 256    author 256                   AGREE
    max_replay    sac_args 1e5    author 1e6
    hidden_size   sac_args 256    author 1024
    reward_scale  sac_args 1.0    author 0.1
Unlike DHPG there is nothing to reconcile: mico.gin and mico_saclr.gin
bracket your lr, and tau/gamma/batch agree outright. hidden_size and the
reward scale differ but are regime under your taxonomy, so they stay on the
harness. `author_hidden_units` (default False) flips the first if you want to
check it; the second is just --reward_scaling 0.1. See [BOUNDARY] in main().

=========================================================================
THE ALGORITHM, EXACTLY  (grep "[MICO")
=========================================================================
[M1] THE DISTANCE.  U_w(x, y) = (||phi(x)||^2 + ||phi(y)||^2)/2
                                + beta * theta(phi(x), phi(y))
    where theta is the ANGLE between the two vectors (arctan2 form, not
    1 - cos) and beta = 0.1. The norm term is what lets a diffuse metric with
    non-zero self-distance be represented at all; it is not a regulariser and
    cannot be dropped in favour of a plain Euclidean or cosine distance.

[M2] THE TARGET.  T^U(x, y) = |r_x - r_y| + gamma^n * U_target(x', y').
    Taken over ALL B^2 ORDERED PAIRS in the minibatch - the second state y is
    not a separate draw, it is every other element of the same batch. Hence no
    permutation index, unlike DHPG's lax-bisimulation term, and hence a B x B
    grid where one bad sample contaminates a whole row and column ([S1]).

[M3] WHICH NETWORK SUPPLIES WHICH REPRESENTATION. The detail most
    reimplementations get wrong, and metric_sac_agent.py differs from
    metric_dqn_agent.py here:
        online_dist = representation_distances(phi_online(x), phi_frozen(x))
        target_dist = target_distances(phi_target(x'), r, gamma^n)
    - first argument: the ONLINE encoder on the batch states, WITH gradient.
    - second argument: the SAME encoder on the SAME states, evaluated under
      `frozen_params` - i.e. stop-gradient, NOT the target network. (The DQN
      agent uses the target network here; the SAC agent does not, and the SAC
      agent is the continuous-control one.) Consequence: exactly half the
      gradient of each pairwise term reaches the encoder. Deliberate.
    - the target uses the TARGET encoder on the NEXT states. That is the same
      tensor the critic target already needs, so it is computed once.

[M4] HUBER, NOT MSE, delta = 1.0, elementwise over all B^2 pairs, then meaned.
    The paper is explicit that MSE fails: "larger distances tended to
    overwhelm the optimization process, thereby degrading performance".

[M5] COMBINATION.  loss = (1 - w) * L_SAC + w * L_MICo.
    (a) THE DEFAULT w = 1e-5 IS THE AUTHOR'S LITERAL VALUE for SAC on
        DM-Control (MetricSACAgent.__init__). It is NOT the Atari value: the
        author used 0.01 for DQN/Rainbow and 0.5 for the quantile agents.
    (b) THOSE THREE VALUES ARE THE POINT. The paper states plainly why they
        differ - the author attributes them to the differing magnitudes of the
        categorical, quantile and non-distributional TD losses, and retunes w
        per agent family accordingly (Sec. 6). So w is NOT a universal
        constant of the method; it is a scale-matching coefficient between
        L_TD and L_MICo, and the author's own procedure is to re-pick it when
        the TD loss changes. Our L_TD is not his pixel-SAC's: different reward
        scale, different observation modality, different network. Re-picking w
        here FOLLOWS the author's method rather than departing from it.
    (c) HOW TO PICK IT, and what to report. MICo/Grad_ratio (logged every
        line) is w*||d L_MICo / d phi|| / ((1-w)*||d L_TD / d phi||), i.e. the
        fraction of the encoder's update direction that MICo is responsible
        for. If that number is ~1e-4, the loss is decorative and the run is
        the mico_weight=0 control wearing a different name. Sweep w over
        {0.0, 1e-5, 1e-2, 0.5} - the control plus the three values the author
        himself used - on ONE task, pick by Grad_ratio landing somewhere
        non-trivial (1e-2 to 1e-1 is a sane target) and by Eval/Return, then
        fix that w for every task and report the sweep. Override without
        touching sac_args via the environment:
            MICO_WEIGHT=0.01 python -m mico --task CheetahRun ...
        The resolved value is echoed in the [algorithm] line.
    (d) Your sac_train_step already uses 0.5 * mean(square(q_error)), and the
        author's L_SAC is 0.5*critic + 1.0*actor + 1.0*alpha. These coincide,
        so nothing needs reweighting.
    (e) The (1 - w) factor is applied to all three SAC losses so the ratio is
        the author's. With per-parameter Adam a global scale is very nearly a
        no-op anyway; what matters is the RELATIVE weight of the critic and
        MICo terms inside the ENCODER's gradient. That ratio is exact here:
        the two encoder gradients are computed separately and summed with
        weights (1-w) and w, which is algebraically identical to one backward
        over the combined loss but lets us measure each side (see (c)).

[M6] THE REPRESENTATION IS phi(x), A FUNCTION OF STATE ALONE. It cannot be a
    hidden layer of your EnsembleCritic, which consumes concat([obs, act]) -
    U(x, y) would then depend on which actions happened to be stored. Hence
    the explicit encoder and the factorisation Q(x, a) = psi(phi(x), a),
    exactly the one the paper assumes ("Q_{xi,omega}(x, .) = psi_xi(phi_omega
    (x))", Sec. 5). EnsembleCritic is reused unchanged, constructed with
    obs_dim = feature_dim so it consumes concat([z, a]).

[M7] THE ENCODER, ported from SACEncoderNetwork (sac_from_pixels):
        pixels:  conv x4 -> flatten -> Dense(50) -> LayerNorm -> tanh
        vectors: Dense(h) -> relu  -> Dense(50) -> LayerNorm -> tanh
    The conv stack has no state-vector analogue, so it becomes one hidden
    layer; everything after it is unchanged. Two properties are load-bearing:
    - the tanh bound. ||phi||^2 <= feature_dim, so the norm term of U cannot
      run away. An unbounded encoder makes the MICo target diverge - watch
      MICo/Rep_sq_norm.
    - the actor reads a SEPARATE head off a DETACHED trunk
      (actor_z = tanh(LN(Dense(stop_grad(h))))), so the policy loss never
      shapes the representation the metric is defined on. SAC-AE's convention,
      which the author kept.
    Switchable via encoder_layer_norm / encoder_tanh / encoder_hidden_layers,
    but the defaults are the author's.
    NOTE this does make the actor one layer deeper than your sac_single.py's
    actor, and bottlenecked at 50 units. That is why the mico_weight=0 control
    exists; do not compare MICo against sac_single.py alone.

[M8] N-STEP. The author runs update_horizon = 1 (mico.gin), but n-step needs
    no new machinery: Dopamine passes cumulative_gamma = gamma^update_horizon
    into target_distances and the replay buffer supplies the summed n-step
    reward, so the target at horizon n is
        |r^(n)_x - r^(n)_y| + gamma^n * U_target(x'_n, y'_n).
    Both halves come from your shared nstep helpers: `reward` is already the
    discounted n-step sum, and cumulative_gamma is gamma**config.nstep.
    DEFAULT IS nstep=3 with the walker override to 1, i.e. identical to your
    sac_single.py and dhpg.py - matched baselines beat literal fidelity here.
    Set nstep=1 for the author's value; whichever you choose, choose it in all
    three files.
    ASYMMETRY, and it is the author's: the CRITIC target uses the per-sample
    `discount` (which carries gamma^n AND termination), while the MICo target
    uses a SCALAR gamma^n with no termination factor. In Dopamine
    cumulative_gamma is a Python float and `terminals` is simply never passed
    to target_distances. On playground DMC tasks nothing truly terminates, so
    the two agree; set mico_discount_from_sample=True to use sqrt(d_i * d_j)
    instead.

=========================================================================
SYNCED FROM YOUR sac_single.py  (grep "[SYNC")
=========================================================================
[S1] TRUNCATION MASKING, PAIRWISE. Your critic loss does
    `q_error *= (1 - truncation)[..., None]`; that line is copied unchanged.
    The MICo loss needs the same mask in PAIRWISE form. BraxAutoResetWrapper
    overwrites state.obs on done, so at a truncated window `next_observation`
    is the first obs of the NEXT episode and `reward` is a partial k<n sum;
    the MICo target reads BOTH of those, for BOTH members of every pair. So a
    single contaminated sample poisons an entire ROW and an entire COLUMN of
    the B x B grid, and one-sided masking would let half of them through.
    utils/mico_utils.pair_mask builds valid[i] * valid[j] in squarify's flat
    index order (verified numerically - run that file directly).
    The ONLINE side reads only `obs` and needs no mask.
    Like your SAC, the masked terms use a plain masked mean rather than
    renormalising by the valid count.
[S2] REWARD SCALING, and one deliberate divergence. Your critic target uses
    `reward * config.reward_scaling`; copied. The author does the same in the
    SAC target (reward_scale_factor = 0.1) but passes the RAW, UNSCALED reward
    to target_distances - see metric_sac_agent.py, `reward_scale_factor *
    rewards` in `targets` versus plain `rewards` in `target_distances`. We
    follow the author. This is the OPPOSITE of dhpg.py's [S2], where you scale
    the homomorphism targets too; the divergence is the author's, not a slip.
    mico_scale_reward=True makes it consistent with DHPG instead. With the
    default reward_scaling=1.0 the question is moot.
[S3] UPDATE ORDER copied exactly: all four gradients computed against the
    PRE-update parameters, then alpha / critic+encoder / actor+actor_head
    stepped, then Polyak. `alpha` inside the critic and actor losses is the
    pre-update value; `alpha_post` is reported, as in your file.
[S4] alpha_opt keeps its HARDCODED lr=3e-4 (not config["lr"]), as in yours.
[S5] target_entropy = -0.5 * act_dim, set in main() before the config freeze.
[S6] The seed phase runs the POLICY (not uniform random), as in yours -
    prefill_buffer is passed the SACPolicy wrapper.
[S7] evaluate(), the monotone next_save checkpointing, `steps =
    int(buffer.size(...))`, the normalizer-at-point-of-use, the walker nstep
    override, and the logger keys are all copied verbatim.

=========================================================================
LOG-KEY MAPPING vs dhpg.py  (grep "[LOGSYNC")
=========================================================================
Identical key, directly overlayable:
    Train/Steps                 same counter, same units (env transitions),
                                same starting value (buffer.size after the
                                seed phase), same window
                                (log_freq * num_envs transitions per line)
    Eval/Return                 IDENTICAL evaluate(), deterministic policy,
                                num_eval_envs parallel episodes, alive-masked
    Norm/actor_model            both = ||actor params||
    Norm/critic_model           both = ||critic params||
    Norm/state_encoder_model    DHPG's f_phi vs MICo's phi. Renamed here from
                                Norm/encoder_model to match.

Identical key, DIFFERENT quantity - do NOT plot on shared axes:
    Loss/Loss_critic            DHPG: plain F.mse_loss, ONE Q head, no 0.5.
                                MICo: 0.5*mean(square(.)) over TWO heads, i.e.
                                your SAC's form. Roughly 0.25*(e1^2+e2^2) vs
                                e^2 - a scale difference by construction, not
                                a behavioural one.
    Loss/Loss_actor             DHPG: -(Q + Q_bar), so it tracks 2x the value
                                scale and goes to -90 on a task where Q ~ 45.
                                MICo: alpha*log_pi - min(q1,q2), the SAC form.
                                Both are unbounded below and neither is a
                                progress signal; use Eval/Return.

Present in one file only (no counterpart exists):
    DHPG only:  Loss/Loss_abstract_critic, Loss/Loss_lax_bisimulation,
                Loss/Loss_transition, Loss/Loss_reward, DHPG/Q_abstract_mean,
                DHPG/Value_equivalence, Norm/abstract_critic_model,
                Norm/action_encoder_model
    MICo only:  Loss/Loss_alpha, Loss/Loss_mico, SAC/Alpha, SAC/LogPi_mean,
                MICo/Online_dist, MICo/Target_dist, MICo/Norm_term,
                MICo/Angle_term, MICo/Rep_sq_norm, Norm/actor_head_model

One rename needed in your plotting code, not in either script:
    DHPG/Q_actual_mean   <->   SAC/Q1_mean
Both are mean(Q(s, a)) over the sampled batch under the online critic. MICo
additionally reports SAC/Q2_mean; the two twin heads track each other, so
either one is the fair counterpart to DHPG's single head.

Also note, and this is true of BOTH files: every scalar except Train/Steps
and Eval/Return is the value from the LAST inner training iteration in the
log window, not an average over it. That is dhpg.py's behaviour and this file
reproduces it, so the noise level is comparable - but neither is a smoothed
statistic, so read them as spot checks.
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

# [M8] the same n-step helpers sac_single.py and dhpg.py use, so the three
# baselines cannot drift on how returns are computed.
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

# The actor and critic are IMPORTED, not reimplemented, so they are identical
# to sac_single.py's. Only their input dimension changes (feature_dim, not
# obs_dim), because both now sit downstream of the encoder.
from utils.mico_utils import (
    cosine_distance,
    huber_loss,
    pair_mask,
    representation_distances,
    target_distances,
)
# from utils.algo_models import EnsembleCritic, SACGaussianActor, Scalar, get_tree_norm
from utils.models import EnsembleCritic, SACGaussianActor, Scalar, get_tree_norm
from utils.types import Transition
from utils.utils import make_static_config_from_dict, sac_args

default_cfg = {
    # ---- identical to sac_single.py's default_cfg ----
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
    "nstep": 3,  # [M8] 1 reproduces the old 1-step behaviour EXACTLY
    "bootstrap_on_truncation": True,
    # ---- MICo; author-fixed, NOT exposed to the CLI ----
    "mico_weight": 0.01,  # [M5] metric_sac_agent.py default. NOT the Atari value.
    "mico_beta": 0.1,  # [M1] weight on the angular term inside U
    "huber_delta": 1.0,  # [M4]
    "feature_dim": 50,  # [M7] Dense(50) in SACEncoderNetwork
    "encoder_hidden_layers": 1,  # [M7] stands in for the conv stack
    "encoder_layer_norm": True,  # [M7]
    "encoder_tanh": True,  # [M7] bounds ||phi||^2 by feature_dim
    "mico_scale_reward": False,  # [S2] the author uses the RAW reward here
    "mico_discount_from_sample": False,  # [M8] the author uses a scalar gamma^n
    # ---- regime switches ----
    "author_hidden_units": False,  # [BOUNDARY] True -> hidden_size = 1024
    "normalize_obs": True,  # True = SAC parity; the author uses raw obs
}


def orthogonal_linear(rngs, in_dim, out_dim):
    """SACEncoderNetwork uses nn.initializers.orthogonal() throughout."""
    return nnx.Linear(
        in_dim,
        out_dim,
        kernel_init=jax.nn.initializers.orthogonal(),
        bias_init=nnx.initializers.zeros,
        rngs=rngs,
    )


@struct.dataclass
class IdentityNormalizer:
    """True pass-through, for normalize_obs=False. An un-updated RunningMeanStd
    is NOT one: .normalize() clips to +-10 unconditionally, saturating the
    cheetah/quadruped/walker velocity components."""

    def update(self, x: jnp.ndarray) -> "IdentityNormalizer":
        return self

    def normalize(self, x: jnp.ndarray, clip: float = 10.0) -> jnp.ndarray:
        return x


# --------------------------------------------------------------------------- #
# [M6][M7] The encoder. phi(x) is what the MICo loss shapes, and the only new
# parameters in this file besides the actor head.
# --------------------------------------------------------------------------- #
class StateEncoder(nnx.Module):
    """phi(x). Port of SACEncoderNetwork's critic branch to state vectors:

        trunk:    obs -> (Dense -> relu) x encoder_hidden_layers -> h
        phi(x) =  tanh(LayerNorm(Dense_{feature_dim}(h)))

    Trained by the critic loss and by the MICo loss, and by nothing else.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        hidden_size: int,
        feature_dim: int,
        n_hidden: int = 1,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
    ):
        # [FLAX] The trunk layers are stored as INDIVIDUAL attributes
        # (trunk_0, trunk_1, ...) rather than in a Python list. Newer flax.nnx
        # treats a bare `list` as a STATIC attribute and refuses to hold
        # Modules in it ("Found data on value of type '<class 'list'>'
        # assigned to static attribute"). nnx.data([...]) / nnx.List([...])
        # would also work but only exist on recent versions; setattr works on
        # every version, so it is what we use.
        dims = [obs_dim] + [hidden_size] * n_hidden
        for i in range(n_hidden):
            setattr(self, f"trunk_{i}", orthogonal_linear(rngs, dims[i], dims[i + 1]))
        self.n_hidden = n_hidden  # plain int: static, as it should be
        self.trunk_dim = dims[-1]
        self.head = orthogonal_linear(rngs, self.trunk_dim, feature_dim)
        self.ln = nnx.LayerNorm(feature_dim, rngs=rngs) if use_layer_norm else None
        self.use_tanh = use_tanh
        self.feature_dim = feature_dim

    def trunk_features(self, obs):
        x = obs
        for i in range(self.n_hidden):
            x = nnx.relu(getattr(self, f"trunk_{i}")(x))
        return x

    def __call__(self, obs):
        z = self.head(self.trunk_features(obs))
        if self.ln is not None:
            z = self.ln(z)
        if self.use_tanh:
            z = jnp.tanh(z)
        return z


class ActorHead(nnx.Module):
    """[M7] The policy's own projection, read off a DETACHED trunk:

        actor_z = tanh(LayerNorm(Dense_{feature_dim}(stop_grad(h))))

    Separate parameters from the critic head, trained ONLY by the policy loss.
    The stop_gradient is what keeps the policy from shaping phi.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        trunk_dim: int,
        feature_dim: int,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
    ):
        self.head = orthogonal_linear(rngs, trunk_dim, feature_dim)
        self.ln = nnx.LayerNorm(feature_dim, rngs=rngs) if use_layer_norm else None
        self.use_tanh = use_tanh

    def __call__(self, h):
        z = self.head(jax.lax.stop_gradient(h))
        if self.ln is not None:
            z = self.ln(z)
        if self.use_tanh:
            z = jnp.tanh(z)
        return z


class SACPolicy(nnx.Module):
    """encoder + actor head + SACGaussianActor behind the interface the rest of
    the harness expects: (obs, key) -> (action, log_prob) for
    utils.acting.actor_step, plus .mean_action / .sample for evaluate().
    Holds references only; owns no parameters of its own."""

    def __init__(self, encoder: StateEncoder, actor_head: ActorHead, actor):
        self.encoder = encoder
        self.actor_head = actor_head
        self.actor = actor

    def z(self, obs):
        return self.actor_head(self.encoder.trunk_features(obs))

    def __call__(self, obs, key):
        return self.actor(self.z(obs), key)

    def mean_action(self, obs):
        return self.actor.mean_action(self.z(obs))

    def sample(self, obs, key=None):
        return self.actor.sample(self.z(obs), key)


def polyak_update(target_model, curr_model, tau: float):
    """Copied from sac_single.py."""
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target = jax.tree_util.tree_map(
        lambda t, c: (1.0 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target)
    return target_model


# --------------------------------------------------------------------------- #
# mico_train_step: sac_train_step with the encoder threaded through and the
# MICo loss summed into the critic backward. Compare side by side with your
# sac_train_step - the algebra of every SAC term is unchanged.
# --------------------------------------------------------------------------- #
def mico_train_step(
    encoder: StateEncoder,
    encoder_opt: nnx.Optimizer,
    target_encoder: StateEncoder,
    actor_head: ActorHead,
    actor_head_opt: nnx.Optimizer,
    actor: SACGaussianActor,
    actor_opt: nnx.Optimizer,
    critic: EnsembleCritic,
    critic_opt: nnx.Optimizer,
    target_critic: EnsembleCritic,
    log_alpha: Scalar,
    alpha_opt: nnx.Optimizer,
    data: Transition,
    config,
    key: jnp.ndarray,
):
    # [SHAPE] Collapse the replay buffer's spurious singleton axis.
    #
    # UniformSamplingQueue is built from a dummy_data_sample that carries a
    # LEADING DIM OF 1 (dummy_obs = jnp.zeros((1, obs_dim)), dummy_zero =
    # jnp.zeros((1,))). ravel_pytree therefore records that 1 as part of each
    # leaf's shape, and the vmapped unflatten on sample() reinstates it, so a
    # sampled batch comes back as
    #     observation (B, 1, obs_dim)   reward    (B, 1)
    #     action      (B, 1, act_dim)   discount  (B, 1)
    #     truncation  (B, 1)
    # rather than (B, obs_dim) / (B,).
    #
    # Every operation in your sac_train_step and dhpg_train_step is elementwise
    # or a full-array mean, so that extra axis is completely harmless there -
    # which is why neither script has ever complained. It is NOT harmless here.
    # metric_utils.squarify branches on x.ndim: a (B, 1) reward takes the 2-D
    # path, producing a (B, B, 1) grid whose transpose is (1, B, B), and
    # target_distances then tries to reshape it to shape[0]**2 = 1. That is the
    # "cannot reshape array of shape (1, 512, 512) into shape 1" you hit.
    #
    # Reshaping once here is a no-op when the batch is already canonical, so
    # this is safe whichever way your buffer is configured, and it changes no
    # number in the SAC terms (mean over B*1 == mean over B).
    batch_size = data.reward.shape[0]
    obs = jnp.reshape(data.observation, (batch_size, -1))
    act = jnp.reshape(data.action, (batch_size, -1))
    next_obs = jnp.reshape(data.next_observation, (batch_size, -1))
    reward = jnp.reshape(data.reward, (batch_size,))
    discount = jnp.reshape(data.discount, (batch_size,))
    truncation = jnp.reshape(
        data.extras["state_extras"]["truncation"], (batch_size,)
    )
    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
    alpha = jnp.exp(log_alpha())

    policy = SACPolicy(encoder, actor_head, actor)

    # [SYNC][S1] window-level truncation flag from nstep_aggregate.
    valid = 1.0 - truncation
    # [S2] the SAC target uses the scaled reward; the MICo target uses the raw
    # one, per metric_sac_agent.py.
    mico_reward = reward * config.reward_scaling if config.mico_scale_reward else reward
    # [M8] the MICo target's discount is a SCALAR gamma^n in the author's code.
    if config.mico_discount_from_sample:
        pair_gamma = jnp.sqrt(
            jnp.maximum(discount[:, None] * discount[None, :], 0.0)
        ).reshape(-1)
    else:
        pair_gamma = config.gamma**config.nstep

    # ---- alpha loss: identical to sac_train_step ------------------------
    def alpha_loss_fn(log_alpha):
        _, log_prob = policy(obs, key_alpha)
        a = jnp.exp(log_alpha())
        loss = jnp.mean(a * jax.lax.stop_gradient(-log_prob - config.target_entropy))
        return (1.0 - config.mico_weight) * loss, loss

    (_, alpha_loss), alpha_grads = nnx.value_and_grad(alpha_loss_fn, has_aux=True)(
        log_alpha
    )

    # ---- targets --------------------------------------------------------
    # Your sac_train_step computes these inside critic_loss_fn and relies on
    # differentiating w.r.t. `critic` only. Here the backward also covers
    # `encoder`, so they are hoisted out and stop-gradiented explicitly. Same
    # numbers, no leakage.
    next_act, next_log_prob = policy(next_obs, key_critic)
    next_act = jax.lax.stop_gradient(next_act)
    next_log_prob = jax.lax.stop_gradient(next_log_prob)
    # [M3] the target encoder on the next states serves BOTH the critic target
    # and the MICo target, so it is computed once.
    target_next_r = jax.lax.stop_gradient(target_encoder(next_obs))
    q1_t, q2_t = target_critic(jnp.concatenate([target_next_r, next_act], axis=-1))
    next_v = jnp.minimum(q1_t, q2_t) - alpha * next_log_prob
    # [M8] `discount` comes from nstep_aggregate and ALREADY CONTAINS gamma^n,
    # so no extra * gamma - exactly as in your n-step sac_train_step.
    target_q = jax.lax.stop_gradient(reward * config.reward_scaling + discount * next_v)

    # ---- critic + MICo ---------------------------------------------------
    # [M5](e) The two encoder gradients are computed SEPARATELY and summed
    # with weights (1 - w) and w. That is algebraically identical to one
    # backward over (1 - w)*critic_loss + w*mico_loss - the MICo term touches
    # no critic parameter and the critic term touches no pairwise quantity -
    # but it lets us measure the two contributions independently, which is the
    # only direct way to answer "is this w doing anything at all?". Cost is
    # one extra encoder forward/backward; the expensive B^2 distance
    # computation is not duplicated.
    def critic_loss_fn(critic, encoder):
        z = encoder(obs)
        q1, q2 = critic(jnp.concatenate([z, act], axis=-1))
        q_error = jnp.stack([q1, q2], axis=-1) - target_q[..., None]
        q_error = q_error * valid[..., None]
        critic_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return critic_loss, (jnp.mean(q1), jnp.mean(q2))

    (
        (critic_loss, (q1_mean, q2_mean)),
        (critic_grads, encoder_grads_critic),
    ) = nnx.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
        critic, encoder
    )

    def mico_loss_fn(encoder):
        z = encoder(obs)
        # [M3] second argument: the SAME encoder on the SAME states, frozen.
        frozen_r = jax.lax.stop_gradient(z)
        online_dist, norm_term, angle_term = representation_distances(
            z,
            frozen_r,
            cosine_distance,
            beta=config.mico_beta,
            return_distance_components=True,
        )
        # [M2] target over all B^2 ordered pairs
        target_dist = target_distances(
            target_next_r, mico_reward, cosine_distance, pair_gamma
        )
        # [M4] Huber elementwise, [S1] pairwise truncation mask, then mean.
        mico_loss = jnp.mean(
            pair_mask(valid) * huber_loss(online_dist, target_dist, config.huber_delta)
        )
        return mico_loss, (
            jnp.mean(online_dist),
            jnp.mean(target_dist),
            jnp.mean(norm_term),
            jnp.mean(config.mico_beta * angle_term),
            jnp.mean(jnp.sum(z**2, axis=-1)),
        )

    (
        (
            mico_loss,
            (
                online_dist_mean,
                target_dist_mean,
                norm_term_mean,
                angle_term_mean,
                rep_sq_norm_mean,
            ),
        ),
        encoder_grads_mico,
    ) = nnx.value_and_grad(mico_loss_fn, has_aux=True)(encoder)

    def _tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))

    w = config.mico_weight
    critic_grads = jax.tree_util.tree_map(lambda g: (1.0 - w) * g, critic_grads)
    encoder_grads = jax.tree_util.tree_map(
        lambda gc, gm: (1.0 - w) * gc + w * gm,
        encoder_grads_critic,
        encoder_grads_mico,
    )

    # [M5](c) The diagnostic that decides whether this run is MICo or is the
    # mico_weight=0 control under another name. ~1e-4 means decorative.
    mico_grad_ratio = (w * _tree_norm(encoder_grads_mico)) / (
        (1.0 - w) * _tree_norm(encoder_grads_critic) + 1e-12
    )

    # ---- actor loss: identical to sac_train_step, plus the actor head ----
    # Gradient w.r.t. (actor, actor_head) only, so the critic is untouched -
    # the author gets the same effect with `frozen_params`. phi is
    # stop-gradiented on both paths: through ActorHead's stop_grad on the
    # trunk, and explicitly on z_c below.
    def actor_loss_fn(actor, actor_head):
        z_a = actor_head(encoder.trunk_features(obs))
        pi, log_pi = actor(z_a, key_actor)
        z_c = jax.lax.stop_gradient(encoder(obs))
        q1, q2 = critic(jnp.concatenate([z_c, pi], axis=-1))
        loss = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
        return (1.0 - config.mico_weight) * loss, (loss, jnp.mean(log_pi))

    (
        (_, (actor_loss, log_pi_mean)),
        (actor_grads, actor_head_grads),
    ) = nnx.value_and_grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(
        actor, actor_head
    )

    # [SYNC][S3] all grads computed against pre-update params, then step.
    alpha_opt.update(log_alpha, alpha_grads)
    critic_opt.update(critic, critic_grads)
    encoder_opt.update(encoder, encoder_grads)
    actor_opt.update(actor, actor_grads)
    actor_head_opt.update(actor_head, actor_head_grads)

    polyak_update(target_critic, critic, config.update_tau)
    polyak_update(target_encoder, encoder, config.update_tau)
    alpha_post = jnp.exp(log_alpha())

    return (
        critic_loss,
        actor_loss,
        alpha_loss,
        alpha_post,
        log_pi_mean,
        q1_mean,
        q2_mean,
        mico_loss,
        online_dist_mean,
        target_dist_mean,
        norm_term_mean,
        angle_term_mean,
        rep_sq_norm_mean,
        mico_grad_ratio,
    )


N_METRICS = 14


@functools.partial(nnx.jit, static_argnames=("env", "buffer"))
def train_n_steps(
    env,
    env_state,
    buffer_state,
    buffer,
    running_state,
    obs_normalizer,
    encoder,
    encoder_opt,
    target_encoder,
    actor_head,
    actor_head_opt,
    actor,
    actor_opt,
    critic,
    critic_opt,
    target_critic,
    log_alpha,
    alpha_opt,
    config,
    nstep_fifo,
    nstep_count,
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
            nstep_fifo,
            nstep_count,
            models,
            val,
        ) = carry
        (
            encoder,
            encoder_opt,
            target_encoder,
            actor_head,
            actor_head_opt,
            actor,
            actor_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
        ) = models

        key, env_key = jax.random.split(key)
        n_env_state, transition = actor_step(
            env,
            env_state,
            SACPolicy(encoder, actor_head, actor),
            obs_normalizer,
            env_key,
            extra_fields=("truncation",),
        )
        # [M8] the window is built on the ROLLOUT side: push the raw 1-step
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
        if config.normalize_obs:
            obs_normalizer = obs_normalizer.update(transition.observation)
        running_state = RunningStatistics.insert_reward(
            running_state, n_env_state.reward
        )

        def do_train(j, carry):
            key, env_state, buffer_state, obs_normalizer, models, _ = carry
            (
                encoder,
                encoder_opt,
                target_encoder,
                actor_head,
                actor_head_opt,
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            ) = models

            buffer_state, batch = buffer.sample(buffer_state)
            if config.normalize_obs:  # normalize at point of use, not storage
                batch = batch._replace(
                    observation=obs_normalizer.normalize(batch.observation),
                    next_observation=obs_normalizer.normalize(batch.next_observation),
                )
            key, train_key = jax.random.split(key)

            val = mico_train_step(
                encoder,
                encoder_opt,
                target_encoder,
                actor_head,
                actor_head_opt,
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
                encoder,
                encoder_opt,
                target_encoder,
                actor_head,
                actor_head_opt,
                actor,
                actor_opt,
                critic,
                critic_opt,
                target_critic,
                log_alpha,
                alpha_opt,
            )
            return (key, env_state, buffer_state, obs_normalizer, models, val)

        init_val = (jnp.zeros((), jnp.float32),) * N_METRICS
        models = (
            encoder,
            encoder_opt,
            target_encoder,
            actor_head,
            actor_head_opt,
            actor,
            actor_opt,
            critic,
            critic_opt,
            target_critic,
            log_alpha,
            alpha_opt,
        )
        key, _, buffer_state, obs_normalizer, models, val = nnx.fori_loop(
            0,
            config.train_per_step,
            do_train,
            (key, n_env_state, buffer_state, obs_normalizer, models, init_val),
        )
        return (
            key,
            n_env_state,
            buffer_state,
            running_state,
            obs_normalizer,
            nstep_fifo,
            nstep_count,
            models,
            val,
        )

    init_val = (jnp.zeros((), jnp.float32),) * N_METRICS
    init_carry = (
        key,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        nstep_fifo,
        nstep_count,
        (
            encoder,
            encoder_opt,
            target_encoder,
            actor_head,
            actor_head_opt,
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

    (
        _,
        env_state,
        buffer_state,
        running_state,
        obs_normalizer,
        nstep_fifo,
        nstep_count,
        models,
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


# --------------------------------------------------------------------------- #
# [SYNC][S7] evaluate() copied VERBATIM from sac_single.py.
# --------------------------------------------------------------------------- #
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
    """Copied from sac_single.py. [S6] `policy` is the (untrained) SACPolicy,
    not a uniform-random policy, exactly as in your file."""

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
                fifo, count, config.gamma, config.nstep, config.bootstrap_on_truncation
            ),
        )
        if config.normalize_obs:
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


# ===========================================================================
# MAIN
# ===========================================================================
def main(args, cfg_env=None):
    # ── reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    prng_key = jax.random.PRNGKey(args.seed)

    rngs = nnx.Rngs(default=args.seed, params=args.seed + 3, dropout=args.seed + 5)

    # ── device ────────────────────────────────────────────────────────────
    jax.default_device = jax.devices(args.device)[args.device_id]

    # ── build config: UNCONDITIONAL, identical to sac_single.py ───────────
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
            "num_envs": args.num_envs,
            "num_eval_envs": args.num_eval_envs,
            "reward_scaling": args.reward_scaling,
        }
    )

    # [M8] per-task override, copied from sac_single.py / dhpg.py so all three
    # baselines use the SAME return length on every task. Must run BEFORE the
    # config is frozen.
    if args.task.lower().startswith("walker"):
        config["nstep"] = 1

    # ── [BOUNDARY] the only place harness and algorithm touch ─────────────
    # mico.gin agrees with sac_args on tau (5e-3), gamma (0.99) and batch
    # (256), and brackets your lr (mico.gin 1e-3, mico_saclr.gin 3e-4), so
    # unlike DHPG there is nothing to reconcile. hidden_units (author 1024)
    # and reward_scale_factor (author 0.1) differ but are regime, so they stay
    # on the harness. Flip the first here to check it; for the second pass
    # --reward_scaling 0.1.
    if config["author_hidden_units"]:
        config["hidden_size"] = 1024

    # ── [SWEEP] MICo hyperparameters, overridable WITHOUT sac_args ────────
    # The algorithm keys are deliberately kept off the shared CLI so that a
    # SAC-shaped flag can never silently change them (dhpg.py's rule). But w
    # has to be sweepable - see [M5](b): the author retunes it whenever the TD
    # loss scale changes, and ours is not his. The environment is the narrowest
    # channel that does not touch the shared parser:
    #     MICO_WEIGHT=0.01 MICO_BETA=0.1 python -m mico --task CheetahRun ...
    # Resolved values are echoed in the [algorithm] line and saved by
    # logger.save_config, so a run is always self-describing.
    for _key, _cast in (
        ("mico_weight", float),
        ("mico_beta", float),
        ("feature_dim", int),
        ("nstep", int),
    ):
        _env = os.environ.get(_key.upper())
        if _env is not None:
            config[_key] = _cast(_env)

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
    obs_normalizer = (
        RunningMeanStd.init((obs_dim,))
        if config["normalize_obs"]
        else IdentityNormalizer()
    )

    # [S5] identical to sac_single.py
    config["target_entropy"] = float(act_dim) * -0.5

    config_data = make_static_config_from_dict("MICoConfig", config)()

    # ── networks ──────────────────────────────────────────────────────────
    # [M6] the actor and critic are YOUR classes, unchanged; only their input
    # width changes, because both now sit downstream of the encoder.
    feature_dim = config["feature_dim"]

    encoder = StateEncoder(
        rngs=rngs,
        obs_dim=obs_dim,
        hidden_size=config["hidden_size"],
        feature_dim=feature_dim,
        n_hidden=config["encoder_hidden_layers"],
        use_layer_norm=config["encoder_layer_norm"],
        use_tanh=config["encoder_tanh"],
    )
    encoder_opt = nnx.Optimizer(
        model=encoder, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )
    target_encoder = deepcopy(encoder)

    actor_head = ActorHead(
        rngs=rngs,
        trunk_dim=encoder.trunk_dim,
        feature_dim=feature_dim,
        use_layer_norm=config["encoder_layer_norm"],
        use_tanh=config["encoder_tanh"],
    )
    actor_head_opt = nnx.Optimizer(
        model=actor_head, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    actor = SACGaussianActor(
        rngs=rngs,
        obs_dim=feature_dim,  # consumes actor_z, not obs
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )
    actor_opt = nnx.Optimizer(
        model=actor, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    critic = EnsembleCritic(
        rngs=rngs,
        obs_dim=feature_dim,  # consumes concat([phi(x), a])
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )
    critic_opt = nnx.Optimizer(
        model=critic, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )
    target_critic = deepcopy(critic)

    log_alpha = Scalar(float(jnp.log(config["init_temperature"])))
    # [S4] hardcoded 3e-4, as in sac_single.py - NOT config["lr"].
    alpha_opt = nnx.Optimizer(
        model=log_alpha, tx=optax.adam(learning_rate=config["lr"]), wrt=nnx.Param
    )

    policy = SACPolicy(encoder, actor_head, actor)

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

    # ── [M8] n-step FIFO state, identical construction to sac_single.py ───
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
        policy=policy,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        config=config_data,
        num_itr=warmup_iters,
        nstep_fifo=nstep_fifo,
        nstep_count=nstep_count,
    )

    # ── main training loop ───────────────────────────────────────────────
    logger.log("Start MICo training")
    logger.log(f"{config}")
    # [LOGSYNC] Same two lines dhpg.py prints, same field order and format, so
    # a MICo run log and a DHPG run log can be diffed head-to-head to confirm
    # the regimes actually matched before comparing any curve.
    logger.log(
        f"[harness] num_envs={config['num_envs']} "
        f"total_env_steps={config['total_env_steps']} "
        f"batch={config['batch_size']} train_per_step={config['train_per_step']} "
        f"UTD={config['train_per_step'] / config['num_envs']:.4f} "
        f"seed={warmup_iters * config['num_envs']} transitions "
        f"log_every={config['log_freq'] * config['num_envs']} transitions "
        f"normalize_obs={config['normalize_obs']}"
    )
    logger.log(
        f"[algorithm] lr={config['lr']} tau={config['update_tau']} "
        f"nstep={config['nstep']} "
        f"cumulative_gamma={config['gamma'] ** config['nstep']:.6f} "
        f"bootstrap_on_truncation={config['bootstrap_on_truncation']} "
        f"mico_weight={config['mico_weight']} beta={config['mico_beta']} "
        f"feature_dim={feature_dim} "
        f"encoder=(hidden_layers={config['encoder_hidden_layers']},"
        f"ln={config['encoder_layer_norm']},tanh={config['encoder_tanh']}) "
        f"scale_reward={config['mico_scale_reward']} "
        f"discount_from_sample={config['mico_discount_from_sample']} "
        f"alpha_lr=3e-4 init_temperature={config['init_temperature']} "
        f"target_entropy={config['target_entropy']}"
    )
    if config["mico_weight"] == 0.0:
        logger.log("[mico] weight=0 -> this run is the ARCHITECTURE CONTROL, not MICo")

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
            encoder=encoder,
            encoder_opt=encoder_opt,
            target_encoder=target_encoder,
            actor_head=actor_head,
            actor_head_opt=actor_head_opt,
            actor=actor,
            actor_opt=actor_opt,
            critic=critic,
            critic_opt=critic_opt,
            target_critic=target_critic,
            log_alpha=log_alpha,
            alpha_opt=alpha_opt,
            config=config_data,
            nstep_fifo=nstep_fifo,
            nstep_count=nstep_count,
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
            mico_loss,
            online_dist_mean,
            target_dist_mean,
            norm_term_mean,
            angle_term_mean,
            rep_sq_norm_mean,
            mico_grad_ratio,
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

        # ── logging: sac_single.py's keys, plus Loss_mico and the MICo block
        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_critic", critic_loss.item())
        logger.log_tabular("Loss/Loss_actor", actor_loss.item())
        logger.log_tabular("Loss/Loss_alpha", alpha_loss.item())
        logger.log_tabular("Loss/Loss_mico", mico_loss.item())

        logger.log_tabular("SAC/Alpha", alpha.item())
        logger.log_tabular("SAC/LogPi_mean", log_pi_mean.item())
        logger.log_tabular("SAC/Q1_mean", q1_mean.item())
        logger.log_tabular("SAC/Q2_mean", q2_mean.item())

        # The two diagnostics that matter:
        #  - Rep_sq_norm should sit well below feature_dim (50). If it pins at
        #    50 the tanh has saturated and the metric has collapsed.
        #  - Online_dist should track Target_dist. A persistent gap means
        #    mico_weight is too small for the metric to be learnable at all,
        #    which is the thing to rule out before concluding "MICo does not
        #    help on this task".
        logger.log_tabular("MICo/Online_dist", online_dist_mean.item())
        logger.log_tabular("MICo/Target_dist", target_dist_mean.item())
        logger.log_tabular("MICo/Norm_term", norm_term_mean.item())
        logger.log_tabular("MICo/Angle_term", angle_term_mean.item())
        logger.log_tabular("MICo/Rep_sq_norm", rep_sq_norm_mean.item())
        # [M5](c) fraction of the encoder's update direction owed to MICo.
        # ~1e-4 => this run is the mico_weight=0 control under another name.
        logger.log_tabular("MICo/Grad_ratio", mico_grad_ratio.item())

        logger.log_tabular(
            "Norm/actor_model", get_tree_norm(nnx.state(actor, nnx.Param))
        )
        logger.log_tabular(
            "Norm/critic_model", get_tree_norm(nnx.state(critic, nnx.Param))
        )
        # [LOGSYNC] dhpg.py calls its phi "Norm/state_encoder_model"; use the
        # same key so the two encoder-norm curves overlay without remapping.
        logger.log_tabular(
            "Norm/state_encoder_model", get_tree_norm(nnx.state(encoder, nnx.Param))
        )
        logger.log_tabular(
            "Norm/actor_head_model", get_tree_norm(nnx.state(actor_head, nnx.Param))
        )

        prng_key, eval_key = jax.random.split(prng_key)
        eval_return, eval_std = evaluate(
            env=env,
            actor=SACPolicy(encoder, actor_head, actor),
            obs_normalizer=obs_normalizer,
            key=eval_key,
            episode_length=config["episode_length"],
            num_eval_envs=config["num_eval_envs"],
            deterministic=True,
        )

        logger.log_tabular("Eval/Return", float(eval_return))

        logger.dump_tabular()

        # ── periodic checkpoint, sac_single.py's monotone form ────────────
        if steps >= next_save:
            logger.nn_model_save(itr=steps, nn_model_saver_element=actor, prefix="actor")
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=critic, prefix="critic"
            )
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=encoder, prefix="encoder"
            )
            logger.nn_model_save(
                itr=steps, nn_model_saver_element=actor_head, prefix="actor_head"
            )
            while next_save <= steps:
                next_save += config["save_freq"]

    # ── final save ────────────────────────────────────────────────────────
    logger.nn_model_save(itr=steps, nn_model_saver_element=actor, prefix="actor")
    logger.nn_model_save(itr=steps, nn_model_saver_element=critic, prefix="critic")
    logger.nn_model_save(itr=steps, nn_model_saver_element=encoder, prefix="encoder")
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=actor_head, prefix="actor_head"
    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = sac_args()

    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "seed-" + str(args.seed).zfill(3)
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]  # "mico"
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