import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

EPS = 1e-6
MIN_STD = 1e-3
KERNEL_INIT = nnx.initializers.lecun_uniform()

def get_tree_norm(tree):
    """Computes the L2 norm across all the leaves of a JAX Pytree"""
    squares = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree)
    total = jax.tree_util.tree_reduce(lambda a, x: a + x, squares)
    return jnp.sqrt(total)


class Scalar(nnx.Module):
    def __init__(self, val: float):
        self.val = nnx.Param(jnp.array(val, dtype=jnp.float32))

    def __call__(self) -> jnp.ndarray:
        return self.val

class SACCritic(nnx.Module):
    """One Q head: obs+act -> 256 -> 256 -> 1.
 
    brax/training/networks.py :: make_q_network builds
        MLP(layer_sizes=list(hidden_layer_sizes) + [1],
            activation=relu,
            kernel_init=lecun_uniform(),
            layer_norm=layer_norm)
 
    and brax's MLP forward loop is
 
        hidden = Dense(size)(hidden)
        if i != last or activate_final:
            hidden = activation(hidden)       # <-- activation FIRST
            if layer_norm:
                hidden = LayerNorm()(hidden)  # <-- then LayerNorm
 
    So the ordering is Dense -> ReLU -> LayerNorm (POST-activation), and the
    final layer gets neither. Your original had Linear -> LayerNorm -> ReLU.
    """
 
    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        layer_norm: bool = True,  # playground: q_network_layer_norm=True
    ):
        self.layer_norm = layer_norm
 
        self.dense_0 = nnx.Linear(
            obs_dim + act_dim, hidden_size, kernel_init=KERNEL_INIT, rngs=rngs
        )
        self.dense_1 = nnx.Linear(
            hidden_size, hidden_size, kernel_init=KERNEL_INIT, rngs=rngs
        )
        self.dense_2 = nnx.Linear(
            hidden_size, 1, kernel_init=KERNEL_INIT, rngs=rngs
        )
 
        if layer_norm:
            self.norm_0 = nnx.LayerNorm(hidden_size, rngs=rngs)
            self.norm_1 = nnx.LayerNorm(hidden_size, rngs=rngs)
 
    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        x = self.dense_0(obs_act)
        x = nnx.relu(x)
        if self.layer_norm:
            x = self.norm_0(x)
 
        x = self.dense_1(x)
        x = nnx.relu(x)
        if self.layer_norm:
            x = self.norm_1(x)
 
        x = self.dense_2(x)          # no activation, no LayerNorm on the output
        return jnp.squeeze(x, axis=-1)
 
 
class EnsembleCritic(nnx.Module):
    """n_critics = 2 (brax make_q_network default).
 
    brax runs both heads inside one QModule and concatenates to (batch, 2);
    returning a tuple is mathematically identical.
    """
 
    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        layer_norm: bool = True,
    ):
        self.q1 = SACCritic(rngs, obs_dim, act_dim, hidden_size, layer_norm)
        self.q2 = SACCritic(rngs, obs_dim, act_dim, hidden_size, layer_norm)
 
    def __call__(self, obs_act: jnp.ndarray):
        return self.q1(obs_act), self.q2(obs_act)
 
 
# ===========================================================================
# Actor
# ===========================================================================
 
class SACGaussianActor(nnx.Module):
    """Tanh-Gaussian policy: obs -> 256 -> 256 -> 2*act_dim.
 
    brax/training/agents/sac/networks.py :: make_sac_networks
        parametric_action_distribution = NormalTanhDistribution(event_size=action_size)
        policy_network = make_policy_network(
            parametric_action_distribution.param_size,   # == 2 * action_size
            hidden_layer_sizes=(256, 256),
            activation=relu,
            layer_norm=policy_network_layer_norm)        # False by default
 
    => a SINGLE output layer of width 2*act_dim, split into (loc, raw_scale).
       Your original used two separate heads off a shared trunk.
 
    brax/training/distribution.py :: NormalTanhDistribution.create_dist
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = (softplus(scale) + min_std) * var_scale   # min_std=1e-3, var_scale=1
 
    => replaces exp(clip(log_std, -20, 2)). At init raw_scale ~= 0 gives
       sigma = softplus(0) + 1e-3 = 0.694, not 1.0.
    """
 
    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        layer_norm: bool = False,  # brax: policy_network_layer_norm=False
        min_std: float = MIN_STD,
        var_scale: float = 1.0,
    ):
        self.act_dim = act_dim
        self.min_std = min_std
        self.var_scale = var_scale
        self.layer_norm = layer_norm
 
        self.dense_0 = nnx.Linear(
            obs_dim, hidden_size, kernel_init=KERNEL_INIT, rngs=rngs
        )
        self.dense_1 = nnx.Linear(
            hidden_size, hidden_size, kernel_init=KERNEL_INIT, rngs=rngs
        )
        self.dense_2 = nnx.Linear(
            hidden_size, 2 * act_dim, kernel_init=KERNEL_INIT, rngs=rngs
        )
 
        if layer_norm:
            self.norm_0 = nnx.LayerNorm(hidden_size, rngs=rngs)
            self.norm_1 = nnx.LayerNorm(hidden_size, rngs=rngs)
 
    def _get_dist_params(self, obs: jnp.ndarray):
        x = self.dense_0(obs)
        x = nnx.relu(x)
        if self.layer_norm:
            x = self.norm_0(x)
 
        x = self.dense_1(x)
        x = nnx.relu(x)
        if self.layer_norm:
            x = self.norm_1(x)
 
        params = self.dense_2(x)     # (..., 2 * act_dim)
 
        loc, raw_scale = jnp.split(params, 2, axis=-1)
        scale = (jax.nn.softplus(raw_scale) + self.min_std) * self.var_scale
        return loc, scale
 
    def _log_prob(self, x_t, loc, scale) -> jnp.ndarray:
        """brax _NormalDistribution.log_prob + ParametricDistribution.log_prob.
 
            log_unnormalized  = -0.5 * (x/scale - loc/scale)**2
            log_normalization = 0.5*log(2pi) + log(scale)
            lp  = log_unnormalized - log_normalization
            lp -= TanhBijector.forward_log_det_jacobian(x)
            lp  = sum(lp, axis=-1)
        """
        log_unnormalized = -0.5 * jnp.square(x_t / scale - loc / scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        log_prob = log_unnormalized - log_normalization
        # TanhBijector.forward_log_det_jacobian
        log_prob -= 2.0 * (jnp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t))
        return jnp.sum(log_prob, axis=-1)
 
    def mean_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        """ParametricDistribution.mode() == postprocess(dist.mode()) == tanh(loc).
 
        NOTE: brax's SAC defaults to deterministic_eval=False and
        mujoco_playground never overrides it, so THE PAPER'S CURVES DO NOT USE
        THIS. Keep it for deployment / videos, not for the reported metric.
        """
        loc, _ = self._get_dist_params(obs)
        return jnp.tanh(loc)
 
    def sample(self, obs: jnp.ndarray, key: jnp.ndarray):
        loc, scale = self._get_dist_params(obs)
        eps = jax.random.normal(key, shape=loc.shape)
        x_t = loc + eps * scale
        action = jnp.tanh(x_t)
        return action, self._log_prob(x_t, loc, scale)
 
    def __call__(self, obs: jnp.ndarray, key: jnp.ndarray):
        return self.sample(obs, key)


#################################
## Representation Modules #######
#################################


class RepNet(nnx.Module):
    def __init__(
        self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, rep_dim: int, hidden_dim: int
    ):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        zero_init = nnx.initializers.zeros

        self.trunk = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            nnx.relu,
        )

        self.state_head = nnx.Linear(hidden_dim, rep_dim, rngs=rngs)

        self.state_action_head = nnx.Linear(hidden_dim + act_dim, rep_dim, rngs=rngs)

        self.state_rep_hidden_head = nnx.Sequential(
            nnx.Linear(rep_dim, hidden_dim, rngs=rngs),
            nnx.elu,
        )

    def state_rep(self, obs: jnp.ndarray) -> jnp.ndarray:

        h = self.trunk(obs)
        z = self.state_head(h)

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def state_action_rep(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:

        h = self.trunk(obs)

        z = self.state_action_head(jnp.concatenate([h, act], axis=-1))

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def state_rep_action_rep(self, state_rep: jnp.ndarray, act: jnp.array) -> jnp.array:
        h = self.state_rep_hidden_head(state_rep)
        z = self.state_action_head(jnp.concatenate([h, act], axis=-1))
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


class SACCriticRep(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, rep_dim: int, hidden_size: int = 256):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(rep_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(self.model(obs_act), axis=-1)


class EnsembleCriticRep(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, rep_dim: int, hidden_size: int = 256):
        self.q1 = SACCriticRep(rngs, rep_dim, hidden_size)
        self.q2 = SACCriticRep(rngs, rep_dim, hidden_size)

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        return self.q1(obs_act), self.q2(obs_act)


class SACGaussianActorRep(nnx.Module):
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def __init__(
        self, rngs: nnx.Rngs, rep_dim: int, act_dim: int, hidden_size: int = 256
    ):
        self.act_dim = act_dim
        self.trunk = nnx.Sequential(
            nnx.Linear(rep_dim, hidden_size, rngs=rngs),
            # nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.relu,
        )
        self.mean_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)

    def _get_dist_params(self, obs: jnp.ndarray):
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = jnp.clip(
            self.log_std_head(features), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return mean, log_std

    def _log_prob(
        self, x_t: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
    ) -> jnp.ndarray:
        std = jnp.exp(log_std)
        log_prob = -0.5 * (
            ((x_t - mean) / (std + EPS)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)

        log_prob -= jnp.sum(
            2.0 * (jnp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t)),
            axis=-1,
        )
        return log_prob

    def sample(self, obs: jnp.ndarray, key: jnp.ndarray):

        mean, log_std = self._get_dist_params(obs)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, shape=mean.shape)
        x_t = mean + eps * std
        action = jnp.tanh(x_t)
        return action, self._log_prob(x_t, mean, log_std)

    def __call__(self, obs: jnp.ndarray, key: jnp.ndarray):
        return self.sample(obs, key)
