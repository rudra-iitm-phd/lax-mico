import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

EPS = 1e-6


# Heterogenous state and action projections : The idea is to respect the information associated with the raw state and action information
# and apply appropriate compression to the state and action spaces


class RepNet(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        state_rep_dim: int,
        act_rep_dim: int,
        hidden_dim: int,
    ):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        zero_init = nnx.initializers.zeros

        self.trunk = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.elu,
        )

        self.state_head = nnx.Linear(hidden_dim, state_rep_dim, rngs=rngs)

        self.state_action_head = nnx.Linear(
            hidden_dim + act_dim, state_rep_dim + act_rep_dim, rngs=rngs
        )

    def state_rep(self, obs: jnp.ndarray) -> jnp.ndarray:

        # Map from obs_dim --> state_rep dim

        h = self.trunk(obs)
        z = self.state_head(h)

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def state_action_rep(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:

        # Map from obs_dim + act_dim --> (state_rep_dim) + (act_rep_dim)

        h = self.trunk(obs)

        z = self.state_action_head(jnp.concatenate([h, act], axis=-1))

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


class SACCriticRep(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        state_rep_dim: int,
        act_rep_dim: int,
        hidden_size: int = 256,
    ):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(state_rep_dim + act_rep_dim, hidden_size, rngs=rngs),
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
    def __init__(
        self,
        rngs: nnx.Rngs,
        state_rep_dim: int,
        act_rep_dim: int,
        hidden_size: int = 256,
    ):
        self.q1 = SACCriticRep(rngs, state_rep_dim, act_rep_dim, hidden_size)
        self.q2 = SACCriticRep(rngs, state_rep_dim, act_rep_dim, hidden_size)

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        return self.q1(obs_act), self.q2(obs_act)


class SACGaussianActorRep(nnx.Module):
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def __init__(
        self, rngs: nnx.Rngs, state_rep_dim: int, act_dim: int, hidden_size: int = 256
    ):
        self.act_dim = act_dim
        self.trunk = nnx.Sequential(
            nnx.Linear(state_rep_dim, hidden_size, rngs=rngs),
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
