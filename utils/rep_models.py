import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


#### copied from gen_pe
class StateAsymmetricMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = jnp.e

    def __init__(self, rngs: nnx.Rngs, rep_dim: int, hidden_size: int):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(rep_dim + rep_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, state1_rep: jnp.ndarray, state2_rep: jnp.ndarray) -> jnp.ndarray:
        log_metric = jnp.squeeze(
            self.model(jnp.concatenate([state1_rep, state2_rep], axis=-1)), axis=-1
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)


class EnsembleStateMetric(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, rep_dim: int, hidden_size: int = 256):
        self.g = StateAsymmetricMetric(rngs, rep_dim, hidden_size)

    def __call__(self, state1_rep: jnp.ndarray, state2_rep: jnp.ndarray) -> jnp.ndarray:
        return self.g(state1_rep, state2_rep), self.g(state2_rep, state1_rep)


class StateActionDiffuseMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = jnp.e

    def __init__(self, rngs: nnx.Rngs, rep_dim: int, hidden_size: int):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(rep_dim + rep_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(
        self, state_action_1_rep: jnp.ndarray, state_action_2_rep: jnp.ndarray
    ) -> jnp.ndarray:

        log_metric = jnp.squeeze(
            self.model(
                jnp.concatenate([state_action_1_rep, state_action_2_rep], axis=-1)
            ),
            axis=-1,
        )

        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)

        return jnp.exp(log_metric)


class EnsembleStateActionMetric(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, rep_dim: int, hidden_size: int):
        self.d = StateActionDiffuseMetric(rngs, rep_dim, hidden_size)

    def __call__(
        self, state_action_1_rep: jnp.ndarray, state_action_2_rep: jnp.ndarray
    ) -> jnp.ndarray:
        return self.d(state_action_1_rep, state_action_2_rep), self.d(
            state_action_2_rep, state_action_1_rep
        )


class MinStateActiontoStateMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = jnp.e

    def __init__(
        self,
        rngs: nnx.Rngs,
        state_action_rep_dim: int,
        state_rep_dim: int,
        hidden_size: int,
    ):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(state_action_rep_dim + state_rep_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(
        self, state_action_rep: jnp.ndarray, state_rep: jnp.ndarray
    ) -> jnp.ndarray:
        log_metric = jnp.squeeze(
            self.model(jnp.concatenate([state_action_rep, state_rep], axis=-1)), axis=-1
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)


class RepNetUnified(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        source_obs_dim: int,
        source_act_dim: int,
        target_obs_dim: int,
        target_act_dim: int,
        rep_dim: int,
        hidden_dim: int,
    ):

        self.source_obs_dim = source_obs_dim
        self.source_act_dim = source_act_dim

        self.target_obs_dim = target_obs_dim
        self.target_act_dim = target_act_dim

        self.hidden_dim = hidden_dim

        zero_init = nnx.initializers.zeros

        self.source_proj = nnx.Linear(source_obs_dim, hidden_dim, rngs=rngs)
        self.target_proj = nnx.Linear(target_obs_dim, hidden_dim, rngs=rngs)

        self.trunk = nnx.Sequential(
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.elu,
        )

        self.state_head = nnx.Linear(hidden_dim, rep_dim, rngs=rngs)

        self.state_action_head_source = nnx.Linear(
            hidden_dim + source_act_dim, rep_dim, rngs=rngs
        )

        self.state_action_head_target = nnx.Linear(
            hidden_dim + target_act_dim, rep_dim, rngs=rngs
        )

        self.rep_source_proj = nnx.Linear(rep_dim, hidden_dim, rngs=rngs)

    def source_state_rep(self, obs: jnp.ndarray) -> jnp.ndarray:

        h = self.source_proj(obs)
        h = self.trunk(h)
        z = self.state_head(h)

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def source_state_action_rep(
        self, obs: jnp.ndarray, act: jnp.ndarray
    ) -> jnp.ndarray:

        h = self.trunk(self.source_proj(obs))
        z = self.state_action_head_source(jnp.concatenate([h, act], axis=-1))

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def target_state_rep(self, obs: jnp.ndarray) -> jnp.ndarray:

        h = self.trunk(self.target_proj(obs))
        z = self.state_head(h)

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def target_state_action_rep(
        self, obs: jnp.ndarray, act: jnp.ndarray
    ) -> jnp.ndarray:

        h = self.trunk(self.target_proj(obs))

        z = self.state_action_head_target(jnp.concatenate([h, act], axis=-1))

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def target_state_source_action_rep(
        self, target_obs: jnp.ndarray, source_act: jnp.ndarray
    ) -> jnp.ndarray:
        h = self.target_trunk(target_obs)
        z = self.state_action_head_source(jnp.concatenate([h, source_act], axis=-1))
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def rep_source_action_rep(
        self, source_rep: jnp.ndarray, source_act: jnp.ndarray
    ) -> jnp.ndarray:

        h = self.trunk(self.rep_source_proj(source_rep))

        z = self.state_action_head_source(jnp.concatenate([h, source_act], axis=-1))

        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


class SACGaussianActorRep(nnx.Module):
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0
    EPS = 1e-6

    def __init__(
        self,
        rngs: nnx.Rngs,
        rep_dim: int,
        act_dim: int,
        target_act_dim: int,
        hidden_size: int = 256,
    ):
        self.act_dim = act_dim
        self.trunk = nnx.Sequential(
            nnx.Linear(rep_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
        )
        self.mean_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)

        self.target_mean_head = nnx.Linear(hidden_size, target_act_dim, rngs=rngs)
        self.target_log_std_head = nnx.Linear(hidden_size, target_act_dim, rngs=rngs)

    def _get_dist_params(self, obs: jnp.ndarray):
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = jnp.clip(
            self.log_std_head(features), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return mean, log_std

    def _get_dist_params_target(self, target_obs: jnp.ndarray):
        features = jax.lax.stop_gradient((self.trunk(target_obs)))
        mean = self.target_mean_head(features)
        log_std = jnp.clip(
            self.target_log_std_head(features), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return mean, log_std

    def _log_prob(
        self, x_t: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
    ) -> jnp.ndarray:
        std = jnp.exp(log_std)
        log_prob = -0.5 * (
            ((x_t - mean) / (std + self.EPS)) ** 2
            + 2.0 * log_std
            + jnp.log(2.0 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)

        log_prob -= jnp.sum(
            2.0 * (jnp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t)),
            axis=-1,
        )
        return log_prob

    def _log_prob_target(
        self, x_t: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
    ) -> jnp.ndarray:
        std = jnp.exp(log_std)
        log_prob = -0.5 * (
            ((x_t - mean) / (std + self.EPS)) ** 2
            + 2.0 * log_std
            + jnp.log(2.0 * jnp.pi)
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

    def sample_target(self, target_obs: jnp.ndarray, key: jnp.ndarray):
        mean, log_std = self._get_dist_params_target(target_obs)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, shape=mean.shape)
        x_t = mean + eps * std
        action = jnp.tanh(x_t)
        return action, self._log_prob_target(x_t, mean, log_std)

    def __call__(self, obs: jnp.ndarray, key: jnp.ndarray):
        return self.sample(obs, key)
