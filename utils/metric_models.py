import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


#### copied from gen_pe
class StateAsymmetricMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, hidden_size: int):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(obs_dim * 2, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, obs_1: jnp.ndarray, obs_2: jnp.ndarray) -> jnp.ndarray:
        log_metric = jnp.squeeze(
            self.model(jnp.concatenate([obs_1, obs_2], axis=-1)), axis=-1
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)


class EnsembleStateMetric(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, obs_dim: int, hidden_size: int = 256):
        self.g = StateAsymmetricMetric(rngs, obs_dim, hidden_size)

    def __call__(self, obs_1: jnp.ndarray, obs_2: jnp.ndarray) -> jnp.ndarray:
        return self.g(obs_1, obs_2), self.g(obs_2, obs_1)


class StateActionDiffuseMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear((obs_dim + act_dim) * 2, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, obs_act_1: jnp.ndarray, obs_act_2: jnp.ndarray) -> jnp.ndarray:

        log_metric = jnp.squeeze(
            self.model(jnp.concatenate([obs_act_1, obs_act_2], axis=-1)),
            axis=-1,
        )

        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)

        return jnp.exp(log_metric)


class EnsembleStateActionMetric(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int):
        self.d = StateActionDiffuseMetric(rngs, obs_dim, act_dim, hidden_size)

    def __call__(self, obs_act_1: jnp.ndarray, obs_act_2: jnp.ndarray) -> jnp.ndarray:
        return self.d(obs_act_1, obs_act_2), self.d(obs_act_2, obs_act_1)


class MinStateActiontoStateMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        hidden_size: int,
    ):
        zero_init = nnx.initializers.zeros
        self.model = nnx.Sequential(
            nnx.Linear(obs_dim + act_dim + obs_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, obs_act: jnp.ndarray, obs_prime: jnp.ndarray) -> jnp.ndarray:
        log_metric = jnp.squeeze(
            self.model(jnp.concatenate([obs_act, obs_prime], axis=-1)), axis=-1
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)
