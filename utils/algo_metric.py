import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

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
class StateAsymmetricMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(self, rngs: nnx.Rngs, obs_dim: int, hidden_size: int):
        self.l1 = orthogonal_linear(rngs, obs_dim * 2, hidden_size)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_1:jnp.ndarray, obs_2:jnp.ndarray):
            x = nnx.gelu(self.ln1(self.l1(jnp.concatenate([obs_1, obs_2], axis=-1))))
            x = nnx.gelu(self.ln2(self.l2(x)))
            x = self.l3(x)
            log_metric = jnp.clip(jnp.squeeze(x, axis=-1), self.LOG_MIN, self.LOG_MAX)
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
        self.l1 = orthogonal_linear(rngs, (obs_dim + act_dim)*2, hidden_size)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act_1: jnp.ndarray, obs_act_2: jnp.ndarray) -> jnp.ndarray:
        x = nnx.gelu(self.ln1(self.l1(jnp.concatenate([obs_act_1, obs_act_2], axis=-1))))
        x = nnx.gelu(self.ln2(self.l2(x)))
        x = self.l3(x)
        log_metric = jnp.squeeze(
           x,axis=-1,
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
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim+obs_dim, hidden_size)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act: jnp.ndarray, obs_prime: jnp.ndarray) -> jnp.ndarray:
        x = nnx.gelu(self.ln1(self.l1(jnp.concatenate([obs_act, obs_prime], axis=-1))))
        x = nnx.gelu(self.ln2(self.l2(x)))
        x = self.l3(x)
        log_metric = jnp.squeeze(
            x, axis=-1
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)