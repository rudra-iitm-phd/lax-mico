import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

EPS = 1e-6


class DroQCritic(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        dropout_rate: float = 0.01,
    ):
        zero_init = nnx.initializers.zeros
        self.dropout_rate = dropout_rate

        self.model = nnx.Sequential(
            nnx.Linear(obs_dim + act_dim, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs, deterministic=False),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs, deterministic=False),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, 1, rngs=rngs),
        )

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(self.model(obs_act), axis=-1)


class EnsembleCritic(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        dropout_rate: float = 0.01,
        num_ensembles: int = 2,
    ):
        self.num_ensembles = num_ensembles  # fine as static (plain int)
        self.critics = nnx.data(
            [
                DroQCritic(rngs, obs_dim, act_dim, hidden_size, dropout_rate)
                for _ in range(num_ensembles)
            ]
        )

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        q_values = jnp.stack([critic(obs_act) for critic in self.critics], axis=0)
        return q_values  # Shape: (M, batch_size)

    def get_min(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        q_values = self.__call__(obs_act)
        return jnp.min(q_values, axis=0)

    def get_mean(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        q_values = self.__call__(obs_act)
        return jnp.mean(q_values, axis=0)
