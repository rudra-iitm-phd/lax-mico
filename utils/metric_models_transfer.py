import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


#### copied from gen_pe
class StateAsymmetricMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(
        self, rngs: nnx.Rngs, source_obs_dim: int, target_obs_dim: int, hidden_size: int
    ):
        zero_init = nnx.initializers.zeros
        self.proj_source_source = nnx.Linear(source_obs_dim * 2, hidden_size, rngs=rngs)
        self.proj_target_target = nnx.Linear(target_obs_dim * 2, hidden_size, rngs=rngs)
        self.proj_source_target = nnx.Linear(
            source_obs_dim + target_obs_dim, hidden_size, rngs=rngs
        )
        self.proj_target_source = nnx.Linear(
            target_obs_dim + source_obs_dim, hidden_size, rngs=rngs
        )
        self.trunk = nnx.Sequential(
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def get_source_state_distance(
        self, obs_1: jnp.ndarray, obs_2: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_source_source(jnp.concatenate([obs_1, obs_2], axis=-1))
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_target_state_distance(
        self, obs_1: jnp.ndarray, obs_2: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_target_target(jnp.concatenate([obs_1, obs_2], axis=-1))
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_source_target_distance(
        self, source: jnp.ndarray, target: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_source_target(jnp.concatenate([source, target], axis=-1))
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_target_source_distance(
        self, target: jnp.ndarray, source: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_target_source(jnp.concatenate([target, source], axis=-1))
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)


class EnsembleStateMetric(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        source_obs_dim: int,
        target_obs_dim: int,
        hidden_size: int = 256,
    ):
        self.g = StateAsymmetricMetric(
            rngs, source_obs_dim, target_obs_dim, hidden_size
        )

    def get_source_distance(
        self, source1: jnp.ndarray, source2: jnp.ndarray
    ) -> jnp.ndarray:
        return self.g.get_source_state_distance(
            source1, source2
        ), self.g.get_source_state_distance(source2, source1)

    def get_target_distance(
        self, target1: jnp.ndarray, target2: jnp.ndarray
    ) -> jnp.ndarray:
        return self.g.get_target_state_distance(
            target1, target2
        ), self.g.get_target_state_distance(target2, target1)

    def get_cross_distance(
        self, source: jnp.ndarray, target: jnp.ndarray
    ) -> jnp.ndarray:
        return self.g.get_source_target_distance(
            source, target
        ), self.g.get_target_source_distance(target, source)


class StateActionDiffuseMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(
        self,
        rngs: nnx.Rngs,
        source_obs_dim: int,
        source_act_dim: int,
        target_obs_dim: int,
        target_act_dim: int,
        hidden_size: int,
    ):
        zero_init = nnx.initializers.zeros
        self.proj_source_source = nnx.Linear(
            (source_obs_dim + source_act_dim) * 2, hidden_size, rngs=rngs
        )
        self.proj_target_target = nnx.Linear(
            (target_obs_dim + target_act_dim) * 2, hidden_size, rngs=rngs
        )
        self.proj_source_target = nnx.Linear(
            (source_obs_dim + source_act_dim + target_obs_dim + target_act_dim),
            hidden_size,
            rngs=rngs,
        )
        self.proj_target_source = nnx.Linear(
            (source_obs_dim + source_act_dim + target_obs_dim + target_act_dim),
            hidden_size,
            rngs=rngs,
        )
        self.trunk = nnx.Sequential(
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def get_source_source_distance(
        self, source_obs_act_1: jnp.ndarray, source_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_source_source(
            jnp.concatenate([source_obs_act_1, source_obs_act_2], axis=-1)
        )
        log_metric = jnp.squeeze(
            self.trunk(proj),
            axis=-1,
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_target_target_distance(
        self, target_obs_act_1: jnp.ndarray, target_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_target_target(
            jnp.concatenate([target_obs_act_1, target_obs_act_2], axis=-1)
        )
        log_metric = jnp.squeeze(
            self.trunk(proj),
            axis=-1,
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_source_target_distance(
        self, source_obs_act_1: jnp.ndarray, target_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_source_target(
            jnp.concatenate([source_obs_act_1, target_obs_act_2], axis=-1)
        )
        log_metric = jnp.squeeze(
            self.trunk(proj),
            axis=-1,
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_target_source_distance(
        self, target_obs_act_1: jnp.ndarray, source_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_target_source(
            jnp.concatenate([target_obs_act_1, source_obs_act_2], axis=-1)
        )
        log_metric = jnp.squeeze(
            self.trunk(proj),
            axis=-1,
        )
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)


class EnsembleStateActionMetric(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        source_obs_dim: int,
        source_act_dim: int,
        target_obs_dim: int,
        target_act_dim: int,
        hidden_size: int,
    ):
        self.d = StateActionDiffuseMetric(
            rngs,
            source_obs_dim,
            source_act_dim,
            target_obs_dim,
            target_act_dim,
            hidden_size,
        )

    def get_source_distance(
        self, source_obs_act_1: jnp.ndarray, source_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        return self.d.get_source_source_distance(
            source_obs_act_1, source_obs_act_2
        ), self.d.get_source_source_distance(source_obs_act_2, source_obs_act_1)

    def get_target_distance(
        self, target_obs_act_1: jnp.ndarray, target_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        return self.d.get_target_target_distance(
            target_obs_act_1, target_obs_act_2
        ), self.d.get_target_target_distance(target_obs_act_2, target_obs_act_1)

    def get_cross_distance(
        self, source_obs_act_1: jnp.ndarray, target_obs_act_2: jnp.ndarray
    ) -> jnp.ndarray:
        return self.d.get_source_target_distance(
            source_obs_act_1, target_obs_act_2
        ), self.d.get_target_source_distance(target_obs_act_2, source_obs_act_1)


class MinStateActiontoStateMetric(nnx.Module):
    LOG_MIN: float = -20.0
    LOG_MAX: float = 2.0

    def __init__(
        self,
        rngs: nnx.Rngs,
        source_obs_dim: int,
        source_act_dim: int,
        target_obs_dim: int,
        target_act_dim: int,
        hidden_size: int,
    ):
        zero_init = nnx.initializers.zeros

        self.proj_source_act_source = nnx.Linear(
            (source_obs_dim + source_act_dim + source_obs_dim), hidden_size, rngs=rngs
        )
        self.proj_source_act_target = nnx.Linear(
            (source_obs_dim + source_act_dim + target_obs_dim), hidden_size, rngs=rngs
        )
        self.proj_target_act_source = nnx.Linear(
            (target_obs_dim + target_act_dim + source_obs_dim), hidden_size, rngs=rngs
        )
        self.proj_target_act_target = nnx.Linear(
            (target_obs_dim + target_act_dim + target_obs_dim), hidden_size, rngs=rngs
        )

        self.trunk = nnx.Sequential(
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def get_source_source_distance(
        self, source_obs_act: jnp.ndarray, source_obs_prime: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_source_act_source(
            jnp.concatenate([source_obs_act, source_obs_prime], axis=-1)
        )
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_source_target_distance(
        self, source_obs_act: jnp.ndarray, target_obs_prime: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_source_act_target(
            jnp.concatenate([source_obs_act, target_obs_prime], axis=-1)
        )
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_target_source_distance(
        self, target_obs_act: jnp.ndarray, source_obs_prime: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_target_act_source(
            jnp.concatenate([target_obs_act, source_obs_prime], axis=-1)
        )
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)

    def get_target_target_distance(
        self, target_obs_act: jnp.ndarray, target_obs_prime: jnp.ndarray
    ) -> jnp.ndarray:
        proj = self.proj_target_act_target(
            jnp.concatenate([target_obs_act, target_obs_prime], axis=-1)
        )
        log_metric = jnp.squeeze(self.trunk(proj), axis=-1)
        log_metric = jnp.clip(log_metric, self.LOG_MIN, self.LOG_MAX)
        return jnp.exp(log_metric)
