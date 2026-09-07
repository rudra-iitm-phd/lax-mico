import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

EPS = 1e-6


def get_tree_norm(tree):
    """Computes the L2 norm across all the leaves of a JAX Pytree"""
    squares = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree)
    total = jax.tree_util.tree_reduce(lambda a, x: a + x, squares)
    return jnp.sqrt(total)


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


class Scalar(nnx.Module):
    def __init__(self, val: float):
        self.val = nnx.Param(jnp.array(val, dtype=jnp.float32))

    def __call__(self) -> jnp.ndarray:
        return self.val


class SACCritic(nnx.Module):
    def __init__(
        self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int = 256
    ):
        self.l1 = orthogonal_linear(rngs, obs_dim + act_dim, hidden_size)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.l3 = orthogonal_linear(rngs, hidden_size, 1)

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.ln1(self.l1(obs_act)))
        x = nnx.relu(self.ln2(self.l2(x)))
        return jnp.squeeze(self.l3(x), axis=-1)


class EnsembleCritic(nnx.Module):
    def __init__(
        self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int = 256
    ):
        self.q1 = SACCritic(rngs, obs_dim, act_dim, hidden_size)
        self.q2 = SACCritic(rngs, obs_dim, act_dim, hidden_size)

    def __call__(self, obs_act: jnp.ndarray) -> jnp.ndarray:
        return self.q1(obs_act), self.q2(obs_act)


class SACGaussianActor(nnx.Module):
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def __init__(
        self, rngs: nnx.Rngs, obs_dim: int, act_dim: int, hidden_size: int = 256
    ):
        self.act_dim = act_dim
        self.l1 = orthogonal_linear(rngs, obs_dim, hidden_size)
        self.l2 = orthogonal_linear(rngs, hidden_size, hidden_size)
        # self.mean_head = orthogonal_linear(rngs, hidden_size, act_dim)
        self.mean_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)
        # self.log_std_head = orthogonal_linear(rngs, hidden_size, act_dim)
        self.log_std_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)

    def _get_dist_params(self, obs: jnp.ndarray):
        features = nnx.relu(self.l2(nnx.relu(self.l1(obs))))
        mean = self.mean_head(features)
        log_std = jnp.clip(
            self.log_std_head(features), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return mean, log_std

    def _log_prob(
        self, x_t: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
    ) -> jnp.ndarray:
        std = jnp.exp(log_std)
        # log_prob = -0.5 * (
        #     ((x_t - mean) / (std + EPS)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
        # )
        log_prob = -0.5 * (
            ((x_t - mean) / std) ** 2
            + 2.0 * log_std
            + jnp.log(2.0 * jnp.pi)  # CHANGED: removed +EPS
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

    def mean_action(self, obs: jnp.ndarray):
        mean, _ = self._get_dist_params(obs)
        action = jnp.tanh(mean)
        return action

    def __call__(self, obs: jnp.ndarray, key: jnp.ndarray):
        return self.sample(obs, key)
