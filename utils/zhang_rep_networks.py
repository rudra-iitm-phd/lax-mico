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


class Scalar(nnx.Module):
    def __init__(self, val: float):
        self.val = nnx.Param(jnp.array(val, dtype=jnp.float32))

    def __call__(self) -> jnp.ndarray:
        return self.val


class Critic(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        state_dim: int,
        act_dim: int,
        hidden_size: int = 256,
    ):
        zero_init = nnx.initializers.zeros
        self.state_encoder = nnx.Sequential(
            nnx.Linear(state_dim, hidden_size, rngs=rngs),
            # nnx.relu,
            # nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )

        # self.state_action_encoder = nnx.Sequential(
        #     nnx.Linear(hidden_size + act_dim, hidden_size, rngs=rngs),
        #     nnx.relu,
        # )

        self.Q_function1 = nnx.Sequential(
            nnx.Linear(hidden_size + act_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )
        self.Q_function2 = nnx.Sequential(
            nnx.Linear(hidden_size + act_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )
        # self.model = nnx.Sequential(
        #     nnx.Linear(state_dim + act_rep_dim, hidden_size, rngs=rngs),
        #     nnx.LayerNorm(hidden_size, rngs=rngs),
        #     nnx.relu,
        #     nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        #     nnx.LayerNorm(hidden_size, rngs=rngs),
        #     nnx.relu,
        #     nnx.Linear(
        #         hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
        #     ),
        # )

    def encoder_state(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.state_encoder(obs)

    # def encoder_state_action(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
    #     phi_s = self.state_encoder(obs)
    #     phi_sa = self.state_action_encoder(jnp.concatenate([phi_s, act], axis=-1))
    #     return phi_sa

    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        phi_s = nnx.relu(self.encoder_state(obs))
        phi_sa = jnp.concatenate([phi_s, act], axis=-1)
        q1 = jnp.squeeze(self.Q_function1(phi_sa), axis=-1)
        q2 = jnp.squeeze(self.Q_function2(phi_sa), axis=-1)
        return q1, q2


class EnsembleCritic(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        state_dim: int,
        act_dim: int,
        hidden_size: int = 256,
    ):
        self.q1 = Critic(rngs, state_dim, act_dim, hidden_size)
        self.q2 = Critic(rngs, state_dim, act_dim, hidden_size)

    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        return self.q1(obs, act), self.q2(obs, act)


class Actor(nnx.Module):
    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def __init__(
        self,
        rngs: nnx.Rngs,
        # state_dim: int,
        act_dim: int,
        # critic: Critic,
        hidden_size: int = 256,
    ):
        self.act_dim = act_dim
        # self.state_encoder = critic.state_encoder
        # self.state_encoder = nnx.Sequential(
        #     nnx.Linear(state_dim, hidden_size, rngs=rngs),
        #     nnx.relu(),
        #     nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        # )
        # self.trunk = nnx.Sequential(
        #     nnx.Linear(state_rep_dim, hidden_size, rngs=rngs),
        #     # nnx.LayerNorm(hidden_size, rngs=rngs),
        #     nnx.relu,
        #     nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        #     nnx.relu,
        # )
        self.trunk = nnx.Sequential(
            # nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.relu,
        )
        self.mean_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_size, act_dim, rngs=rngs)

    def _get_dist_params(self, obs: jnp.ndarray):
        # phi_s = jax.lax.stop_gradient(self.state_encoder(obs))
        phi_s = obs
        features = self.trunk(phi_s)
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
