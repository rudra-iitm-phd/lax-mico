# Copyright 2026 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import abc
from typing import Generic, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp
from jax import flatten_util

from utils.types import Transition

State = TypeVar("State")
Sample = TypeVar("Sample")


@flax.struct.dataclass
class ReplayBufferState:
    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: jnp.ndarray


class QueueBase(abc.ABC, Generic[Sample]):
    def __init__(self, max_replay_size: int, dummy_data_sample, sample_batch_size: int):
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flat, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        self._data_shape = (max_replay_size, len(dummy_flat))
        self._data_type = dummy_flat.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0

    def init(self, key: jnp.ndarray) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_type),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def check_can_insert(self, buffer_state, samples, shards=1):
        assert isinstance(shards, int)
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                f"Insert size {insert_size} exceeds max_replay_size {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def insert(self, buffer_state: ReplayBufferState, samples) -> ReplayBufferState:
        self.check_can_insert(buffer_state, samples, 1)
        return self._insert_internal(buffer_state, samples)

    def _insert_internal(
        self, buffer_state: ReplayBufferState, samples
    ) -> ReplayBufferState:
        update = self._flatten_fn(samples)
        data = buffer_state.data
        position = buffer_state.insert_position
        # Roll buffer if end is reached (circular FIFO)
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)
        return buffer_state.replace(
            data=data, insert_position=position, sample_position=sample_position
        )

    def sample(self, buffer_state: ReplayBufferState):
        return self._sample_internal(buffer_state)

    @abc.abstractmethod
    def _sample_internal(self, buffer_state: ReplayBufferState): ...

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position


class UniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """
    Standard replay buffer: uniform random sampling without replacement.

    SAC is purely off-policy and works well with uniform sampling.
    (gpe uses a priority queue; we don't need that complexity for SAC.)
    """

    def _sample_internal(self, buffer_state: ReplayBufferState):
        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._sample_batch_size,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )
        batch = jnp.take(buffer_state.data, idx, axis=0, mode="wrap")
        return buffer_state.replace(key=key), self._unflatten_fn(batch)


@flax.struct.dataclass
class RunningStatisticsState:
    reward_state: ReplayBufferState


class RunningStatistics:
    @staticmethod
    def init(reward_shape, key) -> RunningStatisticsState:
        reward_state = ReplayBufferState(
            data=jnp.zeros(reward_shape, jnp.float32),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )
        return RunningStatisticsState(reward_state=reward_state)

    @staticmethod
    def insert_reward(
        running_state: RunningStatisticsState, reward: jnp.ndarray
    ) -> RunningStatisticsState:
        rs = running_state.reward_state
        data, position = rs.data, rs.insert_position
        roll = jnp.minimum(0, len(data) - position - len(reward))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll
        data = jax.lax.dynamic_update_slice_in_dim(data, reward, position, axis=0)
        position = (position + len(reward)) % (len(data) + 1)
        sample_position = jnp.maximum(0, rs.sample_position + roll)
        rs = rs.replace(
            data=data, insert_position=position, sample_position=sample_position
        )
        return running_state.replace(reward_state=rs)


@flax.struct.dataclass
class RunningMeanStd:
    """Welford-style running mean/variance, used to normalize observations."""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    STD_MIN: float = flax.struct.field(pytree_node=False, default=1e-6)
    STD_MAX: float = flax.struct.field(pytree_node=False, default=1e6)

    @staticmethod
    def init(shape) -> "RunningMeanStd":
        return RunningMeanStd(
            mean=jnp.zeros(shape, jnp.float32),
            var=jnp.ones(shape, jnp.float32),
            count=jnp.array(0.0, jnp.float32),
        )

    def update(self, x: jnp.ndarray) -> "RunningMeanStd":
        # x: (batch, dim) — raw, unnormalized observations
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        return self.replace(mean=new_mean, var=new_var, count=tot_count)

    def std(self) -> jnp.ndarray:
        return jnp.clip(
            jnp.sqrt(jnp.maximum(self.var, 0.0)), self.STD_MIN, self.STD_MAX
        )

    def normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.mean) / self.std()


# def nstep_fifo_init(template: Transition, nstep: int):
#     """Zero-filled FIFO. `template` is one 1-step transition with a leading
#     num_envs dim (see the nstep_template block in either main())."""
#     return jax.tree_util.tree_map(
#         lambda x: jnp.zeros((nstep,) + x.shape, x.dtype), template
#     )


# def nstep_fifo_push(fifo, transition: Transition):
#     """Drop the oldest entry, append the newest. The astype guards against a
#     dtype mismatch between the template and what actor_step actually returns
#     (e.g. an int32 truncation flag)."""
#     return jax.tree_util.tree_map(
#         lambda buf, x: jnp.concatenate([buf[1:], x.astype(buf.dtype)[None]], axis=0),
#         fifo,
#         transition,
#     )


# def nstep_aggregate(fifo, count, gamma: float, nstep: int, bootstrap_on_truncation):
#     """Collapse the FIFO into one n-step transition.

#     Returns a Transition whose `discount` ALREADY CONTAINS gamma^n - do not
#     multiply by gamma again in the train step.
#     """
#     start = nstep - count  # traced scalar index of the oldest valid entry

#     def take(x):
#         return jnp.take(x, start, axis=0)

#     reward = jnp.zeros_like(fifo.reward[0])
#     discount = jnp.ones_like(fifo.discount[0])
#     alive = jnp.ones_like(fifo.discount[0])
#     next_obs = fifo.next_observation[nstep - 1]

#     for i in range(nstep):  # static unroll; nstep is a compile-time constant
#         valid = (start <= i).astype(reward.dtype)
#         m = valid * alive
#         r_i = fifo.reward[i]
#         d_i = fifo.discount[i]
#         t_i = fifo.extras["state_extras"]["truncation"][i].astype(reward.dtype)

#         if bootstrap_on_truncation:
#             d_boot = d_i + (1.0 - d_i) * t_i
#         else:
#             d_boot = d_i

#         reward = reward + m * discount * r_i
#         next_obs = jnp.where((m > 0)[..., None], fifo.next_observation[i], next_obs)
#         discount = jnp.where(m > 0, discount * gamma * d_boot, discount)
#         alive = alive * jnp.where(valid > 0, d_i * (1.0 - t_i), 1.0)

#     return Transition(
#         observation=take(fifo.observation),
#         action=take(fifo.action),
#         reward=reward,
#         discount=discount,
#         next_observation=next_obs,
#         extras=jax.tree_util.tree_map(take, fifo.extras),
#     )


# def nstep_template_from_dims(num_envs: int, obs_dim: int, act_dim: int) -> Transition:
#     """Convenience constructor for the FIFO template, so both scripts build it
#     identically. Mirrors the dummy_transition passed to UniformSamplingQueue,
#     but with a leading num_envs dim instead of 1."""
#     return Transition(
#         observation=jnp.zeros((num_envs, obs_dim), jnp.float32),
#         action=jnp.zeros((num_envs, act_dim), jnp.float32),
#         reward=jnp.zeros((num_envs,), jnp.float32),
#         discount=jnp.zeros((num_envs,), jnp.float32),
#         next_observation=jnp.zeros((num_envs, obs_dim), jnp.float32),
#         extras={"state_extras": {"truncation": jnp.zeros((num_envs,), jnp.float32)}},
#     )

"""
Shared n-step return accumulation for the parallel-env harness.

Single source of truth for BOTH sac_single.py and dhpg.py, so the two
baselines can never drift on how returns are computed.

WHY THIS EXISTS
---------------
The DHPG author's replay buffer (utils/replay_buffer.py in
sahandrez/homomorphic_policy_gradient) is episode-indexed and builds the
n-step return at SAMPLE time:

    reward = 0; discount = 1
    for i in range(nstep):
        reward   += discount * episode['reward'][idx + i]
        discount *= episode['discount'][idx + i] * gamma
    return (obs[idx-1], action[idx], reward, discount, obs[idx+nstep-1])

Two things follow, and both are easy to get wrong:

  1. The STORED discount already contains gamma^n. A train step that does
     `reward + config.gamma * discount * Q` would therefore be applying
     gamma^(n+1). Consumers of this module must use `discount` RAW.

  2. Because idx is drawn from [0, len - nstep], a window can never straddle
     an episode boundary.

Our UniformSamplingQueue is a flat circular FIFO with no episode structure,
so we build the window on the ROLLOUT side instead: push each 1-step
transition into a FIFO and insert the aggregated n-step transition. The
buffer needs no changes at all.

TRUNCATION vs TERMINATION
-------------------------
utils/acting.py sets `discount = 1 - n_state.done`, and brax's
EpisodeWrapper raises done=1 at the episode_length time limit as well as at
true termination. Naively that emits a discount=0 sample every 1000 steps,
cutting the bootstrap on a purely artificial boundary. DMC tasks never
actually terminate (the author's time_step.discount is always 1.0), so the
target should be r + gamma*Q ALWAYS. `truncation` is carried in
state_extras precisely so this can be undone. Per step:

    bootstrap factor : d_i + (1 - d_i) * t_i   -> 0 only on TRUE termination
    window continues : d_i * (1 - t_i)         -> stops on done of any kind
    window truncated : max over included steps of t_i

Note that when the consumer masks truncated transitions out of the TD loss
(as sac_single.py now does), `bootstrap_on_truncation` no longer affects
the TD target for those samples - they are dropped entirely. It still
affects any OTHER use of `discount`, e.g. DHPG's lax-bisimulation target
`r_dist + discount * transition_dist`, which is not masked.

which reproduces both of the author's properties (always bootstrap, never
cross a reset) explicitly rather than getting them from buffer structure.
Set bootstrap_on_truncation=False to recover the old, biased behaviour.

FIFO LAYOUT
-----------
Leaves have shape (nstep, num_envs, ...); index 0 is the oldest entry,
index nstep-1 the newest. `count` is the number of valid entries so far,
clipped to nstep, so the oldest valid index is `nstep - count`. Early in a
run the window is simply SHORTER than nstep - never malformed, and never
contaminated by the zero-fill.

n=1 is an exact no-op: reward = r_t, discount = gamma * d_t, truncation
= t_t - precisely the raw transition with `config.gamma *` folded in.
"""

import jax
import jax.numpy as jnp

from utils.types import Transition


def nstep_fifo_init(template: Transition, nstep: int):
    """Zero-filled FIFO. `template` is one 1-step transition with a leading
    num_envs dim (see the nstep_template block in either main())."""
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros((nstep,) + x.shape, x.dtype), template
    )


def nstep_fifo_push(fifo, transition: Transition):
    """Drop the oldest entry, append the newest. The astype guards against a
    dtype mismatch between the template and what actor_step actually returns
    (e.g. an int32 truncation flag)."""
    return jax.tree_util.tree_map(
        lambda buf, x: jnp.concatenate([buf[1:], x.astype(buf.dtype)[None]], axis=0),
        fifo,
        transition,
    )


def nstep_aggregate(fifo, count, gamma: float, nstep: int, bootstrap_on_truncation):
    """Collapse the FIFO into one n-step transition.

    `discount` in the result ALREADY CONTAINS gamma^n - do not multiply by
    gamma again in the train step.

    `extras["state_extras"]["truncation"]` is 1.0 iff the window ended at a
    time-limit truncation, so consumers can mask it out of the TD loss the
    way sac_single.py's critic loss does:
        q_error = q_error * (1.0 - truncation)[..., None]
    For n=1 this reduces exactly to the raw per-step flag.
    """
    start = nstep - count  # traced scalar index of the oldest valid entry

    def take(x):
        return jnp.take(x, start, axis=0)

    reward = jnp.zeros_like(fifo.reward[0])
    discount = jnp.ones_like(fifo.discount[0])
    alive = jnp.ones_like(fifo.discount[0])
    truncated = jnp.zeros_like(fifo.discount[0])
    next_obs = fifo.next_observation[nstep - 1]

    for i in range(nstep):  # static unroll; nstep is a compile-time constant
        valid = (start <= i).astype(reward.dtype)
        m = valid * alive
        r_i = fifo.reward[i]
        d_i = fifo.discount[i]
        t_i = fifo.extras["state_extras"]["truncation"][i].astype(reward.dtype)

        if bootstrap_on_truncation:
            d_boot = d_i + (1.0 - d_i) * t_i
        else:
            d_boot = d_i

        reward = reward + m * discount * r_i
        next_obs = jnp.where((m > 0)[..., None], fifo.next_observation[i], next_obs)
        discount = jnp.where(m > 0, discount * gamma * d_boot, discount)
        truncated = jnp.where(m > 0, jnp.maximum(truncated, t_i), truncated)
        alive = alive * jnp.where(valid > 0, d_i * (1.0 - t_i), 1.0)

    return Transition(
        observation=take(fifo.observation),
        action=take(fifo.action),
        reward=reward,
        discount=discount,
        next_observation=next_obs,
        extras={"state_extras": {"truncation": truncated}},
    )


def nstep_template_from_dims(num_envs: int, obs_dim: int, act_dim: int) -> Transition:
    """Convenience constructor for the FIFO template, so both scripts build it
    identically. Mirrors the dummy_transition passed to UniformSamplingQueue,
    but with a leading num_envs dim instead of 1."""
    return Transition(
        observation=jnp.zeros((num_envs, obs_dim), jnp.float32),
        action=jnp.zeros((num_envs, act_dim), jnp.float32),
        reward=jnp.zeros((num_envs,), jnp.float32),
        discount=jnp.zeros((num_envs,), jnp.float32),
        next_observation=jnp.zeros((num_envs, obs_dim), jnp.float32),
        extras={"state_extras": {"truncation": jnp.zeros((num_envs,), jnp.float32)}},
    )
