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

    @staticmethod
    def init(shape) -> "RunningMeanStd":
        return RunningMeanStd(
            mean=jnp.zeros(shape, jnp.float32),
            var=jnp.ones(shape, jnp.float32),
            count=jnp.array(1e-4, jnp.float32),
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

    def normalize(self, x: jnp.ndarray, clip: float = 10.0) -> jnp.ndarray:
        normed = (x - self.mean) / jnp.sqrt(self.var + 1e-8)
        return jnp.clip(normed, -clip, clip)
