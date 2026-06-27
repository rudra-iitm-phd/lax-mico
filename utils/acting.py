from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper

from utils.buffer import RunningMeanStd
from utils.types import Transition


def wrap_env_for_training(
    env, episode_length: int, action_repeat: int = 1, full_reset: bool = False
):
    env = brax_training.VmapWrapper(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env, full_reset)
    return env


# def actor_step(
#     env, env_state, policy, key: jnp.ndarray, extra_fields: Sequence[str] = ()
# ) -> Tuple:
#     obs = env_state.obs
#     action, _ = policy(obs, key)
#     n_state = env.step(env_state, action)
#     state_extras = {x: n_state.info[x] for x in extra_fields}
#     return n_state, Transition(
#         observation=obs,
#         action=action,
#         reward=n_state.reward,
#         discount=1 - n_state.done,
#         next_observation=n_state.obs,
#         extras={"state_extras": state_extras},
#     )


def actor_step(
    env,
    env_state,
    policy,
    obs_normalizer: RunningMeanStd,
    key: jnp.ndarray,
    extra_fields: Sequence[str] = (),
) -> Tuple:
    obs = env_state.obs
    norm_obs = obs_normalizer.normalize(obs)  # CHANGED: feed normalized obs to policy
    action, _ = policy(norm_obs, key)
    n_state = env.step(env_state, action)
    state_extras = {x: n_state.info[x] for x in extra_fields}
    return n_state, Transition(
        observation=obs,  # CHANGED: store RAW obs in the transition
        action=action,
        reward=n_state.reward,
        discount=1 - n_state.done,
        next_observation=n_state.obs,  # raw next_obs too
        extras={"state_extras": state_extras},
    )


# def actor_step(
#     env, env_state, policy, critic, key: jnp.ndarray, extra_fields: Sequence[str] = ()
# ) -> Tuple:
#     obs = env_state.obs
#     action, _ = policy(critic.encoder_state(obs), key)
#     n_state = env.step(env_state, action)
#     state_extras = {x: n_state.info[x] for x in extra_fields}
#     return n_state, Transition(
#         observation=obs,
#         action=action,
#         reward=n_state.reward,
#         discount=1 - n_state.done,
#         next_observation=n_state.obs,
#         extras={"state_extras": state_extras},
#     )


def actor_step_rep(
    env,
    env_state,
    repnet,
    policy,
    key: jnp.ndarray,
    extra_fields: Sequence[str] = (),
) -> Tuple:
    obs = env_state.obs
    obs_rep = repnet.state_rep(obs)
    action, _ = policy(obs_rep, key)
    n_state = env.step(env_state, action)
    state_extras = {x: n_state.info[x] for x in extra_fields}
    return n_state, Transition(
        observation=obs,
        action=action,
        reward=n_state.reward,
        discount=1 - n_state.done,
        next_observation=n_state.obs,
        extras={"state_extras": state_extras},
    )


def actor_step_rep_source(
    env,
    env_state,
    repnet,
    policy,
    key: jnp.ndarray,
    extra_fields: Sequence[str] = (),
) -> Tuple:
    obs = env_state.obs
    obs_rep = repnet.source_state_rep(obs)
    action, _ = policy(obs_rep, key)
    n_state = env.step(env_state, action)
    state_extras = {x: n_state.info[x] for x in extra_fields}
    return n_state, Transition(
        observation=obs,
        action=action,
        reward=n_state.reward,
        discount=1 - n_state.done,
        next_observation=n_state.obs,
        extras={"state_extras": state_extras},
    )


def actor_step_rep_target(
    env,
    env_state,
    repnet,
    policy,
    key: jnp.ndarray,
    extra_fields: Sequence[str] = (),
) -> Tuple:
    obs = env_state.obs
    obs_rep = repnet.target_state_rep(obs)
    action, _ = policy.sample_target(obs_rep, key)
    n_state = env.step(env_state, action)
    state_extras = {x: n_state.info[x] for x in extra_fields}
    return n_state, Transition(
        observation=obs,
        action=action,
        reward=n_state.reward,
        discount=1 - n_state.done,
        next_observation=n_state.obs,
        extras={"state_extras": state_extras},
    )
