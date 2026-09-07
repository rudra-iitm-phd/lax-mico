"""
MICo metric utilities.

Line-for-line port of
    google-research/google-research @ master :: mico/atari/metric_utils.py
with three additions that are NOT in the author's file and are marked [HARNESS]:
    huber_loss          - copied from dopamine.jax.losses so this module has no
                          dopamine dependency (identical formula, delta=1.0)
    pair_mask           - outer-product truncation mask, flattened to B**2 in
                          the SAME order squarify uses
    masked_mico_loss    - the one-line loss the agent actually calls

Nothing else deviates. In particular:

  * `_sqrt` is the neural-tangents custom_jvp square root the author imported
    specifically to stop `jnp.arccos` / `jnp.sqrt` blowing up at 0 gradient.
    Do not "simplify" it to jnp.sqrt: cosine_distance evaluates
    sqrt(1 - cos^2) which is exactly 0 whenever two representations are
    parallel - including at init, and including every diagonal pair (i, i),
    which is B of the B**2 entries in every batch. Plain jnp.sqrt gives NaN
    gradients there.

  * `beta = 0.1` is the author's default (paper Section 5, "in our results we
    use beta = 0.1").

  * `representation_distances` is ASYMMETRIC in intent: the caller passes the
    ONLINE representation first and a STOP-GRADIENTED one second, so only the
    first argument receives gradient. See metric_sac_agent.py::loss_fn.

  * `target_distances` takes a SCALAR `cumulative_gamma`. In Dopamine that is
    gamma ** update_horizon, which is how n-step enters the MICo target: the
    metric target is |r_x - r_y| + gamma^n * U_target(x'_n, y'_n), with the
    n-step reward already summed by the replay buffer. There is no terminal
    or truncation masking in the author's version - see [HARNESS] below.

=========================================================================
SHAPES  (this bites - see the [SHAPE] note in mico.py)
=========================================================================
representation_distances expects representations of shape EXACTLY (B, d), and
target_distances expects rewards of shape EXACTLY (B,). squarify branches on
x.ndim, so a reward of shape (B, 1) - which is what a UniformSamplingQueue
built from a dummy_data_sample with a leading dim of 1 hands you - silently
takes the 2-D path, builds a (B, B, 1) grid, and then dies in the reshape
with "cannot reshape array of shape (1, B, B) into shape 1". Squeeze the batch
to canonical shapes BEFORE calling anything in this module; these functions
are kept as a faithful port and deliberately do not squeeze for you.

=========================================================================
INDEX ORDER  (verified numerically, see test at the bottom of this file)
=========================================================================
squarify() has DIFFERENT semantics for 1-D and 2-D inputs, which looks like a
bug and is not:
    x of shape (B, d):  squarify(x)[i, j] = x[i]
    x of shape (B,):    squarify(x)[i, j] = x[j]
representation_distances squarifies the first argument and squarifies+
transposes the second, so flat index i*B + j pairs (first[i], second[j]).
target_distances squarifies the rewards and also uses the transpose, and takes
|.| of the difference, so the 1-D/2-D mismatch cancels: reward_diffs[i*B + j]
= |r_i - r_j| either way. pair_mask therefore uses valid[i] * valid[j], which
is symmetric and correct under both conventions.
"""

import functools

import jax
import jax.numpy as jnp
from jax import custom_jvp

EPSILON = 1e-9


# The following two functions were borrowed by the author from
# https://github.com/google/neural-tangents/blob/master/neural_tangents/stax.py
# as they resolve the instabilities observed when using `jnp.arccos`.
@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.0):
    return jnp.sqrt(jnp.maximum(x, tol))


@_sqrt.defjvp
def _sqrt_jvp(tol, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    safe_tol = max(tol, 1e-30)
    square_root = _sqrt(x, safe_tol)
    return square_root, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.0)


def l2(x, y):
    return _sqrt(jnp.sum(jnp.square(x - y)))


def cosine_distance(x, y):
    """Angular distance in [0, pi]. This is the author's default distance_fn."""
    numerator = jnp.sum(x * y)
    denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
    cos_similarity = numerator / (denominator + EPSILON)
    return jnp.arctan2(_sqrt(1.0 - cos_similarity**2), cos_similarity)


def squarify(x):
    batch_size = x.shape[0]
    if len(x.shape) > 1:
        representation_dim = x.shape[-1]
        return jnp.reshape(
            jnp.tile(x, batch_size), (batch_size, batch_size, representation_dim)
        )
    return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))


def representation_distances(
    first_representations,
    second_representations,
    distance_fn,
    beta=0.1,
    return_distance_components=False,
):
    """U_omega(x, y) = (||phi(x)||^2 + ||phi(y)||^2) / 2 + beta * theta(x, y).

    Returns a flat (B**2,) vector; entry i*B + j is the pair
    (first_representations[i], second_representations[j]).
    """
    batch_size = first_representations.shape[0]
    representation_dim = first_representations.shape[-1]
    first_squared_reps = squarify(first_representations)
    first_squared_reps = jnp.reshape(
        first_squared_reps, [batch_size**2, representation_dim]
    )
    second_squared_reps = squarify(second_representations)
    second_squared_reps = jnp.transpose(second_squared_reps, axes=[1, 0, 2])
    second_squared_reps = jnp.reshape(
        second_squared_reps, [batch_size**2, representation_dim]
    )
    base_distances = jax.vmap(distance_fn, in_axes=(0, 0))(
        first_squared_reps, second_squared_reps
    )
    norm_average = 0.5 * (
        jnp.sum(jnp.square(first_squared_reps), -1)
        + jnp.sum(jnp.square(second_squared_reps), -1)
    )
    if return_distance_components:
        return norm_average + beta * base_distances, norm_average, base_distances
    return norm_average + beta * base_distances


def absolute_reward_diff(r1, r2):
    return jnp.abs(r1 - r2)


def target_distances(representations, rewards, distance_fn, cumulative_gamma):
    """Target distance using the metric operator.

    T^U(x, y) = |r_x - r_y| + cumulative_gamma * U_target(x', y').
    `representations` are the TARGET-network representations of the NEXT
    states; `cumulative_gamma` is gamma ** nstep.
    """
    next_state_similarities = representation_distances(
        representations, representations, distance_fn
    )
    squared_rews = squarify(rewards)
    squared_rews_transp = jnp.transpose(squared_rews)
    squared_rews = squared_rews.reshape((squared_rews.shape[0] ** 2))
    squared_rews_transp = squared_rews_transp.reshape(
        (squared_rews_transp.shape[0] ** 2)
    )
    reward_diffs = absolute_reward_diff(squared_rews, squared_rews_transp)
    return jax.lax.stop_gradient(
        reward_diffs + cumulative_gamma * next_state_similarities
    )


# --------------------------------------------------------------------------- #
# [HARNESS] Everything below is ours, not the author's.
# --------------------------------------------------------------------------- #
def huber_loss(targets, predictions, delta: float = 1.0):
    """dopamine.jax.losses.huber_loss, inlined verbatim.

    Note this is numerically identical to torch's smooth_l1_loss with beta=1.
    The author found it important to use Huber rather than MSE here: "larger
    distances tended to overwhelm the optimization process" (paper, Sec. 6).
    """
    x = jnp.abs(targets - predictions)
    return jnp.where(x <= delta, 0.5 * x**2, 0.5 * delta**2 + delta * (x - delta))


def pair_mask(valid):
    """(B,) validity flags -> (B**2,) pairwise mask, in squarify's flat order.

    Entry i*B + j is valid[i] * valid[j]. The MICo target for a pair reads the
    reward AND the next-state representation of BOTH members, so one
    contaminated sample poisons an entire row and an entire column of the
    B x B grid; masking one side only would leave half of them through.
    """
    return (valid[:, None] * valid[None, :]).reshape(-1)


def masked_mico_loss(
    online_representations,
    frozen_representations,
    target_next_representations,
    rewards,
    cumulative_gamma,
    valid=None,
    distance_fn=cosine_distance,
    beta: float = 0.1,
):
    """The full MICo loss, exactly as assembled in metric_sac_agent.py.

    Args:
      online_representations:     phi_omega(x) for the batch states, WITH grad.
      frozen_representations:     phi_omega(x) for the same states, stop-grad'd
                                  (the author uses `frozen_params`, i.e. the
                                  online weights held constant, NOT the target
                                  network - see metric_sac_agent.py; the DQN
                                  variant uses the target network here).
      target_next_representations: phi_target(x') for the batch next states.
      rewards:                    (B,) rewards; n-step sum if nstep > 1.
      cumulative_gamma:           scalar gamma ** nstep.
      valid:                      optional (B,) 1.0/0.0 truncation mask.
    Returns:
      (loss, online_dist, target_dist) - the two distance vectors are returned
      for logging; both are (B**2,).
    """
    online_dist = representation_distances(
        online_representations, frozen_representations, distance_fn, beta=beta
    )
    target_dist = target_distances(
        target_next_representations, rewards, distance_fn, cumulative_gamma
    )
    elementwise = huber_loss(online_dist, target_dist)
    if valid is None:
        return jnp.mean(elementwise), online_dist, target_dist
    mask = pair_mask(valid)
    return jnp.mean(mask * elementwise), online_dist, target_dist


if __name__ == "__main__":
    # Index-order and consistency checks referenced in the docstring.
    import numpy as np

    B, D = 4, 3
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    reps = jax.random.normal(k1, (B, D))
    nreps = jax.random.normal(k2, (B, D))
    rews = jax.random.normal(k3, (B,))

    # 2-D squarify: [i, j] = x[i]
    sq = squarify(reps)
    assert np.allclose(sq[2, 0], reps[2]) and np.allclose(sq[2, 3], reps[2])
    # 1-D squarify: [i, j] = x[j]
    sq1 = squarify(rews)
    assert np.allclose(sq1[2, 0], rews[0]) and np.allclose(sq1[0, 2], rews[2])

    # representation_distances pairs (first[i], second[j]) at flat index i*B+j
    d = representation_distances(reps, nreps, cosine_distance)
    for i in range(B):
        for j in range(B):
            manual = 0.5 * (np.sum(reps[i] ** 2) + np.sum(nreps[j] ** 2)) + 0.1 * float(
                cosine_distance(reps[i], nreps[j])
            )
            assert abs(float(d[i * B + j]) - manual) < 1e-4, (i, j)

    # target_distances reward term is |r_i - r_j| at flat index i*B+j
    t = target_distances(nreps, rews, cosine_distance, 0.99)
    u = representation_distances(nreps, nreps, cosine_distance)
    for i in range(B):
        for j in range(B):
            manual = abs(float(rews[i] - rews[j])) + 0.99 * float(u[i * B + j])
            assert abs(float(t[i * B + j]) - manual) < 1e-4, (i, j)

    # gradients are finite even for identical (parallel) representations
    same = jnp.tile(reps[0], (B, 1))
    g = jax.grad(lambda r: jnp.mean(representation_distances(r, r, cosine_distance)))(
        same
    )
    assert np.all(np.isfinite(np.asarray(g))), "custom _sqrt failed"

    # masking with all-valid must equal the unmasked loss
    l_masked, _, _ = masked_mico_loss(reps, reps, nreps, rews, 0.99, jnp.ones((B,)))
    l_plain, _, _ = masked_mico_loss(reps, reps, nreps, rews, 0.99, None)
    assert abs(float(l_masked) - float(l_plain)) < 1e-6

    print("mico_utils: all checks passed")
