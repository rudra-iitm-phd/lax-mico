# rep_lr/utils/viz.py
#
# Two visualisation utilities:
#
#   Task 1 – tsne_rep_plot()
#       Sample a batch from buffer_e1, compute representations, run t-SNE,
#       colour by state value, draw arrows to N sampled frames.
#       Saved to  <log_dir>/tsne/step_<N>.png  (NOT pushed to TensorBoard —
#       each image is ~1-4 MB and saving every log step would bloat disk and
#       cause expensive CPU↔GPU copies inside the training loop).
#
#   Task 2 – alignment_scalars()
#       Pick an anchor state, compute v1(t) = mean(f(anchor, batch)) and
#       v2(t) = mean(h(anchor, batch)).  Returns two Python floats that the
#       caller logs to TensorBoard via logger.log_tabular().
#
# Neither function is JIT-compiled.  They are called from the outer Python
# loop in main.py, OUTSIDE train_n_steps, so there is no JAX tracing issue.
# The CPU↔GPU transfer (np.array(jnp_array)) happens here, explicitly.

import io
import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # no display needed — we save to file
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# =============================================================================
# Task 1 — t-SNE representation plot
# =============================================================================


def tsne_rep_plot(
    buffer_state,
    buffer,
    phi,  # representation network (callable: obs -> z)
    critic,  # used to get V(s) ≈ mean(Q1, Q2) for colouring
    log_dir: str,
    step: int,
    n_vis_frames: int = 4,  # number of frames to show with arrows
    tsne_batch_size: int = 1000,  # how many states to embed
    tsne_perplexity: float = 30.0,
    tsne_seed: int = 0,
    get_frame_fn=None,  # optional: obs -> RGB image (H,W,3) uint8
    # if None, arrows are drawn but no inset image
):
    """
    Sample `tsne_batch_size` observations from the replay buffer,
    compute phi(obs), run t-SNE, colour by V(s), draw arrows to
    `n_vis_frames` randomly selected observations.

    Saves to  <log_dir>/tsne/step_<step>.png

    CHANGE POINT in main.py
    -----------------------
    Called inside the `if steps % config["vis_freq"] == 0:` block.
    Requires:  pip install scikit-learn matplotlib
    """
    # ── pull a batch from the buffer (CPU-side) ───────────────────────────
    # We call buffer._sample_internal directly to get a fixed-size batch
    # without modifying buffer_state permanently.
    # (buffer.sample returns a new buffer_state; we throw it away here.)
    _, batch = buffer.sample(buffer_state)
    obs = np.array(batch.observation)  # (B, obs_dim)  on CPU

    # If tsne_batch_size > batch size, we just use what we have
    B = obs.shape[0]
    idx = np.random.choice(B, size=min(tsne_batch_size, B), replace=False)
    obs_sub = obs[idx]  # (N, obs_dim)

    # ── compute representations on GPU, transfer to CPU ───────────────────
    # phi may be the identity if you are not using a rep net —
    # just pass phi = lambda x: x in that case.
    z_jnp = phi(jnp.array(obs_sub))  # (N, rep_dim)
    z = np.array(z_jnp)  # CPU

    # ── compute V(s) = mean(Q1(s,a'), Q2(s,a')) for colour ───────────────
    # We use a zero action as a quick proxy; replace with a sampled action
    # if you want a proper value estimate.
    dummy_act = jnp.zeros(
        (
            obs_sub.shape[0],
            critic.q1.model[-1].out_features if hasattr(critic, "q1") else 1,
        )
    )
    obs_jnp = jnp.array(obs_sub)
    # try to get act_dim from critic input shape
    try:
        q1_vals = np.array(critic.q1(jnp.concatenate([obs_jnp, dummy_act], axis=-1)))
        q2_vals = np.array(critic.q2(jnp.concatenate([obs_jnp, dummy_act], axis=-1)))
        values = 0.5 * (q1_vals + q2_vals)  # (N,)
    except Exception:
        values = np.zeros(len(obs_sub))

    # ── t-SNE ─────────────────────────────────────────────────────────────
    tsne = TSNE(
        n_components=2, perplexity=tsne_perplexity, random_state=tsne_seed, n_jobs=-1
    )
    z_2d = tsne.fit_transform(z)  # (N, 2)

    # ── plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))

    if get_frame_fn is not None and n_vis_frames > 0:
        # leave room on the right for inset frames
        ax = fig.add_axes([0.05, 0.05, 0.65, 0.90])
    else:
        ax = fig.add_axes([0.05, 0.05, 0.88, 0.90])

    sc = ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=values,
        cmap="viridis",
        s=30,
        alpha=0.7,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="V(s) estimate")
    ax.set_title(f"t-SNE of φ(s)  —  step {step}", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # ── arrows + inset frames ─────────────────────────────────────────────
    if n_vis_frames > 0:
        vis_idx = np.random.choice(
            len(obs_sub), size=min(n_vis_frames, len(obs_sub)), replace=False
        )
        # evenly space inset axes on the right
        for rank, vi in enumerate(vis_idx):
            px, py = z_2d[vi, 0], z_2d[vi, 1]

            if get_frame_fn is not None:
                # compute inset position in figure coordinates
                inset_left = 0.73
                inset_width = 0.24
                inset_h = 0.80 / n_vis_frames
                inset_bot = (
                    0.05 + rank * inset_h + 0.5 * (0.80 / n_vis_frames - inset_h)
                )

                ax_inset = fig.add_axes(
                    [inset_left, inset_bot, inset_width, inset_h * 0.85]
                )
                frame = get_frame_fn(obs_sub[vi])  # (H, W, 3) uint8
                ax_inset.imshow(frame)
                ax_inset.axis("off")

                # convert inset centre to data coords for the arrow tip
                # use annotate with xycoords='figure fraction' for the tail
                inset_cx_fig = inset_left + inset_width / 2
                inset_cy_fig = inset_bot + inset_h * 0.85 / 2

                # draw arrow from scatter point to left edge of inset
                ax.annotate(
                    "",
                    xy=(1.0, inset_cy_fig),
                    xycoords=("axes fraction", "figure fraction"),
                    xytext=(px, py),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
                )
            else:
                # no frame function — just mark the point
                ax.scatter([px], [py], s=120, c="red", zorder=5, marker="*")
                ax.annotate(f"#{rank}", (px, py), fontsize=8, color="red")

    # ── save ──────────────────────────────────────────────────────────────
    tsne_dir = os.path.join(log_dir, "tsne")
    os.makedirs(tsne_dir, exist_ok=True)
    save_path = os.path.join(tsne_dir, f"step_{step:010d}.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# Task 2 — alignment scalar tracker  v1(t) and v2(t)
# =============================================================================


def compute_alignment_scalars(
    buffer_state,
    buffer,
    f_fn,  # callable: (anchor_obs, batch_obs) -> scalar jnp array
    h_fn,  # callable: (anchor_obs, batch_obs) -> scalar jnp array
    anchor_obs=None,  # jnp array (1, obs_dim); if None, sampled once from buffer
):
    """
    Sample a batch from the buffer.
    Compute v1 = mean(f(anchor, batch))  and  v2 = mean(h(anchor, batch)).
    Returns (v1: float, v2: float, anchor_obs: jnp.ndarray).

    anchor_obs is returned so you can KEEP THE SAME ANCHOR across all steps —
    pass it back in on the next call.  On the very first call, pass None and
    an anchor will be sampled automatically.

    CHANGE POINT in main.py
    -----------------------
    Called every log_freq steps inside the logging block.
    anchor_obs is stored in main() between iterations.

    f_fn and h_fn signature
    -----------------------
        def f_fn(anchor: jnp.ndarray, batch: jnp.ndarray) -> jnp.ndarray:
            # anchor: (1, obs_dim) or (1, rep_dim)
            # batch:  (B, obs_dim) or (B, rep_dim)
            # return: scalar (jnp.ndarray shape ())
            ...

    You define f_fn and h_fn in main.py (or in a separate file) and pass
    them here.  They can close over phi, actor, critic, etc.
    """
    _, batch = buffer.sample(buffer_state)
    batch_obs = batch.observation  # (B, obs_dim)

    if anchor_obs is None:
        # pick the first element of this batch as the fixed anchor
        anchor_obs = batch_obs[:1]  # (1, obs_dim)

    v1 = float(jnp.mean(f_fn(anchor_obs, batch_obs)))
    v2 = float(jnp.mean(h_fn(anchor_obs, batch_obs)))

    return v1, v2, anchor_obs
