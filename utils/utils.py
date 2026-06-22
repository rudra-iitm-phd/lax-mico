import argparse
from distutils.util import strtobool

from flax import struct


def make_static_config_from_dict(name: str, d: dict):
    """
    << copied from claude >>
    Convert a plain Python dict to an immutable Flax struct dataclass.

    Why immutable?
    ──────────────
    jax.jit traces a function once and caches the compiled XLA computation.
    If a Python dict were used as config, JAX would re-trace every time the
    dict *object* changes (even if values stay the same), or worse, it might
    cache a stale compilation if values change silently.

    By making the config a Flax struct with pytree_node=False, JAX treats
    every field as a *static* (compile-time) constant. If a value changes,
    JAX knows to re-compile. If it doesn't change, it reuses the cache.

    Usage:
        Config = make_static_config_from_dict("Config", {"lr": 3e-4})
        cfg = Config()      # instantiate
        cfg.lr              # 3e-4
    """
    annotations = {}
    defaults = {}
    for k, v in d.items():
        annotations[k] = type(v)
        defaults[k] = struct.field(default=v, pytree_node=False)
    cls = type(name, (), {"__annotations__": annotations, **defaults})
    return struct.dataclass(cls)


def sac_args():
    """Parse command-line arguments for the SAC training script."""
    from argparse import ArgumentParser

    p = ArgumentParser(description="SAC on MuJoCo Playground (JAX/NNX)")

    # reproducibility
    p.add_argument("--seed", type=int, default=0)
    # environment
    p.add_argument("--task", type=str, default="HumanoidStand-v0")
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--num_envs", type=int, default=128)
    # logging
    p.add_argument("--experiment", type=str, default="sac")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--write-terminal", type=lambda x: bool(strtobool(x)), default=True)
    # compute
    p.add_argument("--device", type=str, default="gpu", help="'gpu' or 'cpu'")
    p.add_argument("--device-id", type=int, default=0)
    # replay buffer
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-replay-size", type=int, default=int(4e6))
    p.add_argument("--warmup-samples", type=int, default=int(5e3))
    # training schedule
    p.add_argument("--total-env-steps", type=int, default=int(5e6))
    p.add_argument("--log-freq", type=int, default=int(1e3))
    p.add_argument("--save-freq", type=int, default=int(5e4))
    p.add_argument("--train-per-step", type=int, default=8)
    # SAC hyperparameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--update-tau", type=float, default=0.005)
    p.add_argument("--init-temperature", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--hidden-size", type=int, default=256)
    # eval
    p.add_argument("--eval-episode-freq", type=int, default=10)
    # transfer
    p.add_argument("--target_task", type=str, default="CheetahRun")
    p.add_argument("--transfer_freq", type=int, default=int(1e3))
    p.add_argument("--transfer_steps", type=int, default=int(10))
    p.add_argument("--grad_steps", type=int, default=int(50))
    p.add_argument("--env2_warmup", type=int, default=int(2e3))
    p.add_argument(
        "--vis-freq",
        type=int,
        default=int(1e5),
        help="Steps between t-SNE saves (Task 1). Much larger than log-freq.",
    )
    p.add_argument(
        "--n-vis-frames",
        type=int,
        default=4,
        help="Number of annotated frames on the t-SNE plot (Task 1).",
    )

    return p.parse_args(), {}
