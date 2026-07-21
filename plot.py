"""
rliable analysis for training-log CSVs like the ones you posted
(columns: Train/Steps, Loss/Loss_critic, ..., Eval/Return, ...).

ASSUMPTION: each CSV = one seed/run of the SAME config. If you're
comparing multiple configs/algorithms instead, just change ALGO_RUNS
below to map {algo_name: [list of csv paths]} for each config.

Install once:  pip install rliable seaborn matplotlib pandas numpy
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import metrics, plot_utils

# NOTE: we deliberately avoid `from rliable import library as rly` — rliable's
# get_interval_estimates depends on the `arch` package, which currently has no
# version that's simultaneously compatible with both modern pandas AND
# rliable's expected `random_state` handling. We reimplement the same
# stratified-bootstrap-over-runs logic in plain numpy instead (this is exactly
# what `arch.StratifiedBootstrap` does under the hood: resample runs with
# replacement, recompute the metric, repeat, take percentiles).


def get_interval_estimates(score_dict, func, reps=2000, ci_size=0.95, seed=0):
    """Drop-in replacement for rliable.library.get_interval_estimates that
    doesn't depend on `arch`. score_dict: {name: array of shape (num_runs, ...)}.
    func: takes an array shaped like the input, returns a 1D array of metrics.
    """
    rng = np.random.RandomState(seed)
    point_estimates, interval_estimates = {}, {}
    for key, scores in score_dict.items():
        point_estimates[key] = func(scores)
        n_runs = scores.shape[0]
        boot_vals = []
        for _ in range(reps):
            idx = rng.randint(0, n_runs, size=n_runs)  # resample runs w/ replacement
            boot_vals.append(func(scores[idx]))
        boot_vals = np.stack(boot_vals)  # (reps, num_metrics)
        alpha = (1 - ci_size) / 2
        lower = np.percentile(boot_vals, 100 * alpha, axis=0)
        upper = np.percentile(boot_vals, 100 * (1 - alpha), axis=0)
        interval_estimates[key] = np.stack([lower, upper])  # (2, num_metrics)
    return point_estimates, interval_estimates


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
ALGO_RUNS = {
    "t3": sorted(glob.glob("../rep_lr_logs/t3_hstand_*.csv")),
    "vsac": sorted(glob.glob("../rep_lr_logs/vsac_hstand_*.csv")),
}

ATTRIBUTES = [
    "Loss/Loss_critic",
    "Loss/Loss_actor",
    "SAC/Alpha",
    "Eval/Return",
]

N_POINTS = 200  # points on the shared x-axis for the curves
BOOTSTRAP_REPS = 2000  # rliable's default (50000) is slow; fine to raise later
TAIL_FRAC = 0.1  # last 10% of steps averaged = "final score" per seed
OUT_DIR = "rliable_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load + align onto a common step grid (seeds rarely log identical steps)
# ---------------------------------------------------------------------------
algo_dfs = {algo: [pd.read_csv(p) for p in paths] for algo, paths in ALGO_RUNS.items()}
all_dfs = [df for dfs in algo_dfs.values() for df in dfs]

lo = max(df["Train/Steps"].min() for df in all_dfs)
hi = min(df["Train/Steps"].max() for df in all_dfs)
grid = np.linspace(lo, hi, N_POINTS)


def stack_attribute(dfs, attr):
    """-> array of shape (num_runs, num_points), interpolated onto `grid`."""
    return np.stack([np.interp(grid, df["Train/Steps"], df[attr]) for df in dfs])


def final_scores(dfs, attr, tail_frac=TAIL_FRAC):
    """-> array of shape (num_runs,): mean of the last tail_frac of each run."""
    out = []
    for df in dfs:
        n = max(1, int(len(df) * tail_frac))
        out.append(df[attr].tail(n).mean())
    return np.array(out)


# ---------------------------------------------------------------------------
# 3. Sample-efficiency curves: IQM + 95% CI band across seeds, per attribute
#    rliable wants shape (num_runs, num_tasks, num_points); num_tasks=1 here.
# ---------------------------------------------------------------------------
iqm_over_time = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[..., t]) for t in range(scores.shape[-1])]
)

for attr in ATTRIBUTES:
    score_dict = {
        algo: stack_attribute(dfs, attr)[:, None, :] for algo, dfs in algo_dfs.items()
    }
    point_est, interval_est = get_interval_estimates(
        score_dict, iqm_over_time, reps=BOOTSTRAP_REPS
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_utils.plot_sample_efficiency_curve(
        grid,
        point_est,
        interval_est,
        algorithms=list(ALGO_RUNS.keys()),
        xlabel="Training Steps",
        ylabel=f"{attr} (IQM)",
        ax=ax,
    )
    safe_name = attr.replace("/", "_")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{safe_name}_curve.png"), dpi=150)
    plt.close(fig)
    print(f"saved {safe_name}_curve.png")

# ---------------------------------------------------------------------------
# 4. Final-performance bars: IQM/Median/Mean + CI, per attribute
#    rliable wants shape (num_runs, num_tasks); num_tasks=1 here.
# ---------------------------------------------------------------------------
agg_funcs = {
    "IQM": metrics.aggregate_iqm,
    "Median": metrics.aggregate_median,
    "Mean": metrics.aggregate_mean,
}
agg_all = lambda s: np.array([f(s) for f in agg_funcs.values()])

for attr in ATTRIBUTES:
    score_dict = {
        algo: final_scores(dfs, attr)[:, None] for algo, dfs in algo_dfs.items()
    }
    point_est, interval_est = get_interval_estimates(
        score_dict, agg_all, reps=BOOTSTRAP_REPS
    )

    # plot_interval_estimates builds its own figure (no `ax` kwarg) and returns it
    fig, axes = plot_utils.plot_interval_estimates(
        point_est,
        interval_est,
        metric_names=list(agg_funcs.keys()),
        algorithms=list(ALGO_RUNS.keys()),
        xlabel=attr,
    )
    safe_name = attr.replace("/", "_")
    fig.savefig(
        os.path.join(OUT_DIR, f"{safe_name}_final_agg.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"saved {safe_name}_final_agg.png")
