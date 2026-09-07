#!/usr/bin/env python3
"""
Compare RL algorithms with rliable and emit publication-quality figures.

Expects logs laid out as:

    <logs_dir>/<Env>/<algo>/seed-<NNN>-<timestamp>/progress.csv

with a step column (default "Train/Steps") and a metric column
(default "Eval/Return").

Examples
--------
# single environment
python rliable_compare.py --logs-dir logs_fresh --envs BallInCup

# aggregate across every environment (the comparison rliable is designed for)
python rliable_compare.py --logs-dir logs_fresh --envs all

# camera-ready names, mild smoothing, PDF only
python rliable_compare.py --logs-dir logs_fresh --envs all \
    --label algo3="Ours" --label dhpg_fix2="DHPG" --smooth 3 --formats pdf
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


def _patch_arch_for_rliable():
    """Make rliable work with arch >= 8.

    rliable 1.2.0 pins arch<8 and calls IIDBootstrap.__init__ with
    ``random_state=...``. arch 8 renamed that argument to ``seed`` and treats
    any unknown keyword as bootstrap *data*, so the call dies with
    "Input `random_state` has type <class 'NoneType'>". Translate the old name
    to the new one; on arch < 8 this is a no-op.
    """
    import inspect

    import arch.bootstrap.base as arch_base

    params = inspect.signature(arch_base.IIDBootstrap.__init__).parameters
    if "random_state" in params or "seed" not in params:
        return False

    original_init = arch_base.IIDBootstrap.__init__

    def init(self, *args, random_state=None, **kwargs):
        if random_state is not None:
            kwargs.setdefault("seed", random_state)
        original_init(self, *args, **kwargs)

    arch_base.IIDBootstrap.__init__ = init
    return True


_PATCHED_ARCH = _patch_arch_for_rliable()

from rliable import library as rly
from rliable import metrics

SEED_RE = re.compile(r"seed-(\d+)")

# Palette, markers and grid matched to the reference figures.
PALETTE = [
    "#e69f00",
    "#7a8f00",
    "#b57bb3",
    "#d97662",
    "#0072b2",
    "#009e73",
    "#56b4e9",
    "#cc79a7",
]
MARKERS = ["s", "D", "p", "^", "o", "v", "P", "X"]

SERIF = [
    "Times New Roman",
    "Nimbus Roman",
    "Liberation Serif",
    "STIXGeneral",
    "DejaVu Serif",
]
SANS = ["Helvetica", "Nimbus Sans", "Liberation Sans", "Arial", "DejaVu Sans"]


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def setup_style(args):
    """Matplotlib defaults matching the reference figure style."""
    serif = args.font == "serif"
    plt.rcParams.update(
        {
            "font.family": "serif" if serif else "sans-serif",
            "font.serif": SERIF,
            "font.sans-serif": SANS,
            "mathtext.fontset": "stix" if serif else "dejavusans",
            "font.size": args.font_size,
            "axes.titlesize": args.font_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size - 1,
            "ytick.labelsize": args.font_size - 1,
            "legend.fontsize": args.font_size,
            # full box around a very light panel
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "axes.axisbelow": True,
            # dashed grey grid; at lw 1.2 the "--" dashes render as 4.44/1.92
            "grid.color": "#b0b0b0",
            "grid.linestyle": "--",
            "grid.linewidth": 1.2,
            "grid.alpha": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.edgecolor": "#cccccc",
            "legend.facecolor": "white",
            "legend.framealpha": 1.0,
            "legend.borderpad": 0.5,
            "legend.handlelength": 1.9,
            "legend.columnspacing": 1.6,
            "legend.handletextpad": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": args.dpi,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            # embed TrueType rather than Type 3, which paper checkers reject
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    warnings.filterwarnings("ignore", message="findfont")


def save(fig, out_dir, name, formats, tight=True):
    paths = []
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        # bbox_inches="tight" crops surrounding whitespace, which would undo any
        # margin reserved via apply_margins(); disable it when margins are set
        fig.savefig(path, bbox_inches="tight" if tight else None)
        paths.append(path)
    plt.close(fig)
    return paths


def apply_margins(fig, args):
    """Reserve left/right margin inside the canvas. Returns whether to crop."""
    width = fig.get_figwidth()
    left = args.margin_left / width
    right = args.margin_right / width
    if left <= 0 and right <= 0:
        return True
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(left, 0.0, 1.0 - left - right, 1.0))
    return False


def style_map(algos, args):
    """Stable colour/marker assignment, honouring --color overrides."""
    colors, markers = {}, {}
    for i, algo in enumerate(algos):
        colors[algo] = args.color_overrides.get(algo, PALETTE[i % len(PALETTE)])
        markers[algo] = MARKERS[i % len(MARKERS)]
    return colors, markers


def marker_every(n_points, i, n_algos, per_curve=7):
    """Space markers out and stagger them so curves stay readable."""
    step = max(n_points // per_curve, 1)
    return (int(i * step / max(n_algos, 1)), step)


def step_scale(max_step):
    """Factor the step axis, e.g. (1e6, r"($\\times 10^6$)")."""
    if max_step >= 1e6:
        return 1e6, r"($\times 10^{6}$)"
    if max_step >= 1e3:
        return 1e3, r"($\times 10^{3}$)"
    return 1.0, ""


def pretty_env(env):
    """BallInCup -> Ball In Cup; FingerTurnHard -> Finger Turn Hard."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", env)


# --------------------------------------------------------------------------- #
# discovery + loading
# --------------------------------------------------------------------------- #
def discover(logs_dir, envs, algos):
    """-> {env: {algo: {seed_id: run_dir}}}, keeping the newest dir per seed."""
    if envs == ["all"]:
        envs = sorted(
            d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))
        )

    tree = {}
    for env in envs:
        env_dir = os.path.join(logs_dir, env)
        if not os.path.isdir(env_dir):
            raise FileNotFoundError(f"no such environment directory: {env_dir}")

        env_algos = algos or sorted(
            d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))
        )

        tree[env] = {}
        for algo in env_algos:
            algo_dir = os.path.join(env_dir, algo)
            if not os.path.isdir(algo_dir):
                warnings.warn(f"[{env}] missing algo directory: {algo}")
                continue

            by_seed = defaultdict(list)
            for run_dir in sorted(glob.glob(os.path.join(algo_dir, "seed-*"))):
                if not os.path.isfile(os.path.join(run_dir, "progress.csv")):
                    warnings.warn(f"no progress.csv in {run_dir}, skipping")
                    continue
                m = SEED_RE.search(os.path.basename(run_dir))
                if m is None:
                    warnings.warn(f"cannot parse seed id from {run_dir}, skipping")
                    continue
                by_seed[int(m.group(1))].append(run_dir)

            runs = {}
            for seed, dirs in by_seed.items():
                if len(dirs) > 1:
                    warnings.warn(
                        f"[{env}/{algo}] seed {seed:03d} has {len(dirs)} runs; "
                        f"using the most recent ({os.path.basename(sorted(dirs)[-1])})"
                    )
                runs[seed] = sorted(dirs)[-1]

            if runs:
                tree[env][algo] = dict(sorted(runs.items()))
            else:
                warnings.warn(f"[{env}] no usable runs for algo {algo}")

        if not tree[env]:
            raise RuntimeError(f"no runs found for environment {env}")
    return tree


def load_run(run_dir, args):
    """Read one progress.csv -> (steps, raw values, smoothed values)."""
    csv_path = os.path.join(run_dir, "progress.csv")
    df = pd.read_csv(csv_path)

    for col in (args.step_col, args.metric_col):
        if col not in df.columns:
            raise KeyError(
                f"{csv_path} has no column {col!r}. Available: {list(df.columns)}"
            )

    df = df[[args.step_col, args.metric_col]].dropna()
    df = df.sort_values(args.step_col).drop_duplicates(
        subset=args.step_col, keep="last"
    )

    steps = df[args.step_col].to_numpy(dtype=np.float64)
    vals = df[args.metric_col].to_numpy(dtype=np.float64)

    if args.drop_leading_zeros:
        nz = np.flatnonzero(vals != 0.0)
        if nz.size:
            steps, vals = steps[nz[0] :], vals[nz[0] :]

    if steps.size == 0:
        raise ValueError(f"{csv_path}: no usable rows for {args.metric_col!r}")

    if args.smooth > 1:
        smooth = (
            pd.Series(vals)
            .rolling(args.smooth, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
    else:
        smooth = vals
    return steps, vals, smooth


# --------------------------------------------------------------------------- #
# score matrices
# --------------------------------------------------------------------------- #
def build_matrices(tree, args):
    """
    final  : {algo: (n_seeds, n_tasks)}              end-of-training score
    curves : {algo: (n_seeds, n_tasks, n_points)}    learning curves
    frames : (n_points,) shared step grid
    tasks  : env names in matrix-column order
    """
    found = {a for env in tree for a in tree[env]}
    # --algos fixes the display order (legend, colour assignment, table rows);
    # without it, fall back to alphabetical
    algos = [a for a in args.algos if a in found] if args.algos else sorted(found)
    tasks = sorted(tree)

    n_seeds = None
    for env in tasks:
        for algo in algos:
            if algo not in tree[env]:
                raise RuntimeError(
                    f"algo {algo!r} is missing for env {env!r}; either rerun it "
                    f"or restrict the comparison with --algos"
                )
            k = len(tree[env][algo])
            if n_seeds is None:
                n_seeds = k
            elif k != n_seeds:
                raise RuntimeError(
                    f"seed count mismatch: {env}/{algo} has {k} seeds, expected "
                    f"{n_seeds}. rliable needs an equal number of runs per cell."
                )

    raw = {}
    max_step = np.inf
    for env in tasks:
        for algo in algos:
            runs = []
            for seed, run_dir in tree[env][algo].items():
                steps, vals, smooth = load_run(run_dir, args)
                runs.append((steps, vals, smooth))
                max_step = min(max_step, steps[-1])
            raw[(algo, env)] = runs

    if args.max_steps is not None:
        max_step = min(max_step, args.max_steps)
    frames = np.linspace(0.0, max_step, args.curve_points)

    final, curves = {}, {}
    for algo in algos:
        f = np.zeros((n_seeds, len(tasks)))
        c = np.zeros((n_seeds, len(tasks), args.curve_points))
        for j, env in enumerate(tasks):
            for i, (steps, vals, smooth) in enumerate(raw[(algo, env)]):
                keep = steps <= max_step + 1e-9
                s, v, sm = steps[keep], vals[keep], smooth[keep]
                w = min(args.final_window, v.size)
                f[i, j] = v[-w:].mean()  # unsmoothed
                c[i, j] = np.interp(frames, s, sm, left=sm[0], right=sm[-1])
        final[algo] = f / args.max_score
        curves[algo] = c / args.max_score

    return final, curves, frames, tasks


def normalize_curves(c, mode, eps=None):
    """Rescale a (runs, tasks, frames) array so tasks can be pooled.

    "none"            raw units; only meaningful within one environment
    "per_env_minmax"  each task mapped to [0, 1] by its own min/max
    "per_env_max"     each task divided by its own max (keeps zero fixed)
    "per_env_final"   each task divided by its own mean final value
    "per_env_log"     min/max in log10 space, for values spanning decades

    The two modes below rescale nothing per task, so differences in magnitude
    BETWEEN series survive; use them to show that one quantity sits above
    another while still fitting both on one axis.

    "log"             log10 of the values, non-positives clipped to `eps`
    "symlog"          sign(x) * log10(1 + |x|/eps); handles negatives and zero
    """
    c = np.asarray(c, dtype=float)
    if mode == "none":
        return c
    if mode in ("log", "log10"):
        positive = c[c > 0]
        floor = eps if eps is not None else (positive.min() if positive.size else 1e-12)
        n_bad = int((c <= 0).sum())
        if n_bad:
            warnings.warn(
                f"log: clipped {n_bad} non-positive value(s) to "
                f"{floor:g}; pass eps= to control the floor, or use "
                f'mode="symlog" if the metric is genuinely signed'
            )
        return np.log10(np.maximum(c, floor))
    if mode == "symlog":
        scale = eps if eps is not None else 1.0
        return np.sign(c) * np.log10(1.0 + np.abs(c) / scale)
    out = np.empty_like(c, dtype=float)
    for j in range(c.shape[1]):
        block = c[:, j, :]
        if mode == "per_env_minmax":
            lo, hi = block.min(), block.max()
            out[:, j, :] = (block - lo) / (hi - lo) if hi > lo else 0.0
        elif mode == "per_env_max":
            hi = np.abs(block).max()
            out[:, j, :] = block / hi if hi > 0 else 0.0
        elif mode == "per_env_final":
            ref = block[:, -1].mean()
            out[:, j, :] = block / ref if ref != 0 else 0.0
        elif mode == "per_env_log":
            pos = block[block > 0]
            floor = pos.min() if pos.size else 1e-12
            lg = np.log10(np.maximum(block, floor))
            lo, hi = lg.min(), lg.max()
            out[:, j, :] = (lg - lo) / (hi - lo) if hi > lo else 0.0
        else:
            raise ValueError(f"unknown normalize mode {mode!r}")
    return out


def load_series(args, specs, algos=None, normalize="per_env_minmax", eps=None):
    """Load several metric columns as separate series, for one or more algos.

    specs : list of metric column names, or of (column, display label) pairs
    algos : restrict to these algorithms; default = args.algos
    normalize : passed to normalize_curves; "none" keeps raw units and
        "log" keeps the relative magnitude of the series
    eps : floor for "log", or the linear scale for "symlog"

    Returns (series, raw, frames, tasks) where `series` and `raw` are
    {label: (runs, tasks, frames)} dicts, normalized and unnormalized. Feed
    `series` straight to fig_grid or fig_sample_efficiency in place of the
    per-algorithm curves.
    """
    import copy

    pairs = [
        (sp, sp.split("/")[-1]) if isinstance(sp, str) else tuple(sp) for sp in specs
    ]
    algos = algos or args.algos
    series, raw, frames, tasks = {}, {}, None, None

    for column, label in pairs:
        sub = copy.copy(args)
        sub.metric_col = column
        sub.algos = algos
        sub.max_score = 1.0  # these columns are not returns
        tree = discover(sub.logs_dir, sub.envs, sub.algos)
        _, curves, fr, tk = build_matrices(tree, sub)
        if frames is None:
            frames, tasks = fr, tk
        elif not np.allclose(fr, frames):
            raise RuntimeError("metrics have different step grids")
        for algo in curves:
            key = label if len(curves) == 1 else f"{label} ({algo})"
            raw[key] = curves[algo]
            series[key] = normalize_curves(curves[algo], normalize, eps)
    return series, raw, frames, tasks


def settling_steps(series, frames, tasks, frac=0.9, final_window=5):
    """Step at which each run first reaches `frac` of its own final level.

    Progress is measured relative to the run's own start and end, so this works
    for quantities that rise and for ones that fall. Returns a tidy DataFrame
    with one row per (series, env, run).
    """
    rows = []
    for name, c in series.items():
        for j, env in enumerate(tasks):
            for i in range(c.shape[0]):
                v = c[i, j]
                start = v[0]
                end = v[-final_window:].mean()
                if np.isclose(end, start):
                    step = np.nan
                else:
                    progress = (v - start) / (end - start)
                    hit = np.flatnonzero(progress >= frac)
                    step = frames[hit[0]] if hit.size else np.nan
                rows.append(dict(series=name, env=env, run=i, step=step))
    return pd.DataFrame(rows)


def settling_summary(df, frac=0.9):
    """Median settling step per series, with the inter-run spread."""
    out = df.groupby("series")["step"].agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        n_missing=lambda s: int(s.isna().sum()),
    )
    out.attrs["frac"] = frac
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
AGG_NAMES = ["Median", "IQM", "Mean", "Optimality Gap"]


def iqm_over_frames(scores):
    """IQM at each point of a (runs, tasks, frames) array."""
    return np.array(
        [metrics.aggregate_iqm(scores[..., t]) for t in range(scores.shape[-1])]
    )


def aggregate_func(x):
    return np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )


def outside_legend(
    fig,
    algos,
    labels,
    colors,
    markers,
    ncol=None,
    fontsize=None,
    marker_size=6.5,
    line_width=2.0,
):
    """Framed legend floating above the axes, as in the reference figures."""
    handles = [
        Line2D(
            [],
            [],
            color=colors[a],
            marker=markers[a],
            lw=line_width,
            ms=marker_size,
            label=labels[a],
        )
        for a in algos
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol or min(len(algos), 4),
        fontsize=fontsize or plt.rcParams["legend.fontsize"],
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(h_pad=0.06)


def fig_aggregate_metrics(scores, cis, algos, labels, colors, markers, args, tag):
    """Point estimate + 95% CI per aggregate metric, one panel each."""
    n = len(algos)
    height = max(0.42 * n + 1.05, 1.6)
    fig, axes = plt.subplots(
        1,
        len(AGG_NAMES),
        figsize=(args.fig_width, height),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    ypos = np.arange(n)[::-1]  # first algorithm at the top

    for k, (ax, name) in enumerate(zip(axes, AGG_NAMES)):
        lo_all, hi_all = [], []
        for y, algo in zip(ypos, algos):
            pt = scores[algo][k]
            lo, hi = cis[algo][0, k], cis[algo][1, k]
            ax.errorbar(
                pt,
                y,
                xerr=[[pt - lo], [hi - pt]],
                color=colors[algo],
                marker=markers[algo],
                ms=7,
                lw=2.0,
                capsize=3.5,
                capthick=2.0,
                zorder=3,
            )
            lo_all.append(lo)
            hi_all.append(hi)

        span = max(hi_all) - min(lo_all)
        pad = 0.32 * span if span > 0 else 0.05
        ax.set_xlim(min(lo_all) - pad, max(hi_all) + pad)
        ax.set_ylim(-0.6, n - 0.4)
        arrow = r"$\downarrow$" if name == "Optimality Gap" else r"$\uparrow$"
        ax.set_title(f"{name} {arrow}", pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"] - 2)
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([labels[a] for a in algos])
    fig.supxlabel(f"Normalized {tag}", fontsize=plt.rcParams["axes.labelsize"])
    return fig


def fig_performance_profile(
    prof, prof_cis, taus, algos, labels, colors, markers, args, tag
):
    fig, ax = plt.subplots(
        figsize=(args.fig_width * 0.52, args.fig_width * 0.42), constrained_layout=True
    )
    for i, algo in enumerate(algos):
        ax.plot(
            taus,
            prof[algo],
            color=colors[algo],
            marker=markers[algo],
            markevery=marker_every(len(taus), i, len(algos)),
            zorder=3,
        )
        ax.fill_between(
            taus,
            prof_cis[algo][0],
            prof_cis[algo][1],
            color=colors[algo],
            alpha=0.3,
            lw=0,
            zorder=2,
        )
    ax.set_xlabel(rf"Normalized {tag} $(\tau)$")
    ax.set_ylabel(r"Fraction of runs $> \tau$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(taus[0], taus[-1])
    outside_legend(fig, algos, labels, colors, markers)
    return fig


def fig_sample_efficiency(
    frames,
    iqm,
    iqm_cis,
    algos,
    labels,
    colors,
    markers,
    args,
    tag,
    logx=False,
    logy=False,
):
    fig, ax = plt.subplots(
        figsize=(args.fig_width * 0.52, args.fig_width * 0.42), constrained_layout=True
    )
    div, suffix = (1.0, "") if logx else step_scale(frames[-1])
    x = frames / div
    for i, algo in enumerate(algos):
        ax.plot(
            x,
            iqm[algo],
            color=colors[algo],
            marker=markers[algo],
            markevery=marker_every(len(x), i, len(algos)),
            zorder=3,
        )
        ax.fill_between(
            x,
            iqm_cis[algo][0],
            iqm_cis[algo][1],
            color=colors[algo],
            alpha=0.3,
            lw=0,
            zorder=2,
        )
    if logx:
        ax.set_xscale("log")
        positive = x[x > 0]
        ax.set_xlim(positive.min() if positive.size else 1, x[-1])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(0, x[-1])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(f"environment steps {suffix}".strip())
    ax.set_ylabel(f"IQM normalized {tag}")
    outside_legend(fig, algos, labels, colors, markers)
    return fig


def fig_probability_of_improvement(poi_rows, labels, colors, markers, args):
    n = len(poi_rows)
    fig, ax = plt.subplots(
        figsize=(args.fig_width * 0.52, max(0.45 * n + 0.75, 1.1)),
        constrained_layout=True,
    )
    ypos = np.arange(n)[::-1]
    for y, row in zip(ypos, poi_rows):
        color = colors[row["x"]]
        p = row["p_improvement"]
        ax.errorbar(
            p,
            y,
            xerr=[[p - row["ci_low"]], [row["ci_high"] - p]],
            color=color,
            marker=markers[row["x"]],
            ms=7,
            lw=2.0,
            capsize=3.5,
            capthick=2.0,
            zorder=3,
        )
    ax.axvline(0.5, color="#555555", lw=1.2, ls="--", zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{labels[r['x']]} vs. {labels[r['y']]}" for r in poi_rows])
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"P(X $>$ Y)")
    ax.grid(axis="y", visible=False)
    return fig


def fig_grid(
    curves,
    frames,
    tasks,
    algos,
    labels,
    colors,
    markers,
    *,
    tag="Return",
    # --- which panels, and in what order --------------------------
    envs=None,
    titles=None,
    ncols=4,
    # --- geometry, all in inches ----------------------------------
    panel_width=1.35,
    panel_height=1.45,
    extra_width=0.75,
    extra_height=0.95,
    w_pad=0.04,
    h_pad=0.04,
    margin_left=0.0,
    margin_right=0.0,
    # --- what the lines look like ---------------------------------
    line_width=1.8,
    marker_size=5.5,
    markers_per_curve=5,
    show_markers=True,
    band="sem",
    band_alpha=0.3,
    # --- text sizes; None inherits the global style ---------------
    title_size=None,
    tick_size=None,
    label_size=None,
    legend_size=None,
    # --- axes -----------------------------------------------------
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xticks=3,
    yticks=None,
    sharey=False,
    logx=False,
    logy=False,
    # --- legend ---------------------------------------------------
    legend="above",
    legend_ncol=None,
    legend_loc="lower right",
):
    """Per-environment grid of learning curves, with every knob exposed.

    curves : {algo: (runs, tasks, frames)}, already normalized
    envs   : subset and ORDER of environments to draw; default = all of `tasks`
    titles : {env: display title}; default prettifies the directory name
    band   : "sem", "std", "minmax" or None
    ylim   : (lo, hi) applied to every panel, or {env: (lo, hi)}
    yticks : approximate tick count, or {env: count}; None picks by panel width
    legend : "above" (floating, framed), "inside" (first panel), or None

    Returns the figure; nothing is written to disk.
    """
    envs = list(envs) if envs is not None else list(tasks)
    missing = [e for e in envs if e not in tasks]
    if missing:
        raise KeyError(f"unknown environment(s): {missing}. Available: {tasks}")
    col_of = {env: tasks.index(env) for env in envs}

    n = len(envs)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    title_size = title_size or plt.rcParams["axes.titlesize"]
    tick_size = tick_size or plt.rcParams["xtick.labelsize"]
    label_size = label_size or plt.rcParams["axes.labelsize"]
    legend_size = legend_size or plt.rcParams["legend.fontsize"]
    if yticks is None:
        # narrow panels cannot carry four y ticks without the labels colliding
        yticks = 3 if panel_width < 1.6 else 4

    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=sharey,
        squeeze=False,
        constrained_layout=True,
        figsize=(
            ncols * panel_width + extra_width,
            nrows * panel_height + extra_height,
        ),
    )

    # on a log axis the (x10^6) factoring turns ticks into 10^-1, 10^0; plot
    # raw steps instead and let the log ticks speak for themselves
    div, suffix = (1.0, "") if logx else step_scale(frames[-1])
    x = frames / div

    for idx, env in enumerate(envs):
        ax = axes[idx // ncols][idx % ncols]
        j = col_of[env]
        for i, algo in enumerate(algos):
            runs = curves[algo][:, j, :]
            mean = runs.mean(axis=0)
            ax.plot(
                x,
                mean,
                color=colors[algo],
                lw=line_width,
                marker=markers[algo] if show_markers else None,
                ms=marker_size,
                markevery=marker_every(len(x), i, len(algos), markers_per_curve),
                zorder=3,
            )
            if band and runs.shape[0] > 1:
                if band == "sem":
                    d = runs.std(axis=0, ddof=1) / np.sqrt(runs.shape[0])
                    lo, hi = mean - d, mean + d
                elif band == "std":
                    d = runs.std(axis=0, ddof=1)
                    lo, hi = mean - d, mean + d
                elif band == "minmax":
                    lo, hi = runs.min(axis=0), runs.max(axis=0)
                else:
                    raise ValueError(f"unknown band {band!r}")
                ax.fill_between(
                    x, lo, hi, color=colors[algo], alpha=band_alpha, lw=0, zorder=2
                )

        title = (titles or {}).get(env, pretty_env(env))
        ax.set_title(title, fontsize=title_size, pad=4)
        ax.tick_params(labelsize=tick_size)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=xticks))
        nb = yticks[env] if isinstance(yticks, dict) else yticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nb))
        if logx:
            ax.set_xscale("log")
            # a log axis cannot include 0; start at the first recorded step
            positive = x[x > 0]
            ax.set_xlim(
                xlim
                if xlim is not None
                else (positive.min() if positive.size else 1, x[-1])
            )
        else:
            ax.set_xlim(xlim if xlim is not None else (0, x[-1]))
        if logy:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim[env] if isinstance(ylim, dict) else ylim)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    # sharex suppresses tick labels on every non-bottom row, which hides them
    # on the last visible panel of any column the final row does not reach
    for col in range(ncols):
        rows_in_col = [r for r in range(nrows) if r * ncols + col < n]
        if rows_in_col:
            axes[rows_in_col[-1]][col].tick_params(labelbottom=True)

    fig.supxlabel(
        xlabel if xlabel is not None else f"environment steps {suffix}".strip(),
        fontsize=label_size,
    )
    fig.supylabel(
        ylabel if ylabel is not None else f"Normalized {tag}", fontsize=label_size
    )

    if legend == "above":
        outside_legend(
            fig,
            algos,
            labels,
            colors,
            markers,
            ncol=legend_ncol,
            fontsize=legend_size,
            marker_size=marker_size,
            line_width=line_width,
        )
    elif legend == "inside":
        handles = [
            Line2D(
                [],
                [],
                color=colors[a],
                lw=line_width,
                marker=markers[a] if show_markers else None,
                ms=marker_size,
                label=labels[a],
            )
            for a in algos
        ]
        axes[0][0].legend(handles=handles, loc=legend_loc, fontsize=legend_size)
    elif legend is not None:
        raise ValueError(f"unknown legend {legend!r}")

    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=w_pad, h_pad=h_pad)
        width = fig.get_figwidth()
        left, right = margin_left / width, margin_right / width
        if left > 0 or right > 0:
            engine.set(rect=(left, 0.0, 1.0 - left - right, 1.0))
    return fig


def fig_per_env_curves(
    curves, frames, tasks, algos, labels, colors, markers, args, tag
):
    """CLI wrapper: fig_grid with the geometry taken from `args`."""
    ncols = max(1, min(args.grid_cols, len(tasks)))
    panel_w = args.panel_width or (args.fig_width - 0.75) / ncols
    pad = args.panel_pad if args.panel_pad is not None else 0.04
    return fig_grid(
        curves,
        frames,
        tasks,
        algos,
        labels,
        colors,
        markers,
        tag=tag,
        ncols=ncols,
        panel_width=panel_w,
        panel_height=args.panel_height,
        w_pad=pad,
        h_pad=pad,
        margin_left=args.margin_left,
        margin_right=args.margin_right,
    )


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def run_analysis(final, curves, frames, tasks, args):
    os.makedirs(args.out_dir, exist_ok=True)
    algos = list(final)
    np.random.seed(args.seed)  # arch>=8 dropped the random_state kwarg rliable used
    tag = args.metric_col.split("/")[-1]
    labels = {a: args.labels.get(a, a) for a in algos}
    colors, markers = style_map(algos, args)
    written = []

    # ---- 1. aggregate metrics + stratified bootstrap CIs -------------------
    print(
        f"\n=== Aggregate metrics (stratified bootstrap, {args.reps} reps, 95% CI) ==="
    )
    agg_scores, agg_cis = rly.get_interval_estimates(
        final, aggregate_func, reps=args.reps
    )
    rows = []
    for algo in algos:
        for k, name in enumerate(AGG_NAMES):
            rows.append(
                dict(
                    algorithm=algo,
                    label=labels[algo],
                    metric=name,
                    point=agg_scores[algo][k],
                    ci_low=agg_cis[algo][0, k],
                    ci_high=agg_cis[algo][1, k],
                )
            )
    agg_df = pd.DataFrame(rows)
    for algo in algos:
        print(f"\n{labels[algo]}")
        for _, r in agg_df[agg_df.algorithm == algo].iterrows():
            print(
                f"  {r.metric:<15} {r.point:7.4f}  [{r.ci_low:7.4f}, {r.ci_high:7.4f}]"
            )
    agg_df.to_csv(
        os.path.join(args.out_dir, f"aggregate_metrics_{tag}.csv"), index=False
    )
    written += save(
        fig_aggregate_metrics(
            agg_scores, agg_cis, algos, labels, colors, markers, args, tag
        ),
        args.out_dir,
        f"aggregate_metrics_{tag}",
        args.formats,
    )

    # ---- 2. performance profiles ------------------------------------------
    lo = float(min(m.min() for m in final.values()))
    hi = float(max(m.max() for m in final.values()))
    taus = np.linspace(min(lo, 0.0), hi * 1.02 + 1e-8, 81)
    prof, prof_cis = rly.create_performance_profile(final, taus, reps=args.reps)
    written += save(
        fig_performance_profile(
            prof, prof_cis, taus, algos, labels, colors, markers, args, tag
        ),
        args.out_dir,
        f"performance_profile_{tag}",
        args.formats,
    )

    # ---- 3. sample efficiency (IQM over runs x tasks at each step) ---------
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        curves, iqm_over_frames, reps=max(args.reps // 10, 200)
    )
    written += save(
        fig_sample_efficiency(
            frames, iqm_scores, iqm_cis, algos, labels, colors, markers, args, tag
        ),
        args.out_dir,
        f"sample_efficiency_{tag}",
        args.formats,
    )

    pd.DataFrame(
        [
            dict(
                algorithm=algo,
                step=fr,
                iqm=iqm_scores[algo][t],
                ci_low=iqm_cis[algo][0, t],
                ci_high=iqm_cis[algo][1, t],
            )
            for algo in algos
            for t, fr in enumerate(frames)
        ]
    ).to_csv(os.path.join(args.out_dir, f"sample_efficiency_{tag}.csv"), index=False)

    # ---- 4. probability of improvement ------------------------------------
    poi_rows = []
    if len(algos) >= 2:
        pairs = {
            f"{x},{y}": (final[x], final[y])
            for i, x in enumerate(algos)
            for y in algos[i + 1 :]
        }
        poi, poi_cis = rly.get_interval_estimates(
            pairs, metrics.probability_of_improvement, reps=args.reps
        )
        print("\n=== P(X > Y): a random X run beats a random Y run ===")
        for key in pairs:
            x, y = key.split(",")
            p = float(np.ravel(poi[key])[0])
            ci = np.ravel(poi_cis[key])
            print(
                f"  P({labels[x]} > {labels[y]}) = {p:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            )
            poi_rows.append(
                dict(
                    x=x, y=y, p_improvement=p, ci_low=float(ci[0]), ci_high=float(ci[1])
                )
            )
        pd.DataFrame(poi_rows).to_csv(
            os.path.join(args.out_dir, f"probability_of_improvement_{tag}.csv"),
            index=False,
        )
        written += save(
            fig_probability_of_improvement(poi_rows, labels, colors, markers, args),
            args.out_dir,
            f"probability_of_improvement_{tag}",
            args.formats,
        )

    # ---- 5. per-environment learning curves -------------------------------
    grid = fig_per_env_curves(
        curves, frames, tasks, algos, labels, colors, markers, args, tag
    )
    written += save(
        grid,
        args.out_dir,
        f"per_env_curves_{tag}",
        args.formats,
        tight=not (args.margin_left or args.margin_right),
    )

    # ---- 6. per-run table + LaTeX -----------------------------------------
    raw_df = pd.DataFrame(
        [
            dict(
                algorithm=algo,
                env=env,
                run_index=i,
                normalized_score=final[algo][i, j],
                raw_score=final[algo][i, j] * args.max_score,
            )
            for algo in algos
            for i in range(final[algo].shape[0])
            for j, env in enumerate(tasks)
        ]
    )
    raw_df.to_csv(os.path.join(args.out_dir, f"per_run_final_{tag}.csv"), index=False)
    print(f"\n=== Per-run final {tag} (mean of last {args.final_window} evals) ===")
    print(
        raw_df.pivot_table(
            index=["env", "run_index"], columns="algorithm", values="raw_score"
        )
        .round(2)
        .to_string()
    )

    tex = os.path.join(args.out_dir, f"aggregate_metrics_{tag}.tex")
    write_latex_table(agg_df, algos, labels, tex)
    written.append(tex)

    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump(
            {
                "aggregate": agg_df.to_dict("records"),
                "probability_of_improvement": poi_rows,
                "tasks": tasks,
                "algorithms": algos,
            },
            fh,
            indent=2,
        )

    print(
        f"\nWrote {len(written)} figure/table files to {os.path.abspath(args.out_dir)}"
    )
    print(
        "Per-environment panels show the mean over seeds with a ±1 s.e.m. "
        "band; every other interval is a 95% stratified bootstrap CI."
    )


def write_latex_table(agg_df, algos, labels, path):
    r"""A booktabs table of point estimates with CIs, ready to \input."""
    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{tabular}{l" + "c" * len(AGG_NAMES) + "}",
        r"\toprule",
        "Method & "
        + " & ".join(
            n + (r" $\downarrow$" if n == "Optimality Gap" else r" $\uparrow$")
            for n in AGG_NAMES
        )
        + r" \\",
        r"\midrule",
    ]
    for algo in algos:
        sub = agg_df[agg_df.algorithm == algo].set_index("metric")
        cells = []
        for name in AGG_NAMES:
            r = sub.loc[name]
            cells.append(
                f"{r.point:.3f} "
                rf"{{\scriptsize [{r.ci_low:.3f}, {r.ci_high:.3f}]}}"
            )
        lines.append(f"{labels[algo]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def parse_kv(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"error: expected ALGO=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="rliable comparison with publication-quality figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--logs-dir", default="logs_fresh")
    p.add_argument(
        "--envs", nargs="+", default=["all"], help="environment names, or 'all'"
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=None,
        help="algorithms to compare, in the order they should appear "
        "in legends and tables (default: every subdirectory, "
        "alphabetically)",
    )
    p.add_argument("--step-col", default="Train/Steps")
    p.add_argument("--metric-col", default="Eval/Return")
    p.add_argument(
        "--max-score",
        type=float,
        default=1000.0,
        help="divide scores by this; 1000 = DMC max return",
    )
    p.add_argument(
        "--final-window",
        type=int,
        default=5,
        help="average the last N eval points as the final score",
    )
    p.add_argument(
        "--max-steps",
        type=float,
        default=None,
        help="truncate all runs at this step count",
    )
    p.add_argument(
        "--curve-points",
        type=int,
        default=101,
        help="resolution of the shared step grid",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="centred rolling-mean window for curves (1 = off)",
    )
    p.add_argument(
        "--drop-leading-zeros",
        action="store_true",
        help="strip leading zero-valued evals (placeholder rows)",
    )
    p.add_argument("--reps", type=int, default=20000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="rliable_out")

    g = p.add_argument_group("figure style")
    g.add_argument(
        "--label",
        action="append",
        dest="label_pairs",
        default=[],
        metavar="ALGO=NAME",
        help="camera-ready name, e.g. --label algo3=Ours",
    )
    g.add_argument(
        "--color",
        action="append",
        dest="color_pairs",
        default=[],
        metavar="ALGO=HEX",
        help="override a colour, e.g. algo3=#0173B2",
    )
    g.add_argument(
        "--fig-width",
        type=float,
        default=5.5,
        help="text width in inches (5.5 for ICLR)",
    )
    g.add_argument("--font", choices=["serif", "sans"], default="sans")
    g.add_argument("--font-size", type=float, default=11.0)
    g.add_argument(
        "--grid-cols", type=int, default=4, help="columns in the per-environment grid"
    )
    g.add_argument(
        "--panel-width",
        type=float,
        default=None,
        help="width of one grid panel in inches; default splits "
        "--fig-width across the columns",
    )
    g.add_argument(
        "--panel-height",
        type=float,
        default=1.45,
        help="height of one grid panel in inches",
    )
    g.add_argument(
        "--panel-pad",
        type=float,
        default=None,
        help="space between grid panels in inches (default ~0.04)",
    )
    g.add_argument(
        "--margin-right",
        type=float,
        default=0.0,
        help="blank margin reserved on the right, in inches; use it "
        "to balance the y-label space on the left",
    )
    g.add_argument(
        "--margin-left",
        type=float,
        default=0.0,
        help="blank margin reserved on the left, in inches",
    )
    g.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "png", "svg", "eps"],
        help="output formats; pass several to write several",
    )
    g.add_argument("--dpi", type=int, default=400, help="raster dpi")

    return p


def finalize_args(args):
    """Fill in the derived fields the figure code expects."""
    if not isinstance(getattr(args, "labels", None), dict):
        args.labels = parse_kv(args.label_pairs)
    if not isinstance(getattr(args, "color_overrides", None), dict):
        args.color_overrides = parse_kv(args.color_pairs)
    return args


def default_args(**overrides):
    """CLI defaults as a plain namespace, for notebooks. Unknown keys raise."""
    args = finalize_args(build_parser().parse_args([]))
    derived = {"labels", "color_overrides"}  # dicts, not argparse fields
    for key, value in overrides.items():
        if key not in derived and not hasattr(args, key):
            raise KeyError(f"unknown option {key!r}")
        setattr(args, key, value)
    return args


def main():
    args = finalize_args(build_parser().parse_args())
    setup_style(args)

    try:
        tree = discover(args.logs_dir, args.envs, args.algos)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print("Runs found:")
    for env in sorted(tree):
        for algo in sorted(tree[env]):
            seeds = ", ".join(f"{s:03d}" for s in tree[env][algo])
            print(f"  {env:<18} {algo:<12} seeds: {seeds}")

    try:
        final, curves, frames, tasks = build_matrices(tree, args)
    except (RuntimeError, KeyError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    n_seeds = next(iter(final.values())).shape[0]
    print(
        f"\nScore matrices: {n_seeds} runs x {len(tasks)} task(s), "
        f"steps up to {frames[-1]:,.0f}"
    )
    if n_seeds * len(tasks) < 10:
        print(
            "NOTE: with this few runs x tasks the bootstrap CIs are wide and "
            "the IQM is close to a median. Treat them as indicative."
        )

    run_analysis(final, curves, frames, tasks, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
