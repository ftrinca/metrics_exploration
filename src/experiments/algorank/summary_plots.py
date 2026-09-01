"""The chapter-level figures of the ranking chapter.

Each function draws one thesis figure from the caches, so the figures can be
regenerated and diffed against the versions in the text. Everything lands in
plots/algorank/summary/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.ranking import competition_rank

from experiments.algorank import cache
from experiments.algorank import (anchored_visible, draw_rank_grid,
                                  plot_reconstruction)
from experiments.algorank.config import (ALGO_METRICS, ALGO_NAMES,
                                         PATTERNS, RATES, label)
from experiments.algorank.experiments import (RATE_BANDS, agreement_by_condition,
                                              degeneracy, non_blackout,
                                              scenario_agreement, spread,
                                              spread_quartiles, variation_preference)
from experiments.algorank.visualize import _choose_window
from experiments.cis.config import UNSTABLE_THRESHOLD

PATTERN_COLORS = {"mcar": "#4C72B0", "scattered": "#55A868", "blackout": "#C44E52"}

# The thesis figure palette: blues carry the data, the heatmap red marks a
# constant reconstruction, and one orange accent marks the worst case.
DARK_BLUE = "#2C4D76"
MID_BLUE = "#6187B4"
LIGHT_BLUE = "#CAD8EA"
SCATTER_BLUE = "#87A7C9"
BAND_FILL = "#CCD9E6"
FLAT_RED = "#A32A31"
ACCENT_ORANGE = "#A75D23"
BAND_COLORS = {"10-30": LIGHT_BLUE, "40-50": MID_BLUE, "60-80": DARK_BLUE}

# The scenarios the chapter shows as (reconstruction, heatmap) pairs.
THESIS_SCENARIOS = [
    ("forecast-economy", "mcar", 10),
    ("forecast-economy", "mcar", 70),
    ("electricity", "mcar", 70),
    ("temperature", "blackout", 50),
    ("temperature", "mcar", 20),
    ("temperature", "scattered", 20),
    ("drift", "mcar", 70),
    ("chlorine", "scattered", 10),
    ("chlorine", "scattered", 40),
    ("climate", "scattered", 70),
]

# The examples of the separation-measure figure: one wide spread, one narrow,
# and the circled outlier where the algorithms sit far apart and the metrics
# disagree anyway.
SPREAD_WIDE = ("temperature", "scattered", 20)
SPREAD_NARROW = ("climate", "scattered", 70)
SPREAD_OUTLIER = ("drift", "mcar", 50)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   {path}")


# Label placement per dataset: (x offset, y offset, alignment), chosen by hand
# so the long names neither collide with a neighbouring point nor leave the
# axes. The datasets near the right edge get their label on the left.
DATASET_LABELS = {
    "temperature": (0, -14, "center"),
    "chlorine": (-8, -3, "right"),
    "climate": (-8, -3, "right"),
    "electricity": (8, 2, "left"),
    "drift": (8, -3, "left"),
    "forecast-economy": (-8, -3, "right"),
}


def plot_dataset_map(axes: dict[str, tuple[float, float]], path: str) -> None:
    """The six datasets on the two axes that decide what an algorithm can borrow."""
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for dataset, (cross, lag1) in axes.items():
        ax.scatter(cross, lag1, s=36, color="#4C72B0", zorder=3)
        dx, dy, ha = DATASET_LABELS.get(dataset, (8, 2, "left"))
        ax.annotate(dataset, (cross, lag1), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("mean cross-series correlation")
    ax.set_ylabel("lag-1 autocorrelation")
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-0.3, 1.08)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_agreement_rate(matrices: dict[tuple, dict], path: str) -> None:
    """Mean agreement by missing rate, one line per pattern."""
    by_rate = agreement_by_condition(matrices)["by_rate"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    rates = [round(r * 100) for r in RATES]
    for pattern in PATTERNS:
        ax.plot(rates, [by_rate[pattern][r] for r in rates], marker="o", ms=4,
                color=PATTERN_COLORS[pattern], label=pattern)
    ax.set_xlabel("missing rate (%)")
    ax.set_ylabel("mean Kendall $\\tau$")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_separation_measure(suite: dict[tuple, dict],
                            matrices: dict[tuple, dict], path: str) -> None:
    """(a) every algorithm's MAE in a wide and a narrow scenario, as shares of
    the worst; (b) separation against agreement over the 96 non-blackout
    scenarios, with the quarter means."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.4, 5.8),
                                      gridspec_kw={"height_ratios": [1, 2.3]})

    for row, key in ((1.0, SPREAD_WIDE), (0.0, SPREAD_NARROW)):
        scores = dict(suite[key]["scores"]["mae"])
        # The same exclusion the separation itself applies, so the panel shows
        # the scores the number is computed from.
        if suite[key]["std_ratio"].get("BRITS", 0.0) > UNSTABLE_THRESHOLD:
            scores.pop("BRITS", None)
        worst = max(v for v in scores.values() if v is not None)
        values = sorted(v / worst for v in scores.values() if v is not None)
        top.plot([values[0], values[-1]], [row, row], color=BAND_FILL, lw=7,
                 solid_capstyle="round", zorder=1)
        top.scatter(values[:-1], [row] * (len(values) - 1), s=52,
                    facecolor="white", edgecolor=DARK_BLUE, lw=1.5, zorder=3)
        top.scatter(values[-1], row, s=62, color=ACCENT_ORANGE, zorder=3)
        separation = (values[-1] - values[0]) / values[-1]
        top.text(1.06, row, f"separation = {separation:.2f}", va="center",
                 fontsize=8, color="0.35")
        if row == 1.0:
            top.text(values[0], row + 0.55, "best of the six", ha="center",
                     fontsize=8, color=DARK_BLUE)
            top.text(values[-1], row + 0.55, "worst", ha="center",
                     fontsize=8, color=ACCENT_ORANGE)
    top.set_yticks([1.0, 0.0])
    top.set_yticklabels([f"{k[0]}, {k[1]}, {k[2]} %"
                         for k in (SPREAD_WIDE, SPREAD_NARROW)], fontsize=9)
    top.set_xticks([0.0, 0.5, 1.0])
    top.set_xticklabels(["no error", "half the worst score", "the worst score"],
                        fontsize=8)
    top.set_xlabel("each dot is one algorithm's mean absolute error "
                   "in that scenario", fontsize=8, color="0.35")
    top.set_xlim(-0.03, 1.32)
    top.set_ylim(-0.6, 2.1)
    top.tick_params(length=0)
    for side in ("top", "right", "left"):
        top.spines[side].set_visible(False)
    top.text(-0.02, 1.02, "(a)", transform=top.transAxes, fontsize=10)

    keys = list(non_blackout(suite))
    separations = [spread(suite[k]) for k in keys]
    agreements = [scenario_agreement(matrices[k]) for k in keys]
    bottom.scatter(separations, agreements, s=20, color=SCATTER_BLUE,
                   alpha=0.9, edgecolor="none")

    ordered = sorted(zip(separations, agreements))
    quarter = len(ordered) // 4
    quarter_names = ["lowest\nquarter", "second\nquarter",
                     "third\nquarter", "highest\nquarter"]
    for i, q in enumerate(spread_quartiles(suite, matrices)["quarters"]):
        lo = ordered[i * quarter][0]
        hi = ordered[min((i + 1) * quarter, len(ordered) - 1)][0]
        bottom.hlines(q["mean_tau"], lo, hi, color=DARK_BLUE, lw=2.6, zorder=3)
        bottom.text((lo + hi) / 2, q["mean_tau"] + 0.035, f"{q['mean_tau']:.2f}",
                    ha="center", fontsize=9, fontweight="bold", color=DARK_BLUE)
        bottom.text((lo + hi) / 2, 1.06, quarter_names[i], ha="center",
                    va="bottom", fontsize=8, color="0.45", clip_on=False)
        if i < 3:
            bottom.axvline(hi, color="0.7", ls="--", lw=0.9)

    outlier = (spread(suite[SPREAD_OUTLIER]),
               scenario_agreement(matrices[SPREAD_OUTLIER]))
    bottom.scatter(*outlier, s=120, facecolor="none", edgecolor=ACCENT_ORANGE,
                   lw=1.8, zorder=3)
    bottom.annotate(
        f"{SPREAD_OUTLIER[0]}, {SPREAD_OUTLIER[1].upper()}, {SPREAD_OUTLIER[2]} %",
        outlier, textcoords="offset points", xytext=(-6, -18), ha="right",
        fontsize=8, color=ACCENT_ORANGE)
    bottom.set_xlim(0.0, 1.0)
    bottom.set_ylim(0.0, 1.03)
    bottom.set_xlabel("separation: how much worse the weakest reconstruction "
                      "is than the best")
    bottom.set_ylabel("agreement between\nthe eight metrics")
    for side in ("top", "right"):
        bottom.spines[side].set_visible(False)
    bottom.text(-0.02, 1.10, "(b)", transform=bottom.transAxes, fontsize=10)
    fig.tight_layout(h_pad=2.5)
    _save(fig, path)


def plot_variation_preference(suite: dict[tuple, dict],
                              matrices: dict[tuple, dict], path: str) -> None:
    """Where each metric sits between flattening and keeping the variation."""
    vp = variation_preference(suite, matrices)
    order = sorted(ALGO_METRICS, key=lambda m: -vp[m]["60-80"])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for y, metric in enumerate(order):
        values = [vp[metric][band] for band in RATE_BANDS]
        ax.plot([min(values), max(values)], [y, y], color="0.85", lw=1.6,
                zorder=2)
    for band in RATE_BANDS:
        ax.scatter([vp[m][band] for m in order], range(len(order)),
                   s=52, color=BAND_COLORS[band], label=f"{band} %", zorder=3)
    ax.axvline(0, color="0.2", lw=1.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["$R^2$" if m == "r2" else label(m) for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("rewards flattening ←    → "
                  "rewards keeping the variation")
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    _save(fig, path)


def plot_zero_variation(suite: dict[tuple, dict], matrices: dict[tuple, dict],
                        path: str, example=("electricity", "mcar", 70)) -> None:
    """(a) a flat and a varied reconstruction of the same gap, masked positions
    only; (b) the mean rank each metric gives the constant reconstructions and
    the variation-matched ones."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 3.7),
                                      gridspec_kw={"width_ratios": [1.05, 1]})

    dataset, pattern, rate = example
    built = cache.load_scenario(dataset, pattern, rate / 100, seed=0)
    y_true, mask = built["y_true"], built["mask"].astype(bool)
    series = 0
    window_size = min(100, y_true.shape[1])
    start = _choose_window(mask[series], window_size, y_true.shape[1])
    window = slice(start, start + window_size)
    truth = y_true[series][window]
    m = mask[series][window]
    for lo, hi in _true_spans(m):
        left.axvspan(lo - 0.5, hi + 0.5, color="#EFEFEF", zorder=0)
    left.plot(truth, color="black", ls="--", lw=1.1, label="truth")
    left.plot(anchored_visible(truth, built["CDRec"][series][window], m),
              color=DARK_BLUE, lw=1.0, label="keeps the variation")
    left.plot(anchored_visible(truth, built["STMVL"][series][window], m),
              color=FLAT_RED, lw=1.7, label="keeps none")
    left.set_xlabel("timestep")
    left.set_ylabel("value")
    left.legend(fontsize=8, loc="upper left", frameon=False)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)
    left.text(0.02, 1.04, "(a)", transform=left.transAxes, fontsize=10)

    dg = degeneracy(suite, matrices)
    order = sorted(ALGO_METRICS, key=lambda x: -dg["constant"][x])
    ys = range(len(order))
    for y, metric in zip(ys, order):
        right.plot([dg["matched"][metric], dg["constant"][metric]], [y, y],
                   color="0.55", lw=1.2, zorder=2)
    right.scatter([dg["matched"][x] for x in order], ys, s=58,
                  facecolor="white", edgecolor=DARK_BLUE, lw=1.6, zorder=3,
                  label="keeps the variation")
    right.scatter([dg["constant"][x] for x in order], ys, s=72,
                  color=FLAT_RED, zorder=3, label="keeps none")
    right.axvline(3.5, color="0.75", ls="--", lw=1.0)
    right.text(3.5, -0.9, "middle of the six", ha="center", fontsize=8,
               color="0.5", clip_on=False)
    right.set_yticks(list(ys))
    right.set_yticklabels(["$R^2$" if x == "r2" else label(x) for x in order])
    right.invert_yaxis()
    right.set_xlabel("mean rank given to the reconstruction")
    right.tick_params(length=0)
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)
    right.legend(fontsize=8, loc="lower right", frameon=False)
    right.text(0.0, 1.04, "(b)", transform=right.transAxes, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def _true_spans(mask_window: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True in one window, as (start, end) index pairs."""
    idx = np.flatnonzero(mask_window)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def plot_scenario_pair(suite: dict[tuple, dict], matrices: dict[tuple, dict],
                       key: tuple, output_dir: str) -> None:
    """One thesis scenario as its two panels: the reconstructions over a
    window, and the per-metric rank heatmap."""
    dataset, pattern, rate = key
    slug = f"{dataset.replace('-', '_')}_{pattern}_{rate}"

    built = cache.load_scenario(dataset, pattern, rate / 100, seed=0)
    y_true, mask = built["y_true"], built["mask"].astype(bool)
    series = 1 if y_true.shape[0] > 1 else 0
    window_size = min(200, y_true.shape[1])
    start = _choose_window(mask[series], window_size, y_true.shape[1])
    window = slice(start, start + window_size)

    plot_reconstruction(
        {algo: built[algo][series][window]
         for algo in ALGO_NAMES if algo in built},
        y_true[series][window],
        mask[series][window],
        title="",
        output_path=os.path.join(output_dir, f"{slug}.pdf"),
    )

    rm = matrices[key]
    algos = sorted(ALGO_NAMES, key=lambda a: np.mean([rm[m][a] for m in ALGO_METRICS]))
    display = {m: competition_rank(rm[m]) for m in ALGO_METRICS}

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    draw_rank_grid(ax, display, algos)
    _save(fig, os.path.join(output_dir, f"hm_{slug}.pdf"))
