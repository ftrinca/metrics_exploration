"""Damage reactivity — the chapter figure and the redundancy numbers.

Draws the response grid (every metric against every distortion, with the
exact blind spots and the non-monotone cells marked) and writes the column
correlations behind the redundancy claims. Reads the reactivity scores of
both calibration passes where both exist: MAE and ND are pinned by the MAE
target, so their columns only mean something on the RMSE-calibrated pass.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metric_eval.core.metric_config import CATEGORIES, METRIC_DIRECTION

from metric_eval.experiments.injector.config import (CATEGORY_COLOR, DISTORTION_NAMES,
                             INJECTOR_CATEGORIES, INJECTOR_METRICS,
                             METRIC_LABEL, PATTERNS, PLOT_DIR, RATES,
                             REPORT_DIR, pass_filename, rate_dir)
from metric_eval.experiments.injector.reactivity import invariance
from metric_eval.experiments.injector.response.aggregate import _classify, _load_all

# The columns read off the RMSE-calibrated pass, per Chapter 4's design.
RMSE_PASS_METRICS = ("mae", "nd")

# The redundancy claims of the results section, as metric pairs.
REDUNDANCY_PAIRS = [
    ("mae", "nd"), ("rmse", "mse"), ("rmse", "nrmse"), ("rmse", "r2"),
    ("mse", "nrmse"), ("mse", "r2"), ("nrmse", "r2"), ("ba", "cdt"),
]


def _load_grid(damage_metric: str) -> dict[tuple, dict] | None:
    """{(pattern, rate): scores} for one pass, or None when it is not cached."""
    out = {}
    for pattern in PATTERNS:
        for rate in RATES:
            path = os.path.join(rate_dir(pattern, rate),
                                pass_filename("scores.json", damage_metric))
            if not os.path.exists(path):
                return None
            with open(path) as f:
                out[(pattern, rate)] = json.load(f)
    return out


def _cell_values(grid: dict[tuple, dict], metric: str,
                 distortion: str) -> list[float]:
    return [s[metric][distortion] for s in grid.values()
            if s.get(metric, {}).get(distortion) is not None]


def deviation_grid(mae_grid: dict[tuple, dict],
                   rmse_grid: dict[tuple, dict] | None) -> dict[str, dict[str, float]]:
    """Mean distance from the perfect-reconstruction value, per (metric, distortion).

    Lower-is-better metrics are distances already. For the higher-is-better
    ones the deviation is (perfect - value), with perfect = 1 for Pearson and
    R2; MI states no perfect value, so its largest reading on the grid stands
    in for it, which the per-metric normalisation of the figure absorbs.
    """
    out: dict[str, dict[str, float]] = {}
    for metric in INJECTOR_METRICS:
        grid = rmse_grid if (metric in RMSE_PASS_METRICS and rmse_grid) else mae_grid
        if METRIC_DIRECTION[metric] == "higher":
            perfect = (1.0 if metric in ("pearson", "r2") else
                       max(v for d in DISTORTION_NAMES
                           for v in _cell_values(grid, metric, d)))
        out[metric] = {}
        for distortion in DISTORTION_NAMES:
            values = _cell_values(grid, metric, distortion)
            if METRIC_DIRECTION[metric] == "higher":
                values = [perfect - v for v in values]
            out[metric][distortion] = float(np.mean(np.abs(values)))
    return out


def exact_blind_spots() -> set[tuple[str, str]]:
    """(metric, distortion) cells where an invariant pins the perfect value."""
    return {(metric, distortion)
            for distortion in DISTORTION_NAMES
            for metric in invariance.expected(distortion)}


def non_monotone_cells() -> set[tuple[str, str]]:
    """(metric, distortion) cells that do not grow steadily over the sweep."""
    data = _load_all()
    return {(metric, distortion)
            for distortion in DISTORTION_NAMES
            for metric in INJECTOR_METRICS
            if _classify(metric, data[distortion][metric])[0] == "non-monotonic"}


def redundancy(mae_grid: dict[tuple, dict],
               rmse_grid: dict[tuple, dict] | None) -> dict[tuple, float | None]:
    """|Pearson| between two metrics' 192 grid cells.

    A pair whose members are both pinned by the pass they were calibrated on
    (MAE with ND, without the RMSE pass) is reported as None, since the
    correlation of two near-constant columns is noise.
    """
    out = {}
    for a, b in REDUNDANCY_PAIRS:
        needs_rmse_pass = a in RMSE_PASS_METRICS or b in RMSE_PASS_METRICS
        if needs_rmse_pass and rmse_grid is None:
            out[(a, b)] = None
            continue
        grid = rmse_grid if needs_rmse_pass else mae_grid
        xs, ys = [], []
        for d in DISTORTION_NAMES:
            xs += _cell_values(grid, a, d)
            ys += _cell_values(grid, b, d)
        out[(a, b)] = float(abs(np.corrcoef(xs, ys)[0, 1]))
    return out


MARKER_RED = "#A32A31"

# How the category names wrap under the grid, matching the thesis figure.
CATEGORY_DISPLAY = {
    "Pointwise Distance": "Pointwise Distance",
    "Distributional Divergence": "Distributional\nDivergence",
    "Temporal Structure": "Temporal\nStructure",
    "Statistical Agreement": "Statistical Agreement",
}


def plot_response_grid(grid: dict[str, dict[str, float]],
                       circles: set, triangles: set, path: str,
                       rmse_pass_available: bool = True) -> None:
    """Metrics as columns, distortions as rows, each column normalised to its
    own strongest response; circles mark exact blindness, triangles a response
    that does not grow steadily over the sweep.

    Without the RMSE-calibrated pass the MAE and ND columns are pinned by the
    calibration, so they are blanked rather than drawn as uniform noise.
    """
    data = np.zeros((len(DISTORTION_NAMES), len(INJECTOR_METRICS)))
    for j, metric in enumerate(INJECTOR_METRICS):
        if metric in RMSE_PASS_METRICS and not rmse_pass_available:
            data[:, j] = np.nan
            continue
        column = np.array([grid[metric][d] for d in DISTORTION_NAMES])
        top = column.max()
        data[:, j] = column / top if top > 0 else 0.0

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    # Columns read off the RMSE-calibrated pass are drawn in a different hue,
    # so the two calibrations cannot be compared against each other by eye.
    rmse_cols = [j for j, m in enumerate(INJECTOR_METRICS)
                 if m in RMSE_PASS_METRICS]
    mae_data = data.copy()
    rmse_data = np.full_like(data, np.nan)
    for j in rmse_cols:
        rmse_data[:, j] = data[:, j]
        mae_data[:, j] = np.nan
    im = ax.imshow(mae_data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.imshow(rmse_data, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    if not rmse_pass_available:
        for j, metric in enumerate(INJECTOR_METRICS):
            if metric in RMSE_PASS_METRICS:
                ax.text(j, (len(DISTORTION_NAMES) - 1) / 2, "needs the\nRMSE pass",
                        ha="center", va="center", fontsize=6, color="0.45",
                        rotation=90)

    # white separators between the cells
    for j in range(1, len(INJECTOR_METRICS)):
        ax.axvline(j - 0.5, color="white", lw=1.4)
    for i in range(1, len(DISTORTION_NAMES)):
        ax.axhline(i - 0.5, color="white", lw=1.4)

    for j, metric in enumerate(INJECTOR_METRICS):
        for i, distortion in enumerate(DISTORTION_NAMES):
            if (metric, distortion) in circles:
                ax.scatter(j, i, s=42, facecolor="none", edgecolor=MARKER_RED,
                           lw=1.4, zorder=3)
            elif (metric, distortion) in triangles:
                ax.scatter(j, i, s=34, marker="^", color=MARKER_RED, zorder=3)

    metric_category = {m: cat for cat in INJECTOR_CATEGORIES
                       for m in CATEGORIES[cat]}
    ax.set_xticks(range(len(INJECTOR_METRICS)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in INJECTOR_METRICS],
                       rotation=90, fontsize=9)
    for tick, metric in zip(ax.get_xticklabels(), INJECTOR_METRICS):
        tick.set_color(CATEGORY_COLOR[metric_category[metric]])
    ax.set_yticks(range(len(DISTORTION_NAMES)))
    ax.set_yticklabels(DISTORTION_NAMES, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # category bars and names under the metric labels
    col = 0
    for category in INJECTOR_CATEGORIES:
        n = len(CATEGORIES[category])
        colour = CATEGORY_COLOR[category]
        ax.plot([col - 0.35, col + n - 0.65], [-0.32, -0.32], color=colour,
                lw=3.2, clip_on=False, solid_capstyle="butt",
                transform=ax.get_xaxis_transform())
        ax.text(col + (n - 1) / 2, -0.40, CATEGORY_DISPLAY[category],
                ha="center", va="top", fontsize=9, color=colour,
                clip_on=False, transform=ax.get_xaxis_transform())
        col += n

    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    cbar.ax.text(0.5, 1.03, "strongest", ha="center", va="bottom", fontsize=9,
                 transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.03, "blind", ha="center", va="top", fontsize=9,
                 transform=cbar.ax.transAxes)

    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   markeredgecolor=MARKER_RED, markeredgewidth=1.4,
                   markersize=7, label="exactly blind"),
        plt.Line2D([], [], marker="^", ls="", color=MARKER_RED,
                   markersize=7, label="responds, but not monotonically"),
        plt.Rectangle((0, 0), 1, 1, facecolor=plt.get_cmap("Blues")(0.62),
                      label="MAE-calibrated pass"),
        plt.Rectangle((0, 0), 1, 1, facecolor=plt.get_cmap("Oranges")(0.62),
                      label="RMSE-calibrated pass (MAE, ND)"),
    ]
    ax.legend(handles=handles, fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.52), frameon=False,
              columnspacing=1.4, handletextpad=0.6)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   {path}")


def write_report(grid: dict[str, dict[str, float]], circles: set,
                 triangles: set, correlations: dict,
                 rmse_pass_available: bool) -> str:
    L = ["INJECTOR SUMMARY — RESPONSE GRID AND REDUNDANCY", "=" * 78, "",
         "MAE and ND columns from the RMSE-calibrated pass: "
         + ("yes" if rmse_pass_available else
            "NO — pass not cached, their columns are pinned by the "
            "calibration and carry no signal"), ""]

    L += ["1. MEAN DISTANCE FROM THE PERFECT VALUE  (24 settings; "
          "o = exact blind spot, ^ = non-monotone over the sweep)",
          f"   {'metric':<9}" + "".join(f"{d[:9]:>11}" for d in DISTORTION_NAMES)]
    for metric in INJECTOR_METRICS:
        cells = ""
        for d in DISTORTION_NAMES:
            mark = ("o" if (metric, d) in circles else
                    "^" if (metric, d) in triangles else " ")
            cells += f"{grid[metric][d]:>10.4f}{mark}"
        L.append(f"   {metric:<9}" + cells)
    L += [f"   exact blind spots: {len(circles)} cells, "
          f"non-monotone: {len(triangles)} cells", ""]

    L += ["2. REDUNDANCY  (|Pearson| over the 192 grid cells)"]
    for (a, b), r in correlations.items():
        value = "needs the RMSE pass" if r is None else f"{r:.3f}"
        L.append(f"   {METRIC_LABEL[a]:<6} {METRIC_LABEL[b]:<6} {value}")
    return "\n".join(L)


def main() -> None:
    mae_grid = _load_grid("mae")
    if mae_grid is None:
        raise FileNotFoundError(
            "Missing the reactivity scores — run ./run_injector.sh first.")
    rmse_grid = _load_grid("rmse")

    grid = deviation_grid(mae_grid, rmse_grid)
    circles = exact_blind_spots()
    triangles = non_monotone_cells() - circles
    correlations = redundancy(mae_grid, rmse_grid)

    os.makedirs(REPORT_DIR, exist_ok=True)
    text = write_report(grid, circles, triangles, correlations,
                        rmse_grid is not None)
    print(text)
    path = os.path.join(REPORT_DIR, "summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nWritten: {path}")

    plot_response_grid(grid, circles, triangles,
                       os.path.join(PLOT_DIR, "response_grid.pdf"),
                       rmse_pass_available=rmse_grid is not None)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Damage reactivity — response grid and redundancy.").parse_args()
    main()
