from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.metric_config import CATEGORIES
from injector.config import (
    DISTORTION_NAMES, INJECTOR_CATEGORIES, PATTERNS, RANGE_BUCKETS,
)

# Code label -> the name used in the thesis.
METRIC_LABEL = {
    "mae": "MAE", "rmse": "RMSE", "mse": "MSE", "mre": "MRE",
    "smape": "sMAPE", "nrmse": "nRMSE", "nd": "ND",
    "wd": "WD", "jsd": "JSD", "kld": "KLD",
    "acf": "ACF", "dtw": "DTW", "smae": "sMAE",
    "pearson": "Pearson", "mi": "MI", "r2": "R²",
    "tost": "TOST", "ba": "BA", "cdt": "CDT",
}

CATEGORY_LABEL = {
    "Pointwise Error": "Pointwise Distance",
    "Distributional": "Distributional Divergence",
    "Temporal / Shape": "Temporal Structure",
    "Statistical Agreement": "Statistical Agreement",
}

DISTORTION_LABEL = {d: d for d in DISTORTION_NAMES}


def _ordered_metrics():
    """(category, metric) pairs in report order."""
    out = []
    for cat in INJECTOR_CATEGORIES:
        for m in CATEGORIES[cat]:
            out.append((cat, m))
    return out


def plot_heatmap(zscores, title, output_path, spreads=None):
    """Draw a metric x distortion heatmap of signed z-scores, rows grouped by category.

    Rows marked flat in `spreads` are drawn grey and labelled "(flat)" rather
    than coloured, since their z-scores divide by solver residual.
    """
    rows = _ordered_metrics()
    data = np.array([[zscores[m].get(d, 0.0) for d in DISTORTION_NAMES] for _, m in rows])
    flat = np.array([bool(spreads and spreads.get(m, {}).get("flat")) for _, m in rows])

    fig, ax = plt.subplots(figsize=(1.15 * len(DISTORTION_NAMES) + 3.2, 0.38 * len(rows) + 2.4))
    shown = np.where(flat[:, None], np.nan, data)
    lim = float(np.nanmax(np.abs(shown))) if np.isfinite(shown).any() else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.86")
    im = ax.imshow(shown, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(DISTORTION_NAMES)))
    ax.set_xticklabels([DISTORTION_LABEL[d] for d in DISTORTION_NAMES], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([METRIC_LABEL[m] + ("  (flat)" if f else "")
                        for (_, m), f in zip(rows, flat)])

    prev = None
    for i, (cat, _) in enumerate(rows):
        if prev is not None and cat != prev:
            ax.axhline(i - 0.5, color="black", linewidth=1.1)
        prev = cat

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if flat[i]:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.45")
                continue
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.62 * lim else "black")

    ax.set_title(title + "\nequal damage: every distortion at the same MAE", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.75,
                 label="worse than this metric's own average  →")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"   plot -> {output_path}")


def plot_sweep(levels, series_by_distortion, category, output_path):
    """Draw one panel per metric in a category, one line per distortion.

    series_by_distortion[distortion][metric] is the list of raw values, one per
    damage level.
    """
    metrics = CATEGORIES[category]
    n = len(metrics)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)

    for k, metric in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        for d in DISTORTION_NAMES:
            ys = series_by_distortion.get(d, {}).get(metric)
            if ys is None or all(v is None for v in ys):
                continue
            xs = [x for x, y in zip(levels, ys) if y is not None]
            vs = [y for y in ys if y is not None]
            flat = (max(vs) - min(vs)) <= 1e-12
            ax.plot(xs, vs, marker="o", markersize=3.4, linewidth=1.5,
                    linestyle=":" if flat else "-",
                    label=DISTORTION_LABEL[d] + (" (flat)" if flat else ""))
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.set_xlabel("damage (MAE / σ)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(f"{CATEGORY_LABEL[category]} — every distortion on one damage axis", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")


CONDITION_ORDER = [(p, b) for p in PATTERNS for b in RANGE_BUCKETS]

CATEGORY_COLOR = {
    "Pointwise Error": "#1f4e79",
    "Distributional": "#7b3294",
    "Temporal / Shape": "#1b7837",
    "Statistical Agreement": "#b35806",
}


def plot_metric_overview(z_by_condition, output_path, spreads_by_condition=None):
    """Draw one small heatmap per metric, all on a single page.

    z_by_condition maps (pattern, bucket) -> {metric: {distortion: z}}. Inside
    each panel the rows are the conditions and the columns the eight
    distortions, on one colour scale shared by every panel.

    Raises ValueError when z_by_condition holds none of CONDITION_ORDER.
    """
    metrics = _ordered_metrics()
    conditions = [c for c in CONDITION_ORDER if c in z_by_condition]
    if not conditions:
        raise ValueError("no conditions to plot")

    grids, flags = {}, {}
    for _, m in metrics:
        grids[m] = np.array([
            [z_by_condition[c][m].get(d, 0.0) for d in DISTORTION_NAMES]
            for c in conditions
        ])
        flags[m] = bool(
            spreads_by_condition
            and all(spreads_by_condition.get(c, {}).get(m, {}).get("flat") for c in conditions)
        )

    live = [grids[m] for _, m in metrics if not flags[m]]
    lim = float(np.nanmax(np.abs(np.concatenate(live)))) if live else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.86")

    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.6 * nrows), squeeze=False)

    im = None
    for k, (cat, m) in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        grid = np.where(flags[m], np.nan, grids[m])
        im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(METRIC_LABEL[m] + ("  (flat)" if flags[m] else ""),
                     fontsize=10, color=CATEGORY_COLOR[cat], fontweight="bold")
        ax.set_xticks(range(len(DISTORTION_NAMES)))
        ax.set_xticklabels([DISTORTION_LABEL[d][:6] for d in DISTORTION_NAMES],
                           rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([f"{p[:4]} {b[:3]}" for p, b in conditions], fontsize=6.5)
        for i in range(1, len(conditions)):
            if conditions[i][0] != conditions[i - 1][0]:
                ax.axhline(i - 0.5, color="black", linewidth=0.9)

    for k in range(len(metrics), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(
        "Every metric across every condition, at equal damage\n"
        "rows: geometry x rate bucket    columns: the eight distortions",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.025, pad=0.04, shrink=0.45)
    cbar.set_label("worse than this metric's own average  →", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")


def plot_condition_grid(z_by_condition, output_path, spreads_by_condition=None):
    """Draw the full metric x distortion heatmaps, one panel per condition."""
    metrics = _ordered_metrics()
    conditions = [c for c in CONDITION_ORDER if c in z_by_condition]

    grids, flags = {}, {}
    for c in conditions:
        grids[c] = np.array([[z_by_condition[c][m].get(d, 0.0) for d in DISTORTION_NAMES]
                             for _, m in metrics])
        flags[c] = np.array([bool(spreads_by_condition
                                  and spreads_by_condition.get(c, {}).get(m, {}).get("flat"))
                             for _, m in metrics])

    shown = {c: np.where(flags[c][:, None], np.nan, grids[c]) for c in conditions}
    allv = np.concatenate([g[np.isfinite(g)] for g in shown.values()])
    lim = float(np.nanmax(np.abs(allv))) if allv.size else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.86")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)
    im = None
    for k, c in enumerate(conditions):
        ax = axes[k // 3][k % 3]
        im = ax.imshow(shown[c], cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{c[0]} — {c[1]}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(DISTORTION_NAMES)))
        ax.set_xticklabels([DISTORTION_LABEL[d] for d in DISTORTION_NAMES],
                           rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([METRIC_LABEL[m] for _, m in metrics], fontsize=7)
        prev = None
        for i, (cat, _) in enumerate(metrics):
            if prev is not None and cat != prev:
                ax.axhline(i - 0.5, color="black", linewidth=0.9)
            prev = cat

    for k in range(len(conditions), 9):
        axes[k // 3][k % 3].axis("off")

    fig.suptitle("All nine conditions side by side, at equal damage", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.02, pad=0.04, shrink=0.4)
    cbar.set_label("worse than this metric's own average  →", fontsize=8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")
