"""The per-condition reactivity heatmaps and the two diagnostic overviews.

Every panel reads like the chapter's response grid: metrics as columns,
distortions as rows, each column as a share of that metric's own strongest
response, so the appendix heatmaps and the main figure can be compared by eye.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metric_eval.core.metric_config import CATEGORIES

from metric_eval.experiments.injector.config import (
    CATEGORY_COLOR, DISTORTION_NAMES, INJECTOR_CATEGORIES, INJECTOR_METRICS,
    METRIC_LABEL, PATTERNS, RANGE_BUCKETS,
)
from metric_eval.experiments.injector.reactivity import invariance

CONDITION_ORDER = [(p, b) for p in PATTERNS for b in RANGE_BUCKETS]

METRIC_CATEGORY = {m: cat for cat in INJECTOR_CATEGORIES for m in CATEGORIES[cat]}
MARKER_RED = "#A32A31"


def _blind_cells() -> set[tuple[str, str]]:
    """(metric, distortion) cells an invariant pins at the perfect value."""
    return {(metric, distortion)
            for distortion in DISTORTION_NAMES
            for metric in invariance.expected(distortion)}


def _data_matrix(dev: dict[str, dict[str, float] | None]) -> np.ndarray:
    data = np.full((len(DISTORTION_NAMES), len(INJECTOR_METRICS)), np.nan)
    for j, metric in enumerate(INJECTOR_METRICS):
        column = dev.get(metric)
        if column is None:
            continue
        for i, distortion in enumerate(DISTORTION_NAMES):
            data[i, j] = column[distortion]
    return data


def _draw_grid(ax, dev: dict[str, dict[str, float] | None],
               mark_blind: bool = True, fontsize: float = 8.0) -> None:
    """One response-grid panel onto an axis; NaN columns (pinned metrics) stay
    white with a small note."""
    data = _data_matrix(dev)
    ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for j in range(1, len(INJECTOR_METRICS)):
        ax.axvline(j - 0.5, color="white", lw=1.0)
    for i in range(1, len(DISTORTION_NAMES)):
        ax.axhline(i - 0.5, color="white", lw=1.0)

    if mark_blind:
        for metric, distortion in _blind_cells():
            j = INJECTOR_METRICS.index(metric)
            i = DISTORTION_NAMES.index(distortion)
            ax.scatter(j, i, s=26, facecolor="none", edgecolor=MARKER_RED,
                       lw=1.1, zorder=3)
    for j, metric in enumerate(INJECTOR_METRICS):
        if dev.get(metric) is None:
            ax.text(j, (len(DISTORTION_NAMES) - 1) / 2, "needs the\nRMSE pass",
                    ha="center", va="center", fontsize=fontsize - 3,
                    color="0.45", rotation=90)

    ax.set_xticks(range(len(INJECTOR_METRICS)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in INJECTOR_METRICS],
                       rotation=90, fontsize=fontsize)
    for tick, metric in zip(ax.get_xticklabels(), INJECTOR_METRICS):
        tick.set_color(CATEGORY_COLOR[METRIC_CATEGORY[metric]])
    ax.set_yticks(range(len(DISTORTION_NAMES)))
    ax.set_yticklabels(DISTORTION_NAMES, fontsize=fontsize)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_heatmap(dev: dict[str, dict[str, float] | None],
                 title: str | None, output_path: str) -> None:
    """One condition's response grid for the appendix subfigures."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    _draw_grid(ax, dev)
    if title:
        ax.set_title(title, fontsize=10)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")


def plot_metric_overview(dev_by_condition: dict[tuple, dict], output_path: str) -> None:
    """One small panel per metric: conditions as rows, distortions as columns.

    A diagnostic, kept out of the thesis: the appendix heatmaps carry the same
    content per condition instead of per metric.
    """
    conditions = [c for c in CONDITION_ORDER if c in dev_by_condition]
    if not conditions:
        raise ValueError("no conditions to plot")

    ncols = 4
    nrows = int(np.ceil(len(INJECTOR_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows),
                             squeeze=False)

    for k, metric in enumerate(INJECTOR_METRICS):
        ax = axes[k // ncols][k % ncols]
        grid = np.full((len(conditions), len(DISTORTION_NAMES)), np.nan)
        for i, c in enumerate(conditions):
            column = dev_by_condition[c].get(metric)
            if column is None:
                continue
            grid[i] = [column[d] for d in DISTORTION_NAMES]
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(METRIC_LABEL[metric], fontsize=9,
                     color=CATEGORY_COLOR[METRIC_CATEGORY[metric]],
                     fontweight="bold")
        ax.set_xticks(range(len(DISTORTION_NAMES)))
        ax.set_xticklabels([d[:6] for d in DISTORTION_NAMES],
                           rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([f"{p[:4]} {b[:3]}" for p, b in conditions],
                           fontsize=6.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(1, len(conditions)):
            if conditions[i][0] != conditions[i - 1][0]:
                ax.axhline(i - 0.5, color="white", lw=1.6)

    for k in range(len(INJECTOR_METRICS), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("Every metric across every condition, at equal damage",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")


def plot_condition_grid(dev_by_condition: dict[tuple, dict], output_path: str) -> None:
    """All nine conditions side by side. A diagnostic, kept out of the thesis."""
    conditions = [c for c in CONDITION_ORDER if c in dev_by_condition]

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), squeeze=False)
    for k, c in enumerate(conditions):
        ax = axes[k // 3][k % 3]
        _draw_grid(ax, dev_by_condition[c], mark_blind=False, fontsize=6.0)
        ax.set_title(f"{c[0]} — {c[1]}", fontsize=10, fontweight="bold")

    for k in range(len(conditions), 9):
        axes[k // 3][k % 3].axis("off")

    fig.suptitle("All nine conditions side by side, at equal damage", fontsize=13)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")
