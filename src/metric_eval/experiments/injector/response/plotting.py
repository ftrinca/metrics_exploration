"""The damage-response sweep figures: one standalone plot per metric.

Each plot shows the metric's raw values across the seven damage levels, one
line per distortion, with numbered axes so a value can be read off. The
colour legend is shared and drawn once, into its own file, so the per-metric
plots stay uncluttered and a figure built from several of them needs the
legend only once.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metric_eval.core.metric_config import CATEGORIES
from metric_eval.experiments.injector.config import (CATEGORY_COLOR, DISTORTION_LABEL,
                             DISTORTION_NAMES, INJECTOR_CATEGORIES,
                             INJECTOR_METRICS, METRIC_LABEL)

# One colour per distortion, shared by every sweep figure.
DISTORTION_COLORS = {
    "noise": "#4C72B0", "bias": "#DD8452", "reorder": "#55A868",
    "discretise": "#C44E52", "lag": "#8172B2", "smooth": "#937860",
    "spikes": "#DA8BC3", "rescale": "#8C8C8C",
}

METRIC_CATEGORY = {m: cat for cat in INJECTOR_CATEGORIES for m in CATEGORIES[cat]}


def plot_metric_sweep(metric: str, levels, series_by_distortion,
                      output_path: str) -> None:
    """One metric's sweep as a standalone plot, raw values on numbered axes."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for d in DISTORTION_NAMES:
        ys = series_by_distortion.get(d, {}).get(metric)
        if ys is None or all(v is None for v in ys):
            continue
        xs = [x for x, y in zip(levels, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        ax.plot(xs, vs, lw=1.1, color=DISTORTION_COLORS[d],
                label=DISTORTION_LABEL[d])
    ax.set_title(METRIC_LABEL[metric], fontsize=10,
                 color=CATEGORY_COLOR[METRIC_CATEGORY[metric]])
    ax.set_xlabel("damage ($\\sigma$)", fontsize=8)
    ax.set_xticks(levels)
    ax.tick_params(labelsize=7, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, alpha=0.25)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")


def plot_sweep_legend(output_path: str) -> None:
    """The distortion colour key, on its own so the sweep plots share one."""
    fig, ax = plt.subplots(figsize=(7.0, 0.4))
    handles = [plt.Line2D([], [], color=DISTORTION_COLORS[d], lw=1.6,
                          label=DISTORTION_LABEL[d]) for d in DISTORTION_NAMES]
    ax.legend(handles=handles, ncol=8, loc="center", fontsize=8,
              frameon=False, handlelength=1.6, columnspacing=1.2)
    ax.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")


def plot_all_sweeps(levels, series_by_distortion, output_dir: str) -> None:
    """Every metric's sweep plot, plus the shared legend."""
    for metric in INJECTOR_METRICS:
        plot_metric_sweep(metric, levels, series_by_distortion,
                          os.path.join(output_dir, f"sweep_{metric}.pdf"))
    plot_sweep_legend(os.path.join(output_dir, "sweep_legend.pdf"))
