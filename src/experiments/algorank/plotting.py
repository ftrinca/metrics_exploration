import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from experiments.algorank.config import ALGO_CATEGORIES, ALGO_METRICS, label
from core.ranking import competition_rank

# The thesis heatmap palette, one colour per competition rank (1 = best).
RANK_COLORS = ["#346960", "#77A29B", "#C6D1C1", "#E5C9B0", "#CD8470", "#A32A31"]

# One colour per algorithm, shared across every reconstruction figure.
ALGO_COLORS = {
    "CDRec": "#4C72B0", "ROSL": "#DD8452", "DynaMMo": "#55A868",
    "MPIN": "#8C8C1A", "STMVL": "#C44E52", "BRITS": "#B5A3D4",
}

CATEGORY_SHORT = {
    "Pointwise Distance": "Pointwise",
    "Distributional Divergence": "Divergence",
    "Temporal Structure": "Temporal",
    "Statistical Agreement": "Agreement",
}


def _metric_tick(metric: str) -> str:
    return "$R^2$" if metric == "r2" else label(metric)


def draw_rank_grid(ax, display_matrix: dict[str, dict[str, int]],
                   algos: list[str]) -> None:
    """Draw the rank cells, labels and category headers onto one axis.

    `display_matrix` holds competition ranks per {metric: {algo: rank}}; the
    caller decides the row order. Shared by the per-condition heatmaps and the
    per-scenario ones so every heatmap in the thesis reads the same way.
    """
    n_algos, n_metrics = len(algos), len(ALGO_METRICS)
    pad = 0.05
    for i, algo in enumerate(algos):
        for j, metric in enumerate(ALGO_METRICS):
            rank = int(display_matrix[metric][algo])
            colour = RANK_COLORS[min(rank, len(RANK_COLORS)) - 1]
            ax.add_patch(Rectangle((j + pad, i + pad), 1 - 2 * pad, 1 - 2 * pad,
                                   facecolor=colour, edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, str(rank), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if rank in (1, n_algos) else "#262626")

    ax.set_xlim(0, n_metrics)
    ax.set_ylim(n_algos, 0)
    ax.set_xticks([j + 0.5 for j in range(n_metrics)])
    ax.set_xticklabels([_metric_tick(m) for m in ALGO_METRICS], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(n_algos)])
    ax.set_yticklabels(algos, fontsize=9)
    # Push the metric names below the per-category rule drawn under the cells.
    ax.tick_params(axis="x", length=0, pad=13)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    col = 0
    for category, metrics in ALGO_CATEGORIES.items():
        n = len(metrics)
        ax.plot([col + 0.12, col + n - 0.12], [-0.30, -0.30],
                color="#8A8A8A", lw=2.4, clip_on=False,
                solid_capstyle="butt")
        ax.text(col + n / 2, -0.48, CATEGORY_SHORT.get(category, category),
                ha="center", va="bottom", color="#737373", fontsize=9,
                clip_on=False)
        ax.plot([col + 0.12, col + n - 0.12],
                [n_algos + 0.10, n_algos + 0.10],
                color="#4A4A4A", lw=1.0, clip_on=False,
                solid_capstyle="butt")
        col += n


def anchored_visible(y_true: np.ndarray, y_pred: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """A reconstruction restricted to the masked positions, NaN elsewhere,
    with the observed truth at each gap's edges so the segments connect."""
    visible = np.full(len(y_pred), np.nan)
    visible[mask] = y_pred[mask]
    for i in range(len(mask)):
        if not mask[i]:
            if i + 1 < len(mask) and mask[i + 1]:
                visible[i] = y_true[i]
            if i - 1 >= 0 and mask[i - 1]:
                visible[i] = y_true[i]
    return visible


def plot_reconstruction(
    imputations: dict[str, np.ndarray],
    y_true: np.ndarray,
    mask: np.ndarray,
    title: str,
    output_path: str,
    figsize: tuple = (9.0, 3.8),
) -> None:
    """Plot every algorithm's imputed values over the true signal for one series.

    A reconstruction is drawn at the masked positions only, anchored to the
    observed truth at each gap's edges so the segments connect; where it
    agrees with the truth by construction there is nothing to show. The y
    range follows the truth, and a reconstruction that leaves it is clipped
    and named in the footnote rather than allowed to flatten everything else.
    """
    fig, ax = plt.subplots(figsize=figsize)

    indices = np.where(mask)[0]
    if len(indices):
        spans, start = [], indices[0]
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                spans.append((start, indices[i - 1]))
                start = indices[i]
        spans.append((start, indices[-1]))
        for span_start, span_end in spans:
            ax.axvspan(span_start - 0.5, span_end + 0.5, color="grey",
                       alpha=0.15, linewidth=0)
        ax.axvspan(0, 0, color="grey", alpha=0.3, label="evaluated")

    ax.plot(y_true, label="ground truth", color="black", linewidth=1.2,
            linestyle="--")

    visible_by_name = {}
    for name, y_pred in imputations.items():
        visible = anchored_visible(y_true, y_pred, mask)
        visible_by_name[name] = visible
        ax.plot(visible, label=name, linewidth=0.9,
                color=ALGO_COLORS.get(name))

    finite_true = y_true[np.isfinite(y_true)]
    y_lo, y_hi = float(np.min(finite_true)), float(np.max(finite_true))
    pad = max((y_hi - y_lo) * 0.5, 0.25)
    ylim = (y_lo - pad, y_hi + pad)
    ax.set_ylim(ylim)

    clipped = []
    for name, visible in visible_by_name.items():
        vals = visible[np.isfinite(visible)]
        if vals.size and (vals.min() < ylim[0] or vals.max() > ylim[1]):
            peak = vals[np.argmax(np.abs(vals))]
            clipped.append(f"{name} to {peak:+.0f}")

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("timestep")
    ax.set_ylabel("value")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    if clipped:
        ax.text(
            0.5, -0.22, "clipped: " + ", ".join(clipped),
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#a33333",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Written: {output_path}")


def plot_algorank_heatmap(
    rank_matrix: dict[str, dict[str, float]],
    algos: list[str],
    title: str | None,
    output_path: str,
    figsize: tuple = (7.2, 5.0),
) -> None:
    """Draw the algorithm x metric rank heatmap, rank 1 = best.

    `algos` is the row order, best to worst, computed once by the caller.
    """
    display_matrix = {m: competition_rank(rank_matrix[m]) for m in ALGO_METRICS}

    fig, ax = plt.subplots(figsize=figsize)
    draw_rank_grid(ax, display_matrix, algos)
    if title:
        # The category labels sit above the grid, so the title needs the pad.
        ax.set_title(title, fontsize=11, pad=48)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Written: {output_path}")
