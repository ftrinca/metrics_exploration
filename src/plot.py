"""Plotting helpers: one ground-truth-vs-reconstruction plot per series, and
one ranking heatmap per dataset.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from metric_config import CATEGORIES


def _save_or_show(fig: plt.Figure, output_path: str | None) -> None:
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Written: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_imputation(
    imputations: dict[str, np.ndarray],
    y_true: np.ndarray,
    title: str = "Ground Truth vs Imputations",
    xlabel: str = "Index",
    ylabel: str = "Value",
    figsize: tuple = (10, 6),
    output_path: str | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """Plot all imputation series alongside the ground truth.

    If mask is provided, a shaded band marks the missing positions, i.e. the
    regions that are evaluated. Works for both scattered (MCAR) and
    contiguous (block / blackout) patterns.

    If output_path is given the figure is saved there; otherwise it is shown.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # shade missing regions first so they sit behind the lines
    if mask is not None:
        indices = np.where(mask)[0]
        if len(indices):
            # group consecutive indices into spans for clean shading
            spans, start = [], indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    spans.append((start, indices[i - 1]))
                    start = indices[i]
            spans.append((start, indices[-1]))
            for s, e in spans:
                ax.axvspan(s - 0.5, e + 0.5, color="grey", alpha=0.15, linewidth=0)
            # single legend entry for all shaded regions
            ax.axvspan(0, 0, color="grey", alpha=0.3, label="missing (evaluated)")

    ax.plot(y_true, label="ground truth", color="black", linewidth=2, linestyle="--")
    for name, y_pred in imputations.items():
        if mask is not None:
            # draw only in missing region, but anchor each span to the
            # neighbouring observed points so the line visually connects
            # back to the ground truth at both edges
            visible = np.full(len(y_pred), np.nan)
            visible[mask] = y_pred[mask]
            # include one boundary point on each side of every masked span
            for i in range(len(mask)):
                if not mask[i]:
                    if (i + 1 < len(mask) and mask[i + 1]):   # left edge
                        visible[i] = y_true[i]
                    if (i - 1 >= 0 and mask[i - 1]):          # right edge
                        visible[i] = y_true[i]
            ax.plot(visible, label=name, linewidth=0.9)
        else:
            ax.plot(y_pred, label=name, linewidth=0.9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_ranking(
    rank_matrix: dict[str, dict[str, int]],
    title: str = "Algorithm Ranking per Metric",
    figsize: tuple = (14, 6),
    output_path: str | None = None,
) -> None:
    """Heatmap of algorithm rankings across all metrics, grouped by category.

    Axes:  X = metrics (ordered by category),  Y = algorithms
    Color: green = rank 1 (best),  red = rank N (worst)
    Each cell is annotated with the numeric rank.
    Category labels appear above the heatmap, separated by vertical dividers.

    If output_path is given the figure is saved there; otherwise it is shown.
    """
    metrics_ = list(rank_matrix.keys())
    algos_   = list(rank_matrix[metrics_[0]].keys())
    n_metrics, n_algos = len(metrics_), len(algos_)

    data = np.array(
        [[rank_matrix[m][a] for m in metrics_] for a in algos_],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=figsize)

    # RdYlGn_r: rank 1 (low) → green, rank N (high) → red
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=1, vmax=n_algos, aspect="auto")

    # ── rank numbers inside each cell ──────────────────────────────────────
    for row in range(n_algos):
        for col in range(n_metrics):
            rank = int(data[row, col])
            ax.text(
                col, row, str(rank),
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if rank in (1, n_algos) else "black",
            )

    # ── x-axis: metric names ───────────────────────────────────────────────
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([m.upper() for m in metrics_], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_algos))
    ax.set_yticklabels(algos_, fontsize=10)

    # ── category dividers and labels (above the heatmap) ───────────────────
    # metrics_ may omit some metrics (see ranking.build_rank_matrix /
    # generate_reports.applicable_metrics - e.g. crps/nll dropped for
    # all-deterministic datasets), so count only the metrics actually present
    # for each category, and skip categories with none.
    col_cursor = 0
    for category, cat_metrics in CATEGORIES.items():
        n = sum(1 for m in cat_metrics if m in metrics_)
        if n == 0:
            continue
        mid = col_cursor + n / 2 - 0.5

        # vertical divider before each category (except the first)
        if col_cursor > 0:
            ax.axvline(col_cursor - 0.5, color="white", linewidth=2.5)

        # category label above the top edge of the heatmap
        ax.text(
            mid, 1.01, category,
            ha="center", va="bottom",
            fontsize=8, fontstyle="italic", color="#333333",
            transform=ax.get_xaxis_transform(),
        )

        col_cursor += n

    # ── colorbar ───────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("rank  (1 = best)", fontsize=9)
    cbar.set_ticks(range(1, n_algos + 1))

    ax.set_title(title, fontsize=13, pad=22)
    # explicit margins instead of tight_layout — avoids the warning caused by
    # category labels sitting outside the normal axes bounding box
    fig.subplots_adjust(left=0.12, right=0.92, top=0.82, bottom=0.28)
    _save_or_show(fig, output_path)
