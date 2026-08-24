"""Figures for the Algorithm Ranking part: the algorithm by metric ranking
heatmap, and the ground-truth-versus-reconstruction line plot.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from algo_ranking.config import ALGO_CATEGORIES, ALGO_METRICS, label
from algo_ranking.ranking_report import category_consensus, global_consensus
from core.ranking import competition_rank


def plot_reconstruction(
    imputations: dict[str, np.ndarray],
    y_true: np.ndarray,
    mask: np.ndarray,
    title: str,
    output_path: str,
    figsize: tuple = (14, 7),
) -> None:
    """Plot every algorithm's imputed values over the true signal for one
    series, with the missing (evaluated) positions shaded.

    y_true and mask are 1-D, already reduced to one series and one window by the
    caller; imputations maps {algo_name: 1-D reconstruction} of the same length.
    Creates the parent directory of output_path.
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
        for s, e in spans:
            ax.axvspan(s - 0.5, e + 0.5, color="grey", alpha=0.15, linewidth=0)
        ax.axvspan(0, 0, color="grey", alpha=0.3, label="missing (evaluated)")

    ax.plot(y_true, label="ground truth", color="black", linewidth=2, linestyle="--")

    visible_by_name = {}
    for name, y_pred in imputations.items():
        visible = np.full(len(y_pred), np.nan)
        visible[mask] = y_pred[mask]
        # anchor each masked span to its neighbouring observed points so the
        # line visually connects back to the ground truth at both edges
        for i in range(len(mask)):
            if not mask[i]:
                if i + 1 < len(mask) and mask[i + 1]:
                    visible[i] = y_true[i]
                if i - 1 >= 0 and mask[i - 1]:
                    visible[i] = y_true[i]
        visible_by_name[name] = visible
        ax.plot(visible, label=name, linewidth=1.1)

    # Clip to the ground truth's own range rather than autoscaling, because one
    # algorithm diverging by orders of magnitude otherwise compresses every
    # other line into a flat band near zero. The 2.5x padding was picked against
    # real cached reconstructions to clear an ordinarily good algorithm's worst
    # moment while still cutting off genuine divergence. Nothing is hidden:
    # matplotlib still draws the clipped lines out to the plot edge, and any
    # algorithm that leaves the visible range is named below the plot so a
    # clipped line is never read as a well-behaved one.
    finite_true = y_true[np.isfinite(y_true)]
    y_lo, y_hi = float(np.min(finite_true)), float(np.max(finite_true))
    pad = max((y_hi - y_lo) * 2.5, 1.0)
    ylim = (y_lo - pad, y_hi + pad)

    off_scale = []
    for name, visible in visible_by_name.items():
        vals = visible[np.isfinite(visible)]
        if vals.size and (vals.min() < ylim[0] or vals.max() > ylim[1]):
            peak = vals[np.argmax(np.abs(vals))]
            off_scale.append(f"{name} (peak {peak:+.0f})")

    ax.set_ylim(ylim)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    if off_scale:
        ax.text(
            0.5, -0.12, "off-scale (clipped): " + ", ".join(off_scale),
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#a33333",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Written: {output_path}")


def plot_algo_ranking_heatmap(
    rank_matrix: dict[str, dict[str, float]],
    title: str,
    output_path: str,
    figsize: tuple = (12, 6),
) -> None:
    """Draw the algorithm by metric rank heatmap, rows sorted best to worst by
    global consensus and columns grouped by category. Rank 1 = best.

    Cells show competition ranks rather than build_rank_matrix's average ranks,
    so a tied group reads as one repeated whole number the way a leaderboard
    shows ties. That conversion is display-only and feeds back into no
    aggregation. Creates the parent directory of output_path.
    """
    cat_consensus = category_consensus(rank_matrix)
    glob_consensus = global_consensus(cat_consensus)
    algos = [a for a, _ in sorted(glob_consensus.items(), key=lambda x: x[1])]
    n_algos = len(algos)
    n_metrics = len(ALGO_METRICS)

    display_matrix = {m: competition_rank(rank_matrix[m]) for m in ALGO_METRICS}
    data = np.array(
        [[display_matrix[m][a] for m in ALGO_METRICS] for a in algos],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=1, vmax=n_algos, aspect="auto")

    for row in range(n_algos):
        for col in range(n_metrics):
            rank = int(data[row, col])
            ax.text(
                col, row, str(rank),
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if rank in (1, n_algos) else "black",
            )

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([label(m) for m in ALGO_METRICS], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_algos))
    ax.set_yticklabels(algos, fontsize=10)

    col_cursor = 0
    for category, cat_metrics in ALGO_CATEGORIES.items():
        n = len(cat_metrics)
        mid = col_cursor + n / 2 - 0.5
        if col_cursor > 0:
            ax.axvline(col_cursor - 0.5, color="white", linewidth=2.5)
        ax.text(
            mid, 1.01, category,
            ha="center", va="bottom",
            fontsize=8, fontstyle="italic", color="#333333",
            transform=ax.get_xaxis_transform(),
        )
        col_cursor += n

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("rank  (1 = best)", fontsize=9)
    cbar.set_ticks(range(1, n_algos + 1))

    ax.set_title(title, fontsize=13, pad=22)
    fig.subplots_adjust(left=0.16, right=0.92, top=0.82, bottom=0.28)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Written: {output_path}")
