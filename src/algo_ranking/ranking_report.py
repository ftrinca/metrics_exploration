from __future__ import annotations

import math
import os

import numpy as np
from scipy.stats import spearmanr

from core.metric_config import METRIC_DIRECTION
from core.ranking import competition_rank, rank_algorithms

from algo_ranking.config import ALGO_CATEGORIES, ALGO_METRICS, label


def build_rank_matrix(
    value_table: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    """{metric: {algo: rank}}, 1 = best. Tied algorithms share one average rank."""
    return {
        metric: rank_algorithms(value_table[metric], METRIC_DIRECTION[metric])
        for metric in ALGO_METRICS
    }


def category_consensus(
    rank_matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """{category: {algo: mean rank across that category's metrics}}."""
    algos = list(next(iter(rank_matrix.values())).keys())
    out: dict[str, dict[str, float]] = {}
    for category, metrics in ALGO_CATEGORIES.items():
        out[category] = {
            algo: float(np.mean([rank_matrix[m][algo] for m in metrics]))
            for algo in algos
        }
    return out


def global_consensus(cat_consensus: dict[str, dict[str, float]]) -> dict[str, float]:
    """{algo: mean rank across the category consensus scores}."""
    algos = list(next(iter(cat_consensus.values())).keys())
    return {
        algo: float(np.mean([cat_consensus[c][algo] for c in ALGO_CATEGORIES]))
        for algo in algos
    }


def _spearman_matrix(rank_matrix: dict[str, dict[str, float]], algos: list[str]) -> np.ndarray:
    """Pairwise Spearman correlation between the metrics' rankings."""
    metrics = ALGO_METRICS
    rank_array = np.array([[rank_matrix[m][a] for a in algos] for m in metrics], dtype=float)
    n = len(metrics)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(rank_array[i], rank_array[j])
            corr[i, j] = corr[j, i] = r
    return corr


def _fmt_comp_rank(r: int) -> str:
    return str(r)


def _positions_with_ties(ordered: list[tuple[str, float]]) -> list[tuple[str, float, int]]:
    """Turn ascending (algo, score) pairs into (algo, score, position) under competition ranking."""
    positions = []
    i, n = 0, len(ordered)
    while i < n:
        j = i
        while j + 1 < n and math.isclose(
            ordered[j + 1][1], ordered[i][1], rel_tol=1e-9, abs_tol=1e-9
        ):
            j += 1
        for k in range(i, j + 1):
            positions.append((ordered[k][0], ordered[k][1], i + 1))
        i = j + 1
    return positions


def write_ranking_report(
    dataset: str,
    pattern: str,
    range_name: str,
    rank_matrix: dict[str, dict[str, float]],
    value_table: dict[str, dict[str, float | None]],
    output_dir: str,
) -> None:
    """Write one scenario's ranking summary into output_dir, creating it if needed.

    Holds the per-metric table grouped by category, the per-category consensus
    profile, the global consensus, the profile spread and the Spearman agreement
    matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    algos = list(next(iter(rank_matrix.values())).keys())
    n_algos = len(algos)
    cat_consensus = category_consensus(rank_matrix)
    glob_consensus = global_consensus(cat_consensus)
    corr = _spearman_matrix(rank_matrix, algos)

    lines = []
    col_w = max(len(a) for a in algos) + 2
    label_w = 12

    lines += [
        f"ALGORITHM RANKING SUMMARY — {dataset} / {range_name} missingness ({pattern.upper()})",
        "=" * 70,
        "rank 1 = best algorithm for that metric/category. Per-metric rows below use",
        "competition ranking, the same convention as the ranking heatmap: a tied group",
        "all shows the group's best position and the next distinct value resumes after",
        "the whole group, e.g. 1, 1, 1, 1, 5, 6 for a 4-way tie spanning positions 1-4",
        "(core/ranking.py's competition_rank). The '-> mean' row, and every consensus/",
        "spread/agreement number below it, use the mathematically-correct average rank",
        "instead (e.g. 2.50 for that same 4-way tie) - see core/ranking.py's",
        "rank_algorithms - since that is what the category/global averaging and the",
        "Spearman correlations actually need; competition rank is a display-only",
        "convention, not a value used in any calculation.",
        "",
        f"{'METRIC':<{label_w}}" + "".join(f"{a:^{col_w}}" for a in algos) + "  direction",
        "-" * (label_w + col_w * n_algos + 12),
    ]
    for category, metrics in ALGO_CATEGORIES.items():
        lines.append(f"  {category.upper()}")
        for metric in metrics:
            ranks = rank_matrix[metric]
            comp_ranks = competition_rank(ranks)
            row = f"  {label(metric):<{label_w - 2}}"
            row += "".join(f"{_fmt_comp_rank(comp_ranks[a]):^{col_w}}" for a in algos)
            row += f"  {METRIC_DIRECTION[metric]}"
            lines.append(row)
        cons = cat_consensus[category]
        row = f"  {'-> mean':<{label_w - 2}}"
        row += "".join(f"{cons[a]:^{col_w}.2f}" for a in algos)
        lines.append(row)
        lines.append("-" * (label_w + col_w * n_algos + 12))

    lines += [
        "",
        "PER-CATEGORY CONSENSUS  (mean rank of that category's 2 metrics; lower = better)",
        "=" * 70,
    ]
    for category in ALGO_CATEGORIES:
        cons = cat_consensus[category]
        ordered = sorted(cons.items(), key=lambda x: x[1])
        lines.append(f"  {category}:")
        for algo, score, pos in _positions_with_ties(ordered):
            lines.append(f"    {pos}. {algo}  (mean rank: {score:.2f})")

    lines += [
        "",
        "GLOBAL CONSENSUS  (mean rank across all 4 categories)",
        "=" * 70,
    ]
    glob_ordered = sorted(glob_consensus.items(), key=lambda x: x[1])
    for algo, score, pos in _positions_with_ties(glob_ordered):
        lines.append(f"  {pos}. {algo}  (mean rank: {score:.2f})")

    lines += [
        "",
        "PROFILE SPREAD  (std of the 4 category-consensus scores per algorithm)",
        "  high = specialist (strong on some properties, weak on others)",
        "  low  = generalist (similarly ranked on every property)",
        "=" * 70,
    ]
    spread = {
        algo: float(np.std([cat_consensus[c][algo] for c in ALGO_CATEGORIES]))
        for algo in algos
    }
    for algo, s in sorted(spread.items(), key=lambda x: -x[1]):
        lines.append(f"  {algo:<14} std={s:.2f}")

    short = [label(m) for m in ALGO_METRICS]
    cell = 8
    lines += [
        "",
        "METRIC AGREEMENT  (Spearman rank correlation across the 8 kept metrics)",
        "=" * 70,
        " " * label_w + "".join(f"{s:^{cell}}" for s in short),
    ]
    for i, metric in enumerate(ALGO_METRICS):
        row = f"{short[i]:<{label_w}}" + "".join(f"{corr[i, j]:^{cell}.2f}" for j in range(len(ALGO_METRICS)))
        lines.append(row)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{pattern}_{range_name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {path}")
