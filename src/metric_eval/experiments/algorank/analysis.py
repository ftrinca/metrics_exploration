from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

from metric_eval.core.metric_config import METRIC_DIRECTION
from metric_eval.core.ranking import rank_algorithms

from metric_eval.experiments.algorank.config import ALGO_CATEGORIES, ALGO_METRICS


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


def consensus_order(rank_matrix: dict[str, dict[str, float]]) -> list[str]:
    """Algorithm names sorted best to worst by global consensus."""
    cons = global_consensus(category_consensus(rank_matrix))
    return [a for a, _ in sorted(cons.items(), key=lambda x: x[1])]


def profile_spread(cat_consensus: dict[str, dict[str, float]]) -> dict[str, float]:
    """Std of each algorithm's four category-consensus scores.

    High means a specialist, strong on some properties and weak on others; low
    means a generalist.
    """
    algos = list(next(iter(cat_consensus.values())).keys())
    return {algo: float(np.std([cat_consensus[c][algo] for c in ALGO_CATEGORIES]))
            for algo in algos}
