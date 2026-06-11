"""Turns the raw scores from generate_reports.compute_all_scores into
per-metric rankings, a consensus ranking, and a Spearman agreement matrix
between metrics, written to a ranking summary report.
"""

import os

import numpy as np
from scipy.stats import spearmanr

from metric_config import CATEGORIES, METRIC_DIRECTION, METRIC_LIST
from generate_reports import applicable_metrics, compute_all_scores


def _rank_algorithms(metric_scores: dict[str, float | None], direction: str) -> dict[str, int]:
    """Rank algorithms for a single metric. Rank 1 = best.

    Algorithms whose score is None (metric failed) are placed last.
    """
    valid = {algo: score for algo, score in metric_scores.items() if score is not None}
    failed = [algo for algo, score in metric_scores.items() if score is None]

    sorted_algos = sorted(valid, key=lambda a: valid[a], reverse=(direction == "higher"))

    ranks = {algo: rank for rank, algo in enumerate(sorted_algos, start=1)}
    last = len(metric_scores)
    for algo in failed:
        ranks[algo] = last
    return ranks


def build_rank_matrix(
    scores: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, int]]:
    """Build {metric: {algo: rank}} from pre-computed scores.

    Only metrics in applicable_metrics(scores) are included - i.e. metrics
    that are None for every algorithm (crps/nll on an all-deterministic
    dataset, or acf/dtw/smae if every algorithm were probabilistic) are
    omitted entirely, rather than showing a meaningless "everyone tied last"
    ranking.
    """
    return {
        metric: _rank_algorithms(scores[metric], METRIC_DIRECTION[metric])
        for metric in applicable_metrics(scores)
    }


def _spearman_matrix(
    rank_matrix: dict[str, dict[str, int]],
    algo_names: list[str],
    metric_list: list[str],
) -> np.ndarray:
    """Compute a (n_metrics × n_metrics) Spearman correlation matrix over
    metric_list (a subset of METRIC_LIST - see applicable_metrics)."""
    rank_array = np.array(
        [[rank_matrix[m][a] for a in algo_names] for m in metric_list],
        dtype=float,
    )
    n = len(metric_list)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(rank_array[i], rank_array[j])
            corr[i, j] = corr[j, i] = r
    return corr


def generate_ranking_report(
    y_true: np.ndarray,
    imputations: dict,
    output_dir: str = "reports",
    dataset_name: str = "ranking",
    mask: np.ndarray | None = None,
) -> None:
    """Write ranking_summary.txt with four sections:

    1. Per-metric ranking table
    2. Consensus ranking (average rank across all metrics)
    3. Spearman correlation matrix between metrics
    4. Most disagreeing / most agreeing metric pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_names = list(imputations.keys())
    scores = compute_all_scores(y_true, imputations, mask)
    rank_matrix = build_rank_matrix(scores)
    metric_list = list(rank_matrix.keys())
    corr = _spearman_matrix(rank_matrix, algo_names, metric_list)

    lines = []

    # ── 1. per-metric ranking table ─────────────────────────────────────────
    col_w = max(len(a) for a in algo_names) + 2
    label_w = 10
    sep = "-" * (label_w + col_w * len(algo_names) + 12)

    omitted = [m for m in METRIC_LIST if m not in rank_matrix]
    if mask is not None:
        n_missing = int(mask.sum())
        n_total   = int(mask.size)
        eval_note = (
            f"Evaluation: missing positions only  "
            f"({n_missing}/{n_total} = {n_missing/n_total*100:.1f}%)  |  "
            f"acf, dtw, smae: full series"
        )
    else:
        eval_note = "Evaluation: full series (no mask)"
    if omitted:
        eval_note += (
            "  |  omitted (not applicable to any algorithm here): "
            + ", ".join(m.upper() for m in omitted)
        )

    lines += [
        "RANKING SUMMARY",
        "=" * 60,
        "rank 1 = best algorithm for that metric",
        eval_note,
        "",
        f"{'METRIC':<{label_w}}" + "".join(f"{a:^{col_w}}" for a in algo_names) + "  direction",
        sep,
    ]
    for category, cat_metrics in CATEGORIES.items():
        present = [m for m in cat_metrics if m in rank_matrix]
        if not present:
            continue
        lines.append(f"  {category.upper()}")
        for metric in present:
            ranks = rank_matrix[metric]
            row = f"  {metric.upper():<{label_w - 2}}"
            row += "".join(f"{ranks[a]:^{col_w}}" for a in algo_names)
            row += f"  {METRIC_DIRECTION[metric]}"
            lines.append(row)
        lines.append(sep)

    # ── 2. consensus ranking ─────────────────────────────────────────────────
    avg_ranks = {
        algo: np.mean([rank_matrix[m][algo] for m in metric_list])
        for algo in algo_names
    }
    lines += [
        "",
        "CONSENSUS RANKING  (average rank across all metrics)",
        "=" * 60,
    ]
    for pos, (algo, avg) in enumerate(
        sorted(avg_ranks.items(), key=lambda x: x[1]), start=1
    ):
        lines.append(f"  {pos}. {algo}  (avg rank: {avg:.2f})")

    # ── 3. Spearman correlation matrix ───────────────────────────────────────
    short = [m[:6].upper() for m in metric_list]
    cell = 8
    lines += [
        "",
        "METRIC AGREEMENT  (Spearman rank correlation between metrics)",
        "=" * 60,
        "  1.0 = always agree on algorithm order",
        " -1.0 = always disagree",
        "",
        " " * 8 + "".join(f"{s:^{cell}}" for s in short),
        " " * 8 + "-" * (cell * len(metric_list)),
    ]
    for i, metric in enumerate(metric_list):
        row = f"{short[i]:<8}" + "".join(f"{corr[i, j]:^{cell}.2f}" for j in range(len(metric_list)))
        lines.append(row)

    # ── 4. most disagreeing / most agreeing pairs ────────────────────────────
    n = len(metric_list)
    pairs = sorted(
        [
            (metric_list[i], metric_list[j], corr[i, j])
            for i in range(n)
            for j in range(i + 1, n)
        ],
        key=lambda x: x[2],
    )
    lines += [
        "",
        "MOST DISAGREEING METRIC PAIRS  (lowest Spearman correlation)",
        "=" * 60,
    ]
    for m1, m2, r in pairs[:10]:
        lines.append(f"  {m1.upper():<8} vs {m2.upper():<8}  r = {r:+.2f}")

    lines += [
        "",
        "MOST AGREEING METRIC PAIRS  (highest Spearman correlation)",
        "=" * 60,
    ]
    for m1, m2, r in reversed(pairs[-10:]):
        lines.append(f"  {m1.upper():<8} vs {m2.upper():<8}  r = {r:+.2f}")

    path = os.path.join(output_dir, f"{dataset_name}_ranking_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {path}")
