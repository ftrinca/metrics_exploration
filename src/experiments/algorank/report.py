from __future__ import annotations

import os

from core.metric_config import METRIC_DIRECTION
from core.ranking import competition_rank

from experiments.algorank import (
    _positions_with_ties, _spearman_matrix, category_consensus, global_consensus,
    profile_spread,
)
from experiments.algorank.config import ALGO_CATEGORIES, ALGO_METRICS, label


def _fmt_comp_rank(r: int) -> str:
    return str(r)


def write_ranking_report(
    dataset: str,
    pattern: str,
    range_name: str,
    rank_matrix: dict[str, dict[str, float]],
    value_table: dict[str, dict[str, float | None]],
    output_dir: str,
    coverage: str | None = None,
) -> None:
    """Write one scenario's ranking summary into output_dir, creating it if needed.

    `range_name` names the file; `coverage` is how the rates it covers read in
    the title, defaulting to `range_name` followed by "missingness".
    """
    coverage = coverage or f"{range_name} missingness"
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
        f"ALGORITHM RANKING SUMMARY — {dataset} / {coverage} ({pattern.upper()})",
        "=" * 70,
        "rank 1 = best. Per-metric rows use competition ranking (a 4-way tie reads",
        "1, 1, 1, 1, 5); '-> mean' and every number below it use average ranks.",
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
        "PER-CATEGORY CONSENSUS  (mean rank of that category's 2 metrics)",
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
        "PROFILE SPREAD  (std of the 4 category-consensus scores; "
        "high = specialist, low = generalist)",
        "=" * 70,
    ]
    spread = profile_spread(cat_consensus)
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
