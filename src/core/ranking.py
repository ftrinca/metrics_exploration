"""Turning per-algorithm metric scores into a per-metric ranking."""

import math


def rank_algorithms(metric_scores: dict[str, float | None], direction: str) -> dict[str, float]:
    """Rank algorithms for one metric, 1 = best. Algorithms scoring None are
    placed last.

    Ties get the average of the positions their group spans, so the ranks in a
    scenario sum to the same total however many ties occur. The consensus
    averaging and the Spearman correlations downstream both rely on that.
    Ties are real here rather than a numerical accident: under blackout
    missingness several algorithms return the same reconstruction, and ranking
    them by insertion order would manufacture a ranking out of a genuine tie.

    direction is "lower" or "higher", from metric_config.METRIC_DIRECTION.
    """
    valid = {algo: score for algo, score in metric_scores.items() if score is not None}
    failed = [algo for algo, score in metric_scores.items() if score is None]

    sorted_algos = sorted(valid, key=lambda a: valid[a], reverse=(direction == "higher"))

    ranks: dict[str, float] = {}
    n = len(sorted_algos)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and math.isclose(
            valid[sorted_algos[j + 1]], valid[sorted_algos[i]], rel_tol=1e-9, abs_tol=1e-9
        ):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[sorted_algos[k]] = avg_rank
        i = j + 1

    last = len(metric_scores)
    for algo in failed:
        ranks[algo] = float(last)
    return ranks


def competition_rank(avg_ranks: dict[str, float]) -> dict[str, int]:
    """Convert average ranks into competition ranking, where every member of a
    tied group shows the group's best position and the next distinct value
    resumes after the whole group: a four-way tie for positions 1 to 4 reads
    1, 1, 1, 1, 5.

    Display only. Every numeric aggregation uses rank_algorithms' average
    ranks, which competition ranking would distort.
    """
    ordered = sorted(avg_ranks.items(), key=lambda x: x[1])
    comp: dict[str, int] = {}
    i, n = 0, len(ordered)
    pos = 1
    while i < n:
        j = i
        while j + 1 < n and math.isclose(
            ordered[j + 1][1], ordered[i][1], rel_tol=1e-9, abs_tol=1e-9
        ):
            j += 1
        group_size = j - i + 1
        for k in range(i, j + 1):
            comp[ordered[k][0]] = pos
        pos += group_size
        i = j + 1
    return comp
