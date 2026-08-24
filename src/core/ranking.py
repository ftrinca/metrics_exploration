import math


def rank_algorithms(metric_scores: dict[str, float | None], direction: str) -> dict[str, float]:
    """Rank algorithms for one metric, 1 = best. Algorithms scoring None are placed last.

    Ties get the average of the positions their group spans, so the ranks in a
    scenario sum to the same total however many ties occur. `direction` is
    "lower" or "higher", from metric_config.METRIC_DIRECTION.
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
    """Convert average ranks into competition ranking, e.g. 1, 1, 1, 1, 5 for a 4-way tie.

    Display only. Every numeric aggregation uses rank_algorithms' average ranks.
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
