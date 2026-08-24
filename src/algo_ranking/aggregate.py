"""Algorithm Ranking aggregate phase: mean-aggregate the cached scores of each
range bucket, rank the algorithms per metric, and write the heatmap and text
report for every (dataset, pattern, range).

Usage:
  python algo_ranking/aggregate.py
  python algo_ranking/aggregate.py --datasets climate
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from algo_ranking.algorithms import ALGORITHMS
from algo_ranking.config import (
    ALGO_METRICS, DATASETS, PATTERNS, RANGE_BUCKETS, RATES,
    heatmap_dir, rate_dir, report_dir,
)
from algo_ranking.score import ensure_scored
from algo_ranking.ranking_report import build_rank_matrix, write_ranking_report
from algo_ranking.plotting import plot_algo_ranking_heatmap


def aggregate_bucket(
    raw_scores: dict[float, dict[str, dict[str, float | None]]],
    rates_in_bucket: list[float],
) -> dict[str, dict[str, float | None]]:
    """Mean over rates_in_bucket, per (metric, algorithm). None values are left
    out of the mean, and a bucket where every value is None is itself None.

    The algorithm list is the union of every algorithm present at any rate in
    the bucket, for the same reason as score._average_scores: an algorithm
    missing at one rate, say from a one-off subprocess failure during the build,
    should not drop out of a bucket where it scored at the other rates.
    """
    present: set[str] = set()
    for r in rates_in_bucket:
        for metric in ALGO_METRICS:
            present.update(raw_scores[r].get(metric, {}).keys())
    algos = [name for name, _, _ in ALGORITHMS if name in present]

    bucketed: dict[str, dict[str, float | None]] = {}
    for metric in ALGO_METRICS:
        bucketed[metric] = {}
        for algo in algos:
            vals = [
                raw_scores[r][metric][algo]
                for r in rates_in_bucket
                if raw_scores[r][metric].get(algo) is not None
            ]
            bucketed[metric][algo] = float(np.mean(vals)) if vals else None
    return bucketed


def aggregate_phase(datasets: list[str], patterns: list[str]) -> None:
    """Aggregate and write the heatmap and report for every (dataset, pattern,
    range). Requires build.py to have produced every rate in config.RATES for
    each (dataset, pattern).

    Scoring is ensured rather than assumed, so an unscored scenario is scored
    and a cache predating a change to config.ALGO_CATEGORIES is topped up with
    only the newly selected metrics. Changing the metric set is therefore an
    aggregate run rather than a re-score.
    """
    for dataset in datasets:
        print(f"=== dataset: {dataset} " + "=" * 50)
        for pattern in patterns:
            print(f"  -- pattern: {pattern} --")
            raw_scores: dict[float, dict] = {}
            for rate in RATES:
                if not os.path.exists(os.path.join(rate_dir(dataset, pattern, rate), "data.json")) \
                        and not os.path.exists(os.path.join(rate_dir(dataset, pattern, rate), "scores.json")):
                    raise FileNotFoundError(
                        f"Nothing cached for dataset={dataset!r} pattern={pattern!r} "
                        f"rate={rate} - run algo_ranking/build.py first."
                    )
                raw_scores[rate] = ensure_scored(dataset, pattern, rate)

            for range_name, rates_in_bucket in RANGE_BUCKETS.items():
                print(f"    -- aggregating range '{range_name}' from rates {rates_in_bucket} --")
                value_table = aggregate_bucket(raw_scores, rates_in_bucket)
                rank_matrix = build_rank_matrix(value_table)

                plot_algo_ranking_heatmap(
                    rank_matrix,
                    title=f"Algorithm Ranking — {dataset} / {range_name} missingness ({pattern.upper()})",
                    output_path=os.path.join(
                        heatmap_dir(dataset), f"{pattern}_{range_name}.png"),
                )
                write_ranking_report(
                    dataset, pattern, range_name, rank_matrix, value_table,
                    output_dir=report_dir(dataset),
                )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm Ranking (Part 2) — aggregate phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    args = parser.parse_args()

    aggregate_phase(args.datasets, args.patterns)
