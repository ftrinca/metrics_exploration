import argparse
import os

from core.buckets import bucket_mean, subjects_present

from algo_ranking import cache
from algo_ranking.algorithms import ALGORITHMS
from algo_ranking.config import (
    ALGO_METRICS, DATASETS, PATTERNS, RANGE_BUCKETS, RATES,
    heatmap_dir, rate_dir, report_dir,
)
from algo_ranking.score import ensure_scored
from algo_ranking.analysis import build_rank_matrix, consensus_order
from algo_ranking.report import write_ranking_report
from algo_ranking.plotting import plot_algo_ranking_heatmap


def aggregate_bucket(
    raw_scores: dict[float, dict[str, dict[str, float | None]]],
    rates_in_bucket: list[float],
) -> dict[str, dict[str, float | None]]:
    """Mean over the rates in one bucket, per (metric, algorithm)."""
    order = [name for name, _, _ in ALGORITHMS]
    algos = subjects_present(raw_scores, rates_in_bucket, ALGO_METRICS, order)
    return bucket_mean(raw_scores, rates_in_bucket, ALGO_METRICS, algos)


def aggregate_phase(datasets: list[str], patterns: list[str]) -> None:
    """Write the heatmap and report for every (dataset, pattern, range)."""
    for dataset in datasets:
        print(f"=== dataset: {dataset} " + "=" * 50)
        for pattern in patterns:
            print(f"  -- pattern: {pattern} --")
            raw_scores: dict[float, dict] = {}
            for rate in RATES:
                built = os.path.exists(cache.deterministic_path(dataset, pattern, rate))
                scored = os.path.exists(os.path.join(rate_dir(dataset, pattern, rate), "scores.json"))
                if not (built or scored):
                    raise FileNotFoundError(
                        f"Nothing cached for dataset={dataset!r} pattern={pattern!r} "
                        f"rate={rate} - run python -m algo_ranking.build first."
                    )
                raw_scores[rate] = ensure_scored(dataset, pattern, rate)

            for range_name, rates_in_bucket in RANGE_BUCKETS.items():
                print(f"    -- aggregating range '{range_name}' from rates {rates_in_bucket} --")
                value_table = aggregate_bucket(raw_scores, rates_in_bucket)
                rank_matrix = build_rank_matrix(value_table)
                algos = consensus_order(rank_matrix)

                plot_algo_ranking_heatmap(
                    rank_matrix, algos,
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
    parser = argparse.ArgumentParser(description="Algorithm ranking — aggregate phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    args = parser.parse_args()

    aggregate_phase(args.datasets, args.patterns)
