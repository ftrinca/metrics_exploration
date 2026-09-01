import argparse
import os

from core.buckets import bucket_mean, subjects_present

from experiments.algorank import cache
from experiments.algorank.config import (
    ALGO_METRICS, ALGO_NAMES, DATASETS, PATTERNS, RANGE_BUCKETS, RATES,
    heatmap_dir, rate_dir, rate_heatmap_dir, rate_report_dir, report_dir,
)
from experiments.algorank import ensure_scored
from experiments.algorank import build_rank_matrix, consensus_order
from experiments.algorank import write_ranking_report
from experiments.algorank import plot_algorank_heatmap


def aggregate_bucket(
    raw_scores: dict[float, dict[str, dict[str, float | None]]],
    rates_in_bucket: list[float],
) -> dict[str, dict[str, float | None]]:
    """Mean over the rates in one bucket, per (metric, algorithm)."""
    # config.ALGO_NAMES, not algorithms.ALGORITHMS: this stage only reads
    # cached scores, so it must not pull in ImputeGAP.
    algos = subjects_present(raw_scores, rates_in_bucket, ALGO_METRICS, ALGO_NAMES)
    return bucket_mean(raw_scores, rates_in_bucket, ALGO_METRICS, algos)


def _emit(dataset, pattern, slug, coverage, value_table, heat_dir, rep_dir) -> None:
    """Rank one value table, then draw its heatmap and write its report.

    `slug` names both output files; `coverage` is how the rates behind them read
    in the titles.
    """
    rank_matrix = build_rank_matrix(value_table)
    algos = consensus_order(rank_matrix)
    # No title: the thesis subcaptions carry the condition, and a title would
    # say the same thing twice while costing figure height.
    plot_algorank_heatmap(
        rank_matrix, algos, title=None,
        output_path=os.path.join(heat_dir, f"{pattern}_{slug}.png"),
    )
    write_ranking_report(dataset, pattern, slug, rank_matrix, value_table,
                         output_dir=rep_dir, coverage=coverage)


def aggregate_phase(datasets: list[str], patterns: list[str]) -> None:
    """Write the heatmap and report for every bucket and every individual rate."""
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
                        f"rate={rate} - run python -m metric_eval.experiments.algorank.build first."
                    )
                raw_scores[rate] = ensure_scored(dataset, pattern, rate)

            for range_name, rates_in_bucket in RANGE_BUCKETS.items():
                print(f"    -- aggregating range '{range_name}' from rates {rates_in_bucket} --")
                _emit(dataset, pattern, range_name, f"{range_name} missingness",
                      aggregate_bucket(raw_scores, rates_in_bucket),
                      heatmap_dir(dataset), report_dir(dataset))

            for rate in RATES:
                print(f"    -- rate {rate:.0%} on its own --")
                _emit(dataset, pattern, f"{round(rate * 100):02d}pct", f"{rate:.0%} missingness",
                      aggregate_bucket(raw_scores, [rate]),
                      rate_heatmap_dir(dataset), rate_report_dir(dataset))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm ranking — aggregate phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    args = parser.parse_args()

    aggregate_phase(args.datasets, args.patterns)
