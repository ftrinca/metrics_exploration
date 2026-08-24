import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from injector import analysis, invariance
from injector.config import (
    DISTORTION_NAMES, EQUAL_PLOT_DIR, EQUAL_REPORT_DIR, PATTERNS,
    RANGE_BUCKETS, RATES, TARGET_DAMAGE, rate_dir,
)
from injector.plotting import (
    METRIC_LABEL, plot_condition_grid, plot_heatmap, plot_metric_overview,
)


def bucket_mean(raw_scores, rates_in_bucket):
    """Mean score per (metric, distortion) over the rates in one bucket."""
    out = {}
    for metric in analysis.INJECTOR_METRICS:
        out[metric] = {}
        for d in DISTORTION_NAMES:
            vals = [raw_scores[r][metric][d] for r in rates_in_bucket
                    if raw_scores[r][metric].get(d) is not None]
            out[metric][d] = float(np.mean(vals)) if vals else None
    return out


def agreement_table(agree) -> str:
    """Render the metric-by-metric Spearman agreement matrix."""
    metrics = analysis.INJECTOR_METRICS
    lines = [
        "METRIC AGREEMENT OVER THE EIGHT DISTORTIONS",
        "=" * 78,
        "Spearman correlation between each pair of metrics' orderings of the",
        "eight equally damaging distortions. Near 1 means the two rank the",
        "kinds of damage the same way and are redundant with each other here.",
        "-" * 78,
        f"{'':<9}" + "".join(f"{METRIC_LABEL[m][:6]:>7}" for m in metrics),
    ]
    for a in metrics:
        row = "".join(
            f"{'  nan':>7}" if np.isnan(agree[a][b]) else f"{agree[a][b]:>7.2f}"
            for b in metrics
        )
        lines.append(f"{METRIC_LABEL[a]:<9}" + row)
    return "\n".join(lines)


def aggregate_phase(patterns):
    """Bucket the cached scores and write the tables, heatmaps and invariance checks."""
    os.makedirs(EQUAL_REPORT_DIR, exist_ok=True)

    z_by_condition: dict[tuple, dict] = {}
    spreads_by_condition: dict[tuple, dict] = {}

    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        raw = {}
        for rate in RATES:
            path = os.path.join(rate_dir(pattern, rate), "scores.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path} — run injector/score.py first.")
            with open(path) as f:
                raw[rate] = json.load(f)

        for bucket, rates_in in RANGE_BUCKETS.items():
            print(f"  -- bucket '{bucket}' from rates {rates_in} --")
            table = bucket_mean(raw, rates_in)
            spreads = analysis.spread(table)
            z = analysis.zscores(table)
            agree = analysis.agreement(table)

            z_by_condition[(pattern, bucket)] = z
            spreads_by_condition[(pattern, bucket)] = spreads

            plot_heatmap(
                z,
                title=f"{pattern} — {bucket} missingness",
                output_path=os.path.join(EQUAL_PLOT_DIR, f"{pattern}_{bucket}_heatmap.png"),
                spreads=spreads,
            )

            # an exact prediction is checked against one rate, not a mean over rates
            inv_rate = rates_in[len(rates_in) // 2]
            inv_rows = invariance.check(raw[inv_rate])

            report = "\n\n".join([
                f"INJECTOR v2 — {pattern}, {bucket} missingness"
                f"\ntarget damage = {TARGET_DAMAGE} sigma"
                f"\nrates averaged: {rates_in}",
                analysis.summary_table(table, spreads),
                agreement_table(agree),
                invariance.table(inv_rows, title=f"{pattern}, rate {inv_rate:.0%}"),
            ])
            out_path = os.path.join(EQUAL_REPORT_DIR, f"{pattern}_{bucket}.txt")
            with open(out_path, "w") as f:
                f.write(report + "\n")
            print(f"   report -> {out_path}")

            failures = [r for r in inv_rows if not r["passed"]]
            if failures:
                print(f"   !! {len(failures)} invariance prediction(s) FAILED — see the report")
        print()

    if set(patterns) == set(PATTERNS):
        print("=== side-by-side overviews " + "=" * 35)
        plot_metric_overview(
            z_by_condition,
            os.path.join(EQUAL_PLOT_DIR, "metric_overview.png"),
            spreads_by_condition,
        )
        plot_condition_grid(
            z_by_condition,
            os.path.join(EQUAL_PLOT_DIR, "condition_grid.png"),
            spreads_by_condition,
        )
    else:
        print("Skipping the side-by-side overviews (they need all three "
              "patterns — re-run without --patterns to include them).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Injector v2 — aggregate phase.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    a = ap.parse_args()
    aggregate_phase(a.patterns)
