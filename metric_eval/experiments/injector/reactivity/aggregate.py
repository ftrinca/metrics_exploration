import argparse
import json
import os

import numpy as np

from metric_eval.core.buckets import bucket_mean as _bucket_mean
from metric_eval.experiments.injector.reactivity import analysis
from metric_eval.experiments.injector.reactivity import invariance
from metric_eval.experiments.injector.config import (
    DISTORTION_NAMES, REACTIVITY_PLOT_DIR, REACTIVITY_REPORT_DIR, PATTERNS,
    RANGE_BUCKETS, RATES, TARGET_DAMAGE, pass_filename, rate_dir,
)
from metric_eval.experiments.injector.reactivity.plotting import (
    METRIC_LABEL, plot_condition_grid, plot_heatmap, plot_metric_overview,
)


def bucket_mean(raw_scores, rates_in_bucket):
    """Mean score per (metric, distortion) over the rates in one bucket."""
    return _bucket_mean(raw_scores, rates_in_bucket,
                        analysis.INJECTOR_METRICS, DISTORTION_NAMES)


def agreement_table(agree) -> str:
    """Render the metric-by-metric Spearman agreement matrix."""
    metrics = analysis.INJECTOR_METRICS
    lines = [
        "METRIC AGREEMENT  (Spearman, over the eight distortions)",
        "=" * 78,
        f"{'':<9}" + "".join(f"{METRIC_LABEL[m][:6]:>7}" for m in metrics),
    ]
    for a in metrics:
        row = "".join(
            f"{'  nan':>7}" if np.isnan(agree[a][b]) else f"{agree[a][b]:>7.2f}"
            for b in metrics
        )
        lines.append(f"{METRIC_LABEL[a]:<9}" + row)
    return "\n".join(lines)


def _load_pass(pattern: str, damage_metric: str) -> dict | None:
    """{rate: scores} for one calibration pass, or None when any rate is missing.

    The MAE pass is required; the RMSE pass is optional, and without it the
    MAE and ND columns are drawn as pinned rather than as noise.
    """
    out = {}
    for rate in RATES:
        path = os.path.join(rate_dir(pattern, rate),
                            pass_filename("scores.json", damage_metric))
        if not os.path.exists(path):
            if damage_metric == "mae":
                raise FileNotFoundError(
                    f"Missing {path} — run injector/score.py first.")
            return None
        with open(path) as f:
            out[rate] = json.load(f)
    return out


def aggregate_phase(patterns):
    """Bucket the cached scores and write the tables, heatmaps and invariance checks."""
    os.makedirs(REACTIVITY_REPORT_DIR, exist_ok=True)

    dev_by_condition: dict[tuple, dict] = {}

    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        raw = _load_pass(pattern, "mae")
        raw_rmse = _load_pass(pattern, "rmse")

        for bucket, rates_in in RANGE_BUCKETS.items():
            print(f"  -- bucket '{bucket}' from rates {rates_in} --")
            table = bucket_mean(raw, rates_in)
            spreads = analysis.spread(table)
            agree = analysis.agreement(table)

            # MAE and ND are pinned by the MAE-calibrated pass, so their
            # columns come from the RMSE pass where it has been run.
            if raw_rmse is not None:
                table_rmse = bucket_mean(raw_rmse, rates_in)
                table["mae"], table["nd"] = table_rmse["mae"], table_rmse["nd"]
                pinned = ()
            else:
                pinned = ("mae", "nd")
            dev = analysis.deviations(table, pinned)
            dev_by_condition[(pattern, bucket)] = dev

            # No title: the thesis subcaptions carry the condition.
            plot_heatmap(
                dev, title=None,
                output_path=os.path.join(REACTIVITY_PLOT_DIR, f"{pattern}_{bucket}_heatmap.png"),
            )

            # an exact prediction is checked against one rate, not a mean over rates
            inv_rate = rates_in[len(rates_in) // 2]
            inv_rows = invariance.check(raw[inv_rate])

            report = "\n\n".join([
                f"DAMAGE REACTIVITY — {pattern}, {bucket} missingness"
                f"\ntarget damage = {TARGET_DAMAGE} sigma"
                f"\nrates averaged: {rates_in}",
                analysis.summary_table(table, spreads),
                agreement_table(agree),
                invariance.table(inv_rows, title=f"{pattern}, rate {inv_rate:.0%}"),
            ])
            out_path = os.path.join(REACTIVITY_REPORT_DIR, f"{pattern}_{bucket}.txt")
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
            dev_by_condition,
            os.path.join(REACTIVITY_PLOT_DIR, "metric_overview.png"),
        )
        plot_condition_grid(
            dev_by_condition,
            os.path.join(REACTIVITY_PLOT_DIR, "condition_grid.png"),
        )
    else:
        print("Skipping the side-by-side overviews (they need all three "
              "patterns — re-run without --patterns to include them).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Damage reactivity — aggregate phase.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    a = ap.parse_args()
    aggregate_phase(a.patterns)
