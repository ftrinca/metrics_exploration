"""Algorithm ranking — summarize phase.

Reads the cached scores through the CIS build (which also carries the
standard-deviation ratios), writes the chapter-level report, and draws the
chapter-level figures. Run `python -m cis.build` first.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from metric_eval.experiments.algorank import cache
from metric_eval.experiments.algorank.config import DATASETS, PLOT_DIR, REPORT_DIR
from metric_eval.experiments.algorank.experiments import rank_matrices
from metric_eval.experiments.algorank.summary_report import write_report
from metric_eval.experiments.algorank.summary_plots import THESIS_SCENARIOS, plot_agreement_rate, plot_dataset_map, plot_scenario_pair, plot_separation_measure, plot_variation_preference, plot_zero_variation
from metric_eval.experiments.cis.gate import load_cache

SUMMARY_PLOT_DIR = os.path.join(PLOT_DIR, "summary")


def dataset_axes() -> dict[str, tuple[float, float]]:
    """The two dataset-map axes, computed on the complete series before masking.

    Which scenario the truth is read from does not matter, since every scenario
    of a dataset shares the same ground truth.
    """
    out = {}
    for dataset in DATASETS:
        # The deterministic cache alone carries the truth, so the axes do not
        # need any seed to have been built.
        with open(cache.deterministic_path(dataset, "mcar", 0.1)) as f:
            y = np.array(json.load(f)["y_true"])
        corr = np.corrcoef(y)
        n = corr.shape[0]
        cross = float(np.mean([abs(corr[i, j])
                               for i in range(n) for j in range(i + 1, n)]))
        lag1 = float(np.mean([np.corrcoef(y[s, :-1], y[s, 1:])[0, 1]
                              for s in range(n)]))
        out[dataset] = (cross, lag1)
    return out


def main(datasets: list[str], skip_plots: bool) -> None:
    """Write the chapter-level report and figures from the built caches."""
    suite = load_cache(datasets)
    matrices = rank_matrices(suite)
    axes = dataset_axes()

    os.makedirs(REPORT_DIR, exist_ok=True)
    text = write_report(suite, matrices, axes)
    print(text)
    path = os.path.join(REPORT_DIR, "algorank_experiments.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nWritten: {path}")

    if skip_plots:
        return
    plot_dataset_map(axes, os.path.join(SUMMARY_PLOT_DIR, "dataset_map.pdf"))
    plot_agreement_rate(matrices,
                        os.path.join(SUMMARY_PLOT_DIR, "agreement_rate.pdf"))
    plot_separation_measure(suite, matrices,
                            os.path.join(SUMMARY_PLOT_DIR, "separation_measure.pdf"))
    plot_variation_preference(suite, matrices,
                              os.path.join(SUMMARY_PLOT_DIR,
                                           "variation_preference.pdf"))
    plot_zero_variation(suite, matrices,
                        os.path.join(SUMMARY_PLOT_DIR, "zero_variation.pdf"))
    for key in THESIS_SCENARIOS:
        if key[0] not in datasets:
            continue
        try:
            plot_scenario_pair(suite, matrices, key, SUMMARY_PLOT_DIR)
        except FileNotFoundError as exc:
            print(f"   SKIP {'/'.join(map(str, key))}: {exc}")
    print(f"Written: {SUMMARY_PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Algorithm ranking — chapter-level report and figures.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    main(args.datasets, args.skip_plots)
