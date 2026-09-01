import argparse
import os

from metric_eval.experiments.algorank.config import DATASETS

from metric_eval.experiments.cis.config import CIS_PLOT_DIR, CIS_REPORT_DIR
from metric_eval.experiments.cis.gate import load_cache
from metric_eval.experiments.cis.injector_data import load_damage_sweep, load_equal_damage
from metric_eval.experiments.cis.plotting import (plot_coverage, plot_gate_distribution,
                          plot_known_damage, plot_variation_axis)
from metric_eval.experiments.cis.report import write_report


def main(datasets: list[str], skip_plots: bool) -> None:
    """Read the built caches, write the report and the figures."""
    cache = load_cache(datasets)
    conditions = load_equal_damage()
    sweep = load_damage_sweep()
    if not (conditions and sweep):
        raise FileNotFoundError(
            "Missing Experiment 1's cache. Run ./run_injector.sh first.")

    os.makedirs(CIS_REPORT_DIR, exist_ok=True)
    text = write_report(cache, conditions, sweep)
    print(text)
    path = os.path.join(CIS_REPORT_DIR, "cis_experiments.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nWritten: {path}")

    if skip_plots:
        return
    os.makedirs(CIS_PLOT_DIR, exist_ok=True)
    plot_gate_distribution(cache, os.path.join(CIS_PLOT_DIR, "cis_gate.pdf"))
    plot_coverage(cache, os.path.join(CIS_PLOT_DIR, "cis_coverage.pdf"))
    plot_variation_axis(cache, os.path.join(CIS_PLOT_DIR, "cis_variation_axis.pdf"))
    plot_known_damage(conditions, sweep,
                      os.path.join(CIS_PLOT_DIR, "cis_known_damage.pdf"))
    print(f"Written: {CIS_PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3 (CIS) — report and figures from the built caches.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    main(args.datasets, args.skip_plots)
