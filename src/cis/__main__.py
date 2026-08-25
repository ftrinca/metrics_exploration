import argparse
import os

from algo_ranking.config import DATASETS, PATTERNS, RATES

from cis.config import CIS_PLOT_DIR, CIS_REPORT_DIR, FLAT_THRESHOLD, UNSTABLE_THRESHOLD
from cis.gate import collect_all_scenarios
from cis.plotting import (plot_cis_vs_consensus, plot_reactivity_response,
                          plot_gate_distribution)
from cis.experiments import reactivity_response, supporting_experiments
from cis.injector_data import load_injector_reactivity
from cis.validation import (derive_component_scales, derive_gate_thresholds,
                            excluded_scenario_breakdown, validation_summary)


def main(datasets: list[str], patterns: list[str], rates: list[float]) -> None:
    """Gate and score every scenario, then write the reports and figures."""
    rows, scenario_scores, n_timesteps = collect_all_scenarios(datasets, patterns, rates)

    os.makedirs(CIS_REPORT_DIR, exist_ok=True)

    lines = [validation_summary(rows, scenario_scores), ""]

    reactivity_cache = load_injector_reactivity()
    derived = derive_component_scales(reactivity_cache)
    lines.append("COMPONENT SCALE DERIVATION"
                 "  (mean over Experiment 1's eight calibrated distortions)")
    if not derived:
        lines.append("  skipped: no Injector cache found, adopted constants left unchecked")
    else:
        for metric, info in derived.items():
            lines.append(f"  {metric:5s} n={info['n']:4d}  mean={info['mean']:.4f}  "
                         f"median={info['median']:.4f}   (adopted {info['adopted']})")
    lines.append("")

    thr = derive_gate_thresholds(rows)
    lines.append("GATE THRESHOLD EXPLORATORY CHECK (KDE density valleys, log10 iqr_ratio)")
    lines.append(f"  bw={thr['bw']}  n_nonzero={thr['n_nonzero']}")
    lines.append(f"  flat_valley={thr['flat_valley']!r}"
                 f"  (adopted FLAT_THRESHOLD={FLAT_THRESHOLD})")
    lines.append(f"  unstable_valley={thr['unstable_valley']!r}"
                 f"  (adopted UNSTABLE_THRESHOLD={UNSTABLE_THRESHOLD})")
    lines.append("")

    excl = excluded_scenario_breakdown(rows)
    lines.append("EXCLUDED-SCENARIO BREAKDOWN (scenarios with fewer than 3 gate survivors)")
    lines.append(f"  n_excluded={excl['n_excluded_scenarios']}  "
                 f"blackout={excl['n_blackout_excluded']}/{excl['n_blackout_total']}")
    lines.append(f"  failing (scenario,algo) pairs: decisive={excl['n_failing_decisive']}  "
                 f"moderate={excl['n_failing_moderate']}")

    summary = "\n".join(lines)
    print(summary)
    report_path = os.path.join(CIS_REPORT_DIR, "cis_validation_summary.txt")
    with open(report_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nWritten: {report_path}")

    supporting = supporting_experiments(rows, scenario_scores, n_timesteps,
                                        reactivity_cache)
    supporting_path = os.path.join(CIS_REPORT_DIR, "cis_supporting_experiments.txt")
    with open(supporting_path, "w") as f:
        f.write(supporting + "\n")
    print(f"Written: {supporting_path}")

    plot_gate_distribution(rows, os.path.join(CIS_PLOT_DIR, "cis_gate_distribution.png"))
    plot_cis_vs_consensus(rows, os.path.join(CIS_PLOT_DIR, "cis_gated_vs_consensus.png"))

    if reactivity_cache:
        plot_reactivity_response(
            reactivity_response(reactivity_cache),
            os.path.join(CIS_PLOT_DIR, "cis_reactivity_response.png"))
    else:
        print("Skipped cis_reactivity_response.png: no Injector cache found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3 (CIS) — gate and composite score.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    args = parser.parse_args()

    main(args.datasets, args.patterns, args.rates)
