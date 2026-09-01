import numpy as np

from metric_eval.experiments.algorank.config import ALGO_METRICS, ALGO_NAMES, PATTERNS, label

from metric_eval.experiments.cis.config import (ADOPTED_POWER, CIS_METRICS, FLAT_THRESHOLD,
                        UNSTABLE_THRESHOLD)
from metric_eval.experiments.cis.experiments import (RATE_BANDS, agreement_with_metrics, blind_spots,
                             component_profile, coverage, damage_sweep,
                             equal_damage_response, gate_outcome,
                             instrument_comparison, threshold_sensitivity,
                             variant_coverage, variation_preference)


def _name(ranking: str) -> str:
    return "CIS" if ranking == "CIS" else label(ranking)


def write_report(cache: dict[tuple, dict], conditions: dict, sweep: dict) -> str:
    """The text behind every table in the CIS chapter."""
    L = ["CIS EXPERIMENTS", "=" * 78, "",
         f"scenarios {len(cache)}, algorithms {len(ALGO_NAMES)}, "
         f"components {', '.join(label(m) for m in CIS_METRICS)}, "
         f"gate [{FLAT_THRESHOLD}, {UNSTABLE_THRESHOLD}] on the standard-deviation "
         f"ratio, exponent p={ADOPTED_POWER}", ""]

    g = gate_outcome(cache)
    total = g["kept"] + g["flat"] + g["unstable"]
    L += ["1. GATE OUTCOME",
          f"   of {total} (scenario, algorithm) pairs: {g['kept']} kept, "
          f"{g['flat']} flat, {g['unstable']} unstable",
          "   " + f"{'algorithm':12s}" + "".join(f"{p:>26}" for p in PATTERNS)]
    for algo in ALGO_NAMES:
        cells = "".join(
            f"{g['per_algo'][algo][p]['passed']:9d}/{g['per_algo'][algo][p]['total']:<4d}"
            f"({g['per_algo'][algo][p]['median_ratio']:8.3f})" for p in PATTERNS)
        L.append(f"   {algo:12s}{cells}")
    L.append("")

    ic = instrument_comparison(cache)
    L += ["2. SPREAD MEASURE: STANDARD DEVIATION AGAINST INTERQUARTILE RANGE",
          f"   the two disagree on {ic['n_reclassified']} pairs: " +
          ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(ic["transitions"].items())),
          f"   IQR reads exactly 0 for {ic['zero_iqr_but_varied']} reconstructions "
          f"that are not constant",
          f"   their distinct-value counts run from {ic['unique_range'][0]:.0f} to "
          f"{ic['unique_range'][1]:.0f}",
          f"   genuinely constant reconstructions: {ic['n_constant']}", ""]

    ts = threshold_sensitivity(cache)
    L += ["3. THRESHOLD SENSITIVITY", "   flat cut (upper held at 3.0):"]
    for cut, info in ts["flat"].items():
        L.append(f"      {cut:5.2f}  survivors {info['survivors']:4d}  "
                 f"rankable {info['rankable']:3d}")
    L.append("   upper cut (flat held at 0.15):")
    for cut, info in ts["unstable"].items():
        L.append(f"      {cut:5.1f}  survivors {info['survivors']:4d}  "
                 f"rankable {info['rankable']:3d}")
    L.append("")

    cv = coverage(cache)
    L += ["4. COVERAGE",
          f"   rankable scenarios {cv['n_rankable']} of {len(cache)}",
          f"   by geometry {cv['by_geometry']}",
          f"   by rate     {cv['by_rate']}",
          f"   by dataset  {cv['by_dataset']}",
          f"   survivors per scenario {cv['survivor_histogram']}", ""]

    cp = component_profile(cache)
    L += ["5. COMPONENT DISTANCES OVER THE SURVIVORS  (1.0 = the reference)",
          f"   {'component':12s}{'p10':>9}{'median':>9}{'p90':>9}"]
    for metric, info in cp.items():
        L.append(f"   {label(metric):12s}{info['p10']:9.3f}{info['median']:9.3f}"
                 f"{info['p90']:9.3f}")
    L.append("")

    for gated, title in ((False, "ungated, non-blackout"), (True, "gate survivors")):
        vp = variation_preference(cache, gated=gated)
        L += [f"6. VARIATION PREFERENCE, {title.upper()}"
              "   (positive: keeps the variation first)",
              "   " + f"{'ranking':10s}" + "".join(f"{b:>10}" for b in RATE_BANDS)]
        for ranking in list(ALGO_METRICS) + ["CIS"]:
            L.append(f"   {_name(ranking):10s}" +
                     "".join(f"{vp[ranking][b]:10.2f}" for b in RATE_BANDS))
        L.append("")

    am = agreement_with_metrics(cache)
    L += ["7. CIS AGAINST EACH SINGLE METRIC  (gate survivors)",
          "   " + "  ".join(f"{label(m)} {v:.2f}" for m, v in
                            sorted(am.items(), key=lambda x: -x[1])), ""]

    bs = blind_spots(conditions)
    distortions = list(next(iter(bs.values())))
    L += ["8. WHAT EACH METRIC SEES UNDER EQUAL POINTWISE DAMAGE",
          "   (distance to the truth; near zero is a blind spot)",
          "   " + f"{'metric':8s}" + "".join(f"{d[:9]:>10}" for d in distortions)]
    for metric in ALGO_METRICS:
        L.append(f"   {label(metric):8s}" +
                 "".join(f"{bs[metric][d]:10.3f}" for d in distortions))
    L.append("")

    ed = equal_damage_response(conditions)
    L += ["9. THE COMPOSITE UNDER EQUAL POINTWISE DAMAGE",
          "   " + f"{'distortion':12s}" +
          "".join(f"{label(m):>9}" for m in CIS_METRICS) + f"{'CIS':>9}"]
    for distortion in sorted(ed["per_distortion"], key=ed["per_distortion"].get):
        L.append(f"   {distortion:12s}" +
                 "".join(f"{ed['per_component'][distortion][m]:9.3f}"
                         for m in CIS_METRICS) +
                 f"{ed['per_distortion'][distortion]:9.3f}")
    L += [f"   coverage {ed['coverage']:.3f} (least detected: {ed['least_detected']}), "
          f"spread {ed['spread']:.2f}",
          f"   gate fired on {ed['gate_fired']} of {ed['n']} distortion-conditions", ""]

    vc = variant_coverage(conditions)
    L += ["10. COMPONENT AND EXPONENT VARIANTS ON THE EQUAL-DAMAGE RUN",
          f"   {'variant':36s}{'coverage':>10}{'spread':>9}  least detected"]
    for variant, info in vc.items():
        L.append(f"   {variant:36s}{info['coverage']:10.3f}{info['spread']:9.2f}"
                 f"  {info['least_detected']}")
    L.append("")

    sw = damage_sweep(sweep)
    levels = next(iter(sw.values()))["damage_levels"]
    L += ["11. THE COMPOSITE ACROSS THE DAMAGE SWEEP",
          "   " + f"{'distortion':12s}" + "".join(f"{l:>7}" for l in levels) +
          f"{'monotone':>10}{'rise':>8}"]
    for distortion, info in sw.items():
        L.append(f"   {distortion:12s}" + "".join(f"{v:7.2f}" for v in info["cis"]) +
                 f"{str(info['monotone']):>10}{info['rise']:8.3f}")
    L.append(f"   monotone in {sum(1 for v in sw.values() if v['monotone'])} of {len(sw)}")

    return "\n".join(L)
