import numpy as np
from scipy.stats import spearmanr

from core.metric_config import METRIC_DIRECTION
from core.ranking import rank_algorithms
from algo_ranking.analysis import build_rank_matrix, category_consensus, global_consensus
from algo_ranking.config import ALGO_CATEGORIES, PATTERNS

from cis.config import (ALGO_NAMES, CIS_METRICS, FLAT_THRESHOLD,
                        UNSTABLE_THRESHOLD)
from cis.gate import component_values, variant_cis
from cis.injector_data import load_injector_response


def _metric_rank(scores: dict[str, dict[str, float]], metric: str,
                 subjects: list[str]) -> dict[str, float]:
    """Rank `subjects` on one metric, in that metric's own direction."""
    return rank_algorithms({s: scores[metric][s] for s in subjects},
                           direction=METRIC_DIRECTION[metric])


def within_category_agreement(
    scenario_scores: dict[tuple, dict],
    categories: dict[str, list[str]] = ALGO_CATEGORIES,
    algorithms: list[str] = ALGO_NAMES,
) -> dict[str, dict]:
    """Mean Spearman correlation between the two metrics inside each category.

    CIS keeps one metric per category, which is free of consequence only where
    the two members rank algorithms the same way. Reported for all scenarios and
    split by geometry.
    """
    out = {}
    for category, metrics in categories.items():
        if len(metrics) != 2:
            continue
        x, y = metrics

        def mean_rho(keys) -> float:
            vals = []
            for key in keys:
                scores = scenario_scores[key]
                present = [a for a in algorithms
                           if scores[x].get(a) is not None and scores[y].get(a) is not None]
                if len(present) < 3:
                    continue
                rx = _metric_rank(scores, x, present)
                ry = _metric_rank(scores, y, present)
                rho, _ = spearmanr([rx[a] for a in present], [ry[a] for a in present])
                if not np.isnan(rho):
                    vals.append(float(rho))
            return float(np.mean(vals)) if vals else float("nan")

        keys = list(scenario_scores)
        out[category] = {
            "pair": (x, y),
            "all": mean_rho(keys),
            "non_blackout": mean_rho([k for k in keys if k[1] != "blackout"]),
            "blackout": mean_rho([k for k in keys if k[1] == "blackout"]),
        }
    return out


def component_spread(per_subject: dict[str, dict[str, float]]) -> dict[str, dict]:
    """Range and relative range of each component across whatever it was computed over.

    A component whose relative spread is far below the others contributes almost
    nothing to CIS's ordering.
    """
    out = {}
    for component in ("M", "D", "T", "I"):
        vals = [c[component] for c in per_subject.values()]
        lo, hi, mean = min(vals), max(vals), float(np.mean(vals))
        out[component] = {"min": lo, "max": hi,
                          "relative_spread": (hi - lo) / mean if mean else 0.0}
    return out


def variant_agreement(
    rows: list[dict],
    scenario_scores: dict[tuple, dict],
    n_timesteps: dict[tuple, int],
    slots: tuple[str, ...],
    min_survivors: int = 3,
) -> dict:
    """Agreement between a CIS variant's survivor ranking and the panel consensus.

    Scenarios with fewer than `min_survivors` are skipped, because a rank
    correlation over two points is either +1 or -1 whatever the values are.
    """
    by_scenario: dict[tuple, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault((r["dataset"], r["pattern"], r["rate"]), []).append(r)

    rhos, skipped = [], 0
    for key, scenario_rows in by_scenario.items():
        survivors = [r["algo"] for r in scenario_rows if r["passes_gate"]]
        if len(survivors) < min_survivors:
            skipped += 1
            continue
        scores, n = scenario_scores[key], n_timesteps[key]
        cis_rank = rank_algorithms(
            {a: variant_cis(scores, a, n, slots) for a in survivors}, direction="higher")
        restricted = {m: {a: scores[m][a] for a in survivors} for m in scores}
        cons_rank = global_consensus(category_consensus(build_rank_matrix(restricted)))
        rho, _ = spearmanr([cis_rank[a] for a in survivors],
                           [cons_rank[a] for a in survivors])
        if not np.isnan(rho):
            rhos.append(float(rho))

    v = np.array(rhos)
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "min": float(v.min()), "pct_at_least_0_9": float(100 * np.mean(v >= 0.9)),
            "skipped": skipped}


def reactivity_response(conditions: dict[str, dict]) -> dict:
    """CIS and the gate ratio for each of Experiment 1's eight distortions.

    Experiment 1 solves every distortion to the same mean absolute error, so the M
    component is pinned and whatever variation is left in CIS is variation in
    the kind of damage rather than its size.
    """
    per_condition, per_distortion = {}, {}
    for condition, payload in conditions.items():
        scores, n = payload["scores"], payload["n_timesteps"]
        names = [d for d in payload["iqr_ratio"] if scores["mae"].get(d) is not None]
        values = {d: variant_cis(scores, d, n, CIS_METRICS) for d in names}
        per_condition[condition] = {
            "cis": values,
            "iqr_ratio": {d: payload["iqr_ratio"][d] for d in names},
            "relative_spread": (max(values.values()) - min(values.values()))
                               / float(np.mean(list(values.values()))),
            "components": {d: component_values(scores, d, n) for d in names},
        }
        for d, v in values.items():
            per_distortion.setdefault(d, []).append(v)
    return {"per_condition": per_condition,
            "mean_per_distortion": {d: float(np.mean(v)) for d, v in per_distortion.items()}}


def response_curve(sweep: dict[str, dict]) -> dict:
    """CIS at each of Experiment 1's damage levels, per distortion.

    A composite that is not monotone in damage would report a more damaged
    reconstruction as the better one, so this is a correctness check.
    """
    out = {}
    for distortion, payload in sweep.items():
        scores, n = payload["scores"], payload["n_timesteps"]
        levels = sorted(payload["iqr_ratio"])
        values = [variant_cis(scores, L, n, CIS_METRICS) for L in levels]
        out[distortion] = {
            "damage_levels": payload.get("damage_levels"),
            "cis": values,
            "iqr_ratio": [payload["iqr_ratio"][L] for L in levels],
            "monotone": all(b < a for a, b in zip(values, values[1:])),
            "drop": values[0] - values[-1],
        }
    return out


def gate_outcome_table(rows: list[dict]) -> dict[str, dict]:
    """{algorithm: {pattern: {passed, total, median_ratio}}}."""
    out = {}
    for algo in ALGO_NAMES:
        out[algo] = {}
        for pattern in PATTERNS:
            subset = [r for r in rows if r["algo"] == algo and r["pattern"] == pattern]
            if not subset:
                continue
            out[algo][pattern] = {
                "passed": sum(1 for r in subset if r["passes_gate"]),
                "total": len(subset),
                "median_ratio": float(np.median([r["iqr_ratio"] for r in subset])),
            }
    return out


def gate_and_mi_overlap(rows: list[dict], scenario_scores: dict[tuple, dict]) -> dict:
    """How much of the I component's job the gate has already done.

    MI is exactly 0 for a constant reconstruction, which the flat check also
    catches. The two are not redundant if MI still varies among the survivors,
    which the correlation below measures.
    """
    def mi_of(r):
        return scenario_scores[(r["dataset"], r["pattern"], r["rate"])]["mi"][r["algo"]]

    flat = [r for r in rows if not r["passes_gate"] and r["iqr_ratio"] < FLAT_THRESHOLD]
    survivors = [r for r in rows if r["passes_gate"]]
    mi_flat = [mi_of(r) for r in flat]
    mi_pass = [mi_of(r) for r in survivors]
    rho, _ = spearmanr([r["iqr_ratio"] for r in survivors], mi_pass)
    return {
        "n_flat_failures": len(mi_flat),
        "n_survivors": len(mi_pass),
        "mi_zero_among_flat_failures": sum(1 for v in mi_flat if v == 0.0),
        "mi_zero_among_survivors": sum(1 for v in mi_pass if v == 0.0),
        "median_mi_flat_failures": float(np.median(mi_flat)) if mi_flat else float("nan"),
        "median_mi_survivors": float(np.median(mi_pass)) if mi_pass else float("nan"),
        "rho_iqr_ratio_vs_mi_among_survivors": float(rho),
    }

COMPONENT_VARIANTS = [
    ("adopted (MAE, WD, DTW, MI)", CIS_METRICS),
    ("RMSE in place of MAE", ("rmse", "wd", "dtw", "mi")),
    ("JSD in place of WD", ("mae", "jsd", "dtw", "mi")),
    ("sMAE in place of DTW", ("mae", "wd", "smae", "mi")),
    ("R2 in place of MI", ("mae", "wd", "dtw", "r2")),
    ("three components, MI dropped", ("mae", "wd", "dtw")),
    ("two components, MAE and DTW", ("mae", "dtw")),
    ("one component, MAE alone", ("mae",)),
]


def supporting_experiments(rows, scenario_scores, n_timesteps, conditions) -> str:
    """The text report behind the CIS chapter's design section."""
    L = ["CIS SUPPORTING EXPERIMENTS", "=" * 78, ""]

    L.append("1. WITHIN-CATEGORY AGREEMENT  (Spearman between each category's two metrics)")
    L.append(f"   {'category':30s}{'pair':16s}{'all':>8}{'non-blackout':>15}{'blackout':>11}")
    for cat, info in within_category_agreement(scenario_scores).items():
        x, y = info["pair"]
        L.append(f"   {cat:30s}{x + '/' + y:16s}{info['all']:8.2f}"
                 f"{info['non_blackout']:15.2f}{info['blackout']:11.2f}")
    L.append("")

    L.append("2. COMPONENT SUBSTITUTIONS (gated survivors, against the 8-metric consensus)")
    L.append(f"   {'variant':32s}{'n':>4}{'mean':>8}{'median':>9}{'min':>8}{'%>=0.9':>9}")
    for label, slots in COMPONENT_VARIANTS:
        r = variant_agreement(rows, scenario_scores, n_timesteps, slots)
        L.append(f"   {label:32s}{r['n']:4d}{r['mean']:8.3f}{r['median']:9.3f}"
                 f"{r['min']:8.3f}{r['pct_at_least_0_9']:8.1f}%")
    L.append("")

    L.append("3. GATE OUTCOME BY ALGORITHM AND GEOMETRY (passed/total, median iqr_ratio)")
    table = gate_outcome_table(rows)
    L.append(f"   {'algorithm':12s}" + "".join(f"{p:>24}" for p in PATTERNS))
    for algo, per_pattern in table.items():
        cells = "".join(f"{per_pattern[p]['passed']:8d}/{per_pattern[p]['total']:<3d}"
                        f"({per_pattern[p]['median_ratio']:8.3f})" for p in PATTERNS
                        if p in per_pattern)
        L.append(f"   {algo:12s}{cells}")
    L.append("")

    L.append("4. GATE AND THE I COMPONENT")
    ov = gate_and_mi_overlap(rows, scenario_scores)
    L.append(f"   flat failures {ov['n_flat_failures']}, survivors {ov['n_survivors']}")
    L.append(f"   MI exactly 0 in {ov['mi_zero_among_flat_failures']} flat failures "
             f"and {ov['mi_zero_among_survivors']} survivors")
    L.append(f"   median MI: {ov['median_mi_flat_failures']:.4f} among flat failures, "
             f"{ov['median_mi_survivors']:.4f} among survivors")
    L.append(f"   rho(iqr_ratio, MI) among survivors = "
             f"{ov['rho_iqr_ratio_vs_mi_among_survivors']:.3f}")
    L.append("")

    if not conditions:
        L.append("5. DAMAGE-REACTIVITY RESPONSE: skipped, no Injector cache found.")
        L.append("")
    else:
        eq = reactivity_response(conditions)
        names = list(next(iter(eq["per_condition"].values()))["cis"])
        L.append("5. RESPONSE UNDER EQUAL DAMAGE"
                 "  (Experiment 1's eight distortions at equal MAE)")
        L.append(f"   {'condition':18s}" + "".join(f"{d[:9]:>10}" for d in names)
                 + f"{'spread':>9}")
        for condition in sorted(eq["per_condition"]):
            c = eq["per_condition"][condition]
            L.append(f"   {condition:18s}" + "".join(f"{c['cis'][d]:10.4f}" for d in names)
                     + f"{c['relative_spread']:9.3f}")
        mean = eq["mean_per_distortion"]
        L.append("   mean per distortion, least penalised first:")
        for d in sorted(mean, key=mean.get, reverse=True):
            L.append(f"      {d:12s} {mean[d]:.4f}")
        L.append(f"   worst/best ratio: {max(mean.values()) / min(mean.values()):.2f}")

        merged = {d: {c: float(np.mean([cond["components"][d][c]
                                        for cond in eq["per_condition"].values()]))
                      for c in "MDTI"} for d in names}
        L.append("   component spread across the eight (mean over conditions):")
        for comp, info in component_spread(merged).items():
            L.append(f"      {comp}  {info['min']:.4f} to {info['max']:.4f}   "
                     f"relative spread {info['relative_spread']:.3f}")
        failed = sum(1 for c in eq["per_condition"].values() for v in c["iqr_ratio"].values()
                     if not (FLAT_THRESHOLD <= v <= UNSTABLE_THRESHOLD))
        total = sum(len(c["iqr_ratio"]) for c in eq["per_condition"].values())
        L.append(f"   gate: {failed} of {total} distortion-conditions fail")
        L.append("")

    sweep = load_injector_response()
    if not sweep:
        L.append("6. DAMAGE-SWEEP RESPONSE: skipped, no Injector sweep cache found.")
    else:
        sw = response_curve(sweep)
        levels = next(iter(sw.values()))["damage_levels"] or []
        L.append("6. RESPONSE ACROSS THE DAMAGE-RESPONSE CURVE")
        L.append(f"   {'distortion':13s}" + "".join(f"{l:>9}" for l in levels)
                 + f"{'monotone':>10}{'drop':>8}")
        for distortion, info in sw.items():
            L.append(f"   {distortion:13s}" + "".join(f"{v:9.4f}" for v in info["cis"])
                     + f"{str(info['monotone']):>10}{info['drop']:8.3f}")
        L.append(f"   monotone in {sum(1 for v in sw.values() if v['monotone'])}"
                 f"/{len(sw)} sweeps")
        L.append("   gate over the same sweep:")
        for distortion, info in sw.items():
            L.append(f"   {distortion:13s}" + "".join(f"{v:9.3f}" for v in info["iqr_ratio"]))

    return "\n".join(L)
