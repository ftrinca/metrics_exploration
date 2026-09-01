"""Render the chapter-level statistics as one text report.

Every number Chapter 5 cites outside the per-condition reports comes from
here, so the thesis can be checked against one file.
"""
from __future__ import annotations

from experiments.algorank.config import ALGO_METRICS, ALGO_NAMES, PATTERNS, RATES, label
from experiments.algorank.experiments import (POINTWISE_METRICS, RATE_BANDS,
                                              SPREAD_RULERS, agreement_by_condition,
                                              agreement_by_dataset,
                                              algo_ratio_medians, blackout_identity,
                                              degeneracy, departure, flat_consensus,
                                              headline_numbers, jsd_bin_robustness,
                                              pearson_substitution,
                                              pointwise_stability, ranking_diversity,
                                              spread_quartiles, spread_rulers,
                                              variation_pairs, variation_preference,
                                              worked_examples)


def write_report(suite: dict[tuple, dict], matrices: dict[tuple, dict],
                 dataset_axes: dict[str, tuple[float, float]]) -> str:
    """The text behind every chapter-level table of the ranking chapter."""
    L = ["ALGORITHM RANKING — CHAPTER-LEVEL STATISTICS", "=" * 78, "",
         f"{len(suite)} scenarios, {len(ALGO_NAMES)} algorithms, "
         f"{len(ALGO_METRICS)} metrics; agreement is Kendall tau_b, "
         "averaged over the 28 metric pairs of a scenario", ""]

    ac = agreement_by_condition(matrices)
    L += ["1. AGREEMENT BY CONDITION",
          "   by pattern: " + "  ".join(f"{p} {ac['by_pattern'][p]:.2f}"
                                        for p in PATTERNS),
          "   by rate:"]
    header = "   " + f"{'pattern':<12}" + "".join(
        f"{round(r * 100):>7}" for r in RATES)
    L.append(header)
    for pattern in PATTERNS:
        L.append("   " + f"{pattern:<12}" + "".join(
            f"{ac['by_rate'][pattern][round(r * 100)]:7.2f}" for r in RATES))
    L.append("")

    nb = {k: m for k, m in matrices.items() if k[1] != "blackout"}
    rd = ranking_diversity(nb)
    L += ["2. DISTINCT RANKINGS PER SCENARIO  (non-blackout)",
          "   " + "  ".join(f"{k}:{v}" for k, v in rd["histogram"].items())]
    rd_all = ranking_diversity(matrices)
    L += ["   unanimous scenarios (all 144): " +
          (", ".join("/".join(map(str, k)) for k in rd_all["unanimous"]) or "none"),
          ""]

    fc = flat_consensus(nb)
    L += ["3. CONSENSUS  (flat mean rank over the 8 metrics and the "
          f"{len(nb)} non-blackout scenarios; {fc['judgements']} judgements)",
          f"   {'algorithm':<10}{'mean rank':>11}{'first places':>14}"]
    for algo in sorted(ALGO_NAMES, key=lambda a: fc["mean_rank"][a]):
        L.append(f"   {algo:<10}{fc['mean_rank'][algo]:11.2f}"
                 f"{fc['first_share'][algo]:14.0%}")
    L.append("")

    dp = departure(nb)
    L += ["4. DEPARTURE FROM THE CATEGORY-WEIGHTED CONSENSUS  (non-blackout)",
          f"   {'metric':<7}{'mean tau':>10}{'reproduces':>12}{'CDRec first':>13}"]
    for metric in sorted(ALGO_METRICS, key=lambda m: -dp[m]["mean_tau"]):
        L.append(f"   {label(metric):<7}{dp[metric]['mean_tau']:10.2f}"
                 f"{dp[metric]['reproduces']:9d}/{len(nb)}"
                 f"{dp[metric]['cdrec_first']:10d}/{len(nb)}")
    L.append("")

    sq = spread_quartiles(suite, matrices)
    L += ["5. SPREAD  ((worst - best) / worst on MAE, diverging BRITS dropped)",
          f"   {'quarter':<18}{'mean tau':>10}{'one winner':>12}"]
    for i, q in enumerate(sq["quarters"], start=1):
        L.append(f"   {i} (24 scenarios){'':<2}{q['mean_tau']:>10.2f}"
                 f"{q['one_winner']:>9d}/{q['n']}")
    sr = spread_rulers(suite, matrices)
    L += ["   rho with agreement per ruler: " +
          "  ".join(f"{label(m)} {sr['rho_with_agreement'][m]:.2f}"
                    for m in SPREAD_RULERS)]
    between = sr["between_rulers"]
    lo = min(between, key=between.get)
    hi = max(between, key=between.get)
    L += [f"   rulers against each other: from {between[lo]:.2f} "
          f"({label(lo[0])}-{label(lo[1])}) to {between[hi]:.2f} "
          f"({label(hi[0])}-{label(hi[1])})", ""]

    vp = variation_preference(suite, matrices)
    L += ["6. VARIATION PREFERENCE  (BRITS excluded, non-blackout; "
          "positive: keeps the variation first)",
          "   " + f"{'metric':<7}{'overall':>9}" + "".join(f"{b:>9}" for b in RATE_BANDS)]
    for metric in sorted(ALGO_METRICS, key=lambda m: -vp[m]["overall"]):
        L.append(f"   {label(metric):<7}{vp[metric]['overall']:9.2f}" +
                 "".join(f"{vp[metric][b]:9.2f}" for b in RATE_BANDS))
    pairs = variation_pairs(suite, matrices)
    L += ["   share of MAE-disagreement pairs preferring the more varied "
          "reconstruction:",
          "   " + "  ".join(f"{label(m)} {v:.2f}"
                            for m, v in sorted(pairs.items(), key=lambda x: -x[1])),
          ""]

    dg = degeneracy(suite, matrices)
    L += [f"7. DEGENERATE RECONSTRUCTIONS  ({dg['n_constant']} constant, "
          f"{dg['n_diverging']} diverging, non-blackout; mean ranks, "
          "3.5 = middle of six)",
          f"   {'metric':<7}{'constant':>10}{'diverging':>11}{'matched':>10}"]
    for metric in sorted(ALGO_METRICS, key=lambda m: -dg["constant"][m]):
        L.append(f"   {label(metric):<7}{dg['constant'][metric]:10.2f}"
                 f"{dg['diverging'][metric]:11.2f}{dg['matched'][metric]:10.2f}")
    L.append("")

    h = headline_numbers(suite, matrices)
    L += ["8. HEADLINE NUMBERS  (non-blackout unless stated)",
          f"   MAE and RMSE produce different rankings in "
          f"{h['mae_rmse_differ']} scenarios; RMSE does not take the flatter "
          f"of every disagreed pair in {h['rmse_not_flatter']} of them",
          "   names a different winner from the majority: " +
          "  ".join(f"{label(m)} {h['majority_differs'][m]}"
                    for m in sorted(ALGO_METRICS,
                                    key=lambda x: -h['majority_differs'][x])),
          f"   mean agreement {h['agreement_all']:.3f}; without JSD "
          f"{h['agreement_without_jsd']:.3f} "
          f"(+{(h['agreement_without_jsd'] / h['agreement_all'] - 1):.0%})",
          f"   distinct winners per scenario {h['distinct_first']:.2f}, "
          f"distinct last places {h['distinct_last']:.2f}",
          "   mean rank range across the metrics: " +
          "  ".join(f"{a} {h['rank_range'][a]:.2f}" for a in ALGO_NAMES),
          f"   MPIN keeps under half the variation in "
          f"{h['mpin_flat_share']:.0%} of the scenarios",
          f"   ST-MVL first at rates 60-80%: RMSE "
          f"{h['stmvl_first_high']['rmse']}/{h['stmvl_first_high']['n_high']}, "
          f"R2 {h['stmvl_first_high']['r2']}/{h['stmvl_first_high']['n_high']}",
          f"   MI reads exactly 0 for {h['mi_zero']} reconstructions "
          f"(all 144 scenarios), {h['mi_zero_not_constant']} of them not constant",
          f"   agreement without any degenerate reconstruction "
          f"{h['tau_neither']:.2f} ({h['n_neither']} scenarios), with a constant "
          f"and a diverging one {h['tau_both']:.2f} ({h['n_both']} scenarios)", ""]

    we = worked_examples(suite, matrices)
    L += ["9. WORKED EXAMPLES"]
    for name, value in we.items():
        L.append(f"   {name}: {value:.2f}")
    L.append("")

    L += ["10. DATASET MAP  (complete series, before masking)",
          f"   {'dataset':<18}{'mean |cross-corr|':>19}{'lag-1 autocorr':>16}"]
    for dataset, (cross, lag1) in dataset_axes.items():
        L.append(f"   {dataset:<18}{cross:19.2f}{lag1:16.2f}")
    L.append("")

    ad = agreement_by_dataset(matrices)
    L += ["11. AGREEMENT BY DATASET  (non-blackout)",
          "   " + "  ".join(f"{d} {v:.2f}"
                            for d, v in sorted(ad.items(),
                                               key=lambda x: -x[1]))]
    L.append("")

    ps = pearson_substitution(suite, matrices)
    L += ["12. PEARSON IN PLACE OF R2  (non-blackout)",
          f"   mean agreement {ps['agreement_original']:.3f} original, "
          f"{ps['agreement_substituted']:.3f} substituted",
          "   consensus order original:    " + " > ".join(ps["order_original"]),
          "   consensus order substituted: " + " > ".join(ps["order_substituted"]),
          f"   Pearson and R2 rank identically in "
          f"{ps['identical_rankings']}/{ps['n']} scenarios, "
          f"mean tau {ps['mean_tau_r2_pearson']:.2f}", ""]

    st = pointwise_stability(suite)
    L += ["13. POINTWISE STABILITY  (share of a series' metric carried by its "
          "single largest position;",
          f"    per family, over the reconstructions of {st['scenarios']} "
          "cached scenarios)",
          f"   {'family':<16}{'median':>9}{'p90':>9}{'share > 0.5':>13}"]
    for m in sorted(POINTWISE_METRICS, key=lambda x: -st["median"][x]):
        L.append(f"   {m:<16}{st['median'][m]:9.3f}{st['p90'][m]:9.3f}"
                 f"{st['over_half'][m]:13.1%}")
    if st["missing"]:
        L.append(f"   {st['missing']} scenarios have no build cache on this "
                 "machine — run where the build caches live")
    L.append("")

    am = algo_ratio_medians(suite)
    L += ["14. MEDIAN STANDARD-DEVIATION RATIO PER ALGORITHM  (all 144 scenarios)",
          "   " + "  ".join(f"{a} {v:.2f}"
                            for a, v in sorted(am.items(),
                                               key=lambda x: -x[1])), ""]

    bi = blackout_identity(suite)
    L += ["15. BLACKOUT IDENTITY  (the four deterministic algorithms, at the "
          "masked positions)"]
    if bi["scenarios"]:
        L.append(f"   identical in {bi['identical']}/{bi['scenarios']} blackout "
                 f"scenarios (largest value difference "
                 f"{bi['max_difference']:.2e})")
    if bi["missing"]:
        L.append(f"   {bi['missing']} blackout scenarios have no build cache "
                 "on this machine — run where the build caches live")
    L.append("")

    jb = jsd_bin_robustness(suite)
    L += ["16. JSD HISTOGRAM ROBUSTNESS  (scenarios holding a diverging "
          "reconstruction; how often JSD still puts one first)"]
    for variant, (first, n) in jb["variants"].items():
        if n:
            L.append(f"   {variant:<12} {first}/{n}")
    if jb["missing"]:
        L.append(f"   {jb['missing']} scenarios have no build cache on this "
                 "machine — run where the build caches live")

    return "\n".join(L)
