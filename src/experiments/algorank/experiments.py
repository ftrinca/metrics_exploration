"""Chapter-level statistics over all 144 scenarios.

Everything here reads the caches: the per-scenario scores through the CIS
build (cis.build), which also carries the standard-deviation ratios the
degeneracy and variation analyses need. Run `python -m cis.build` before
`python -m metric_eval.experiments.algorank.summarize`.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

from core.ranking import rank_algorithms

from experiments.algorank import (build_rank_matrix, category_consensus,
                                  global_consensus)
from experiments.algorank.config import ALGO_METRICS, ALGO_NAMES, PATTERNS, RATES

# The cuts of the CIS gate, reused here so that "constant" and "diverging"
# mean the same thing in both chapters.
from experiments.cis.config import FLAT_THRESHOLD, UNSTABLE_THRESHOLD

RATE_BANDS = {"10-30": (10, 30), "40-50": (40, 50), "60-80": (60, 80)}

# The rulers of the spread robustness check (Appendix "The spread of a
# scenario"). MAE first, since it is the one the chapter reads.
SPREAD_RULERS = ("mae", "smae", "wd", "rmse", "dtw")

METRIC_PAIRS = list(itertools.combinations(ALGO_METRICS, 2))


def non_blackout(suite: dict[tuple, dict]) -> dict[tuple, dict]:
    """The 96 scenarios with MCAR or scattered missingness."""
    return {k: p for k, p in suite.items() if k[1] != "blackout"}


def rank_matrices(suite: dict[tuple, dict]) -> dict[tuple, dict[str, dict[str, float]]]:
    """{scenario: {metric: {algo: average rank}}} for every scenario."""
    return {key: build_rank_matrix(payload["scores"]) for key, payload in suite.items()}


def _tau(ranks_a: dict[str, float], ranks_b: dict[str, float]) -> float:
    algos = list(ranks_a)
    return float(kendalltau([ranks_a[a] for a in algos],
                            [ranks_b[a] for a in algos]).statistic)


def scenario_agreement(rank_matrix: dict[str, dict[str, float]]) -> float:
    """Mean Kendall tau_b over the 28 pairs of metrics of one scenario."""
    return float(np.mean([_tau(rank_matrix[a], rank_matrix[b])
                          for a, b in METRIC_PAIRS]))


def _winners(ranks: dict[str, float]) -> frozenset[str]:
    best = min(ranks.values())
    return frozenset(a for a, r in ranks.items() if r == best)


def _losers(ranks: dict[str, float]) -> frozenset[str]:
    worst = max(ranks.values())
    return frozenset(a for a, r in ranks.items() if r == worst)


def agreement_by_condition(matrices: dict[tuple, dict]) -> dict:
    """Mean per-scenario agreement, overall per pattern and per (pattern, rate)."""
    by_pattern = {p: [] for p in PATTERNS}
    by_rate = {p: {round(r * 100): [] for r in RATES} for p in PATTERNS}
    for (_, pattern, rate), rm in matrices.items():
        value = scenario_agreement(rm)
        by_pattern[pattern].append(value)
        by_rate[pattern][rate].append(value)
    return {
        "by_pattern": {p: float(np.mean(v)) for p, v in by_pattern.items()},
        "by_rate": {p: {r: float(np.mean(v)) for r, v in rates.items()}
                    for p, rates in by_rate.items()},
    }


def ranking_diversity(matrices: dict[tuple, dict]) -> dict:
    """How many different rankings the eight metrics produce per scenario."""
    histogram: dict[int, int] = {}
    unanimous = []
    for key, rm in matrices.items():
        distinct = len({tuple(sorted(rm[m].items())) for m in ALGO_METRICS})
        histogram[distinct] = histogram.get(distinct, 0) + 1
        if distinct == 1:
            unanimous.append(key)
    return {"histogram": dict(sorted(histogram.items())), "unanimous": unanimous}


def flat_consensus(matrices: dict[tuple, dict]) -> dict:
    """Mean rank and first-place share per algorithm, flat over metrics and scenarios.

    This is the construction behind the consensus table: every (metric,
    scenario) judgement counts once. A tie for first place splits the credit
    over the tied algorithms.
    """
    total_rank = {a: 0.0 for a in ALGO_NAMES}
    first = {a: 0.0 for a in ALGO_NAMES}
    n = 0
    for rm in matrices.values():
        for metric in ALGO_METRICS:
            n += 1
            for algo in ALGO_NAMES:
                total_rank[algo] += rm[metric][algo]
            tied = _winners(rm[metric])
            for algo in tied:
                first[algo] += 1.0 / len(tied)
    return {
        "mean_rank": {a: total_rank[a] / n for a in ALGO_NAMES},
        "first_share": {a: first[a] / n for a in ALGO_NAMES},
        "judgements": n,
    }


def _consensus_ranks(rank_matrix: dict[str, dict[str, float]]) -> dict[str, float]:
    """The category-weighted consensus of one scenario, as average ranks."""
    return rank_algorithms(global_consensus(category_consensus(rank_matrix)),
                           direction="lower")


def departure(matrices: dict[tuple, dict]) -> dict:
    """How far each metric sits from the category-weighted consensus ranking."""
    out = {m: {"tau": [], "reproduces": 0, "cdrec_first": 0} for m in ALGO_METRICS}
    for rm in matrices.values():
        consensus = _consensus_ranks(rm)
        for metric in ALGO_METRICS:
            cell = out[metric]
            cell["tau"].append(_tau(rm[metric], consensus))
            cell["reproduces"] += rm[metric] == consensus
            cell["cdrec_first"] += _winners(rm[metric]) == frozenset({"CDRec"})
    return {m: {"mean_tau": float(np.mean(c["tau"])),
                "reproduces": c["reproduces"], "cdrec_first": c["cdrec_first"]}
            for m, c in out.items()}


def spread(payload: dict) -> float:
    """(worst - best) / worst over the six MAE scores of one scenario.

    BRITS is dropped where it diverges (standard-deviation ratio above the
    gate's upper cut), because its error would decide the number on its own.
    """
    scores = dict(payload["scores"]["mae"])
    if payload["std_ratio"].get("BRITS", 0.0) > UNSTABLE_THRESHOLD:
        scores.pop("BRITS", None)
    values = [v for v in scores.values() if v is not None]
    worst, best = max(values), min(values)
    return (worst - best) / worst


def _ruler_spread(payload: dict, metric: str) -> float:
    """spread() measured with another lower-is-better metric as the ruler."""
    scores = dict(payload["scores"][metric])
    if payload["std_ratio"].get("BRITS", 0.0) > UNSTABLE_THRESHOLD:
        scores.pop("BRITS", None)
    values = [v for v in scores.values() if v is not None]
    worst, best = max(values), min(values)
    return (worst - best) / worst if worst else 0.0


def spread_quartiles(suite: dict[tuple, dict], matrices: dict[tuple, dict]) -> dict:
    """Agreement and unanimity of the winner, in four groups of 24 by spread."""
    keys = sorted(non_blackout(suite), key=lambda k: spread(suite[k]))
    quarter_size = len(keys) // 4
    out = []
    for i in range(4):
        quarter = keys[i * quarter_size:(i + 1) * quarter_size]
        agreements = [scenario_agreement(matrices[k]) for k in quarter]
        one_winner = sum(
            1 for k in quarter
            if len({_winners(matrices[k][m]) for m in ALGO_METRICS}) == 1
            and all(len(_winners(matrices[k][m])) == 1 for m in ALGO_METRICS))
        out.append({"mean_tau": float(np.mean(agreements)),
                    "one_winner": one_winner, "n": len(quarter)})
    return {"quarters": out}


def spread_rulers(suite: dict[tuple, dict], matrices: dict[tuple, dict]) -> dict:
    """The two appendix checks: each ruler against agreement, and ruler against ruler."""
    keys = list(non_blackout(suite))
    agreements = [scenario_agreement(matrices[k]) for k in keys]
    per_ruler = {m: [_ruler_spread(suite[k], m) for k in keys] for m in SPREAD_RULERS}

    rho_with_agreement = {
        m: float(spearmanr(values, agreements).statistic)
        for m, values in per_ruler.items()}
    between = {}
    for a, b in itertools.combinations(SPREAD_RULERS, 2):
        between[(a, b)] = float(spearmanr(per_ruler[a], per_ruler[b]).statistic)
    return {"rho_with_agreement": rho_with_agreement, "between_rulers": between}


def _band(rate: int) -> str:
    for name, (lo, hi) in RATE_BANDS.items():
        if lo <= rate <= hi:
            return name
    raise ValueError(rate)


def variation_preference(suite: dict[tuple, dict],
                         matrices: dict[tuple, dict]) -> dict:
    """Whether each metric places the reconstructions that keep the variation first.

    Per scenario: the Spearman correlation between each reconstruction's
    standard-deviation ratio and its rank, sign flipped so that positive means
    keeping the variation is rewarded. BRITS is excluded, since its ratios
    would decide the correlation on their own. Non-blackout scenarios only.
    """
    algos = [a for a in ALGO_NAMES if a != "BRITS"]
    out = {m: {"overall": [], **{b: [] for b in RATE_BANDS}} for m in ALGO_METRICS}
    for key in non_blackout(suite):
        rate = key[2]
        payload, rm = suite[key], matrices[key]
        ratios = [payload["std_ratio"][a] for a in algos]
        for metric in ALGO_METRICS:
            ranks = [rm[metric][a] for a in algos]
            r = spearmanr(ratios, [-x for x in ranks]).statistic
            if np.isnan(r):
                continue
            out[metric]["overall"].append(float(r))
            out[metric][_band(rate)].append(float(r))
    return {m: {band: float(np.mean(v)) if v else float("nan")
                for band, v in bands.items()} for m, bands in out.items()}


def variation_pairs(suite: dict[tuple, dict], matrices: dict[tuple, dict]) -> dict:
    """Of the pairs a metric orders differently from MAE, the share where it
    prefers the reconstruction that kept more of the variation."""
    prefers = {m: [0, 0] for m in ALGO_METRICS if m != "mae"}
    for key in non_blackout(suite):
        payload, rm = suite[key], matrices[key]
        for metric in prefers:
            for a, b in itertools.combinations(ALGO_NAMES, 2):
                base = rm["mae"][a] - rm["mae"][b]
                other = rm[metric][a] - rm[metric][b]
                if base * other >= 0:
                    continue
                preferred = a if other < 0 else b
                more_varied = max((a, b), key=lambda x: payload["std_ratio"][x])
                prefers[metric][0] += preferred == more_varied
                prefers[metric][1] += 1
    return {m: (hits / total if total else float("nan"))
            for m, (hits, total) in prefers.items()}


def degeneracy(suite: dict[tuple, dict], matrices: dict[tuple, dict]) -> dict:
    """Mean rank per metric of the constant, diverging and variation-matched
    reconstructions, over the non-blackout scenarios.

    "Constant" and "diverging" follow the CIS gate's cuts on the
    standard-deviation ratio; "matched" is the reconstruction whose ratio sits
    closest to 1 in a scenario that also holds a constant one, which is the
    comparison point of the zero-variation figure.
    """
    constant = {m: [] for m in ALGO_METRICS}
    diverging = {m: [] for m in ALGO_METRICS}
    matched = {m: [] for m in ALGO_METRICS}
    n_constant = n_diverging = 0
    for key in non_blackout(suite):
        payload, rm = suite[key], matrices[key]
        ratios = payload["std_ratio"]
        has_constant = False
        for algo in ALGO_NAMES:
            if ratios[algo] < FLAT_THRESHOLD:
                n_constant += 1
                has_constant = True
                for m in ALGO_METRICS:
                    constant[m].append(rm[m][algo])
            elif ratios[algo] > UNSTABLE_THRESHOLD:
                n_diverging += 1
                for m in ALGO_METRICS:
                    diverging[m].append(rm[m][algo])
        if has_constant:
            closest = min(ALGO_NAMES, key=lambda a: abs(ratios[a] - 1.0))
            for m in ALGO_METRICS:
                matched[m].append(rm[m][closest])
    return {
        "n_constant": n_constant, "n_diverging": n_diverging,
        "constant": {m: float(np.mean(v)) for m, v in constant.items()},
        "diverging": {m: float(np.mean(v)) for m, v in diverging.items()},
        "matched": {m: float(np.mean(v)) for m, v in matched.items()},
    }


def headline_numbers(suite: dict[tuple, dict], matrices: dict[tuple, dict]) -> dict:
    """The single numbers the chapter text cites outside its tables."""
    keys = list(non_blackout(suite))

    mae_rmse_differ = [k for k in keys if matrices[k]["mae"] != matrices[k]["rmse"]]
    rmse_not_flatter = 0
    for k in mae_rmse_differ:
        payload, rm = suite[k], matrices[k]
        for a, b in itertools.combinations(ALGO_NAMES, 2):
            x = rm["mae"][a] - rm["mae"][b]
            y = rm["rmse"][a] - rm["rmse"][b]
            if x * y < 0:
                rmse_pick = a if y < 0 else b
                flatter = min((a, b), key=lambda v: payload["std_ratio"][v])
                if rmse_pick != flatter:
                    rmse_not_flatter += 1
                    break

    majority_differs = {m: 0 for m in ALGO_METRICS}
    for k in keys:
        rm = matrices[k]
        counts: dict[frozenset, int] = {}
        for m in ALGO_METRICS:
            w = _winners(rm[m])
            counts[w] = counts.get(w, 0) + 1
        majority = max(counts, key=counts.get)
        for m in ALGO_METRICS:
            majority_differs[m] += _winners(rm[m]) != majority

    without_jsd = [m for m in ALGO_METRICS if m != "jsd"]
    def _mean_agreement(metric_set):
        pairs = list(itertools.combinations(metric_set, 2))
        return float(np.mean([
            np.mean([_tau(matrices[k][a], matrices[k][b]) for a, b in pairs])
            for k in keys]))
    agreement_all = _mean_agreement(ALGO_METRICS)
    agreement_wo_jsd = _mean_agreement(without_jsd)

    distinct_first = float(np.mean(
        [len({_winners(matrices[k][m]) for m in ALGO_METRICS}) for k in keys]))
    distinct_last = float(np.mean(
        [len({_losers(matrices[k][m]) for m in ALGO_METRICS}) for k in keys]))

    rank_range = {
        a: float(np.mean([max(matrices[k][m][a] for m in ALGO_METRICS)
                          - min(matrices[k][m][a] for m in ALGO_METRICS)
                          for k in keys]))
        for a in ALGO_NAMES}

    mpin_flat_share = float(np.mean(
        [suite[k]["std_ratio"]["MPIN"] < 0.5 for k in keys]))

    high = [k for k in keys if k[2] >= 60]
    stmvl_first_high = {
        m: sum(1 for k in high if "STMVL" in _winners(matrices[k][m]))
        for m in ("rmse", "r2")}

    mi_zero = mi_zero_not_constant = 0
    for k, payload in suite.items():
        for a in ALGO_NAMES:
            if payload["scores"]["mi"].get(a) == 0.0:
                mi_zero += 1
                mi_zero_not_constant += payload["std_ratio"][a] > 0.0

    neither, both = [], []
    for k in keys:
        ratios = suite[k]["std_ratio"]
        has_c = any(ratios[a] < FLAT_THRESHOLD for a in ALGO_NAMES)
        has_d = any(ratios[a] > UNSTABLE_THRESHOLD for a in ALGO_NAMES)
        if not has_c and not has_d:
            neither.append(k)
        if has_c and has_d:
            both.append(k)

    return {
        "mae_rmse_differ": len(mae_rmse_differ),
        "rmse_not_flatter": rmse_not_flatter,
        "majority_differs": majority_differs,
        "agreement_all": agreement_all,
        "agreement_without_jsd": agreement_wo_jsd,
        "distinct_first": distinct_first, "distinct_last": distinct_last,
        "rank_range": rank_range,
        "mpin_flat_share": mpin_flat_share,
        "stmvl_first_high": {"n_high": len(high), **stmvl_first_high},
        "mi_zero": mi_zero, "mi_zero_not_constant": mi_zero_not_constant,
        "tau_neither": float(np.mean([scenario_agreement(matrices[k]) for k in neither])),
        "tau_both": float(np.mean([scenario_agreement(matrices[k]) for k in both])),
        "n_neither": len(neither), "n_both": len(both),
    }


def worked_examples(suite: dict[tuple, dict], matrices: dict[tuple, dict]) -> dict:
    """The per-scenario numbers the chapter's worked examples cite."""
    def mae_ratio(key, a, b):
        s = suite[key]["scores"]["mae"]
        return s[a] / s[b]

    def worst_over_best(key, exclude_diverging_brits=True):
        s = dict(suite[key]["scores"]["mae"])
        if exclude_diverging_brits and \
                suite[key]["std_ratio"]["BRITS"] > UNSTABLE_THRESHOLD:
            s.pop("BRITS")
        values = list(s.values())
        return max(values) / min(values)

    return {
        "forecast-economy mcar 10, MPIN over CDRec (MAE)":
            mae_ratio(("forecast-economy", "mcar", 10), "MPIN", "CDRec"),
        "temperature scattered 20, worst over best (MAE)":
            worst_over_best(("temperature", "scattered", 20)),
        "drift mcar 50, worst over best (MAE)":
            worst_over_best(("drift", "mcar", 50)),
        "drift mcar 50, agreement":
            scenario_agreement(matrices[("drift", "mcar", 50)]),
        "forecast-economy mcar 70, agreement":
            scenario_agreement(matrices[("forecast-economy", "mcar", 70)]),
        "electricity mcar 70, agreement":
            scenario_agreement(matrices[("electricity", "mcar", 70)]),
        "chlorine scattered 40, BRITS over best (MAE)":
            suite[("chlorine", "scattered", 40)]["scores"]["mae"]["BRITS"]
            / min(v for v in
                  suite[("chlorine", "scattered", 40)]["scores"]["mae"].values()
                  if v is not None),
        "climate scattered 70, worst over best (MAE)":
            worst_over_best(("climate", "scattered", 70)),
    }


def agreement_by_dataset(matrices: dict[tuple, dict]) -> dict[str, float]:
    """Mean per-scenario agreement per dataset, non-blackout scenarios only.

    Blackout is left out for the same reason it is left out of the spread
    analysis: four algorithms tie there, so its near-zero agreement would say
    the same thing about every dataset.
    """
    per_dataset: dict[str, list[float]] = {}
    for key, rm in matrices.items():
        if key[1] == "blackout":
            continue
        per_dataset.setdefault(key[0], []).append(scenario_agreement(rm))
    return {d: float(np.mean(v)) for d, v in per_dataset.items()}


def pearson_substitution(suite: dict[tuple, dict],
                         matrices: dict[tuple, dict]) -> dict:
    """The chapter's headline results with Pearson correlation in place of R².

    Pearson is scored in the cache alongside the chapter's eight metrics, so
    the substituted rankings read the same reconstructions. "Leaves the
    results unchanged" is checked on the three results the chapter builds on:
    the mean agreement, the consensus order, and how often the two candidate
    metrics rank the six algorithms identically.
    """
    from core.metric_config import METRIC_DIRECTION
    keys = list(non_blackout(suite))
    pearson_ranks = {
        k: rank_algorithms(suite[k]["scores"]["pearson"],
                           METRIC_DIRECTION["pearson"]) for k in keys}

    def substituted(k: tuple, metric: str) -> dict[str, float]:
        return pearson_ranks[k] if metric == "r2" else matrices[k][metric]

    agreement_sub = float(np.mean([
        np.mean([_tau(substituted(k, a), substituted(k, b))
                 for a, b in METRIC_PAIRS]) for k in keys]))
    agreement_orig = float(np.mean([scenario_agreement(matrices[k])
                                    for k in keys]))

    def consensus_order(use_pearson: bool) -> tuple[str, ...]:
        total = {a: 0.0 for a in ALGO_NAMES}
        for k in keys:
            for m in ALGO_METRICS:
                ranks = substituted(k, m) if use_pearson else matrices[k][m]
                for a in ALGO_NAMES:
                    total[a] += ranks[a]
        return tuple(sorted(ALGO_NAMES, key=lambda a: total[a]))

    identical = sum(pearson_ranks[k] == matrices[k]["r2"] for k in keys)
    mean_tau = float(np.mean([_tau(pearson_ranks[k], matrices[k]["r2"])
                              for k in keys]))
    return {
        "agreement_original": agreement_orig,
        "agreement_substituted": agreement_sub,
        "order_original": consensus_order(False),
        "order_substituted": consensus_order(True),
        "identical_rankings": identical, "n": len(keys),
        "mean_tau_r2_pearson": mean_tau,
    }


# The families of the largest-position analysis: metrics whose per-position
# terms are rescalings of one another share one row, since their shares are
# identical by construction.
POINTWISE_METRICS = ("mae/nd", "rmse/mse/nrmse", "mre", "smape")


def pointwise_stability(suite: dict[tuple, dict]) -> dict:
    """How much of a pointwise metric one position can decide.

    For every reconstructed series, the share of the metric's per-position
    terms carried by the single largest term. This is the instability the
    chapter attributes to MRE — a near-zero true value produces a ratio large
    enough to decide the average on its own — measured for every family of
    the pointwise category on the same reconstructions. Needs the build
    caches on disk; scenarios without one are skipped and counted.
    """
    import json as _json
    from experiments.algorank import cache as _cache

    shares: dict[str, list[float]] = {m: [] for m in POINTWISE_METRICS}
    scenarios = missing = 0
    for (dataset, pattern, rate) in non_blackout(suite):
        det_path = _cache.deterministic_path(dataset, pattern, rate / 100)
        if not os.path.exists(det_path):
            missing += 1
            continue
        with open(det_path) as f:
            built = _json.load(f)
        y_true = np.array(built["y_true"])
        mask = np.array(built["mask"]).astype(bool)
        recons = [np.array(built[a]) for a in ("CDRec", "ROSL",
                                               "DynaMMo", "STMVL")]
        seed_path = os.path.join(_cache.seed_dir(dataset, pattern,
                                                 rate / 100, 0), "data.json")
        if os.path.exists(seed_path):
            with open(seed_path) as f:
                drawn = _json.load(f)
            recons += [np.array(drawn[a]) for a in ("BRITS", "MPIN")]
        scenarios += 1
        for rec in recons:
            for s in range(y_true.shape[0]):
                y, r = y_true[s][mask[s]], rec[s][mask[s]]
                e = np.abs(y - r)
                if e.sum() == 0:
                    continue
                shares["mae/nd"].append(float(e.max() / e.sum()))
                e2 = e ** 2
                shares["rmse/mse/nrmse"].append(float(e2.max() / e2.sum()))
                nz = y != 0
                t = e[nz] / np.abs(y[nz])
                shares["mre"].append(float(t.max() / t.sum()))
                sm = e / (0.5 * (np.abs(y) + np.abs(r)) + 1e-300)
                shares["smape"].append(float(sm.max() / sm.sum()))
    return {
        "scenarios": scenarios, "missing": missing,
        "median": {m: float(np.median(v)) for m, v in shares.items()},
        "p90": {m: float(np.percentile(v, 90)) for m, v in shares.items()},
        "over_half": {m: float(np.mean(np.array(v) > 0.5))
                      for m, v in shares.items()},
    }


def algo_ratio_medians(suite: dict[tuple, dict]) -> dict[str, float]:
    """Median standard-deviation ratio per algorithm over all 144 scenarios."""
    return {a: float(np.median([p["std_ratio"][a] for p in suite.values()]))
            for a in ALGO_NAMES}


def blackout_identity(suite: dict[tuple, dict]) -> dict:
    """Whether the four deterministic algorithms coincide under a blackout.

    Chapter 5 claims they return the same reconstruction, not merely four
    constant ones, so the check compares the reconstructed values themselves
    at the masked positions. Needs the build caches on disk.
    """
    import json as _json
    from experiments.algorank import cache as _cache

    deterministic = ("CDRec", "ROSL", "DynaMMo", "STMVL")
    scenarios = identical = 0
    worst = 0.0
    missing = []
    for (dataset, pattern, rate) in suite:
        if pattern != "blackout":
            continue
        path = _cache.deterministic_path(dataset, pattern, rate / 100)
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            built = _json.load(f)
        mask = np.array(built["mask"]).astype(bool)
        gap = [np.array(built[a])[mask] for a in deterministic]
        scenarios += 1
        spread_across = float(max(np.max(np.abs(g - gap[0])) for g in gap[1:]))
        worst = max(worst, spread_across)
        identical += spread_across < 1e-9
    return {"scenarios": scenarios, "identical": identical,
            "max_difference": worst, "missing": len(missing)}


def jsd_bin_robustness(suite: dict[tuple, dict]) -> dict:
    """JSD's preference for a diverging reconstruction, across histogram choices.

    For every non-blackout scenario holding a diverging reconstruction, JSD is
    recomputed per series and averaged exactly as the scoring stage does, at
    bin counts 10, 100 and 1000 over the shared range, and once at the default
    count with both series binned over the range of the truth alone. Reported
    is how often a diverging reconstruction still comes out first. Needs the
    build caches on disk.
    """
    import json as _json
    from experiments.algorank import cache as _cache
    from experiments.algorank.config import N_SEEDS

    def jsd_series(y, r, bins, truth_range):
        source = y if truth_range else np.concatenate([y, r])
        lo, hi = float(source.min()), float(source.max())
        p, _ = np.histogram(y, bins=bins, range=(lo, hi))
        q, _ = np.histogram(r, bins=bins, range=(lo, hi))
        p = p.astype(float) + 1e-10
        q = q.astype(float) + 1e-10
        p, q = p / p.sum(), q / q.sum()
        m = (p + q) / 2
        kl = lambda a, b: float(np.sum(a * np.log(a / b)))
        return (kl(p, m) + kl(q, m)) / 2

    variants = {"default": (None, False), "bins 10": (10, False),
                "bins 100": (100, False), "bins 1000": (1000, False),
                "truth-range": (None, True)}
    counts = {v: [0, 0] for v in variants}
    missing = 0
    for key, payload in suite.items():
        dataset, pattern, rate = key
        if pattern == "blackout":
            continue
        diverging = [a for a in ALGO_NAMES
                     if payload["std_ratio"][a] > UNSTABLE_THRESHOLD]
        if not diverging:
            continue
        det_path = _cache.deterministic_path(dataset, pattern, rate / 100)
        if not os.path.exists(det_path):
            missing += 1
            continue
        with open(det_path) as f:
            built = _json.load(f)
        y_true = np.array(built["y_true"])
        mask = np.array(built["mask"]).astype(bool)
        recon = {a: [np.array(built[a])] for a in ("CDRec", "ROSL",
                                                   "DynaMMo", "STMVL")}
        for seed in range(N_SEEDS):
            with open(os.path.join(_cache.seed_dir(dataset, pattern,
                                                   rate / 100, seed),
                                   "data.json")) as f:
                drawn = _json.load(f)
            for a in ("BRITS", "MPIN"):
                recon.setdefault(a, []).append(np.array(drawn[a]))
        for variant, (bins, truth_range) in variants.items():
            scores = {}
            for a, draws in recon.items():
                per_draw = []
                for r in draws:
                    per_series = []
                    for s in range(y_true.shape[0]):
                        y, rr = y_true[s][mask[s]], r[s][mask[s]]
                        b = bins or max(10, int(np.sqrt(len(y))))
                        per_series.append(jsd_series(y, rr, b, truth_range))
                    per_draw.append(float(np.mean(per_series)))
                scores[a] = float(np.mean(per_draw))
            first = min(scores, key=scores.get)
            counts[variant][0] += first in diverging
            counts[variant][1] += 1
    return {"variants": {v: tuple(c) for v, c in counts.items()},
            "missing": missing}
