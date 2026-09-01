import numpy as np
from scipy.stats import spearmanr

from metric_eval.core.metric_config import METRIC_DIRECTION
from metric_eval.core.ranking import rank_algorithms

from metric_eval.experiments.algorank.config import ALGO_METRICS, ALGO_NAMES, PATTERNS

from metric_eval.experiments.cis.config import (ADOPTED_POWER, CIS_METRICS, FLAT_THRESHOLD,
                        POWER_VARIANTS, UNSTABLE_THRESHOLD)
from metric_eval.experiments.cis.gate import MIN_SURVIVORS, passes, survivors
from metric_eval.experiments.cis.score import cis, combine, components

RATE_BANDS = {"10-30": (10, 30), "40-50": (40, 50), "60-80": (60, 80)}

COMPONENT_VARIANTS = [
    ("adopted (MAE, WD, MI)", CIS_METRICS),
    ("RMSE in place of MAE", ("rmse", "wd", "mi")),
    ("JSD in place of WD", ("mae", "jsd", "mi")),
    ("R2 in place of MI", ("mae", "wd", "r2")),
    ("DTW added", ("mae", "wd", "dtw", "mi")),
    ("without the divergence component", ("mae", "mi")),
    ("without the dependency component", ("mae", "wd")),
    ("MAE alone", ("mae",)),
    ("WD alone", ("wd",)),
    ("MI alone", ("mi",)),
]


def _band(rate: int) -> str:
    for name, (lo, hi) in RATE_BANDS.items():
        if lo <= rate <= hi:
            return name
    raise ValueError(rate)


def _metric_rank(payload, metric, subjects):
    return rank_algorithms({a: payload["scores"][metric].get(a) for a in subjects},
                           direction=METRIC_DIRECTION[metric])


def _cis_rank(payload, subjects, metrics=CIS_METRICS, power=ADOPTED_POWER):
    return rank_algorithms(
        {a: cis(payload["scores"], payload["reference"], a, metrics, power)
         for a in subjects}, direction="lower")


def gate_outcome(cache: dict[tuple, dict]) -> dict:
    """Counts of what the standard-deviation gate keeps, by algorithm and geometry."""
    per_algo = {a: {p: {"passed": 0, "total": 0, "ratios": []} for p in PATTERNS}
                for a in ALGO_NAMES}
    flat = unstable = kept = 0
    for (_, pattern, _), payload in cache.items():
        for algo, ratio in payload["std_ratio"].items():
            cell = per_algo[algo][pattern]
            cell["total"] += 1
            cell["ratios"].append(ratio)
            if ratio < FLAT_THRESHOLD:
                flat += 1
            elif ratio > UNSTABLE_THRESHOLD:
                unstable += 1
            else:
                kept += 1
                cell["passed"] += 1
    for algo in per_algo:
        for pattern in per_algo[algo]:
            cell = per_algo[algo][pattern]
            ratios = cell.pop("ratios")
            cell["median_ratio"] = float(np.median(ratios)) if ratios else None
    return {"kept": kept, "flat": flat, "unstable": unstable, "per_algo": per_algo}


def instrument_comparison(cache: dict[tuple, dict]) -> dict:
    """How the two spread measures classify the same reconstructions.

    The interquartile range ignores the tails by construction, so under a
    scattered pattern it reads exactly zero for a reconstruction that is constant
    over most of a burst and varies outside it. The standard deviation does not.
    """
    changed: dict[tuple, int] = {}
    zero_iqr_varied = []
    for key, payload in cache.items():
        for algo, std in payload["std_ratio"].items():
            iqr = payload["iqr_ratio"][algo]
            unique = payload["n_unique"][algo]
            if passes(iqr) != passes(std):
                label = ("flat" if iqr < FLAT_THRESHOLD else
                         "unstable" if iqr > UNSTABLE_THRESHOLD else "pass")
                other = ("flat" if std < FLAT_THRESHOLD else
                         "unstable" if std > UNSTABLE_THRESHOLD else "pass")
                changed[(label, other)] = changed.get((label, other), 0) + 1
            if iqr == 0.0 and unique > 1.0:
                zero_iqr_varied.append((key, algo, unique))
    return {
        "n_reclassified": sum(changed.values()),
        "transitions": changed,
        "zero_iqr_but_varied": len(zero_iqr_varied),
        "unique_range": ((min(u for *_, u in zero_iqr_varied),
                          max(u for *_, u in zero_iqr_varied))
                         if zero_iqr_varied else None),
        "n_constant": sum(1 for p in cache.values()
                          for u in p["n_unique"].values() if u == 1.0),
    }


def threshold_sensitivity(cache: dict[tuple, dict],
                          flats=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
                          uppers=(2.0, 3.0, 5.0, 10.0)) -> dict:
    """Survivors and rankable scenarios over a range of both cuts."""
    out = {"flat": {}, "unstable": {}}
    for flat in flats:
        kept = [len(survivors(p, flat, UNSTABLE_THRESHOLD)) for p in cache.values()]
        out["flat"][flat] = {"survivors": sum(kept),
                             "rankable": sum(1 for k in kept if k >= MIN_SURVIVORS)}
    for upper in uppers:
        kept = [len(survivors(p, FLAT_THRESHOLD, upper)) for p in cache.values()]
        out["unstable"][upper] = {"survivors": sum(kept),
                                  "rankable": sum(1 for k in kept if k >= MIN_SURVIVORS)}
    return out


def coverage(cache: dict[tuple, dict]) -> dict:
    """Where a ranking survives the gate, by geometry, rate and dataset."""
    by_geometry, by_rate, by_dataset, histogram = {}, {}, {}, {}
    for (dataset, pattern, rate), payload in cache.items():
        n = len(survivors(payload))
        histogram[n] = histogram.get(n, 0) + 1
        if n >= MIN_SURVIVORS:
            by_geometry[pattern] = by_geometry.get(pattern, 0) + 1
            by_rate[rate] = by_rate.get(rate, 0) + 1
            by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
    return {"by_geometry": by_geometry, "by_rate": dict(sorted(by_rate.items())),
            "by_dataset": by_dataset,
            "survivor_histogram": dict(sorted(histogram.items())),
            "n_rankable": sum(1 for p in cache.values()
                              if len(survivors(p)) >= MIN_SURVIVORS)}


def component_profile(cache: dict[tuple, dict]) -> dict:
    """Distribution of each component over the gate survivors.

    A component whose distances sit far below the others contributes almost
    nothing to the power mean, whichever exponent is used.
    """
    collected = {m: [] for m in CIS_METRICS}
    for payload in cache.values():
        for algo in survivors(payload):
            values = components(payload["scores"], payload["reference"], algo)
            if values is None:
                continue
            for metric, value in values.items():
                collected[metric].append(value)
    return {m: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90))} for m, v in collected.items()}


def variation_preference(cache: dict[tuple, dict], gated: bool) -> dict:
    """Correlation between a ranking and how much variation a reconstruction keeps.

    Positive means the ranking puts reconstructions that keep the variation of the
    truth first, negative means it puts flattened ones first. Experiment 2 reports
    the eight metrics along this axis, so CIS can be read on the same scale as its
    own components.
    """
    rankings = list(ALGO_METRICS) + ["CIS"]
    out = {name: {band: [] for band in RATE_BANDS} for name in rankings}
    for (_, pattern, rate), payload in cache.items():
        if pattern == "blackout" and not gated:
            continue
        subjects = (survivors(payload) if gated else
                    [a for a, r in payload["std_ratio"].items() if r <= 3.0])
        if len(subjects) < MIN_SURVIVORS:
            continue
        spread = [payload["std_ratio"][a] for a in subjects]
        for name in rankings:
            ranks = (_cis_rank(payload, subjects) if name == "CIS"
                     else _metric_rank(payload, name, subjects))
            r, _ = spearmanr([-ranks[a] for a in subjects], spread)
            if not np.isnan(r):
                out[name][_band(rate)].append(r)
    return {name: {band: float(np.mean(v)) if v else float("nan")
                   for band, v in bands.items()} for name, bands in out.items()}


def agreement_with_metrics(cache: dict[tuple, dict]) -> dict:
    """How close CIS's ranking is to each single metric's, over the gate survivors.

    Reported as a description of where the composite sits among the eight. The
    panel carries no ground truth, so nothing here says any ordering is right.
    """
    out = {}
    for metric in ALGO_METRICS:
        values = []
        for payload in cache.values():
            subjects = survivors(payload)
            if len(subjects) < MIN_SURVIVORS:
                continue
            r, _ = spearmanr(
                [_cis_rank(payload, subjects)[a] for a in subjects],
                [_metric_rank(payload, metric, subjects)[a] for a in subjects])
            if not np.isnan(r):
                values.append(r)
        out[metric] = float(np.mean(values))
    return out


def blind_spots(conditions: dict[str, dict]) -> dict:
    """Each metric's distance per distortion, averaged over Experiment 1's conditions.

    Every distortion in a condition carries the same pointwise damage, so a
    pointwise metric reads them all alike and any spread in another metric is
    that metric telling the kinds apart. A value near zero is a blind spot.
    """
    out = {}
    for metric in ALGO_METRICS:
        per_distortion = {}
        for condition in conditions.values():
            for subject in condition["subjects"]:
                value = components(condition["scores"], condition["reference"],
                                   subject, (metric,))
                if value is not None:
                    per_distortion.setdefault(subject, []).append(value[metric])
        out[metric] = {d: float(np.mean(v)) for d, v in per_distortion.items()}
    return out


def equal_damage_response(conditions: dict[str, dict],
                          metrics: tuple[str, ...] = CIS_METRICS,
                          power: float = ADOPTED_POWER) -> dict:
    """CIS per distortion under equal pointwise damage, and how evenly it covers them.

    `coverage` is the least-detected distortion over the mean and `spread` is the
    most-detected over the least. The two have to be read together: a purely
    pointwise score reads every distortion alike, which gives it a coverage near
    1 and a spread near 1, and tells the kinds of damage apart not at all.
    """
    per_distortion, per_component = {}, {}
    gate_fired = total = 0
    for condition in conditions.values():
        for subject in condition["subjects"]:
            values = components(condition["scores"], condition["reference"],
                                subject, metrics)
            if values is None:
                continue
            per_distortion.setdefault(subject, []).append(combine(values, power))
            for metric, value in values.items():
                per_component.setdefault(subject, {}).setdefault(metric, []).append(value)
            total += 1
            if not passes(condition["std_ratio"][subject]):
                gate_fired += 1
    mean = {d: float(np.mean(v)) for d, v in per_distortion.items()}
    values = np.array(list(mean.values()))
    return {
        "per_distortion": mean,
        "per_component": {d: {m: float(np.mean(v)) for m, v in c.items()}
                          for d, c in per_component.items()},
        "coverage": float(values.min() / values.mean()),
        "spread": float(values.max() / values.min()),
        "least_detected": min(mean, key=mean.get),
        "gate_fired": gate_fired, "n": total,
    }


def variant_coverage(conditions: dict[str, dict]) -> dict:
    """Every component substitution and exponent, measured on the equal-damage run."""
    out = {}
    for label, metrics in COMPONENT_VARIANTS:
        result = equal_damage_response(conditions, metrics)
        out[label] = {"coverage": result["coverage"], "spread": result["spread"],
                      "least_detected": result["least_detected"]}
    for power in POWER_VARIANTS:
        result = equal_damage_response(conditions, CIS_METRICS, power)
        out[f"exponent p={power}"] = {
            "coverage": result["coverage"], "spread": result["spread"],
            "least_detected": result["least_detected"]}
    return out


def damage_sweep(sweep: dict[str, dict], metrics: tuple[str, ...] = CIS_METRICS,
                 power: float = ADOPTED_POWER) -> dict:
    """CIS at each damage level of Experiment 1's sweep, per distortion.

    A composite that does not rise with the damage would call a more damaged
    reconstruction the closer one, so this is a correctness check.
    """
    out = {}
    for distortion, payload in sweep.items():
        levels = sorted(s for s in payload["subjects"] if len(s) == 2 and s.startswith("L"))
        values = []
        for level in levels:
            c = components(payload["scores"], payload["reference"], level, metrics)
            values.append(combine(c, power) if c is not None else float("nan"))
        out[distortion] = {
            "damage_levels": payload.get("damage_levels"),
            "cis": values,
            "monotone": all(b > a for a, b in zip(values, values[1:])),
            "rise": float(values[-1] - values[0]),
        }
    return out
