import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr

from core.ranking import rank_algorithms

from algo_ranking.algorithms import ALGORITHMS, STOCHASTIC_ALGORITHMS
from algo_ranking.config import (ALGO_CATEGORIES, DATASETS, N_SEEDS, PATTERNS, RATES,
                                 rate_dir, seed_dir)
from algo_ranking.ranking_report import build_rank_matrix, category_consensus, global_consensus

import injector.config as injector_config
from core.metric_config import METRIC_DIRECTION

ALGO_NAMES = [name for name, _, _ in ALGORITHMS]

# CIS's four components, one per Part 2 category. COMPONENT_VARIANTS below
# re-measures every single-metric substitution.
CIS_METRICS = ("mae", "wd", "dtw", "mi")

# Each component divides by its scale here, read off Part 1's equal-damage run:
# the mean value the metric takes over the eight calibrated distortions,
# averaged over Part 1's conditions. derive_component_scales recomputes them.
COMPONENT_SCALES = {
    "mae": 0.4964,
    "wd": 0.3338,
    "dtw": 0.0146,   # applied to DTW / n_timesteps, not to raw DTW
    "mi": 0.9400,
}

# Used by the substitution sweep for metrics the adopted score does not use.
FALLBACK_SCALE = 1.0

# Gate thresholds, both applied to the IQR ratio. Picked by eye from
# cis_gate_distribution.png; derive_gate_thresholds checks them against the data.
FLAT_THRESHOLD = 0.15
UNSTABLE_THRESHOLD = 3.0

CIS_PLOT_DIR = os.path.join(SRC, "plots", "cis")
CIS_REPORT_DIR = os.path.join(SRC, "reports", "cis")

ALGO_COLORS = {
    "CDRec": "#4C72B0", "ROSL": "#DD8452", "DynaMMo": "#55A868",
    "STMVL": "#C44E52", "BRITS": "#8172B2", "MPIN": "#937860",
}
PATTERN_COLORS = {"mcar": "#4C72B0", "scattered": "#55A868", "blackout": "#C44E52"}


def _load_scores(dataset: str, pattern: str, rate: float) -> dict[str, dict[str, float | None]]:
    """{metric: {algo: value}}, from Part 2's cached scores.json."""
    path = os.path.join(rate_dir(dataset, pattern, rate), "scores.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run algo_ranking/score.py for "
            f"dataset={dataset!r} pattern={pattern!r} rate={rate} first."
        )
    with open(path) as f:
        return json.load(f)


def _iqr(x: np.ndarray) -> float:
    """Interquartile range."""
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def stability_ratios(dataset: str, pattern: str, rate: float) -> tuple[dict[str, float], int]:
    """Return ({algo: iqr_ratio}, n_timesteps) for one scenario.

    Stochastic algorithms are averaged over the seeds; deterministic ones are
    read from the first seed alone. n_timesteps comes from the cached truth
    because compute_cis divides DTW by it.
    """
    per_seed_iqr: dict[str, list[float]] = {name: [] for name in ALGO_NAMES}
    n_timesteps = None

    for seed in range(N_SEEDS):
        data_path = os.path.join(seed_dir(dataset, pattern, rate, seed), "data.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Missing {data_path}. Run algo_ranking/build.py for "
                f"dataset={dataset!r} pattern={pattern!r} rate={rate} seed={seed} first."
            )
        with open(data_path) as f:
            built = json.load(f)

        y_true = np.array(built["y_true"])
        mask = np.array(built["mask"]).astype(bool)
        if n_timesteps is None:
            n_timesteps = y_true.shape[-1]
        true_vals = y_true[mask]
        true_iqr = _iqr(true_vals)

        for name in ALGO_NAMES:
            if name not in built:
                continue
            if name not in STOCHASTIC_ALGORITHMS and per_seed_iqr[name]:
                continue
            recon_vals = np.array(built[name])[mask]
            per_seed_iqr[name].append(_iqr(recon_vals) / (true_iqr + 1e-12))

    ratios = {name: float(np.mean(per_seed_iqr[name])) for name in ALGO_NAMES if per_seed_iqr[name]}
    return ratios, n_timesteps


def compute_cis(mae: float, wd: float, dtw: float, mi: float, n_timesteps: int) -> float:
    """CIS = (M * D * T * I)^(1/4), in (0, 1), higher is better.

    Each component divides by its COMPONENT_SCALES entry, so all four are in
    units of the same reference damage. DTW is divided by n_timesteps first.
    """
    M = np.exp(-mae / COMPONENT_SCALES["mae"])
    D = np.exp(-wd / COMPONENT_SCALES["wd"])
    T = np.exp(-(dtw / n_timesteps) / COMPONENT_SCALES["dtw"])
    I = 1.0 - np.exp(-mi / COMPONENT_SCALES["mi"])
    return float((M * D * T * I) ** (1 / 4))


def gate_and_score(dataset: str, pattern: str, rate: float) -> dict[str, dict]:
    """Return {algo: {iqr_ratio, passes_gate, cis}} for one scenario.

    CIS is computed whether or not the algorithm passes, so the ungated ranking
    can be compared against the panel separately. Callers filter on passes_gate.
    """
    ratios, n_timesteps = stability_ratios(dataset, pattern, rate)
    scores = _load_scores(dataset, pattern, rate)

    out = {}
    for algo, iqr_ratio in ratios.items():
        passes = FLAT_THRESHOLD <= iqr_ratio <= UNSTABLE_THRESHOLD
        cis = compute_cis(
            scores["mae"][algo], scores["wd"][algo], scores["dtw"][algo], scores["mi"][algo], n_timesteps
        )
        out[algo] = {"iqr_ratio": iqr_ratio, "passes_gate": passes, "cis": cis}
    return out


def collect_all_scenarios(
    datasets: list[str] = DATASETS,
    patterns: list[str] = PATTERNS,
    rates: list[float] = RATES,
) -> tuple[list[dict], dict[tuple, dict], dict[tuple, int]]:
    """Flatten gate_and_score over every scenario into one row per (scenario, algorithm).

    Each row carries that scenario's eight-metric consensus rank. The scores are
    returned because the gated comparison recomputes the consensus over the
    survivors only, and the series lengths because the T component needs them.
    """
    rows = []
    scenario_scores: dict[tuple, dict] = {}
    n_timesteps: dict[tuple, int] = {}
    for dataset in datasets:
        for pattern in patterns:
            for rate in rates:
                scores = _load_scores(dataset, pattern, rate)
                scenario_scores[(dataset, pattern, rate)] = scores

                rank_matrix = build_rank_matrix(scores)
                cat_consensus = category_consensus(rank_matrix)
                glob_consensus = global_consensus(cat_consensus)

                gated = gate_and_score(dataset, pattern, rate)
                n_timesteps[(dataset, pattern, rate)] = stability_ratios(dataset, pattern, rate)[1]
                for algo, info in gated.items():
                    rows.append({
                        "dataset": dataset, "pattern": pattern, "rate": rate,
                        "algo": algo,
                        "iqr_ratio": info["iqr_ratio"],
                        "passes_gate": info["passes_gate"],
                        "cis": info["cis"],
                        "consensus_rank": glob_consensus[algo],
                    })
    return rows, scenario_scores, n_timesteps


def derive_component_scales(conditions: dict[str, dict] | None = None) -> dict:
    """Recompute COMPONENT_SCALES from Part 1's equal-damage cache.

    Returns {} when that cache is absent, since the adopted constants are
    hard-coded and this only exists to keep them checkable.
    """
    conditions = load_injector_equal_damage() if conditions is None else conditions
    if not conditions:
        return {}
    collected: dict[str, list[float]] = {}
    for payload in conditions.values():
        scores, n = payload["scores"], payload["n_timesteps"]
        for metric in COMPONENT_SCALES:
            for distortion in payload["iqr_ratio"]:
                value = scores[metric].get(distortion)
                if value is None or not np.isfinite(value):
                    continue
                collected.setdefault(metric, []).append(
                    value / n if metric == "dtw" else value)
    return {metric: {"n": len(v), "mean": float(np.mean(v)),
                     "median": float(np.median(v)), "adopted": COMPONENT_SCALES[metric]}
            for metric, v in collected.items()}


def derive_gate_thresholds(rows: list[dict], bw: float = 0.15) -> dict:
    """Look for the adopted thresholds in the data, as a check rather than a derivation.

    Exact zeros are dropped and a Gaussian KDE is fit to log10 of the rest; a
    valley between two peaks marks a boundary between populations. `bw` is chosen
    manually and is stable across roughly 0.10 to 0.20.
    """
    vals = np.array([r["iqr_ratio"] for r in rows if r["iqr_ratio"] > 0])
    log_vals = np.log10(vals)
    kde = gaussian_kde(log_vals, bw_method=bw)
    grid = np.linspace(log_vals.min(), log_vals.max(), 3000)
    density = kde(grid)
    valleys = [grid[i] for i in range(1, len(grid) - 1)
               if density[i] < density[i - 1] and density[i] < density[i + 1]]

    near_flat = [10 ** v for v in valleys if -1.0 < v < 0.0]
    near_unstable = [10 ** v for v in valleys if 0.2 < v < 1.0]
    return {
        "bw": bw,
        "n_nonzero": len(vals),
        "flat_valley": near_flat[0] if near_flat else None,
        "unstable_valley": near_unstable[0] if near_unstable else None,
    }


def excluded_scenario_breakdown(rows: list[dict]) -> dict:
    """Split the failures in unrankable scenarios into decisive and marginal ones.

    Decisive means exactly zero or beyond twice the unstable threshold, so it is
    visible whether the thresholds cut close calls.
    """
    by_scenario: dict[tuple, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault((r["dataset"], r["pattern"], r["rate"]), []).append(r)

    excluded = [(k, v) for k, v in by_scenario.items()
                if sum(1 for r in v if r["passes_gate"]) < 3]
    n_blackout_excluded = sum(1 for k, _ in excluded if k[1] == "blackout")
    n_blackout_total = sum(1 for k in by_scenario if k[1] == "blackout")

    decisive, moderate = 0, 0
    for _, scenario_rows in excluded:
        for r in scenario_rows:
            if r["passes_gate"]:
                continue
            if r["iqr_ratio"] == 0.0 or r["iqr_ratio"] > 2 * UNSTABLE_THRESHOLD:
                decisive += 1
            else:
                moderate += 1

    return {
        "n_excluded_scenarios": len(excluded),
        "n_blackout_excluded": n_blackout_excluded,
        "n_blackout_total": n_blackout_total,
        "n_failing_decisive": decisive,
        "n_failing_moderate": moderate,
    }


def validation_summary(rows: list[dict], scenario_scores: dict[tuple, dict]) -> str:
    """Spearman correlation between CIS's ranking and the eight-metric consensus.

    Reported ungated over several scopes, then over gate survivors only.
    """
    lines = ["CIS VALIDATION SUMMARY", "=" * 70, ""]

    by_scenario: dict[tuple, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault((r["dataset"], r["pattern"], r["rate"]), []).append(r)

    def _report(label: str, scope_rows: list[tuple]) -> None:
        rhos = []
        for key, scenario_rows in scope_rows:
            algos = [r["algo"] for r in scenario_rows]
            cis_vals = {r["algo"]: r["cis"] for r in scenario_rows}
            cis_rank = rank_algorithms(cis_vals, direction="higher")
            cons_rank = {r["algo"]: r["consensus_rank"] for r in scenario_rows}
            rho, _ = spearmanr([cis_rank[a] for a in algos], [cons_rank[a] for a in algos])
            if not np.isnan(rho):
                rhos.append(rho)
        rhos = np.array(rhos)
        if len(rhos):
            lines.append(
                f"{label:45s} n={len(rhos):3d}  mean={rhos.mean():.3f}  "
                f"median={np.median(rhos):.3f}  min={rhos.min():.3f}  "
                f"%>=0.9={100*np.mean(rhos>=0.9):5.1f}%"
            )

    all_scope = list(by_scenario.items())
    ms_scope = [(k, v) for k, v in by_scenario.items() if k[1] in ("mcar", "scattered")]
    ms_low_scope = [(k, v) for k, v in ms_scope if k[2] == 0.2]

    _report("All scenarios (ungated)", all_scope)
    _report("mcar+scattered, all rates (ungated)", ms_scope)
    _report("mcar+scattered, 20pct only (ungated)", ms_low_scope)

    # the consensus is recomputed from scratch over the survivor subset, not
    # filtered from the full six-algorithm ranking
    gated_rhos = []
    n_too_few = 0
    for key, scenario_rows in by_scenario.items():
        survivors = [r for r in scenario_rows if r["passes_gate"]]
        if len(survivors) < 3:
            n_too_few += 1
            continue
        algos = [r["algo"] for r in survivors]
        cis_rank = rank_algorithms({r["algo"]: r["cis"] for r in survivors}, direction="higher")

        full_scores = scenario_scores[key]
        restricted_scores = {
            metric: {a: full_scores[metric][a] for a in algos} for metric in full_scores
        }
        cons_rank = global_consensus(category_consensus(build_rank_matrix(restricted_scores)))

        rho, _ = spearmanr([cis_rank[a] for a in algos], [cons_rank[a] for a in algos])
        if not np.isnan(rho):
            gated_rhos.append(rho)
    gated_rhos = np.array(gated_rhos)
    lines.append(
        f"{'Gate applied, survivors only, all scenarios':45s} n={len(gated_rhos):3d}  "
        f"mean={gated_rhos.mean():.3f}  median={np.median(gated_rhos):.3f}  "
        f"min={gated_rhos.min():.3f}  %>=0.9={100*np.mean(gated_rhos>=0.9):5.1f}%"
    )
    lines.append(f"  ({n_too_few} scenarios had fewer than 3 survivors, so the gate alone resolved them)")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Supporting analyses. None of these feeds into a CIS value; they check that
# the design choices behind it hold against the current metric panel and the
# current Injector run.
# ══════════════════════════════════════════════════════════════════════════

def _metric_rank(scores: dict[str, dict[str, float]], metric: str, subjects: list[str]) -> dict[str, float]:
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


def component_values(scores: dict, subject: str, n_timesteps: int) -> dict[str, float]:
    """The four normalized CIS components for one reconstruction.

    `subject` is an algorithm name for Part 2's scores and a distortion name for
    Part 1's, since the two caches share the {metric: {name: value}} shape.
    """
    return {
        "M": float(np.exp(-scores["mae"][subject] / COMPONENT_SCALES["mae"])),
        "D": float(np.exp(-scores["wd"][subject] / COMPONENT_SCALES["wd"])),
        "T": float(np.exp(-(scores["dtw"][subject] / n_timesteps) / COMPONENT_SCALES["dtw"])),
        "I": float(1.0 - np.exp(-scores["mi"][subject] / COMPONENT_SCALES["mi"])),
    }


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


def variant_cis(scores: dict, subject: str, n_timesteps: int, slots: tuple[str, ...]) -> float:
    """CIS with the four component metrics replaced by `slots`.

    Metrics absent from COMPONENT_SCALES fall back to FALLBACK_SCALE. DTW is
    divided by series length first, and R squared enters as its distance from
    1.0. Passing fewer than four slots gives the geometric mean over that many
    components.
    """
    parts = []
    for metric in slots:
        value = scores[metric][subject]
        scale = COMPONENT_SCALES.get(metric, FALLBACK_SCALE)
        if metric == "dtw":
            parts.append(np.exp(-(value / n_timesteps) / scale))
        elif metric in ("mi",):
            parts.append(1.0 - np.exp(-value / scale))
        elif metric == "r2":
            parts.append(np.exp(-abs(1.0 - value) / scale))
        else:
            parts.append(np.exp(-abs(value) / scale))
    return float(np.prod(parts) ** (1.0 / len(parts)))


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


def load_injector_equal_damage() -> dict[str, dict]:
    """Read Part 1's equal-damage cache, or {} when that experiment has not been run.

    Returns {"pattern/rate": {scores, iqr_ratio, n_timesteps}}. The IQR ratio is
    recomputed here, since the gate is CIS's own instrument.
    """
    out = {}
    for pattern in injector_config.PATTERNS:
        for rate in injector_config.RATES:
            folder = injector_config.rate_dir(pattern, rate)
            data_path = os.path.join(folder, "data.json")
            scores_path = os.path.join(folder, "scores.json")
            if not (os.path.exists(data_path) and os.path.exists(scores_path)):
                continue
            with open(data_path) as f:
                data = json.load(f)
            with open(scores_path) as f:
                scores = json.load(f)
            y_true = np.array(data["y_true"])
            mask = np.array(data["mask"]).astype(bool)
            true_iqr = _iqr(y_true[mask])
            names = [d for d in injector_config.DISTORTION_NAMES if d in data]
            out[f"{pattern}/{round(rate * 100):02d}pct"] = {
                "scores": scores,
                "n_timesteps": y_true.shape[-1],
                "iqr_ratio": {d: _iqr(np.array(data[d])[mask]) / (true_iqr + 1e-12)
                              for d in names},
            }
    return out


def load_injector_sweep() -> dict[str, dict]:
    """Read Part 1's damage sweep, or {} when that experiment has not been run."""
    out = {}
    for distortion in injector_config.DISTORTION_NAMES:
        folder = injector_config.sweep_dir(distortion)
        data_path = os.path.join(folder, "data.json")
        scores_path = os.path.join(folder, "scores.json")
        if not (os.path.exists(data_path) and os.path.exists(scores_path)):
            continue
        with open(data_path) as f:
            data = json.load(f)
        with open(scores_path) as f:
            scores = json.load(f)
        y_true = np.array(data["y_true"])
        mask = np.array(data["mask"]).astype(bool)
        true_iqr = _iqr(y_true[mask])
        levels = sorted(k for k in data if len(k) == 2 and k.startswith("L"))
        out[distortion] = {
            "scores": scores,
            "n_timesteps": y_true.shape[-1],
            "damage_levels": data.get("levels"),
            "iqr_ratio": {L: _iqr(np.array(data[L])[mask]) / (true_iqr + 1e-12)
                          for L in levels},
        }
    return out


def equal_damage_response(conditions: dict[str, dict]) -> dict:
    """CIS and the gate ratio for each of Part 1's eight distortions.

    Part 1 solves every distortion to the same mean absolute error, so the M
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


def damage_sweep_response(sweep: dict[str, dict]) -> dict:
    """CIS at each of Part 1's damage levels, per distortion.

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


def supporting_experiments(rows, scenario_scores, n_timesteps) -> str:
    """The text report behind the CIS chapter's design section."""
    L = ["CIS SUPPORTING EXPERIMENTS", "=" * 78, ""]

    L.append("1. WITHIN-CATEGORY AGREEMENT (does one metric per category lose anything?)")
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

    L.append("4. GATE AND THE I COMPONENT (is MI still needed after the flat check?)")
    ov = gate_and_mi_overlap(rows, scenario_scores)
    L.append(f"   flat failures {ov['n_flat_failures']}, survivors {ov['n_survivors']}")
    L.append(f"   MI exactly 0 in {ov['mi_zero_among_flat_failures']} flat failures "
             f"and {ov['mi_zero_among_survivors']} survivors")
    L.append(f"   median MI: {ov['median_mi_flat_failures']:.4f} among flat failures, "
             f"{ov['median_mi_survivors']:.4f} among survivors")
    L.append(f"   rho(iqr_ratio, MI) among survivors = "
             f"{ov['rho_iqr_ratio_vs_mi_among_survivors']:.3f}")
    L.append("")

    conditions = load_injector_equal_damage()
    if not conditions:
        L.append("5. EQUAL-DAMAGE RESPONSE: skipped, no Injector cache found.")
        L.append("")
    else:
        eq = equal_damage_response(conditions)
        names = list(next(iter(eq["per_condition"].values()))["cis"])
        L.append("5. RESPONSE UNDER EQUAL DAMAGE (Part 1's eight distortions, all at the")
        L.append("   same MAE, so the M component is pinned and only kind varies)")
        L.append(f"   {'condition':18s}" + "".join(f"{d[:9]:>10}" for d in names) + f"{'spread':>9}")
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

    sweep = load_injector_sweep()
    if not sweep:
        L.append("6. DAMAGE-SWEEP RESPONSE: skipped, no Injector sweep cache found.")
    else:
        sw = damage_sweep_response(sweep)
        levels = next(iter(sw.values()))["damage_levels"] or []
        L.append("6. RESPONSE ACROSS THE DAMAGE SWEEP (CIS must fall as damage rises)")
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


def _scatter_ratio(ax, rows: list[dict], field: str) -> None:
    """Jittered scatter of one field, grouped by pattern and coloured by algorithm."""
    rng = np.random.default_rng(0)
    for pi, pattern in enumerate(PATTERNS):
        for ai, algo in enumerate(ALGO_NAMES):
            vals = [r[field] for r in rows if r["pattern"] == pattern and r["algo"] == algo]
            if not vals:
                continue
            vals = np.array(vals)
            x = pi + rng.uniform(-0.32, 0.32, size=len(vals)) + (ai - 2.5) * 0.045
            ax.scatter(x, vals, color=ALGO_COLORS[algo], s=22, alpha=0.75,
                       edgecolor="white", linewidth=0.3, label=algo if pi == 0 else None)
    ax.set_xticks(range(len(PATTERNS)))
    ax.set_xticklabels(PATTERNS)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(-0.5, len(PATTERNS) - 1 + 0.8)


def plot_gate_distribution(rows: list[dict], output_path: str) -> None:
    """The IQR ratio for every scenario and algorithm, with both thresholds marked."""
    fig, ax = plt.subplots(figsize=(8, 6))

    _scatter_ratio(ax, rows, "iqr_ratio")
    ax.set_ylim(bottom=-0.02)
    ax.axhline(FLAT_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.text(0.05, FLAT_THRESHOLD, f"flat threshold ({FLAT_THRESHOLD})",
            va="bottom", ha="left", fontsize=8, color="gray", transform=ax.get_yaxis_transform())
    ax.axhline(UNSTABLE_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.text(0.05, UNSTABLE_THRESHOLD, f"unstable threshold ({UNSTABLE_THRESHOLD})",
            va="bottom", ha="left", fontsize=8, color="gray", transform=ax.get_yaxis_transform())
    ax.set_ylabel("IQR(reconstruction) / IQR(truth), masked positions (symlog scale)")
    ax.set_title("Stability gate across all scenarios and algorithms\n"
                 "(near 0 = flat or collapsed, far above 1 = unstable)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {output_path}")


def plot_cis_vs_consensus(rows: list[dict], output_path: str) -> None:
    """CIS rank against consensus rank, one point per (scenario, algorithm).

    CIS is ranked over all six algorithms rather than over survivors, so the
    figure shows which points the gate removes; validation_summary's gated row
    re-ranks over survivors only and is a stricter computation.
    """
    by_scenario: dict[tuple, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault((r["dataset"], r["pattern"], r["rate"]), []).append(r)

    xs_gated, ys_gated = [], []
    xs_survive, ys_survive, colors_survive = [], [], []

    for key, scenario_rows in by_scenario.items():
        pattern = key[1]
        cis_vals = {r["algo"]: r["cis"] for r in scenario_rows}
        cis_rank = rank_algorithms(cis_vals, direction="higher")
        for r in scenario_rows:
            if r["passes_gate"]:
                xs_survive.append(r["consensus_rank"])
                ys_survive.append(cis_rank[r["algo"]])
                colors_survive.append(PATTERN_COLORS[pattern])
            else:
                xs_gated.append(r["consensus_rank"])
                ys_gated.append(cis_rank[r["algo"]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs_gated, ys_gated, c="lightgray", s=45, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.scatter(xs_survive, ys_survive, c=colors_survive, s=45, alpha=0.8, edgecolor="white", linewidth=0.3)
    lo, hi = 1, len(ALGO_NAMES)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
                       label="fails gate (excluded)", markersize=9)]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                        label=f"survives, {p}", markersize=9) for p, c in PATTERN_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    ax.set_xlabel("8-metric global consensus rank (1 = best)")
    ax.set_ylabel("CIS rank (1 = best)")
    ax.set_title("CIS rank vs. full 8-metric consensus rank, gate applied")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {output_path}")


def plot_equal_damage_response(equal_damage: dict, output_path: str) -> None:
    """CIS and its four components across Part 1's eight calibrated distortions.

    Left panel: CIS per distortion, one marker per condition. Right panel: the
    four components, where a component that barely moves shows up as a flat line.
    """
    per_condition = equal_damage["per_condition"]
    mean_cis = equal_damage["mean_per_distortion"]
    names = sorted(mean_cis, key=mean_cis.get)
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for condition, payload in per_condition.items():
        pattern = condition.split("/")[0]
        ax1.scatter(x, [payload["cis"][d] for d in names],
                    color=PATTERN_COLORS.get(pattern, "gray"), s=26, alpha=0.65,
                    edgecolor="white", linewidth=0.3)
    ax1.plot(x, [mean_cis[d] for d in names], color="black", linewidth=1.6,
             marker="o", markersize=5, label="mean over conditions")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("CIS")
    ax1.set_title("CIS under equal damage\n(every distortion at the same MAE)")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=p,
                      markersize=8) for p, c in PATTERN_COLORS.items()]
    handles.append(Line2D([0], [0], color="black", label="mean"))
    ax1.legend(handles=handles, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    styles = {"M": ("#4C72B0", "MAE"), "D": ("#DD8452", "WD"),
              "T": ("#55A868", "DTW"), "I": ("#C44E52", "MI")}
    for comp, (color, source) in styles.items():
        vals = [float(np.mean([p["components"][d][comp] for p in per_condition.values()]))
                for d in names]
        ax2.plot(x, vals, color=color, marker="o", markersize=5, linewidth=1.6,
                 label=f"{comp}  (from {source})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("component value")
    ax2.set_title("The four components over the same eight")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {output_path}")


def main(datasets: list[str], patterns: list[str], rates: list[float]) -> None:
    """Gate and score every scenario, then write the reports and figures."""
    rows, scenario_scores, n_timesteps = collect_all_scenarios(datasets, patterns, rates)

    os.makedirs(CIS_REPORT_DIR, exist_ok=True)

    lines = [validation_summary(rows, scenario_scores), ""]

    derived = derive_component_scales()
    lines.append("COMPONENT SCALE DERIVATION (mean over Part 1's eight calibrated distortions)")
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
    lines.append(f"  flat_valley={thr['flat_valley']!r}  (adopted FLAT_THRESHOLD={FLAT_THRESHOLD})")
    lines.append(f"  unstable_valley={thr['unstable_valley']!r}  (adopted UNSTABLE_THRESHOLD={UNSTABLE_THRESHOLD})")
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

    supporting = supporting_experiments(rows, scenario_scores, n_timesteps)
    supporting_path = os.path.join(CIS_REPORT_DIR, "cis_supporting_experiments.txt")
    with open(supporting_path, "w") as f:
        f.write(supporting + "\n")
    print(f"Written: {supporting_path}")

    plot_gate_distribution(rows, os.path.join(CIS_PLOT_DIR, "cis_gate_distribution.png"))
    plot_cis_vs_consensus(rows, os.path.join(CIS_PLOT_DIR, "cis_gated_vs_consensus.png"))

    equal_damage_cache = load_injector_equal_damage()
    if equal_damage_cache:
        plot_equal_damage_response(
            equal_damage_response(equal_damage_cache),
            os.path.join(CIS_PLOT_DIR, "cis_equal_damage_response.png"))
    else:
        print("Skipped cis_equal_damage_response.png: no Injector cache found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2 (Algorithm Ranking): CIS gate and composite score.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    args = parser.parse_args()

    main(args.datasets, args.patterns, args.rates)
