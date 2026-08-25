import json
import os

import numpy as np

from algo_ranking import cache
from algo_ranking.algorithms import STOCHASTIC_ALGORITHMS
from algo_ranking.config import DATASETS, N_SEEDS, PATTERNS, RATES, rate_dir
from algo_ranking.analysis import build_rank_matrix, category_consensus, global_consensus

from cis.config import (ALGO_NAMES, COMPONENT_SCALES, FALLBACK_SCALE,
                        FLAT_THRESHOLD, UNSTABLE_THRESHOLD)


def _load_scores(dataset: str, pattern: str,
                 rate: float) -> dict[str, dict[str, float | None]]:
    """{metric: {algo: value}}, from Experiment 2's cached scores.json."""
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
    """Return ({algo: iqr_ratio}, n_timesteps) for one scenario. """
    per_seed_iqr: dict[str, list[float]] = {name: [] for name in ALGO_NAMES}
    n_timesteps = None

    for seed in range(N_SEEDS):
        built = cache.load_scenario(dataset, pattern, rate, seed)
        y_true = built["y_true"]
        mask = built["mask"].astype(bool)
        if n_timesteps is None:
            n_timesteps = y_true.shape[-1]
        true_iqr = _iqr(y_true[mask])

        for name in ALGO_NAMES:
            if name not in built:
                continue
            if name not in STOCHASTIC_ALGORITHMS and per_seed_iqr[name]:
                continue
            recon_vals = built[name][mask]
            per_seed_iqr[name].append(_iqr(recon_vals) / (true_iqr + 1e-12))

    ratios = {name: float(np.mean(per_seed_iqr[name]))
              for name in ALGO_NAMES if per_seed_iqr[name]}
    return ratios, n_timesteps


def compute_cis(mae: float, wd: float, dtw: float, mi: float, n_timesteps: int) -> float:
    """CIS = (M * D * T * I)^(1/4), in (0, 1), higher is better."""
    M = np.exp(-mae / COMPONENT_SCALES["mae"])
    D = np.exp(-wd / COMPONENT_SCALES["wd"])
    T = np.exp(-(dtw / n_timesteps) / COMPONENT_SCALES["dtw"])
    I = 1.0 - np.exp(-mi / COMPONENT_SCALES["mi"])
    return float((M * D * T * I) ** (1 / 4))


def gate_and_score(dataset: str, pattern: str, rate: float,
                   scores: dict | None = None) -> tuple[dict[str, dict], int]:
    """Return ({algo: {iqr_ratio, passes_gate, cis}}, n_timesteps) for one scenario. """
    ratios, n_timesteps = stability_ratios(dataset, pattern, rate)
    scores = _load_scores(dataset, pattern, rate) if scores is None else scores

    out = {}
    for algo, iqr_ratio in ratios.items():
        passes = FLAT_THRESHOLD <= iqr_ratio <= UNSTABLE_THRESHOLD
        cis = compute_cis(
            scores["mae"][algo], scores["wd"][algo], scores["dtw"][algo],
            scores["mi"][algo], n_timesteps,
        )
        out[algo] = {"iqr_ratio": iqr_ratio, "passes_gate": passes, "cis": cis}
    return out, n_timesteps


def collect_all_scenarios(
    datasets: list[str] = DATASETS,
    patterns: list[str] = PATTERNS,
    rates: list[float] = RATES,
) -> tuple[list[dict], dict[tuple, dict], dict[tuple, int]]:
    """Flatten gate_and_score over every scenario into one row per (scenario, algorithm)."""
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

                gated, n_t = gate_and_score(dataset, pattern, rate, scores)
                n_timesteps[(dataset, pattern, rate)] = n_t
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



def component_values(scores: dict, subject: str, n_timesteps: int) -> dict[str, float]:
    """The four normalized CIS components for one reconstruction."""
    return {
        "M": float(np.exp(-scores["mae"][subject] / COMPONENT_SCALES["mae"])),
        "D": float(np.exp(-scores["wd"][subject] / COMPONENT_SCALES["wd"])),
        "T": float(np.exp(-(scores["dtw"][subject] / n_timesteps) / COMPONENT_SCALES["dtw"])),
        "I": float(1.0 - np.exp(-scores["mi"][subject] / COMPONENT_SCALES["mi"])),
    }



def variant_cis(scores: dict, subject: str, n_timesteps: int, slots: tuple[str, ...]) -> float:
    """CIS with the four component metrics replaced by `slots`."""
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


