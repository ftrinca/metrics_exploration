import numpy as np
from scipy.stats import gaussian_kde, spearmanr

from core.ranking import rank_algorithms
from algo_ranking.analysis import build_rank_matrix, category_consensus, global_consensus

from cis.config import COMPONENT_SCALES, UNSTABLE_THRESHOLD
from cis.injector_data import load_injector_reactivity


def derive_component_scales(conditions: dict[str, dict] | None = None) -> dict:
    """Recompute COMPONENT_SCALES from Experiment 1's damage-reactivity cache."""
    conditions = load_injector_reactivity() if conditions is None else conditions
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
        cis_rank = rank_algorithms({r["algo"]: r["cis"] for r in survivors},
                                   direction="higher")

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
    lines.append(f"  ({n_too_few} scenarios had fewer than 3 survivors, "
                 f"so the gate alone resolved them)")

    return "\n".join(lines)
