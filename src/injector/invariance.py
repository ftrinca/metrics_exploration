"""Check the exact invariants each distortion declares in config.DISTORTIONS.

Each structural property implies an exact prediction about specific metrics: a
preserved multiset cannot move any statistic computed from the empirical value
distribution alone (WD, JSD, KLD), a preserved mean cannot move a measure of a
difference in means (Bland-Altman, CDT), and a positive-slope affine transform
cannot move a correlation (Pearson). A failed prediction means either the
distortion is not doing what its design claims or the metric implementation
does not have the property it is supposed to have.
"""

from __future__ import annotations

from injector.config import DISTORTIONS

# property -> {metric: (predicted value, absolute tolerance)}
#
# The tolerances differ because core.dataset_io.matrix_to_lists rounds every
# value to four decimal places on its way into the cache, and each invariant
# survives that rounding differently. A permuted multiset of rounded values is
# still the same multiset, so those metrics stay at machine zero; a mean moves,
# because different values round in different directions; a correlation is
# invariant to scale and barely moves. Every tolerance is still orders of
# magnitude tighter than any real reaction, so a metric that has genuinely lost
# an invariance still fails.
#
# "rank" predicts nothing, because no metric in the current set is a pure rank
# statistic. It is recorded because it explains why discretise leaves DTW's
# alignment largely intact.
PREDICTIONS = {
    "multiset": {
        "wd":  (0.0, 1e-9),
        "jsd": (0.0, 1e-9),
        "kld": (0.0, 1e-9),
    },
    "mean": {
        "ba":  (0.0, 1e-4),
        "cdt": (0.0, 1e-4),
    },
    "affine": {
        "pearson": (1.0, 1e-7),
    },
    "rank": {},
}


def expected(distortion: str) -> dict[str, tuple[float, float]]:
    """All exact predictions for one distortion, merged across its properties."""
    out: dict[str, tuple[float, float]] = {}
    for prop in DISTORTIONS[distortion]["preserves"]:
        out.update(PREDICTIONS.get(prop, {}))
    return out


def check(scores: dict[str, dict[str, float | None]]) -> list[dict]:
    """Check every prediction against one scenario's scores.

    scores is the {metric: {distortion: value}} structure produced by
    core.scoring.compute_all_scores. Returns one row per prediction; a metric
    that returned None counts as a failure rather than being skipped.
    """
    rows = []
    for distortion in DISTORTIONS:
        for metric, (predicted, tol) in expected(distortion).items():
            if metric not in scores or distortion not in scores[metric]:
                continue
            observed = scores[metric][distortion]
            if observed is None:
                rows.append({
                    "distortion": distortion, "metric": metric,
                    "predicted": predicted, "observed": None,
                    "deviation": None, "tolerance": tol,
                    "passed": False, "note": "metric returned None",
                })
                continue
            deviation = abs(float(observed) - predicted)
            rows.append({
                "distortion": distortion, "metric": metric,
                "predicted": predicted, "observed": float(observed),
                "deviation": deviation, "tolerance": tol,
                "passed": deviation <= tol,
                "note": "",
            })
    return rows


def table(rows: list[dict], title: str = "") -> str:
    """Render the pass/fail table for check()'s rows."""
    head = ["EXACT INVARIANCE CHECKS" + (f"  ({title})" if title else ""), "=" * 78]
    head.append(
        "Each row is a prediction implied by a distortion's structure, not an "
        "observation.\nA failure means either the distortion or the metric is "
        "not what it claims to be."
    )
    head.append("-" * 78)
    head.append(
        f"{'distortion':<13}{'metric':<9}{'predicted':>10}{'|deviation|':>13}{'tolerance':>12}   ")
    head.append("-" * 78)
    for r in rows:
        dev = "None" if r["deviation"] is None else f"{r['deviation']:.2e}"
        flag = "ok" if r["passed"] else "FAIL"
        note = f"  {r['note']}" if r["note"] else ""
        head.append(
            f"{r['distortion']:<13}{r['metric']:<9}{r['predicted']:>10.1f}"
            f"{dev:>13}{r['tolerance']:>12.0e}   {flag}{note}"
        )
    head.append("-" * 78)
    n_pass = sum(1 for r in rows if r["passed"])
    head.append(f"{n_pass} of {len(rows)} predictions hold.")
    return "\n".join(head)
