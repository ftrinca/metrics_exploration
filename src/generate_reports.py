"""Loads a dataset's JSON file, computes every metric for every algorithm or
distortion, and writes the per-metric, per-algorithm scores report. See
metric_config.py for the metric list and metric_verification.md for formula
details.
"""

import json
import os

import metrics
import numpy as np

from metric_config import FULL_SERIES_METRICS, METRIC_LIST, PROBABILISTIC_METRICS


def load_data(path: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """Load ground truth, imputations/reconstructions, and optional mask from
    a JSON file.

    Expected JSON keys:
      "y_true"  — ground truth (required)
      "mask"    — boolean array, True = missing position (optional)
      <algo>    — one key per algorithm/distortion (any other name)

    Returns:
      y_true       — shape (n_timesteps,) or (n_series, n_timesteps)
      imputations  — {algo_name: array of same shape as y_true}
      mask         — bool array of same shape as y_true, or None if absent
    """
    with open(path, "r") as f:
        data = json.load(f)

    y_true = np.array(data.pop("y_true"), dtype=float)

    mask_raw = data.pop("mask", None)
    mask = np.array(mask_raw, dtype=bool) if mask_raw is not None else None

    imputations = {key: np.array(val, dtype=float) for key, val in data.items()}
    return y_true, imputations, mask


def format_value(metric_name: str, value) -> str:
    """Format a metric return value as a human-readable string.

    ba returns a (mean_diff, loa) tuple; everything else is a scalar.
    """
    if metric_name == "ba":
        mean_diff, loa = value
        return f"(mean_diff={mean_diff:.6f}, loa={loa:.6f})"
    return f"{float(value):.6f}"


def to_scalar(metric_name: str, value) -> float:
    """Convert a metric return value to a single float for ranking.

    ba is the only metric that returns a tuple; rank by |mean_diff| so that
    algorithms with less systematic bias rank higher.
    """
    if metric_name == "ba":
        mean_diff, _ = value
        return abs(mean_diff)
    return float(value)


def is_probabilistic(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """True if y_pred carries an extra "samples" dimension on top of y_true's
    shape, i.e. this algorithm produced a distribution per point rather than
    a single point estimate (the shape crps/nll expect for their
    "posterior samples" branch)."""
    return y_pred.ndim == y_true.ndim + 1


def metric_applies(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """True if `metric_name` can be computed for this algorithm's output.

    - Probabilistic metrics (crps, nll) require posterior-sample output (see
      is_probabilistic) - they are undefined for a single point estimate.
    - Temporal/Shape metrics (acf, dtw, smae) expect a single point-estimate
      series and are undefined for posterior-sample output - the inverse
      condition of the probabilistic metrics.
    - All other metrics apply unconditionally.
    """
    probabilistic = is_probabilistic(y_true, y_pred)
    if metric_name in PROBABILISTIC_METRICS:
        return probabilistic
    if metric_name in FULL_SERIES_METRICS:
        return not probabilistic
    return True


def _apply_metric(
    fn,
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
):
    """Call fn, optionally restricting evaluation to masked (missing) positions.

    mask=None  → evaluate on full series (legacy behaviour).
    mask=array → evaluate only on positions where mask is True, UNLESS the
                 metric is in FULL_SERIES_METRICS (acf, dtw, smae), which
                 always receive the complete series because temporal
                 structure matters.

    Metrics for which metric_applies() is False return None - i.e. "not
    applicable" for this algorithm's output (e.g. crps/nll on a deterministic
    algorithm, or acf/dtw/smae on a probabilistic one). Not computed at all.

    y_true / y_pred shapes:
      1D — (n_timesteps,)            → call fn directly
      2D — (n_series, n_timesteps)   → call fn per series, return mean
    ba returns a (mean_diff, loa) tuple in both cases.
    """
    if not metric_applies(metric_name, y_true, y_pred):
        return None

    use_mask = (mask is not None) and (metric_name not in FULL_SERIES_METRICS)

    if y_true.ndim == 1:
        if use_mask:
            if mask.sum() == 0:
                return None
            return fn(y_true[mask], y_pred[mask])
        return fn(y_true, y_pred)

    # 2D: per-series evaluation
    series_results = []
    for s in range(y_true.shape[0]):
        if use_mask:
            m_s = mask[s]
            if m_s.sum() == 0:
                continue
            series_results.append(fn(y_true[s][m_s], y_pred[s][m_s]))
        else:
            series_results.append(fn(y_true[s], y_pred[s]))

    if not series_results:
        return None

    if metric_name == "ba":
        return (
            float(np.mean([r[0] for r in series_results])),
            float(np.mean([r[1] for r in series_results])),
        )
    return float(np.mean(series_results))


def compute_all_scores(
    y_true: np.ndarray,
    imputations: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute every metric for every algorithm via _apply_metric (see its
    docstring for the masking and applicability rules).

    Returns {metric_name: {algo_name: scalar_score_or_None}}.
    None means either the metric doesn't apply to that algorithm's output, or
    it raised an exception.
    Supports both 1D (univariate) and 2D (multivariate) arrays.
    """
    scores: dict[str, dict[str, float | None]] = {}

    for metric_name in METRIC_LIST:
        fn = getattr(metrics, metric_name)
        scores[metric_name] = {}
        for algo, y_pred in imputations.items():
            try:
                raw = _apply_metric(fn, metric_name, y_true, y_pred, mask)
                scores[metric_name][algo] = to_scalar(metric_name, raw) if raw is not None else None
            except Exception as exc:
                scores[metric_name][algo] = None
                print(f"  WARNING: {metric_name} failed for {algo}: {exc}")

    return scores


def applicable_metrics(scores: dict[str, dict[str, float | None]]) -> list[str]:
    """Return the subset of METRIC_LIST that has at least one non-None score
    - i.e. metrics that were actually computed for at least one algorithm in
    this dataset.

    Used to hide metrics from the ranking heatmap/report when none of the
    algorithms produce output the metric applies to (e.g. crps/nll on an
    all-deterministic dataset, or acf/dtw/smae if every algorithm were
    probabilistic). Order follows METRIC_LIST (and therefore CATEGORIES).
    """
    return [m for m in METRIC_LIST if any(v is not None for v in scores[m].values())]


def generate_metrics_report(
    y_true: np.ndarray,
    imputations: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str = "reports",
    mask: np.ndarray | None = None,
) -> None:
    """Write a single matrix-style metric report for all algorithms.

    Uses the same masking and applicability rules as _apply_metric; metrics
    that do not apply to an algorithm's output are marked "n/a" rather than
    omitted, so the report shows the full picture.
    Rows = metrics, columns = algorithms.
    ba is split into two rows (bias and loa) so no information is lost.
    Saved to <output_dir>/<dataset_name>_metrics.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_name}_metrics.txt")

    algo_names = list(imputations.keys())
    col_w      = max(max(len(a) for a in algo_names), 12) + 2
    label_w    = 12

    # ── compute raw values: {metric: {algo: raw_value}} ────────────────────
    # raw_value is a float, a (mean_diff, loa) tuple (ba), an Exception, or
    # None ("not applicable" - e.g. a probabilistic metric on a deterministic
    # algorithm, or a fully-observed series).
    raw: dict[str, dict[str, object]] = {}
    for metric_name in METRIC_LIST:
        fn = getattr(metrics, metric_name)
        raw[metric_name] = {}
        for algo, y_pred in imputations.items():
            try:
                raw[metric_name][algo] = _apply_metric(fn, metric_name, y_true, y_pred, mask)
            except Exception as exc:
                raw[metric_name][algo] = exc

    # ── build rows: ba becomes two rows, errors become "ERROR", None becomes "n/a" ──
    def fmt(value: object) -> str:
        if isinstance(value, Exception):
            return f"{'ERROR':^{col_w}}"
        if value is None:
            return f"{'n/a':^{col_w}}"
        return f"{float(value):^{col_w}.6f}"

    rows: list[tuple[str, list[str]]] = []
    for metric_name in METRIC_LIST:
        if metric_name == "ba":
            bias_cells, loa_cells = [], []
            for algo in algo_names:
                v = raw["ba"][algo]
                if isinstance(v, Exception):
                    bias_cells.append(f"{'ERROR':^{col_w}}")
                    loa_cells.append(f"{'ERROR':^{col_w}}")
                elif v is None:
                    bias_cells.append(f"{'n/a':^{col_w}}")
                    loa_cells.append(f"{'n/a':^{col_w}}")
                else:
                    mean_diff, loa = v
                    bias_cells.append(f"{float(mean_diff):^{col_w}.6f}")
                    loa_cells.append(f"{float(loa):^{col_w}.6f}")
            rows.append(("BA (bias)", bias_cells))
            rows.append(("BA (loa)", loa_cells))
        else:
            cells = [fmt(raw[metric_name][algo]) for algo in algo_names]
            rows.append((metric_name.upper(), cells))

    # ── format as aligned table ─────────────────────────────────────────────
    sep = "-" * (label_w + col_w * len(algo_names))
    header = f"{'METRIC':<{label_w}}" + "".join(f"{a:^{col_w}}" for a in algo_names)

    if mask is not None:
        n_missing = int(mask.sum())
        n_total   = int(mask.size)
        eval_note = (
            f"Evaluation: missing positions only  "
            f"({n_missing}/{n_total} = {n_missing/n_total*100:.1f}%)  |  "
            f"acf, dtw, smae: full series  |  "
            f"crps, nll: n/a unless the algorithm output is probabilistic"
        )
    else:
        eval_note = (
            "Evaluation: full series (no mask)  |  "
            "crps, nll: n/a unless the algorithm output is probabilistic"
        )

    lines = [
        f"METRIC SCORES  —  {dataset_name}",
        eval_note,
        "=" * len(sep),
        header,
        sep,
    ]
    for label, cells in rows:
        lines.append(f"{label:<{label_w}}" + "".join(cells))

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Written: {path}")
