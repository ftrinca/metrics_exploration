import core.metrics as metrics
import numpy as np

from core.metric_config import FULL_SERIES_METRICS, METRIC_LIST, PROBABILISTIC_METRICS


def to_scalar(metric_name: str, value) -> float:
    """Reduce a metric's return value to one float. `ba` is reduced to |mean_diff|."""
    if metric_name == "ba":
        mean_diff, _ = value
        return abs(mean_diff)
    return float(value)


def is_probabilistic(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Whether y_pred carries an extra samples dimension, i.e. a distribution per point."""
    return y_pred.ndim == y_true.ndim + 1


def metric_applies(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Whether `metric_name` is scored for this output shape.

    The full-series metrics need a point estimate and the probabilistic ones
    are scored only on posterior samples, so the two sets exclude each other.
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
    """Evaluate one metric, returning None where it does not apply.

    With a mask, evaluation is restricted to the True positions, except for the
    full-series metrics. A 2-D (n_series, n_timesteps) input is evaluated per
    series and averaged, which is why RMSE here is not ImputeGAP's pooled RMSE.
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
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Score every reconstruction on every metric.

    Returns {metric: {name: score or None}}, None meaning the metric does not
    apply to that output or raised. `metric_names` restricts the pass to a
    subset of METRIC_LIST, so a cached score file missing one metric can be
    topped up without paying for DTW again.
    """
    wanted = (METRIC_LIST if metric_names is None
              else [m for m in METRIC_LIST if m in set(metric_names)])
    scores: dict[str, dict[str, float | None]] = {}

    for metric_name in wanted:
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
    """The metrics with at least one non-None score, in METRIC_LIST order."""
    return [m for m in METRIC_LIST if any(v is not None for v in scores[m].values())]
