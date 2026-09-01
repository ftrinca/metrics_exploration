import numpy as np

from experiments.cis.config import ADOPTED_POWER, CIS_METRICS

# MI and R2 rise with agreement; every other metric in the panel falls with it.
_SIMILARITY = {"mi", "r2"}


def _ceiling(metric: str, reference: dict) -> float:
    """The value a metric takes when the reconstruction and the truth coincide."""
    return reference["mi_self"] if metric == "mi" else reference["r2_ceiling"]


def component(metric: str, value: float, reference: dict) -> float:
    """One metric as a distance relative to the constant-mean reconstruction.

    0 means the reconstruction and the truth coincide on this metric and 1 means
    it is as far from the truth as the reference, whichever direction the raw
    metric runs in. Dividing by a per-scenario reference is what makes the four
    components comparable to each other and across datasets.
    """
    ref = reference["scores"][metric]
    if metric in _SIMILARITY:
        top = _ceiling(metric, reference)
        span = top - ref
        return float(np.clip((top - value) / span, 0.0, None)) if span > 0 else 0.0
    return float(value / ref) if ref > 0 else 0.0


def components(scores: dict, reference: dict, algo: str,
               metrics: tuple[str, ...] = CIS_METRICS) -> dict[str, float] | None:
    """The relative distances of one reconstruction, or None when a metric is absent."""
    out = {}
    for metric in metrics:
        value = scores[metric].get(algo)
        if value is None or not np.isfinite(value):
            return None
        out[metric] = component(metric, value, reference)
    return out


def combine(distances: dict[str, float], power: float = ADOPTED_POWER) -> float:
    """Power mean of the components, in the same units as a single component.

    An exponent above one stops three small distances from cancelling one large
    one, so a reconstruction that fails on a single axis cannot be rescued by the
    other three.
    """
    values = np.array(list(distances.values()), dtype=float)
    if np.isinf(power):
        return float(values.max())
    if power == 1.0:
        return float(values.mean())
    return float((np.mean(values ** power)) ** (1.0 / power))


def cis(scores: dict, reference: dict, algo: str,
        metrics: tuple[str, ...] = CIS_METRICS,
        power: float = ADOPTED_POWER) -> float | None:
    """CIS of one reconstruction: 0 coincides with the truth, 1 matches the reference."""
    distances = components(scores, reference, algo, metrics)
    return None if distances is None else combine(distances, power)
