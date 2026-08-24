from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from injector.config import SPIKE_RATE


def gap_sigma(y_col: np.ndarray, idx: np.ndarray) -> float:
    """Standard deviation of the true values at this series' missing positions."""
    return float(np.std(y_col[idx]))


def gap_mean(y_col: np.ndarray, idx: np.ndarray) -> float:
    """Mean of the true values at this series' missing positions."""
    return float(np.mean(y_col[idx]))


def damage(y_col: np.ndarray, idx: np.ndarray, values: np.ndarray) -> float:
    """Mean absolute error at the masked positions, in units of sigma."""
    sigma = gap_sigma(y_col, idx)
    if sigma <= 0.0:
        return 0.0
    return float(np.mean(np.abs(values - y_col[idx]))) / sigma


def _rng(seed: int, name: str) -> np.random.Generator:
    """Deterministic generator for one (seed, distortion) pair."""
    return np.random.default_rng(seed + (abs(hash(name)) % 100_000))


def noise(y_col, idx, severity, seed):
    """Add independent Gaussian noise of sd `severity` * sigma at each missing position."""
    sigma = gap_sigma(y_col, idx)
    eps = _rng(seed, "noise").normal(0.0, severity * sigma, size=idx.size)
    return y_col[idx] + eps


def bias(y_col, idx, severity, seed):
    """Shift every missing position by a constant `severity` * sigma."""
    return y_col[idx] + severity * gap_sigma(y_col, idx)


def reorder(y_col, idx, severity, seed):
    """Cyclically permute a fraction `severity` of the gap positions.

    The permutation order is drawn once and is independent of the fraction, and
    the chosen subset is rotated rather than randomly permuted, which keeps
    damage a smooth function of the fraction.
    """
    vals = y_col[idx].copy()
    n = vals.size
    k = int(round(float(severity) * n))
    if k < 2:
        return vals
    order = _rng(seed, "reorder").permutation(n)
    chosen = order[:k]
    vals[chosen] = vals[np.roll(chosen, 1)]
    return vals


def discretise(y_col, idx, severity, seed):
    """Round every missing value onto a uniform grid of step `severity` * sigma."""
    step = float(severity) * gap_sigma(y_col, idx)
    if step <= 0.0:
        return y_col[idx].copy()
    return step * np.round(y_col[idx] / step)


def lag(y_col, idx, severity, seed):
    """Replace each missing value with the true value `severity` steps earlier.

    The lag may be fractional, linearly interpolating between the two
    neighbouring integer lags.
    """
    n_timesteps = y_col.size
    shift = float(severity)
    lo = int(np.floor(shift))
    frac = shift - lo
    a = y_col[(idx - lo) % n_timesteps]
    if frac <= 0.0:
        return a
    b = y_col[(idx - lo - 1) % n_timesteps]
    return (1.0 - frac) * a + frac * b


def smooth(y_col, idx, severity, seed):
    """Substitute a moving average of width `severity` at the missing positions.

    The window may be fractional, blending the two neighbouring integer widths.
    """
    w = max(1.0, float(severity))
    lo = int(np.floor(w))
    frac = w - lo
    a = uniform_filter1d(y_col, size=max(1, lo))[idx]
    if frac <= 0.0:
        return a
    b = uniform_filter1d(y_col, size=max(1, lo + 1))[idx]
    return (1.0 - frac) * a + frac * b


def spikes(y_col, idx, severity, seed):
    """Add a randomly signed spike of `severity` * sigma to config.SPIKE_RATE of the positions."""
    sigma = gap_sigma(y_col, idx)
    vals = y_col[idx].copy()
    n = vals.size
    n_spikes = max(1, min(n, round(SPIKE_RATE * n)))
    rng = _rng(seed, "spikes")
    where = rng.choice(n, size=n_spikes, replace=False)
    sign = rng.choice([-1.0, 1.0], size=n_spikes)
    vals[where] += sign * float(severity) * sigma
    return vals


def rescale(y_col, idx, severity, seed):
    """Expand the missing values around their own mean by (1 + `severity`)."""
    mu = gap_mean(y_col, idx)
    return mu + (1.0 + float(severity)) * (y_col[idx] - mu)


FUNCTIONS = {
    "noise": noise,
    "bias": bias,
    "reorder": reorder,
    "discretise": discretise,
    "lag": lag,
    "smooth": smooth,
    "spikes": spikes,
    "rescale": rescale,
}


def _lag_scan(y_col, idx):
    """Scan points for the lag search, fine below one timestep and coarse above."""
    top = float(min(y_col.size // 2, 200))
    fine = np.arange(0.02, 1.0, 0.02)
    coarse = np.arange(1.0, top + 0.5, 0.5)
    return list(np.concatenate([fine, coarse]))


def _smooth_scan(y_col, idx):
    """Scan points for the smoothing window, fine in [1, 3) and coarse above."""
    top = float(min(y_col.size // 2, 401))
    fine = np.arange(1.0, 3.0, 0.1)
    coarse = np.arange(3.0, top + 1.0, 1.0)
    return list(np.concatenate([fine, coarse]))


# How the solver should search each severity.
#   "continuous"  damage rises monotonically from zero, so bracket and bisect
#   "scan"        damage is not monotone, so walk the scan points for the first
#                 crossing and bisect there
SEVERITY_SPEC = {
    "noise":      {"kind": "continuous", "lo": 1e-4, "hi": 4.0},
    "bias":       {"kind": "continuous", "lo": 1e-4, "hi": 4.0},
    "reorder":    {"kind": "continuous", "lo": 1e-3, "hi": 1.0},
    "discretise": {"kind": "continuous", "lo": 1e-3, "hi": 12.0},
    "lag":        {"kind": "scan", "scan": _lag_scan},
    "smooth":     {"kind": "scan", "scan": _smooth_scan},
    "spikes":     {"kind": "continuous", "lo": 1e-3, "hi": 60.0},
    "rescale":    {"kind": "continuous", "lo": 1e-4, "hi": 12.0},
}


def apply_one(y_true, mask, name, severities, seed):
    """Apply one distortion across every series at its own solved severity.

    `severities` maps series index to severity, as produced by
    calibrate.solve_scenario. Returns a full (n_timesteps, n_series) array that
    equals y_true wherever mask is False; a series missing from `severities` is
    left untouched.
    """
    fn = FUNCTIONS[name]
    out = y_true.copy()
    for series_idx in range(y_true.shape[1]):
        idx = np.where(mask[:, series_idx])[0]
        if idx.size == 0:
            continue
        sev = severities.get(series_idx)
        if sev is None:
            continue
        out[idx, series_idx] = fn(y_true[:, series_idx], idx, sev, seed + 1000 + series_idx)
    return out
