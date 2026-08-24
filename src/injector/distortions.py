"""The eight distortions, each parameterised by its own severity knob.

Every distortion is mask-aware: only positions where mask is True are altered
and observed positions are copied through exactly, which is what lets the
full-series metrics (ACF, DTW, sMAE) measure a reconstruction rather than an
artificially disturbed series. All severities are relative to sigma, the
standard deviation of the TRUE values inside that series' missing block.

Each distortion has the signature fn(y_col, idx, severity, seed) and returns an
array of len(idx), so calibrate.py can treat them interchangeably. The seed
builds a fresh generator INSIDE the function, so repeated calls with the same
arguments return the same array; without that, bisection on a stochastic
distortion would chase its own sampling noise instead of the severity.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from injector.config import SPIKE_RATE


def gap_sigma(y_col: np.ndarray, idx: np.ndarray) -> float:
    """Standard deviation of the true values at this series' missing positions."""
    return float(np.std(y_col[idx]))


def gap_mean(y_col: np.ndarray, idx: np.ndarray) -> float:
    return float(np.mean(y_col[idx]))


def damage(y_col: np.ndarray, idx: np.ndarray, values: np.ndarray) -> float:
    """Mean absolute error at the masked positions, in units of sigma, which is
    the quantity every distortion is calibrated to.

    MAE rather than RMSE because it is linear in the error, which makes it
    monotone in every severity knob below and therefore solvable. RMSE would
    also depend on how the error is spread across positions, which is precisely
    what should stay free to vary between distortions.
    """
    sigma = gap_sigma(y_col, idx)
    if sigma <= 0.0:
        return 0.0
    return float(np.mean(np.abs(values - y_col[idx]))) / sigma


def _rng(seed: int, name: str) -> np.random.Generator:
    """Deterministic generator for one (seed, distortion) pair.

    Mixing the distortion name in means noise and spikes on the same series do
    not share a stream, while repeated calls for the same distortion are
    byte-identical.
    """
    return np.random.default_rng(seed + (abs(hash(name)) % 100_000))


def noise(y_col, idx, severity, seed):
    """Add independent Gaussian noise at each missing position.

    Damage is severity * sigma * sqrt(2/pi) in expectation, so roughly
    0.8 * severity, monotone and near-linear.
    """
    sigma = gap_sigma(y_col, idx)
    eps = _rng(seed, "noise").normal(0.0, severity * sigma, size=idx.size)
    return y_col[idx] + eps


def bias(y_col, idx, severity, seed):
    """Shift every missing position by a constant.

    Damage is exactly severity, since every error equals the same constant.
    """
    return y_col[idx] + severity * gap_sigma(y_col, idx)


def reorder(y_col, idx, severity, seed):
    """Cyclically permute a random fraction of the gap positions, severity being
    that fraction in (0, 1].

    Values are permuted rather than replaced, so the multiset in the gap is
    unchanged whatever the fraction, which makes the WD / JSD / KLD invariance
    exact and keeps the mean exact so Bland-Altman and CDT read zero too.

    Two details make the fraction a usable severity knob. The permutation order
    is drawn once and is independent of the fraction, so the moved set grows
    with the fraction instead of being redrawn each time. The chosen subset is
    rotated rather than randomly permuted, because a random permutation leaves
    an unpredictable number of values in place, which turns damage into a noisy
    step function that bisection cannot solve.

    Damage is roughly severity * E|y_i - y_j| ~ 1.13 * severity * sigma, so a
    full rotation is the ceiling.
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
    """Round every missing value onto a uniform grid of step severity * sigma.

    The step is continuous, which is what makes the distortion calibratable,
    and rounding depends on no fitted model, so the result is exactly
    reproducible. Damage rises with the step and then plateaus near
    E|y - mu| ~ 0.8 sigma, where every value has collapsed onto one grid point.
    """
    step = float(severity) * gap_sigma(y_col, idx)
    if step <= 0.0:
        return y_col[idx].copy()
    return step * np.round(y_col[idx] / step)


def lag(y_col, idx, severity, seed):
    """Replace each missing value with the true value `severity` steps earlier,
    so the replacements are real values from the series sitting at the wrong
    moment.

    The lag may be fractional, linearly interpolating between the two
    neighbouring integer lags, because damage jumps between consecutive integer
    lags by far more than the calibration tolerance. At whole numbers this is
    exactly an integer lag.
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
    """Substitute a moving average of the true series at the missing positions,
    modelling regression to the local mean: numerically close to the truth
    while erasing short-term structure.

    The window may be fractional, blending the two neighbouring integer widths,
    because integer-only windows step in damage by more than the calibration
    tolerance. Damage saturates near E|y - mu| ~ 0.8 sigma once the average is
    effectively the series mean, which is the ceiling config.TARGET_DAMAGE and
    the sweep levels stay below.
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
    """Add a large, randomly signed spike to a fixed fraction of positions.

    The fraction is held at config.SPIKE_RATE so the solver has a single knob.
    Damage is exactly (n_spikes / n) * severity, since spiked positions are
    wrong by exactly severity * sigma and every other position is exact.
    """
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
    """Expand the missing values around their own mean by (1 + severity).

    Shape and order survive intact and every value is off by the same factor,
    so the output is a positive-slope affine transform of the truth with the
    mean left where it was. Damage is exactly severity * E|y - mu|.
    """
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
    """Scan points for the lag search.

    The low end is scanned finely because a fractional lag on a series with any
    high-frequency content already does real damage: at lag 1 a noisy series is
    typically already past 0.4 sigma, so the whole lower half of a damage sweep
    lives between lags 0 and 1 and a scan starting at 0.5 would miss it.
    """
    top = float(min(y_col.size // 2, 200))
    fine = np.arange(0.02, 1.0, 0.02)
    coarse = np.arange(1.0, top + 0.5, 0.5)
    return list(np.concatenate([fine, coarse]))


def _smooth_scan(y_col, idx):
    """Scan points for the smoothing window.

    Damage is non-monotone in practice, because on a series with a slow drift a
    very wide moving average can sit closer to the truth than a middling one.
    Windows below 1 are meaningless, and fractional windows in [1, 2] cover the
    mildest smoothing.
    """
    top = float(min(y_col.size // 2, 401))
    fine = np.arange(1.0, 3.0, 0.1)
    coarse = np.arange(3.0, top + 1.0, 1.0)
    return list(np.concatenate([fine, coarse]))


# How the solver should search each severity.
#   "continuous"  damage rises monotonically from zero, so bracket and bisect
#   "scan"        damage is continuous in the parameter but not monotone, so
#                 walk the scan points for the first crossing and bisect there
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

    severities maps series index to severity, as produced by
    calibrate.solve_scenario. Returns a full (n_timesteps, n_series) array that
    equals y_true wherever mask is False. A series missing from severities is
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
