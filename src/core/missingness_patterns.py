"""Missingness patterns, as boolean masks over an (n_timesteps, n_series)
array where True marks a position treated as missing and therefore evaluated.

Thin wrappers over ImputeGAP's GenGap, which returns a NaN-punched copy rather
than a mask. rate_dataset is fixed at 1.0 everywhere, so every series is
contaminated and contributes to the per-series averages in core.scoring.

"""

import math

import numpy as np

from imputegap.recovery.contamination import GenGap


def full_mask(data: np.ndarray) -> np.ndarray:
    """Every position is missing/evaluated. No contamination is applied."""
    return np.ones_like(data, dtype=bool)


def mcar_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """Pointwise random removal (block_size=1) across every series."""
    contaminated = GenGap.mcar(
        data, rate_dataset=1.0, rate_series=rate, block_size=1, verbose=verbose
    )
    return np.isnan(contaminated)


def aligned_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """One contiguous gap, at the same offset position in every series."""
    contaminated = GenGap.aligned(data, rate_dataset=1.0, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)


def scattered_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """One contiguous gap per series, at an independent random start position."""
    contaminated = GenGap.scattered(
        data, rate_dataset=1.0, rate_series=rate, verbose=verbose
    )
    return np.isnan(contaminated)


def blackout_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """One contiguous gap, aligned at the same position across all series."""
    contaminated = GenGap.blackout(data, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)


def gaussian_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """Per series, removed positions follow a Gaussian distribution centred
    on the series midpoint (positions near the middle are more likely to be
    removed than positions near the edges).
    """
    contaminated = GenGap.gaussian(data, rate_dataset=1.0, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)


def distribution_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """Per series, removed positions are sampled from a uniform distribution
    over the non-offset positions (GenGap.distribution() requires an explicit
    probabilities_list - a uniform one is built here for that purpose).
    """
    n_timesteps, n_series = data.shape
    n_free = n_timesteps - math.ceil(0.1 * n_timesteps)  # default offset = 0.1
    uniform = np.full((n_series, n_free), 1.0 / n_free)
    contaminated = GenGap.distribution(
        data, rate_dataset=1.0, rate_series=rate, probabilities_list=uniform, verbose=verbose
    )
    return np.isnan(contaminated)


def disjoint_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """Per series, one contiguous block; blocks for successive series are
    placed back-to-back along a shared cursor (no randomness).
    """
    contaminated = GenGap.disjoint(data, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)


def overlap_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """Like disjoint_mask, but each series' block is rewound so consecutive
    series' missing blocks overlap (no randomness).
    """
    contaminated = GenGap.overlap(data, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)


PATTERN_FUNCS = {
    "full": full_mask,
    "mcar": mcar_mask,
    "aligned": aligned_mask,
    "scattered": scattered_mask,
    "blackout": blackout_mask,
    "gaussian": gaussian_mask,
    "distribution": distribution_mask,
    "disjoint": disjoint_mask,
    "overlap": overlap_mask,
}


def make_mask(data: np.ndarray, pattern: str, rate: float, verbose: bool = False) -> np.ndarray:
    """Dispatch to the right pattern function. `rate` is ignored for "full"."""
    fn = PATTERN_FUNCS.get(pattern)
    if fn is None:
        raise ValueError(
            f"Unknown pattern {pattern!r}. Choose one of: {sorted(PATTERN_FUNCS)}"
        )
    if pattern == "full":
        return fn(data)
    return fn(data, rate, verbose=verbose)
