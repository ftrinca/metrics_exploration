"""Missingness patterns, built on top of ImputeGAP's GenGap contamination functions.

A "pattern" here is a function that takes a ground-truth matrix of shape
(n_timesteps, n_series) and returns a boolean mask of the same shape, where
True marks a position that is treated as missing (and therefore included in
metric evaluation). This is the format expected by utils.compute_all_scores
and utils.generate_metrics_report (mask=None means "evaluate on everything").

GenGap itself does not return masks - it returns a copy of the input with
NaNs at the removed positions. The wrappers below just call GenGap and then
take np.isnan() of the result.

  full      -> every position is evaluated; no contamination is applied.
  mcar      -> GenGap.mcar() with block_size=1, i.e. individual points are
               removed at random rather than fixed-size blocks (GenGap's
               default block_size=10 groups removals into runs of 10, which
               is closer to the old "block"/"blackout" patterns).
  scattered -> GenGap.scattered(): one contiguous block per series, placed at
               a random start position. Closest equivalent to the old "block"
               pattern (one random gap per series).
  blackout  -> GenGap.blackout(): one contiguous gap at the SAME position in
               every series simultaneously (an "all sensors down at once"
               scenario). Note this differs from the old "blackout", which
               put three smaller gaps spread across a single series.

rate_dataset (GenGap's name for "fraction of series selected for
contamination") is fixed at 1.0 everywhere below, so every series ends up
with at least some missing values and therefore contributes to the per-series
averages in compute_all_scores. rate_series ("fraction of values removed
within a selected series") is the `rate` argument below and is the value that
varies between configurations (10%, 20%, 40%, ...).

Remaining GenGap patterns, added below for completeness:

  aligned      -> one contiguous block per series, all starting at the same
                  offset position. With rate_dataset=1.0 this is the same as
                  blackout (blackout is just aligned with rate_dataset=1).
  gaussian     -> per series, removed positions are sampled (without
                  replacement) from a Gaussian distribution centred on the
                  series midpoint, so the middle of the series is more likely
                  to go missing than the edges.
  distribution -> like gaussian, but the per-position removal probabilities
                  are passed in directly via `probabilities_list`. Below this
                  defaults to a uniform distribution, making it equivalent to
                  removing positions uniformly at random.
  disjoint     -> walks through the series with a shared cursor: series 0
                  loses [P, P+W), series 1 loses [P+W, P+2W), etc., so the
                  missing blocks across series never overlap.
  overlap      -> same as disjoint, but the cursor is rewound by `shift`
                  before each series' block, so consecutive blocks overlap.
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


# Single dispatch table used by both generator scripts and dataset_config.
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
