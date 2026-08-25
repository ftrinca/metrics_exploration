import numpy as np
from imputegap.recovery.contamination import GenGap

def full_mask(data: np.ndarray) -> np.ndarray:
    """Every position is missing. No contamination is applied."""
    return np.ones_like(data, dtype=bool)

def mcar_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """Pointwise random removal across every series."""
    contaminated = GenGap.mcar(data, rate_dataset=1.0, rate_series=rate, block_size=1, verbose=verbose)
    return np.isnan(contaminated)

def scattered_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """One contiguous gap per series, at an independent random start position."""
    contaminated = GenGap.scattered(data, rate_dataset=1.0, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)

def blackout_mask(data: np.ndarray, rate: float, verbose: bool = False) -> np.ndarray:
    """One contiguous gap, aligned at the same position across all series."""
    contaminated = GenGap.blackout(data, rate_series=rate, verbose=verbose)
    return np.isnan(contaminated)

PATTERN_FUNCS = {
    "full": full_mask,
    "mcar": mcar_mask,
    "scattered": scattered_mask,
    "blackout": blackout_mask,
}

def make_mask(data: np.ndarray, pattern: str, rate: float, verbose: bool = False) -> np.ndarray:
    """Build a boolean mask over an (n_timesteps, n_series) array, True marking a missing position."""
    fn = PATTERN_FUNCS.get(pattern)
    if fn is None:
        raise ValueError(f"Unknown pattern {pattern!r}. Choose one of: {sorted(PATTERN_FUNCS)}")
    if pattern == "full":
        return fn(data)
    return fn(data, rate, verbose=verbose)
