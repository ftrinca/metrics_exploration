"""Normalization applied to ground truth before contamination, shared by
synthetic and real-world data alike. Reuses ImputeGAP's own normalizers via
TimeSeries.normalize(data=...), which operates per-series (axis=0) on an
arbitrary (n_timesteps, n_series) array.
"""

import numpy as np
from imputegap.recovery.manager import TimeSeries

# "none"     -> no rescaling, values are used as-is
# "z_score"  -> each series rescaled to mean 0, std 1
# "min_max"  -> each series rescaled to [0, 1]
# "z_lib"    -> z_score via scipy.stats.zscore
# "m_lib"    -> min_max via sklearn MinMaxScaler

def apply_normalization(data: np.ndarray, method: str | None) -> np.ndarray:
    """Return a normalized copy of `data`. method=None or "none" returns
    `data` unchanged."""
    if method is None or method == "none":
        return data
    return TimeSeries(verbose=False).normalize(normalizer=method, data=data, verbose=False)
