import numpy as np
from imputegap.recovery.manager import TimeSeries


def apply_normalization(data: np.ndarray, method: str | None) -> np.ndarray:
    """Return a per-series normalised copy of `data`.

    `method` is one of ImputeGAP's normalizer names ("z_score", "min_max", "z_lib", "m_lib"); None or "none" returns `data` unchanged.
    """
    if method is None or method == "none":
        return data
    return TimeSeries(verbose=False).normalize(normalizer=method, data=data, verbose=False)
