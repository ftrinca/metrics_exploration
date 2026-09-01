import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from contextlib import contextmanager

import numpy as np

from imputegap.recovery.imputation import Imputation

from metric_eval.experiments.algorank import config


@contextmanager
def _suppress_c_output():
    """Silence stdout and stderr at the file-descriptor level for the block."""
    old_out = os.dup(1)
    old_err = os.dup(2)
    devnull  = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_out)
        os.close(old_err)


# (display name, family, the ImputeGAP class). This order is the order every
# report column, heatmap row and averaged score list downstream is built in.
ALGORITHMS = [
    ("CDRec", "Matrix Completion", Imputation.MatrixCompletion.CDRec),
    ("ROSL",  "Matrix Completion", Imputation.MatrixCompletion.ROSL),
    ("DynaMMo", "Pattern Search", Imputation.PatternSearch.DynaMMo),
    ("STMVL",   "Pattern Search", Imputation.PatternSearch.STMVL),
    ("BRITS", "Deep Learning", Imputation.DeepLearning.BRITS),
    ("MPIN", "Deep Learning", Imputation.DeepLearning.MPIN),
]

# BRITS and MPIN vary between seeds; every other entry is deterministic and is
# built once per scenario rather than once per seed.
STOCHASTIC_ALGORITHMS = {"BRITS", "MPIN"}
DETERMINISTIC_ALGORITHMS = {name for name, _, _ in ALGORITHMS if name not in STOCHASTIC_ALGORITHMS}

assert [name for name, _, _ in ALGORITHMS] == config.ALGO_NAMES
assert STOCHASTIC_ALGORITHMS == config.STOCHASTIC_ALGO_NAMES


def _seed_everything(seed: int, seed_torch: bool = True) -> None:
    """Seed numpy, and torch as well when seed_torch is True. A missing torch is ignored."""
    np.random.seed(seed)
    if seed_torch:
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass


def build(y_true: np.ndarray, mask: np.ndarray, name: str, seed: int = 0) -> np.ndarray | None:
    """Run one imputation algorithm and return its reconstruction, or None.

    y_true is contaminated with NaNs wherever mask is True. `seed` reaches the
    algorithm solely through the global numpy/torch RNG state. An algorithm that
    raises, or that leaves NaNs behind, returns None: a partially filled series
    would propagate NaNs into every metric.
    """
    family, AlgoClass = next((f, c) for n, f, c in ALGORITHMS if n == name)
    _seed_everything(seed, seed_torch=name in STOCHASTIC_ALGORITHMS)
    ts_m = np.where(mask, np.nan, y_true)

    try:
        imputer = AlgoClass(ts_m)
        imputer.logs = False
        imputer.verbose = False

        t0 = time.perf_counter()
        with _suppress_c_output():
            imputer.impute()
        elapsed = time.perf_counter() - t0

        if np.isnan(imputer.recov_data).any():
            raise ValueError("imputed data contains NaN")

        print(f"{name:<12} {family:<20} {elapsed:>9.2f}s")
        return imputer.recov_data

    except Exception as e:
        print(f"{name:<12} {family:<20} {'ERROR':>10}  ({e})")
        return None
