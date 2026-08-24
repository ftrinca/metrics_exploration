import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from contextlib import contextmanager

import numpy as np

from imputegap.recovery.imputation import Imputation


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


def _seed_everything(seed: int, seed_torch: bool = True) -> None:
    """Seed numpy, and torch as well when seed_torch is True. A missing torch is ignored."""
    np.random.seed(seed)
    if seed_torch:
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass


def build(
    y_true: np.ndarray, mask: np.ndarray, seed: int = 0, only: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run the imputation algorithms and return {algo_name: reconstruction}.

    y_true is contaminated with NaNs wherever mask is True. `only` restricts the
    run to those algorithm names, defaulting to all of ALGORITHMS. `seed`
    reaches the algorithms solely through the global numpy/torch RNG state. An
    algorithm that raises, or that leaves NaNs behind, is absent from the result.

    ImputeGAP's own RMSE/MAE/MI/CORRELATION are printed as a running sanity
    check; the thesis metrics are computed later by algo_ranking/score.py.
    """
    algos_to_run = ALGORITHMS if only is None else [a for a in ALGORITHMS if a[0] in only]
    needs_torch = any(name in STOCHASTIC_ALGORITHMS for name, _, _ in algos_to_run)
    _seed_everything(seed, seed_torch=needs_torch)
    ts_m = np.where(mask, np.nan, y_true)
    results: dict[str, np.ndarray] = {}

    print(f"{'Algorithm':<24} {'Family':<20} {'Time (s)':>10}  "
          f"{'RMSE':>10}  {'MAE':>10}  {'MI':>10}  {'CORRELATION':>12}")
    print("-" * 104)

    for name, family, AlgoClass in algos_to_run:
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

            imputer.score(y_true, imputer.recov_data, verbose=False)
            rmse        = imputer.metrics.get("RMSE", float("nan"))
            mae         = imputer.metrics.get("MAE", float("nan"))
            mi          = imputer.metrics.get("MI", float("nan"))
            correlation = imputer.metrics.get("CORRELATION", float("nan"))

            print(f"{name:<24} {family:<20} {elapsed:>10.4f}  "
                  f"{rmse:>10.4f}  {mae:>10.4f}  {mi:>10.4f}  {correlation:>12.4f}")

            results[name] = imputer.recov_data

        except Exception as e:
            print(f"{name:<24} {family:<20} {'ERROR':>10}  ({e})")

    return results
