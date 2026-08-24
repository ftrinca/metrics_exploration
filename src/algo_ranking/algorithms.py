"""Run a fixed list of ImputeGAP imputation algorithms against a
NaN-contaminated copy of `y_true` and return each one's reconstruction.

ImputeGAP's own RMSE/MAE/MI/CORRELATION are printed here as a running sanity
check only. The metrics the thesis reports are computed later, by
algo_ranking/score.py, from the saved reconstructions.

Open caveat: no algorithm here returns posterior samples, so the 2-D branches
of core.metrics' crps and nll are never exercised by this pipeline. Feeding
them would also need a third (samples) dimension in
core.dataset_io.matrix_to_lists, whose format stores one value per
[series][timestep].
"""

import os

# Must be set before torch or ImputeGAP's native algorithm libraries load. Both
# bundle their own OpenMP runtime, and loading two of them into one process
# trips OpenMP's duplicate-runtime check, which calls abort() rather than
# raising, so it cannot be caught and surfaces as a bare crash with no
# traceback. Disabling the check is safe here because this project only ever
# runs algorithms sequentially, never concurrently.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from contextlib import contextmanager

import numpy as np

from imputegap.recovery.imputation import Imputation

# ImputeGAP hardcodes seed=42 inside Imputation.DeepLearning.BRITS.impute()
# whatever params it is given, and no other algorithm in ALGORITHMS exposes
# usable seed control through its params dict either, so the global numpy/torch
# RNG state set by _seed_everything is the only lever on run-to-run variation.


@contextmanager
def _suppress_c_output():
    """Silence stdout and stderr at the file-descriptor level for the block.

    Several ImputeGAP algorithms are implemented in C/C++ and write solver
    output straight to fd 1 and fd 2, which `verbose=False` cannot intercept.
    Cosmetic only, with no effect on results.
    """
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


# (display name, family, the ImputeGAP class that implements it). The order
# here is the order every report column, heatmap row and averaged score list
# downstream is built in.
ALGORITHMS = [
    ("CDRec", "Matrix Completion", Imputation.MatrixCompletion.CDRec),
    ("ROSL",  "Matrix Completion", Imputation.MatrixCompletion.ROSL),
    ("DynaMMo", "Pattern Search", Imputation.PatternSearch.DynaMMo),
    ("STMVL",   "Pattern Search", Imputation.PatternSearch.STMVL),
    ("BRITS", "Deep Learning", Imputation.DeepLearning.BRITS),
    ("MPIN", "Deep Learning", Imputation.DeepLearning.MPIN),
]

# BRITS and MPIN vary between seeds because of random weight initialisation and
# training; every other entry returns the same output for the same input. build.py
# uses this split to run the deterministic ones once per (dataset, pattern, rate)
# rather than once per seed, so moving a name between these two sets changes how
# much of the build cache has to be recomputed.
STOCHASTIC_ALGORITHMS = {"BRITS", "MPIN"}
DETERMINISTIC_ALGORITHMS = {name for name, _, _ in ALGORITHMS if name not in STOCHASTIC_ALGORITHMS}


def _seed_everything(seed: int, seed_torch: bool = True) -> None:
    """Seed numpy, and torch as well when seed_torch is True. A missing torch
    is ignored rather than raising.

    seed_torch=False skips importing torch at all, which matters because the
    import is what makes torch initialise its C extensions and its own bundled
    OpenMP runtime. A process that does that and then loads a native ImputeGAP
    algorithm has two OpenMP runtimes in it, which aborts, or with
    KMP_DUPLICATE_LIB_OK set, segfaults later instead.
    """
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
    """Return {algo_name: reconstruction} of y_true with NaNs at every position
    where mask is True. Each reconstruction keeps y_true's own orientation.

    `only` restricts the run to those algorithm names, defaulting to all of
    ALGORITHMS. `seed` reaches the algorithms solely through the global
    numpy/torch RNG state, so a deterministic algorithm returns the same result
    for every seed.

    An algorithm that raises, or that leaves NaNs behind, is simply absent from
    the returned dict.
    """
    algos_to_run = ALGORITHMS if only is None else [a for a in ALGORITHMS if a[0] in only]
    # Only import torch at all when a torch-based algorithm is actually in this
    # call, because the import itself is what causes the OpenMP collision
    # described in _seed_everything.
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

            # Remaining NaNs mean the algorithm did not fill the gaps at all,
            # and every metric would propagate them, so this is the one result
            # that has to be dropped. Everything else finite is kept, including
            # a flat constant fill or an error large enough to trip ImputeGAP's
            # own RMSE cap, because those are real reconstructions that belong
            # in the results as visibly bad scores rather than as an algorithm
            # that silently vanished from the scenario.
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
