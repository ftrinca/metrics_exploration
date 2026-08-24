"""Subprocess worker: run exactly one algorithm on a pre-computed (y_true,
mask) pair and write its reconstruction to a small JSON file.

Spawned by build.py's _run_algorithms_isolated(), one fresh OS process per
algorithm, so that no two algorithms' native libraries or ML frameworks
(torch, tensorflow, ImputeGAP's C/C++ libraries) ever share a process. Two
such collisions were found in practice, a duplicate-OpenMP-runtime crash
between torch and ImputeGAP's native libraries and a hard deadlock between
CDRec and DeepMVI. Isolating every algorithm rules out that whole class of
failure regardless of which algorithms end up in ALGORITHMS, which
enumerating unsafe pairings by hand would not.

Invoked as `python -m algo_ranking._run_algorithm`, not run directly.
"""

import argparse
import json
import sys

import numpy as np

from algo_ranking import algorithms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="JSON file with {'y_true': [...], 'mask': [...]}, native (n_series, n_timesteps) orientation",
    )
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", required=True,
        help="Where to write {'result': [...]} (native orientation) - key absent if the algorithm failed",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    y_true_t = np.array(data["y_true"])
    mask_t = np.array(data["mask"])

    results = algorithms.build(y_true_t, mask_t, seed=args.seed, only={args.algo})

    out = {}
    if args.algo in results:
        out["result"] = results[args.algo].tolist()
    with open(args.output, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
