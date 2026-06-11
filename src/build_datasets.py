"""Single entry point that builds every experiment defined in
experiment_config.ALL_SPECS:

  1. ground truth     - data/synthetic_ground_truth.py or
                         data/real_world_ground_truth.py, depending on
                         spec.source
  2. normalization    - data/normalization.py, depending on
                         spec.normalization
  3. missingness mask - missingness_patterns.make_mask(), depending on
                         spec.missingness_pattern and spec.rate
  4. reconstruction   - reconstruction/synthetic_distortions.py or
                         reconstruction/imputation_algorithms.py, depending on
                         spec.reconstruction

Each experiment's {y_true, mask, **reconstructions} is written to
spec.output_path (see experiment_config.ExperimentSpec.output_path) via
dataset_io.save_dataset(). evaluate_metrics.py reads these files to produce
reports/plots.

Run once (or re-run to regenerate):
  python build_datasets.py
"""

import dataset_io
from data import normalization, real_world_ground_truth, synthetic_ground_truth
from experiment_config import ALL_SPECS
from missingness_patterns import make_mask
from reconstruction import imputation_algorithms, synthetic_distortions

for spec in ALL_SPECS:
    print(f"-- {spec.dataset_name} --------------------------------------------------")

    # ── 1. ground truth ──────────────────────────────────────────────────────
    if spec.is_synthetic:
        y_true = synthetic_ground_truth.generate(n_series=spec.n_series)
    else:
        y_true = real_world_ground_truth.generate(spec.source, spec.n_series)

    # ── 2. normalization ─────────────────────────────────────────────────────
    y_true = normalization.apply_normalization(y_true, spec.normalization)

    # ── 3. missingness mask ──────────────────────────────────────────────────
    mask = make_mask(y_true, spec.missingness_pattern, spec.rate)

    # ── 4. reconstruction ────────────────────────────────────────────────────
    if spec.reconstruction == "synthetic_distortions":
        reconstructions = synthetic_distortions.build(y_true)
    elif spec.reconstruction == "imputation_algorithms":
        reconstructions = imputation_algorithms.build(y_true, mask)
    else:
        raise ValueError(f"Unknown reconstruction {spec.reconstruction!r}")

    # ── save ──────────────────────────────────────────────────────────────────
    json_out = {
        "y_true": dataset_io.matrix_to_lists(y_true),
        "mask": dataset_io.bool_matrix_to_mask(mask),
    }
    for name, mat in reconstructions.items():
        json_out[name] = dataset_io.matrix_to_lists(mat)

    dataset_io.save_dataset(spec.output_path, json_out)

    n_missing = int(mask.sum())
    n_total = int(mask.size)
    print(f"{spec.dataset_name:<24} -> {spec.output_path}  "
          f"({n_missing}/{n_total} = {n_missing / n_total * 100:.1f}% missing)\n")
