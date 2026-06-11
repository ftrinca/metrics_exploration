"""Single source of truth for which experiments exist in this project.

Each ExperimentSpec fully describes one experiment:
  - source             where the ground truth comes from: "synthetic", or an
                        ImputeGAP dataset id (e.g. "eeg-alcohol")
  - normalization      how the ground truth is rescaled before contamination,
                        see data/normalization.py
  - missingness_pattern, rate
                        which positions are treated as missing, see
                        missingness_patterns.make_mask()
  - reconstruction     which kind of "filled in" data to evaluate against the
                        ground truth: the fixed synthetic_distortions (six
                        hand-crafted failure modes) or real ImputeGAP
                        imputation_algorithms - see reconstruction/

build_datasets.py reads ALL_SPECS, builds one JSON per spec, and
evaluate_metrics.py reads the same list to produce reports/plots. Adding,
removing, or renaming an experiment only requires editing this file.

Generated JSONs are organised as:

    time_series/synthetic/<missingness_pattern>/[<rate>/]data.json
    time_series/imputegap/<source>/<missingness_pattern>/[<rate>/]data.json

(the rate folder is omitted for "full", since rate is meaningless there).
"""

import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_SERIES_DIR = os.path.join(HERE, "time_series")


@dataclass(frozen=True)
class ExperimentSpec:
    # "synthetic" for the generated data, or an ImputeGAP dataset id (e.g. "eeg-alcohol")
    source: str
    # missingness pattern, see missingness_patterns.PATTERN_FUNCS
    missingness_pattern: str
    # fraction of values removed per series (rate_series in missingness_patterns); ignored for "full"
    rate: float = 1.0
    # normalization applied to the ground truth before contamination, see data/normalization.py
    # "none" | "z_score" | "min_max" | "z_lib" | "m_lib"
    normalization: str = "none"
    # which reconstruction to run, see reconstruction/
    # "synthetic_distortions" (six fixed failure modes) | "imputation_algorithms" (real ImputeGAP algorithms)
    reconstruction: str = "synthetic_distortions"
    # number of series/channels in this dataset
    n_series: int = 20

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def rate_label(self) -> str:
        """e.g. 0.2 -> '20pct'."""
        return f"{round(self.rate * 100):02d}pct"

    @property
    def dataset_name(self) -> str:
        """Short name used for report/plot filenames, e.g. 'synthetic_mcar_20pct'."""
        if self.is_synthetic:
            parts = ["synthetic", self.missingness_pattern]
        else:
            parts = ["imputegap", self.source, self.missingness_pattern]
        if self.missingness_pattern != "full":
            parts.append(self.rate_label)
        return "_".join(parts)

    @property
    def output_path(self) -> str:
        """Where build_datasets.py writes (and evaluate_metrics.py reads) this experiment's JSON."""
        if self.is_synthetic:
            parts = [TIME_SERIES_DIR, "synthetic", self.missingness_pattern]
        else:
            parts = [TIME_SERIES_DIR, "imputegap", self.source, self.missingness_pattern]
        if self.missingness_pattern != "full":
            parts.append(self.rate_label)
        parts.append("data.json")
        return os.path.join(*parts)


# ── synthetic experiments: six hand-crafted "distortions" as the reconstruction ──
# Same set of missingness configurations as before, now expressed via
# missingness_patterns.PATTERN_FUNCS.
SYNTHETIC_SPECS = [
    ExperimentSpec(source="synthetic", missingness_pattern="full"),
    ExperimentSpec(source="synthetic", missingness_pattern="mcar", rate=0.1),
    ExperimentSpec(source="synthetic", missingness_pattern="mcar", rate=0.2),
    ExperimentSpec(source="synthetic", missingness_pattern="mcar", rate=0.4),
    ExperimentSpec(source="synthetic", missingness_pattern="scattered", rate=0.2),
    ExperimentSpec(source="synthetic", missingness_pattern="blackout", rate=0.2),
]

# ── real-world experiments: real ImputeGAP algorithms as the reconstruction ──
IMPUTEGAP_SPECS = [
    ExperimentSpec(
        source="eeg-alcohol", missingness_pattern="mcar", rate=0.2,
        normalization="z_score", reconstruction="imputation_algorithms",
    ),
]

ALL_SPECS = SYNTHETIC_SPECS + IMPUTEGAP_SPECS
