"""The one place the output locations are decided.

Everything the pipelines write — caches, plots, reports — lands under the
repository's `outputs/` directory, beside `metric_eval/` and never inside it,
so the source tree holds only source. Each experiment's config derives its own
subdirectories from the three roots below.
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

TIME_SERIES_DIR = os.path.join(OUTPUTS_DIR, "time_series")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
