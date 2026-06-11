# Metrics Exploration

This folder contains a small pipeline for testing imputation-evaluation metrics
on time series data. It builds a set of datasets (synthetic or real, with
different missingness patterns), runs some "reconstructions" against each
dataset (either hand-crafted distortions or real ImputeGAP imputation
algorithms), computes a fixed set of metrics for every reconstruction, and
writes out text reports and plots.

The pipeline has two steps:

1. `python src/build_datasets.py` - generates the data and writes one JSON
   file per experiment to `src/time_series/`.
2. `python src/evaluate_metrics.py` - reads those JSON files and writes
   reports to `src/reports/` and plots to `src/plots/`.

Step 1 only needs to be re-run when something changes about *what data*
exists (a new dataset, a new missingness pattern, a different normalization,
...). Step 2 can be re-run on its own whenever something changes about *how
the data is evaluated or displayed* (a new metric, a different plotted
series, ranking logic, ...), since it just reads the JSON files that are
already there.

## Layout

```
README.md               - this file
metric_verification.md  - per-metric formula review
requirements.txt        - pinned dependencies
src/                     - all code, plus generated data/reports/plots
```

## Setup

All dependencies are listed in `requirements.txt`, plus `imputegap` (used for
loading real-world data, normalization, missingness patterns, and the
real imputation algorithms - it isn't pinned in `requirements.txt` because of
a packaging issue, install it separately):

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install imputegap==1.1.2
```

Then run the two scripts above from this directory (they are invoked as
`python src/<script>.py`, with paths resolved relative to `src/`
regardless of the current working directory).

## Where to change things

This section lists the most common things to adjust and where to do it.

**Add, remove, or change a dataset/experiment** (which missingness patterns,
rates, normalizations, and reconstructions exist) -> `src/experiment_config.py`,
the `SYNTHETIC_SPECS` / `IMPUTEGAP_SPECS` lists. Every entry is an
`ExperimentSpec`, e.g.:

```python
ExperimentSpec(source="synthetic", missingness_pattern="mcar", rate=0.2)
```

Adding a line here is enough to add a whole new experiment - `build_datasets.py`
and `evaluate_metrics.py` both loop over `ALL_SPECS` and don't need to be
touched.

**Change which missingness patterns are available** -> `src/missingness_patterns.py`,
the `PATTERN_FUNCS` dictionary. New patterns just need a function that takes
the ground truth and a rate and returns a boolean mask (most of these wrap
ImputeGAP's `GenGap` contamination functions).

**Change normalization** -> `src/data/normalization.py`. The `normalization`
field on an `ExperimentSpec` accepts `"none"`, `"z_score"`, `"min_max"`,
`"z_lib"`, or `"m_lib"`.

**Change the synthetic ground truth** (shapes, length, number of series, noise)
-> `src/data/synthetic_ground_truth.py`. `N_TIMESTEPS` and `N_SERIES` are the
defaults; `BASE_SHAPES` is the list of waveform generators each series is
built from.

**Change the six hand-crafted distortions** (constant offset, random spikes,
etc.) -> `src/reconstruction/synthetic_distortions.py`.

**Change which real ImputeGAP algorithms are run** -> `src/reconstruction/imputation_algorithms.py`,
the `ALGORITHMS` list.

**Add, remove, or re-categorize a metric** -> `src/metric_config.py`. This is
the single place that defines which metrics exist, how they're grouped in
reports/heatmaps (`CATEGORIES`), which direction is "better"
(`METRIC_DIRECTION`), and which metrics need special handling (full-series
input, probabilistic output). The actual formula for a metric goes in
`src/metrics.py`, as a function with the same name.

**Change which series is plotted in the imputation plot** -> `src/evaluate_metrics.py`,
the `PLOT_SERIES_INDEX` constant near the top of the file. Each dataset has
several series (channels); only one is shown at a time so the plot stays
readable.

## File-by-file

All paths below are relative to `src/`.

### `experiment_config.py`

The central list of experiments. An `ExperimentSpec` describes one experiment:
where the ground truth comes from (`source`), what normalization to apply,
which missingness pattern and rate to use, and which kind of reconstruction to
evaluate against it (the synthetic distortions, or real imputation
algorithms). It also derives the dataset name and output file path used
everywhere else, so renaming or adding an experiment only requires editing
this file. `SYNTHETIC_SPECS` and `IMPUTEGAP_SPECS` are combined into
`ALL_SPECS`, which both pipeline scripts iterate over.

### `data/`

- `synthetic_ground_truth.py` - generates the synthetic multivariate time
  series. Each series follows one of four base shapes (sine with a trend, sum
  of two sines, smoothed random walk, smoothed square wave blended with a
  sine), cycled round-robin across series, with per-series randomized phase,
  frequency, trend, and noise. Everything is seeded, so the output is
  reproducible.
- `real_world_ground_truth.py` - loads a real ImputeGAP dataset (e.g.
  `eeg-alcohol`) and cuts it down to the requested number of series.
- `normalization.py` - applies one of the normalization methods to the ground
  truth before contamination, using ImputeGAP's own normalizers.

### `missingness_patterns.py`

Wraps ImputeGAP's `GenGap` contamination functions and converts their output
(a NaN-contaminated copy of the data) into a boolean mask, where `True` means
"this position is missing and should be evaluated". `make_mask()` is the
single entry point used by `build_datasets.py`; it dispatches to one of the
pattern functions in `PATTERN_FUNCS` based on the `missingness_pattern` string
in an `ExperimentSpec`. Currently used patterns are `full` (nothing removed,
everything evaluated), `mcar` (individual points removed at random),
`scattered` (one random gap per series), and `blackout` (one gap at the same
position across all series). A few more patterns (`aligned`, `gaussian`,
`distribution`, `disjoint`, `overlap`) are implemented and ready to use, just
not part of any current experiment.

### `reconstruction/`

- `synthetic_distortions.py` - builds six fixed, hand-crafted "reconstructions"
  from the ground truth, each meant to represent a different failure mode an
  imputation algorithm might have: a constant offset, a few large spikes, a
  time shift, oversmoothing, a shuffled order, and a rescaled amplitude. These
  are deterministic (seeded) and don't depend on the missingness mask - they
  are computed directly from the ground truth.
- `imputation_algorithms.py` - runs a fixed list of real ImputeGAP imputation
  algorithms (simple baselines like mean/zero fill, matrix-completion methods
  like CDRec/SoftImpute/SVT, and STMVL) against a copy of the ground truth with
  the missingness mask applied as NaNs. Algorithms that fail or leave NaNs
  behind are skipped and simply don't appear in the output. ImputeGAP's own
  quick metrics (RMSE, MAE, MI, correlation) are printed during this step as a
  sanity check, but the actual metrics used for reporting are computed later
  by `metrics.py` / `generate_reports.py`.

### `dataset_io.py`

Small helper module shared by `build_datasets.py`. The ground truth and
reconstructions are produced as `(n_timesteps, n_series)` numpy arrays, but
the JSON files (and everything downstream) use a `[series][timestep]` list
format. `matrix_to_lists`, `matrix_to_mask`, and `bool_matrix_to_mask` do that
conversion (and round values to 4 decimals to keep file sizes down);
`save_dataset` writes the resulting dictionary to disk, creating folders as
needed.

### `build_datasets.py`

The first pipeline step. For every `ExperimentSpec` in `ALL_SPECS`, it: generates
the ground truth, applies normalization, builds the missingness mask, runs the
reconstruction (synthetic distortions or real algorithms), and writes
`{y_true, mask, <reconstruction name>: ...}` to a JSON file under
`time_series/`. The path follows the pattern
`time_series/synthetic/<pattern>/[<rate>/]data.json` for synthetic data, or
`time_series/imputegap/<source>/<pattern>/[<rate>/]data.json` for real data
(the rate folder is left out for `full`, since there's no rate there). Run
this once, or again whenever the experiment list or any of the generation code
changes.

### `metric_config.py`

Defines the 22 metrics used throughout the project: which ones exist, how
they're grouped into categories for the reports and heatmaps (`CATEGORIES`),
the order they're shown in (`METRIC_LIST`, derived from `CATEGORIES`), and
whether a lower or higher value is better (`METRIC_DIRECTION`). It also marks
two special groups:

- `FULL_SERIES_METRICS` (ACF, DTW, spectral MAE) - these need the entire
  series to be meaningful, so they're always computed on the full series even
  when a mask is set, and they're skipped for probabilistic algorithm output.
- `PROBABILISTIC_METRICS` (CRPS, NLL) - these only make sense for algorithms
  that output a distribution per point rather than a single value, so they're
  skipped for deterministic output.

### `metrics.py`

The actual formula for each of the 22 metrics, one function per metric (e.g.
`mae`, `rmse`, `wd`, `dtw`, ...), each taking `(y_true, y_pred)` and returning
a float (except `ba`, which returns a `(mean_diff, loa)` tuple). Most of these
are thin wrappers around `sklearn`, `scipy`, `statsmodels`, `dtaidistance`,
`properscoring`, or `pingouin`. Each function has a short comment explaining
what it measures, its range, and which direction is better. A detailed,
metric-by-metric writeup (formulas, sanity checks, comparisons against
reference implementations) is in `metric_verification.md`.

### `generate_reports.py`

Loads a dataset's JSON file (`load_data`) and computes every metric for every
reconstruction (`compute_all_scores`). For each (metric, reconstruction) pair,
`metric_applies()` checks whether that metric makes sense for that
reconstruction's output - probabilistic metrics need probabilistic output,
full-series metrics need deterministic output - and the score is recorded as
`None` if not. `applicable_metrics()` then returns the subset of metrics that
were actually computed for at least one reconstruction in a given dataset;
this is what lets the heatmap and ranking report leave out metrics like CRPS/NLL
entirely when nothing in a dataset is probabilistic, instead of showing them as
"n/a" everywhere. `generate_metrics_report()` writes the raw per-metric,
per-reconstruction scores to `reports/<dataset>_metrics.txt` (here, metrics
that don't apply are still shown, just marked "n/a", since this report is
meant to show the full picture).

### `ranking.py`

Turns the raw scores from `compute_all_scores` into rankings.
`build_rank_matrix()` ranks every reconstruction for each applicable metric
(rank 1 = best, according to `METRIC_DIRECTION`; reconstructions with a `None`
score are ranked last). `generate_ranking_report()` writes
`reports/<dataset>_ranking_summary.txt`, containing the per-metric ranking
table grouped by category, a consensus ranking (average rank across all
metrics), a Spearman correlation matrix between metrics (how often two metrics
agree on the ordering of reconstructions), and the most agreeing/disagreeing
metric pairs. Any metric that wasn't applicable to this dataset is listed
separately as "omitted".

### `plot.py`

Two plotting functions. `plot_imputation()` draws the ground truth and every
reconstruction for a single series, shading the positions that were treated as
missing. `plot_ranking()` draws the heatmap of ranks (reconstructions x
metrics, grouped by category with a divider and label per category). Both
functions either save to a file (if `output_path` is given) or call
`plt.show()`.

### `evaluate_metrics.py`

The second pipeline step, and the main place to look when changing what gets
displayed. For every `ExperimentSpec` in `ALL_SPECS` (skipping any whose
JSON file doesn't exist yet), it: writes the metrics report, plots one series'
imputation comparison (which series is controlled by `PLOT_SERIES_INDEX` at
the top of the file), writes the ranking report, and plots the ranking
heatmap. Output goes to `reports/` and `plots/`, named after
`spec.dataset_name` (e.g. `synthetic_mcar_20pct`, `imputegap_eeg-alcohol_mcar_20pct`).

### `metric_verification.md` (repo root)

A separate writeup, one section per metric, covering the formula, what it
measures, how the code implements it, any deviations from the textbook
definition, and how it compares against reference implementations
(scikit-learn, ImputeGAP, sktime, etc.) where applicable. Useful background if
a metric's value looks surprising and it's unclear whether that's expected or
a bug.

### `requirements.txt` (repo root)

Pinned versions for the non-ImputeGAP dependencies (numpy, scipy,
scikit-learn, statsmodels, dtaidistance, properscoring, pingouin,
matplotlib). `imputegap` is installed separately (see Setup above).

## Output layout

```
src/time_series/<...>/data.json        - generated by build_datasets.py
src/reports/<dataset>_metrics.txt       - raw scores, one row per metric
src/reports/<dataset>_ranking_summary.txt - rankings, consensus, metric agreement
src/plots/<dataset>_imputation.png      - one series, ground truth vs. reconstructions
src/plots/<dataset>_ranking.png         - heatmap of ranks per metric
```

`<dataset>` is `spec.dataset_name`, e.g. `synthetic_mcar_20pct` or
`imputegap_eeg-alcohol_mcar_20pct`.
