# Comparative Analysis of Evaluation Metrics for Time Series Imputation

Code and results for the experiments in a Master's thesis of the Swiss Joint
Master of Computer Science, Universities of Bern, Fribourg and Neuchâtel. The
repository holds the implementations of the 21 evaluation metrics, the three
experiments of the thesis, and every report and figure the thesis cites.

Three experiments, run in order:

| | Experiment                              | Question |
|---|-----------------------------------------|---|
| 1 | **Impact of Synthetic Distortions on Metric Behaviour** | With the amount of damage held constant, do the candidate metrics tell different *kinds* of damage apart? |
| 2 | **Ranking Imputation Algorithms**       | Applied to real algorithms on real data, do the kept metrics agree on which algorithm is best? |
| 3 | **CIS - Combined Imputation Score**     | Can metrics that disagree be combined into one number that no kind of distortion escapes? |

Run from scratch, Experiment 1 takes about ten minutes, Experiment 2 several
hours (its 1,440 algorithm runs dominate the whole pipeline), and Experiment 3
about a minute. Every stage caches its output, so a rerun on existing caches
takes minutes.

---

## Install

Required:

- Python 3.10 or newer (verified on 3.13.2)
- a C compiler: on macOS `xcode-select --install`, on Linux `build-essential`
- on macOS, Homebrew and Armadillo — ImputeGAP's C++ algorithms (CDRec, ROSL,
  DynaMMo, ST-MVL) are linked against
  `/opt/homebrew/opt/armadillo/lib/libarmadillo.14.dylib`:

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # then run the "Next steps" commands the installer prints, and:
  brew install armadillo
  ```

  If Homebrew installs a newer major version of Armadillo, symlink its dylib
  under the name above.

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows: `py -m venv .venv`, then `.venv\Scripts\activate`.

---

## Scripts

`./scripts/run_all.sh` runs the whole pipeline end to end, in the order the
caches depend on each other. `run_injector.sh`, `run_algorank.sh` and
`run_cis.sh` run Experiments 1, 2 and 3 on their own; each experiment's
section below starts with its command.

Every stage skips work already done; `--force` redoes it. The scripts can be
invoked from anywhere and set the import path themselves. To run individual
stages with `python -m`, either install the package (`pip install -e .`) or
run from the repository root.

---

## Experiment 1 — Impact of Synthetic Distortions on Metric Behaviour

```bash
./scripts/run_injector.sh
```

Eight distortions are applied to the same reconstruction task, each damaging
the series in a different way, and each is solved numerically to the same mean
absolute error, so the amount of damage is equal by construction. Every metric
is then scored against every distortion: a value that varies across the eight
is a reaction to the kind of distortion, and an exact zero is a blind spot. The
`reactivity` stages do this at one damage level; the `response` stages sweep
seven levels, so a metric that stays flat over a whole sweep is blind to that
distortion at any size.

Stage by stage:

```bash
python -m metric_eval.experiments.injector.selftest
python -m metric_eval.experiments.injector.reactivity.calibrate
python -m metric_eval.experiments.injector.reactivity.build
python -m metric_eval.experiments.injector.reactivity.score
python -m metric_eval.experiments.injector.reactivity.aggregate

python -m metric_eval.experiments.injector.response.build
python -m metric_eval.experiments.injector.response.score
python -m metric_eval.experiments.injector.response.aggregate
python -m metric_eval.experiments.injector.summary
python -m metric_eval.experiments.injector.panels
```

`selftest` runs on synthetic data, needs no cache, and verifies that every
distortion solves to its damage target and keeps its declared invariants. The
`calibrate`, `build` and `score` stages also take `--damage-metric rmse`,
which solves the same grid against an RMSE target; `injector.summary` reads
the MAE and ND columns off that second pass:

```bash
python -m metric_eval.experiments.injector.reactivity.calibrate --damage-metric rmse
python -m metric_eval.experiments.injector.reactivity.build     --damage-metric rmse
python -m metric_eval.experiments.injector.reactivity.score     --damage-metric rmse
```

---

## Experiment 2 — Ranking Imputation Algorithms

```bash
./scripts/run_algorank.sh
```

Six imputation algorithms reconstruct six real datasets under three
missingness patterns and eight missing rates, 144 scenarios in total. Every
reconstruction is scored with the eight kept metrics, each metric ranks the
six algorithms, and the rankings are compared against each other. `build` is
the expensive stage: it runs each algorithm in its own subprocess, so one
crashing cannot take the others down, and it caches every reconstruction, so
the later stages rerun freely without touching it.

Stage by stage:

```bash
python -m metric_eval.experiments.algorank.build
python -m metric_eval.experiments.algorank.score
python -m metric_eval.experiments.algorank.aggregate
python -m metric_eval.experiments.algorank.visualize
python -m metric_eval.experiments.algorank.summarize
```

`summarize` reads Experiment 3's cache, so on a fresh machine it comes after
`run_cis.sh`; `run_all.sh` orders the stages accordingly.

---

## Experiment 3 — CIS, Combined Imputation Score

```bash
./scripts/run_cis.sh
```

Combines three metrics that disagree by design — MAE, WD and MI — into one
composite score, behind a stability gate that discards constant and diverging
reconstructions before ranking. Reads what Experiment 2 cached for the ranking
half and what Experiment 1 cached for the distortion half, so both need to
have run first.

In two stages:

```bash
python -m metric_eval.experiments.cis.build
python -m metric_eval.experiments.cis
```

---

## Outputs

```
outputs/plots/
├── injector/reactivity/                                              metric x distortion heatmaps, one per condition, plus metric_overview.png and condition_grid.png
├── injector/response/                                                one panel per metric, all eight distortions on a shared damage axis
├── algorank/<dataset>/heatmap/<pattern>_<bucket>.png
├── algorank/<dataset>/heatmap/by_rate/<pattern>_<rate>pct.png
├── algorank/<dataset>/reconstruction/<pattern>_<rate>pct.png
└── cis/                                                              gate, coverage, variation axis, known damage

outputs/reports/
├── injector/reactivity/                                              raw value tables, metric agreement, invariance checks
├── injector/response/                                                flat / monotonic / non-monotonic per metric
├── algorank/<dataset>/<pattern>_<bucket>.txt
├── algorank/<dataset>/by_rate/<pattern>_<rate>pct.txt
└── cis/                                                              gate, coverage, variation axis, known damage

outputs/time_series/                                                  cached reconstructions — large, and regenerable
```

`outputs/time_series/` is around 400 MB and fully reproducible from the code,
so it is gitignored. `outputs/plots/` and `outputs/reports/` are what the
thesis cites and are kept.

---

## The metrics

`metric_eval/core/metrics.py` holds **21 metric functions**, one formula each.
**19 of them are scored**, and those 19 are what
`metric_eval/core/metric_config.py` groups into categories and assigns a
direction:

| Category | Metrics |
|---|---|
| Pointwise Distance | MAE, RMSE, MSE, MRE, sMAPE, nRMSE, ND |
| Distributional Divergence | WD, JSD, KLD |
| Temporal Structure | ACF, DTW, sMAE |
| Statistical Agreement | Pearson, MI, R², TOST, BA, CDT |

The two that are never scored are `crps` and `nll`: on a point estimate CRPS
equals MAE and NLL is a monotone function of RMSE, so on the point estimates
every algorithm here produces they would add a column each and no information.

Each experiment then narrows the set further:

| | Uses | Which |
|---|---|---|
| 1 — Synthetic distortions | 19 | all of them |
| 2 — Algorithm ranking | 8 | two per category: MAE/RMSE, R²/MI, WD/JSD, DTW/sMAE |
| 3 — CIS | 3 | chosen so that no distortion goes undetected: MAE, WD, MI |

Three of the 19 — ACF, DTW and sMAE — are computed on the whole series,
because masking them would destroy what they measure: the autocorrelation
structure, the warping path and the power spectrum all need the series intact.
Every other metric sees the missing positions alone.

Most metrics are lower-is-better; Pearson, MI and R² are
higher-is-better; TOST is a p-value, so lower; and BA is ranked on the
absolute mean difference. `METRIC_DIRECTION` is what every ranking and z-score
in the project reads, so a metric added without an entry there raises an
error.

The test suite pins the implementations to algebraic identities they must
satisfy: ND is a fixed multiple of MAE, nRMSE a fixed multiple of RMSE, MSE is
RMSE squared, Pearson is invariant to a positive-slope affine transform, and
WD, JSD and KLD are invariant to a permutation of the values. Those identities
are load-bearing for the thesis's redundancy and blind-spot arguments, so the
suite asserts them. From the repository root:

```bash
pip install pytest
pytest
```

---

## Reproducibility and configuration

Every stochastic step is seeded. Experiment 1 draws its distortions from a
generator keyed on the seed, the series index and the distortion name, so
repeating a run reproduces it byte for byte. Experiment 2 runs the two
stochastic algorithms (BRITS and MPIN) three times per scenario and averages;
the other four are deterministic and run once.

Two things matter when comparing numbers against another tool:

- **The cache rounds to four decimal places** on the way to disk
  (`metric_eval/core/dataset_io.py`). Anything that has to hold exactly — the
  invariance checks of Experiment 1 — is checked at a tolerance that accounts
  for this.
- **Some conventions differ from ImputeGAP's own.** RMSE is averaged per
  series here and pooled across series there; mutual-information bins use a
  range shared between the two series; Pearson on a constant series returns
  zero rather than NaN. Each is a deliberate choice, and each means a number
  here cannot be set directly beside the same metric computed by ImputeGAP.

Each experiment's `config.py` is the single source of truth for its design and
carries the reasoning for each constant in comments. The values most likely to
be worth changing:

| Where | Constant | Meaning |
|---|---|---|
| `injector/config.py` | `TARGET_DAMAGE` | the damage every distortion is solved to, in σ |
| | `DAMAGE_LEVELS` | the seven levels of the sweep |
| | `PATTERNS`, `RATES` | missingness geometries and rates |
| `algorank/config.py` | `ALGO_CATEGORIES` | the kept metrics, grouped by category |
| | `DATASETS`, `RATES`, `N_SEEDS` | scenario coverage |
| `cis/config.py` | `CIS_METRICS` | the metrics the composite uses |
| | `FLAT_THRESHOLD`, `UNSTABLE_THRESHOLD` | the stability gate |
| | `ADOPTED_POWER` | the exponent of the power mean |

Changing the metric set in Experiment 2 costs an `aggregate` run and nothing
else: `aggregate` checks the cached scores against the current selection and,
where a metric is missing, computes only that metric from the cached
reconstructions (`score.ensure_scored`).

---

## Use of AI

AI assistants by Anthropic (Claude Opus 5) were used in the process of writing the
thesis this repository accompanies, and in building this repository itself:
mainly for the final review of the thesis text, for generating a part of the
plots and illustrations from the author's own material, for the broad
analysis of all scenarios of the two experiments, and to generate parts of
the code.

Every generated part was reviewed and verified by the author. The pipelines
run from the author's own experimental design, the figures read the cached
results of the author's experiments and regenerate without any AI model, and
every number the thesis cites is written into the reports under
`outputs/reports/`. All ideas and final decisions are the author's own, as
is the creative and cognitive process behind the research, the analyses and
the code.
