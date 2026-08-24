# metric_eval

Code and results for the experiments in *Comparative Analysis of Evaluation
Metrics for Time Series Imputation* (Master's thesis, Swiss Joint Master of
Computer Science, Universities of Bern, Fribourg and Neuchâtel).

Three experiments, run in order:

| | Experiment | Question |
|---|---|---|
| 1 | **Injector** | With the amount of damage held constant, do the candidate metrics tell different *kinds* of damage apart? |
| 2 | **Algorithm ranking** | Applied to real algorithms on real data, do the kept metrics agree on which algorithm is best? |
| 3 | **CIS** | Can complementary metrics be combined into one score without losing what made them complementary? |

---

## Install

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optionally install the project itself, which lets you run the modules from
anywhere rather than only from `src/`:

```bash
pip install -e ".[dev]"
```

The `dev` extra adds `pytest`. Plain `pip install -e .` leaves it out.
Installing is never required: the runner scripts and every stage work from
`src/` without it.

`imputegap` supplies both the datasets and the imputation algorithms, so
nothing runs without it. The upper bounds in `requirements.txt` are not
cosmetic — each one is a real incompatibility and the reason is in a comment
next to it.

---

## Running the experiments

All commands are run from `src/`.

```bash
cd src
```

### Quick check first

```bash
python -m injector.selftest
```

Runs on synthetic data. It needs neither ImputeGAP nor any cached output, and
it verifies the two things Experiment 1 rests on: that every distortion can be
solved to its damage target, and that the declared structural invariants hold
before any metric is involved. If this fails, nothing downstream is
trustworthy.

### Experiment 1 — Injector

```bash
./run_injector.sh                       # everything, ~10 minutes
```

or stage by stage:

```bash
python -m injector.calibrate            # solve severities  -> calibration.json
python -m injector.build                # apply them        -> data.json
python -m injector.score                # all 20 metrics    -> scores.json
python -m injector.aggregate            # tables, heatmaps, invariance checks

python -m injector.build_sweep          # the same eight across seven damage levels
python -m injector.score_sweep
python -m injector.aggregate_sweep
```

Read `injector/README.md` for the design and for why each distortion is
defined the way it is.

### Experiment 2 — Algorithm ranking

```bash
./run_algo_ranking.sh                   # everything; the build stage takes hours
```

or stage by stage:

```bash
python -m algo_ranking.build            # SLOW: 6 algorithms x 54 scenarios x seeds
python -m algo_ranking.score            # metrics from the cached reconstructions
python -m algo_ranking.aggregate        # consensus ranks, agreement matrices, heatmaps
python -m algo_ranking.visualize        # reconstruction plots (not part of the ranking)
```

`build` is the only genuinely expensive stage in the project. It runs each
algorithm in its own subprocess so that one crashing cannot take the others
down, and it caches every reconstruction, so `score` and `aggregate` can be
re-run freely afterwards without touching it.

### Experiment 3 — CIS

```bash
./run_cis.sh
```

or:

```bash
python -m cis.cis
```

Reads the scores Experiment 2 produced, so it needs that to have run first.

### Everything

```bash
./run_all.sh
```

### Common flags

Every stage caches its output and skips work already done. To redo it:

```bash
python -m injector.score --force
```

To work on a subset while developing:

```bash
python -m injector.build      --patterns mcar --rates 0.2 0.5
python -m algo_ranking.score  --datasets chlorine --patterns blackout
python -m injector.calibrate  --target 0.4
```

Two figures in Experiment 1 (`metric_overview.png` and `condition_grid.png`)
need all three missingness patterns, so a `--patterns` subset skips them and
says so.

---

## What comes out

```
plots/
├── injector/equal_damage/       metric x distortion heatmaps, one per condition,
│                                plus metric_overview.png and condition_grid.png
├── injector/damage_sweep/       one panel per metric, all eight distortions on
│                                a shared damage axis
├── algo_ranking/<dataset>/heatmap/<pattern>_<bucket>.png
├── algo_ranking/<dataset>/reconstruction/<pattern>_<rate>pct.png
└── cis/                         gate distribution, CIS against the 8-metric consensus

reports/
├── injector/equal_damage/       raw value tables, metric agreement, invariance checks
├── injector/damage_sweep/       flat / monotonic / non-monotonic per metric
├── algo_ranking/<dataset>/<pattern>_<bucket>.txt
└── cis/                         validation, supporting experiments, rejected constructions

time_series/                     cached reconstructions — large, and regenerable
```

`time_series/` is around 350 MB and is fully reproducible from the code, so it
is not worth version-controlling. `plots/` and `reports/` are what the thesis
cites and are worth keeping.

---

## The metric set

`core/metrics.py` holds **22 metric functions**, one formula each. **20 of them
are scored**, and those 20 are what `core/metric_config.py` groups into
categories and assigns a direction:

| Category | Metrics |
|---|---|
| Pointwise Error | MAE, RMSE, MSE, MRE, sMAPE, nRMSE, ND |
| Distributional | WD, JSD, KLD |
| Temporal / Shape | ACF, DTW, sMAE |
| Statistical Agreement | Pearson, MI, R², TOST, BA, CDT |
| Domain-specific | PFC |

The two that are never scored are `crps` and `nll`. Neither is undefined for a
point estimate — CRPS is exactly MAE there, and NLL is a monotone function of
RMSE — so on the point estimates every algorithm here produces they would add a
column each and no information. They are implemented, and their posterior-sample
branches are unverified, because nothing in this project exercises them.

Each experiment then narrows the set further:

| | Uses | Which |
|---|---|---|
| 1 — Injector | 19 | scored on all 20, reported on the four categories above the domain-specific one |
| 2 — Algorithm ranking | 8 | two per category: MAE/RMSE, R²/MI, WD/JSD, DTW/sMAE |
| 3 — CIS | 4 | one per category: MAE, WD, DTW, MI |

Three of the 20 are computed on the **whole series** rather than at the missing
positions only: ACF, DTW and sMAE. Masking them would destroy what they measure,
since the autocorrelation structure, the warping path and the power spectrum all
need the series intact. Every other metric sees the missing positions alone.

Directions are not uniform. Most are lower-is-better; Pearson, MI, R² and PFC
are higher-is-better; TOST is a p-value, so lower; and BA returns a pair and is
ranked on the absolute mean difference. `METRIC_DIRECTION` is what every ranking
and z-score in the project reads, so a metric added without an entry there
raises rather than being silently ranked the wrong way round.

---

## Layout

```
src/
├── run_injector.sh           Experiment 1, end to end
├── run_algo_ranking.sh       Experiment 2, end to end
├── run_cis.sh                Experiment 3, end to end
├── run_all.sh                all three, in dependency order
├── _run_common.sh            sourced by the four above; not run directly
│
├── core/                     shared by all three experiments
│   ├── metrics.py            22 metric functions, one formula each
│   ├── metric_config.py      the 20 that are scored, with categories
│   │                         and directions
│   ├── scoring.py            compute_all_scores — every metric, every reconstruction
│   ├── ranking.py            rank_algorithms
│   ├── missingness_patterns.py
│   ├── dataset_io.py         (T, N) arrays <-> the [series][timestep] JSON cache
│   └── data/                 ground-truth loading and normalisation
│
├── injector/                 Experiment 1
│   ├── config.py             single source of truth for the design
│   ├── distortions.py        the eight distortions
│   ├── calibrate.py          solves each severity to a common damage target
│   ├── build.py  score.py  aggregate.py
│   ├── build_sweep.py  score_sweep.py  aggregate_sweep.py
│   ├── analysis.py           spread, z-scores, metric agreement
│   ├── invariance.py         machine-checked exact predictions
│   ├── plotting.py
│   ├── selftest.py
│   └── README.md             the design, and why it is what it is
│
├── algo_ranking/             Experiment 2
│   ├── config.py             datasets, algorithms, the kept metric set
│   ├── algorithms.py         the six ImputeGAP algorithms
│   ├── build.py  score.py  aggregate.py
│   ├── _run_algorithm.py     one algorithm in one subprocess
│   ├── ranking_report.py  plotting.py  visualize.py
│
├── cis/                      Experiment 3
│   └── cis.py                gate, score, validation, rejected constructions
│
└── time_series/  plots/  reports/       generated output

tests/                        pytest suite over core/metrics.py
pyproject.toml                packaging; dependencies are read from
                              requirements.txt rather than repeated
```

Each experiment is a **build → score → aggregate** pipeline with its own cache,
so any stage can be re-run once its inputs exist. Building is the expensive
part and scoring is cheap, which is why they are separate: changing a metric
costs a scoring pass, not an algorithm pass.

`core/` holds only what more than one experiment uses. Anything used by a
single experiment lives in that experiment's own package, even where two are
similar — the two `plotting.py` files draw different figures and merging them
would help nobody.

---

## Testing

```bash
pytest                        # from the repository root
```

`tests/` checks the algebraic identities the metric implementations are
supposed to satisfy: that ND is a fixed multiple of MAE, that nRMSE is a fixed
multiple of RMSE, that MSE is RMSE squared, that Pearson is invariant to a
positive-slope affine transform, and that WD, JSD and KLD are invariant to a
permutation of the values. Those identities are load-bearing for the thesis's
redundancy and blind-spot arguments, so they are worth asserting rather than
assuming.

`python -m injector.selftest` is the equivalent check for the Experiment 1
design and does not need pytest.

---

## Reproducibility

Every stochastic step is seeded. Experiment 1 draws its distortions from a
generator keyed on the seed, the series index and the distortion name, so
repeating a run reproduces it byte for byte. Experiment 2 runs the two
stochastic algorithms (BRITS and MPIN) three times per scenario and averages;
the other four are deterministic and run once.

Two things are worth knowing before comparing numbers against another tool:

- **The cache rounds to four decimal places** on the way to disk
  (`core/dataset_io.py`). Anything that has to hold exactly — the invariance
  checks in Experiment 1 — is checked at a tolerance that accounts for this,
  and `injector/invariance.py` explains the tolerance chosen for each
  property.
- **Some conventions differ from ImputeGAP's own.** RMSE is averaged per
  series here and pooled across series there; mutual-information bins use a
  range shared between the two series; Pearson on a constant series returns
  zero rather than NaN. None is a bug, and each means a number here is not
  directly comparable with the same metric computed by ImputeGAP.

---

## Configuration

Each experiment's `config.py` is the single source of truth for its design and
carries the reasoning for each constant in comments. The values most likely to
be worth changing:

| Where | Constant | Meaning |
|---|---|---|
| `injector/config.py` | `TARGET_DAMAGE` | the damage every distortion is solved to, in σ |
| | `DAMAGE_LEVELS` | the seven levels of the sweep |
| | `PATTERNS`, `RATES` | missingness geometries and rates |
| `algo_ranking/config.py` | `ALGO_CATEGORIES` | the kept metrics, grouped by category |
| | `DATASETS`, `RATES`, `N_SEEDS` | scenario coverage |
| `cis/cis.py` | `CIS_METRICS` | the four metrics the composite uses |
| | `FLAT_THRESHOLD`, `UNSTABLE_THRESHOLD` | the stability gate |
| | `MI_SCALE` | the empirical scale for the MI component |

Changing the metric set in Experiment 2 costs an `aggregate` run and nothing
else. `aggregate` checks the cached scores against the current selection and,
where a metric is missing, computes only that metric from the cached
reconstructions before continuing (`score.ensure_scored`). The expensive part
of a scoring pass is DTW on 2000-timestep series, and there is no reason to
pay it again to add a spectral distance.
