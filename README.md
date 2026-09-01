# Comparative Analysis of Evaluation Metrics for Time Series Imputation

Code and results for the experiments in a Master's thesis of the Swiss Joint
Master of Computer Science, Universities of Bern, Fribourg and Neuchâtel.

Three experiments, run in order:

| | Experiment | Question |
|---|---|---|
| 1 | **Injector** | With the amount of damage held constant, do the candidate metrics tell different *kinds* of damage apart? |
| 2 | **Algorithm ranking** | Applied to real algorithms on real data, do the kept metrics agree on which algorithm is best? |
| 3 | **CIS** | Can metrics that disagree be combined into one number that no kind of damage escapes? |

---

## Install

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optionally install the project itself, which lets you run the modules from
anywhere without the runner scripts:

```bash
pip install -e ".[dev]"
```

The `dev` extra adds `pytest`. Plain `pip install -e .` leaves it out.
Installing is never required: the runner scripts put the repository root on
`PYTHONPATH` themselves, and `pytest` picks it up from `pyproject.toml`.

`imputegap` supplies both the datasets and the imputation algorithms, so
nothing runs without it. The upper bounds in `requirements.txt` are not
cosmetic — each one is a real incompatibility and the reason is in a comment
next to it.

---

## Running the experiments

The runner scripts can be invoked from anywhere and set the import path
themselves. To run individual stages with `python -m`, either install the
package (`pip install -e .`) or run from the repository root.

### Quick check first

```bash
python -m metric_eval.experiments.injector.selftest
```

Runs on synthetic data. It needs neither ImputeGAP nor any cached output, and
it verifies the two things Experiment 1 rests on: that every distortion can be
solved to its damage target, and that the declared structural invariants hold
before any metric is involved. If this fails, nothing downstream is
trustworthy.

### Experiment 1 — Injector

```bash
./scripts/run_injector.sh                       # everything, ~10 minutes
```

or stage by stage:

```bash
python -m metric_eval.experiments.injector.reactivity.calibrate   # solve severities  -> calibration.json
python -m metric_eval.experiments.injector.reactivity.build       # apply them        -> data.json
python -m metric_eval.experiments.injector.reactivity.score       # all 20 metrics    -> scores.json
python -m metric_eval.experiments.injector.reactivity.aggregate   # tables, heatmaps, invariance checks

python -m metric_eval.experiments.injector.response.build         # the same eight across seven damage levels
python -m metric_eval.experiments.injector.response.score
python -m metric_eval.experiments.injector.response.aggregate
python -m metric_eval.experiments.injector.summary                # response grid and redundancy (both passes)
python -m metric_eval.experiments.injector.panels                 # the eight distortion panels
```

The calibrate, build and score stages also take `--damage-metric rmse`, which
solves the same grid against an RMSE target into `*_rmse.json` caches beside
the MAE ones. MAE and ND are pinned by the MAE target, so `injector.summary`
reads their columns off that second pass.
```bash
python -m metric_eval.experiments.injector.reactivity.calibrate --damage-metric rmse
python -m metric_eval.experiments.injector.reactivity.build     --damage-metric rmse
python -m metric_eval.experiments.injector.reactivity.score     --damage-metric rmse
```

The design behind this experiment, and the reason each distortion is defined
the way it is, is in [The Injector design](#the-injector-design) below.

### Experiment 2 — Algorithm ranking

```bash
./scripts/run_algorank.sh                   # everything; the build stage takes hours
```

or stage by stage:

```bash
python -m metric_eval.experiments.algorank.build            # SLOW: 6 algorithms x 54 scenarios x seeds
python -m metric_eval.experiments.algorank.score            # metrics from the cached reconstructions
python -m metric_eval.experiments.algorank.aggregate        # consensus ranks, agreement matrices, heatmaps
python -m metric_eval.experiments.algorank.visualize        # reconstruction plots (not part of the ranking)
python -m metric_eval.experiments.algorank.summarize        # chapter-level tables and figures (needs cis.build)
```

`build` is the only genuinely expensive stage in the project. It runs each
algorithm in its own subprocess so that one crashing cannot take the others
down, and it caches every reconstruction, so `score` and `aggregate` can be
re-run freely afterwards without touching it.

### Experiment 3 — CIS

```bash
./scripts/run_cis.sh
```

or, in two stages:

```bash
python -m metric_eval.experiments.cis.build     # about a minute, cached under outputs/time_series/cis/
python -m metric_eval.experiments.cis           # report and figures, seconds
```

Reads what Experiment 2 cached for the ranking half and what Experiment 1 cached
for the known-damage half, so both need to have run first. `--force` rebuilds
the cache.

### Everything

```bash
./scripts/run_all.sh
```

### Common flags

Every stage caches its output and skips work already done. To redo it:

```bash
python -m metric_eval.experiments.injector.score --force
```

To work on a subset while developing:

```bash
python -m metric_eval.experiments.injector.build      --patterns mcar --rates 0.2 0.5
python -m metric_eval.experiments.algorank.score  --datasets chlorine --patterns blackout
python -m metric_eval.experiments.injector.calibrate  --target 0.4
```

Two figures in Experiment 1 (`metric_overview.png` and `condition_grid.png`)
need all three missingness patterns, so a `--patterns` subset skips them and
says so.

---

## What comes out

```
outputs/plots/
├── injector/reactivity/         metric x distortion heatmaps, one per condition,
│                                plus metric_overview.png and condition_grid.png
├── injector/response/           one panel per metric, all eight distortions on
│                                a shared damage axis
├── algorank/<dataset>/heatmap/<pattern>_<bucket>.png
├── algorank/<dataset>/heatmap/by_rate/<pattern>_<rate>pct.png
├── algorank/<dataset>/reconstruction/<pattern>_<rate>pct.png
└── cis/                         gate, coverage, variation axis, known damage

outputs/reports/
├── injector/reactivity/         raw value tables, metric agreement, invariance checks
├── injector/response/           flat / monotonic / non-monotonic per metric
├── algorank/<dataset>/<pattern>_<bucket>.txt
├── algorank/<dataset>/by_rate/<pattern>_<rate>pct.txt
└── cis/                         gate, coverage, variation axis, known damage

outputs/time_series/                     cached reconstructions — large, and regenerable
```

`outputs/time_series/` is around 400 MB and is fully reproducible from the code, so it
is not worth version-controlling. `outputs/plots/` and `outputs/reports/` are what the thesis
cites and are worth keeping.

---

## The metric set

`core/metrics.py` holds **22 metric functions**, one formula each. **20 of them
are scored**, and those 20 are what `core/metric_config.py` groups into
categories and assigns a direction:

| Category | Metrics |
|---|---|
| Pointwise Distance | MAE, RMSE, MSE, MRE, sMAPE, nRMSE, ND |
| Distributional Divergence | WD, JSD, KLD |
| Temporal Structure | ACF, DTW, sMAE |
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
| 3 — CIS | 3 | chosen so that no distortion goes undetected: MAE, WD, MI |

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

## The Injector design

Experiment 1 applies eight distortions to the same series, each damaging the
reconstruction in a different way, and scores every metric against all eight.
The comparison only means something if the eight are equally damaging: a metric
that reacts more strongly to one distortion than another has told us nothing if
that distortion was simply larger. Each distortion is therefore applied at a
severity solved numerically so that all eight land on the same mean absolute
error at the masked positions, in units of that series' own sigma. With size
held constant, any variation left in a metric is variation in kind.

### What the design rests on

**MAE is pinned and RMSE is not.** Because the calibration solves against mean
absolute error, MAE reads almost identically for all eight distortions. That is
the intended outcome rather than a weakness: MAE's row is the control that shows
the calibration worked. RMSE is left free, since it depends on how the error is
distributed across positions rather than only on its mean, and the two therefore
separate. In a representative run RMSE reads 0.40 under `bias` and 1.76 under
`spikes` at identical MAE, which is a clean result about error weighting.

**Each distortion declares what it leaves exactly intact.** A distortion that
preserves the multiset of values cannot move any statistic computed from the
value distribution alone, so WD, JSD and KLD must read exactly zero. One that
preserves the mean cannot move Bland-Altman or Cohen's d. A positive-slope
affine transform cannot move Pearson from 1.0. These are predictions implied by
the structure of each distortion, not observations, and `injector/invariance.py`
turns each one into an assertion that the aggregate report prints as a pass/fail
table. A failure means either the distortion is not doing what it claims or the
metric implementation is wrong.

**The sweep runs on damage rather than on each distortion's own parameter.**
Sweeping a lag over timesteps and a smoothing window over widths would give
eight incomparable x-axes needing eight separate figures. Sweeping the damage
itself puts all eight on one axis, so a single panel answers which kind of
damage a given metric is blind to. A metric that stays flat across a whole sweep
is blind to that kind of damage, rather than merely reacting weakly to a smaller
one.

### The eight distortions

| distortion | what it disturbs | severity | preserves exactly |
|---|---|---|---|
| `noise` | pointwise accuracy, at random | noise sd / σ | — |
| `bias` | pointwise accuracy, systematically | offset / σ | affine |
| `reorder` | order in time | fraction of gap positions rotated | multiset, mean |
| `discretise` | shape of the value distribution | grid step / σ | rank |
| `lag` | alignment in time | lag in timesteps | — |
| `smooth` | short-term detail and variance | moving-average window | — |
| `spikes` | the tails, leaving most values exact | spike magnitude / σ | — |
| `rescale` | variance, leaving shape intact | scale factor − 1 | affine, mean |

Two of the definitions are chosen for the solver rather than for convention.
`discretise` rounds onto a uniform grid instead of clustering, because a cluster
count is an integer that bottoms out at 2 while a grid step is continuous, and
because rounding depends on no fitted model and so is exactly reproducible.
`reorder` rotates a fraction of the gap positions instead of shuffling within a
window. Both approaches preserve the value multiset, which is what makes the
WD/JSD/KLD invariance exact, but only a fraction maps directly onto damage: a
random permutation leaves an unpredictable number of values in place, which
turns damage into a noisy step function that bisection cannot solve. The
rotation is what makes the knob usable.

`lag` and `smooth` are continuous, interpolating between neighbouring integer
values, because damage jumps between consecutive integers by far more than the
calibration tolerance. At whole numbers each is exactly the ordinary definition.

### Things worth knowing

**Ceilings.** Smoothing cannot exceed E|y − μ| ≈ 0.8 σ however wide the window,
and reordering cannot exceed E|yᵢ − yⱼ| ≈ 1.13 σ even at a full rotation.
`TARGET_DAMAGE` at 0.5 σ sits comfortably under both, and every solve reports
whether it reached its target rather than silently clipping. Clipping would
quietly turn the target into "as damaged as this distortion can be" for some of
the eight and not for others, which is precisely the confound the equalisation
exists to remove.

The top of `DAMAGE_LEVELS` is a different matter. On airq under a 40 % blackout,
`discretise` tops out at 0.66 σ and `smooth` at 0.66 σ, so both fall about
0.04 σ short of the 0.7 level and the sweep's shared axis does not hold at its
last point. Every mono / non-monotonic / flat verdict in the report is
unchanged when that level is dropped, so the conclusions stand, but a figure
that plots the level as equal damage overstates what was achieved. The
`achieved` field cached beside each level records what each distortion actually
reached.

**Smoothing is not monotone in its window.** On a series with a slow drift a very
wide moving average can sit closer to the truth than a middling one, so damage
rises, falls and rises again. Both `lag` and `smooth` therefore use a
scan-then-bisect solver rather than plain bisection, which would land on
whichever root it happened to bracket.

**The tolerance is 0.01 σ, and `reorder` is why.** The number of rotated
positions is an integer, so reorder's damage moves in steps of roughly
|yᵢ − yⱼ| / n, and for an unlucky pair a single step is about 0.01 σ. Everything
else lands one to two orders of magnitude inside that. Per-series achieved
damage is recorded either way, so nothing is hidden by the choice.

**Run `python -m metric_eval.experiments.injector.selftest` before trusting a run.** It checks the two
things the design rests on: that every distortion can actually be solved to the
target on this data, and that the declared structural invariants hold at the
array level, before any metric is involved. If a target turns out to be
unreachable, that is the signal to lower `config.TARGET_DAMAGE` rather than to
work around it, because the experiment is not equalised if one distortion cannot
reach the target.

---

## Layout

```
scripts/
├── run_injector.sh           Experiment 1, end to end
├── run_algorank.sh           Experiment 2, end to end
├── run_cis.sh                Experiment 3, end to end
├── run_all.sh                all three, in dependency order
└── _run_common.sh            sourced by the four above; not run directly

metric_eval/                  the one installable package
├── paths.py              where the outputs go (everything under outputs/)
│
├── core/                 shared by all three experiments
│   ├── metrics.py        22 metric functions, one formula each
│   ├── metric_config.py  the 20 that are scored, with categories
│   │                     and directions
│   ├── scoring.py        compute_all_scores — every metric, every reconstruction
│   ├── ranking.py        rank_algorithms
│   ├── buckets.py        the rate-bucket mean both experiments use
│   ├── missingness_patterns.py
│   ├── dataset_io.py     (T, N) arrays <-> the [series][timestep] JSON cache
│   └── data/             ground-truth loading and normalisation
│
├── experiments/
│   ├── injector/         Experiment 1
│   │   ├── config.py     single source of truth for the design
│   │   ├── distortions.py  the eight distortions
│   │   ├── selftest.py
│   │   ├── reactivity/   one damage level, eight kinds, 24 scenarios
│   │   │   ├── calibrate.py  solves each severity to a common damage target
│   │   │   ├── build.py  score.py  aggregate.py
│   │   │   ├── analysis.py   spread, z-scores, metric agreement
│   │   │   ├── invariance.py machine-checked exact predictions
│   │   │   └── plotting.py
│   │   ├── response/     seven damage levels, one scenario
│   │   │   ├── build.py  score.py  aggregate.py
│   │   │   └── plotting.py
│   │   ├── summary.py    response grid and redundancy correlations
│   │   └── panels.py     the eight distortion panels of the thesis
│   │
│   ├── algorank/         Experiment 2
│   │   ├── config.py     datasets, algorithms, the kept metric set
│   │   ├── algorithms.py the six ImputeGAP algorithms
│   │   ├── build.py  score.py  aggregate.py
│   │   ├── _run_algorithm.py  one algorithm in one subprocess
│   │   ├── cache.py      reading a scenario back from disk
│   │   ├── analysis.py   ranks, consensus, agreement
│   │   ├── report.py     the text ranking summary
│   │   ├── plotting.py  visualize.py
│   │   ├── experiments.py  the chapter-level statistics
│   │   ├── summary_report.py  summary_plots.py
│   │   └── summarize.py  python -m metric_eval.experiments.algorank.summarize
│   │
│   └── cis/              Experiment 3
│       ├── config.py     components and gate thresholds
│       ├── build.py      reference reconstruction and gate ratios
│       ├── injector_data.py  reads Experiment 1's two caches
│       ├── gate.py       the stability gate
│       ├── score.py      the components and the composite
│       ├── experiments.py  the analyses behind every table
│       ├── report.py  plotting.py
│       └── __main__.py   python -m metric_eval.experiments.cis
│
└── background/           the introduction's figures, from the caches
    └── figures.py        python -m metric_eval.background.figures

outputs/                      everything the code writes; nothing in metric_eval/ is
├── time_series/              cached reconstructions (gitignored, regenerable)
├── plots/                    the figures the thesis includes
└── reports/                  the numbers the thesis cites

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

`python -m metric_eval.experiments.injector.selftest` is the equivalent check for the Experiment 1
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
| `experiments/algorank/config.py` | `ALGO_CATEGORIES` | the kept metrics, grouped by category |
| | `DATASETS`, `RATES`, `N_SEEDS` | scenario coverage |
| `cis/config.py` | `CIS_METRICS` | the metrics the composite uses |
| | `FLAT_THRESHOLD`, `UNSTABLE_THRESHOLD` | the stability gate |
| | `ADOPTED_POWER` | the exponent of the power mean |

Changing the metric set in Experiment 2 costs an `aggregate` run and nothing
else. `aggregate` checks the cached scores against the current selection and,
where a metric is missing, computes only that metric from the cached
reconstructions before continuing (`score.ensure_scored`). The expensive part
of a scoring pass is DTW on 2000-timestep series, and there is no reason to
pay it again to add a spectral distance.

---

## Use of AI

AI assistants by Anthropic (Claude) were used in the process of writing the
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
