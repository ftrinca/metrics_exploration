# Injector

A rebuild of the Injector experiment in which all eight distortions are
calibrated to cause the **same amount of damage**, so that the comparison
between them is about the *kind* of damage rather than its size.

---

## Why the experiment was rebuilt

The first version applied each distortion at a hand-picked constant — `ALPHA_JITTER = 0.3`,
`ALPHA_OFFSET = 0.75`, `SHIFT = 20`, and so on. Those constants were never
equalised against each other. Measured as mean absolute error at the masked
positions, the eight ranged from **0.149 σ (random spikes) to 1.040 σ
(shuffle)** — a factor of seven:

| distortion | old damage (MAE / σ) |
|---|---|
| shuffle | 1.040 |
| constant offset | 0.746 |
| time shift | 0.741 |
| amplitude scaled | 0.482 |
| oversmoothed | 0.340 |
| jitter | 0.240 |
| quantize | 0.173 |
| random spikes | 0.149 |

Reactivity z-scores a metric's eight values against that metric's own mean and
standard deviation, so for any single metric the z-score is just its own eight
numbers rescaled. Sorting the distortions by MAE reactivity therefore
reproduces, **exactly**, the order you get by sorting them by how much raw MAE
each one causes. Verified for MAE and RMSE:

```
MAE, by raw damage   shuffle > offset > time_shift > amplitude > oversmooth > jitter > quantize > spikes
MAE, by reactivity   shuffle > offset > time_shift > amplitude > oversmooth > jitter > quantize > spikes
```

That heatmap was showing distortion **size**. That is why the "negative
controls" all passed — an invariance cannot be faked by magnitude — and why
the "positive controls" mostly did not.

---

## What changed

**Damage is held constant.** Every distortion is applied at a severity solved
numerically so that all eight land on the same `mean(|ŷ − y|) / σ` at the
masked positions, per series. With size controlled, any remaining variation in
a metric is about kind.

Two consequences, both intended:

- **MAE is pinned** and reads almost identically for all eight. That is the
  point. It also means MAE's row in the report is the control that shows the
  calibration worked.
- **RMSE is not pinned**, because it depends on how the error is distributed
  rather than only on its mean. The pointwise pair still separates, and that
  separation is now a clean result about error weighting rather than an
  artefact of severity. In the smoke run RMSE reads 0.40 under `bias` and
  1.76 under `spikes` at identical MAE.

**The positive/negative control framing is dropped.** Each distortion instead
declares the property it disturbs and, where one exists, the property it
leaves *exactly* intact. Those invariants are machine-checked, which turns
the blind-spot findings from an observation into a test.

**The sweep sweeps damage, not each distortion's own parameter.** The old sweep ran lag
over 5–80 timesteps, smoothing over 5–100, cluster count from 16 down to 2 —
eight incomparable x-axes needing eight separate figures. The sweep now runs on the damage
itself, so all eight distortions sit on one axis and a single panel answers
"which kind of damage is this metric blind to".

**Renames**, so each name says what it does:

| old name | new name | severity parameter |
|---|---|---|
| `jitter` | `noise` | noise sd / σ |
| `constant_offset` | `bias` | offset / σ |
| `shuffle` | `reorder` | fraction of gap positions rotated |
| `quantize` | `discretise` | grid step / σ |
| `time_shift` | `lag` | lag in timesteps (fractional) |
| `oversmoothed` | `smooth` | moving-average window (fractional) |
| `random_spikes` | `spikes` | spike magnitude / σ |
| `amplitude_scaled` | `rescale` | scale factor − 1 |

**Two definitions changed**, both because the old knob could not be
calibrated:

- `discretise` rounds to a uniform grid instead of k-means clustering. A
  cluster count is an integer that bottoms out at 2; a grid step is
  continuous. It also removes the dependence on a fitted model and its seed.
- `reorder` rotates a *fraction* of the gap positions instead of shuffling
  within a window. Both preserve the value multiset exactly — which is what
  makes the WD/JSD/KLD invariance exact — but a fraction maps directly onto
  damage. The rotation (rather than a random permutation) matters: a random
  permutation leaves an unpredictable number of values in place, which made
  damage a noisy step function the solver could not converge on.

`lag` and `smooth` were made continuous by interpolating between neighbouring
integer values. At whole numbers each is exactly the ordinary definition.

---

## Running it

```bash
# 0. check the machinery on synthetic data — needs no ImputeGAP, no cache
python injector/selftest.py

# 1. equal-damage experiment
python injector/calibrate.py      # solve severities, write calibration.json
python injector/build.py          # apply them, cache data.json
python injector/score.py          # all 20 metrics, 19 reported
python injector/aggregate.py      # heatmaps, reports, invariance table

# 2. damage sweep
python injector/build_sweep.py
python injector/score_sweep.py
python injector/aggregate_sweep.py
```

Every stage caches and skips work already done; pass `--force` to redo it.
`--patterns` and `--rates` take subsets for quick checks.

**Run `selftest.py` first.** It checks the two things the design rests on: that every
distortion can actually be solved to the target on this data, and that the
declared structural invariants hold at the array level before any metric is
involved. If a target turns out to be unreachable, that is the signal to lower
`config.TARGET_DAMAGE` rather than to work around it.

---

## Outputs

```
reports/injector/equal_damage/<pattern>_<bucket>.txt
    raw metric x distortion table with spread columns   <- the primary result
    metric agreement matrix over the eight distortions
    exact invariance check table

reports/injector/damage_sweep/damage_sweep.txt
    per metric per distortion: flat / monotonic / non-monotonic

plots/injector/equal_damage/<pattern>_<bucket>_heatmap.png
plots/injector/damage_sweep/<category>.png
```

---

## Things worth knowing

**The cache rounds to four decimals.** `core.dataset_io.matrix_to_lists`
rounds on the way to disk, so an "exact" invariant can only be checked to the
precision rounding leaves. Multiset invariance survives it exactly (rounding
is applied to every value, so a permuted multiset of rounded values is the
same multiset) and WD/JSD/KLD stay at machine zero. Mean invariance does not:
different values round in different directions and the mean moves by around
1e-6. The tolerances in `invariance.py` are set per property for this reason
and the file explains each one.

**The tolerance is 0.01 σ, and reorder is why.** The number of rotated
positions is an integer, so reorder's damage moves in steps of roughly
|yᵢ − yⱼ| / n; for an unlucky pair a single step is about 0.01 σ. Everything
else lands one to two orders of magnitude inside that. Per-series achieved
damage is recorded either way, so nothing is hidden by the choice.

**Smoothing is not monotone in its window.** On a series with a slow drift a
very wide moving average can sit closer to the truth than a middling one, so
damage rises, falls and rises again. This is why both `lag` and `smooth` use a
scan-then-bisect solver rather than plain bisection.

**Ceilings.** Smoothing cannot exceed E|y − μ| ≈ 0.8 σ however wide the
window; reordering cannot exceed E|yᵢ − yⱼ| ≈ 1.13 σ even at a full rotation.
`TARGET_DAMAGE` and `DAMAGE_LEVELS` are kept well below the lowest ceiling,
and every solve reports whether it reached its target rather than silently
clipping.
