# Metric Verification Report

This document reviews all 22 metrics implemented in `metrics.py`. For each metric it covers the canonical mathematical formula, what the metric measures, how the code implements it and any deviations from the standard, bugs or naming issues, and reference implementations for cross-checking.

---

## Table of Contents
1. [MAE](#1-mae)
2. [RMSE](#2-rmse)
3. [MSE](#3-mse)
4. [MRE](#4-mre)
5. [sMAPE](#5-smape)
6. [WD (Wasserstein Distance)](#6-wd)
7. [Pearson](#7-pearson)
8. [JSD](#8-jsd)
9. [MI](#9-mi)
10. [R²](#10-r²)
11. [CRPS](#11-crps) ← **= MAE for deterministic algorithms**
12. [ACF](#12-acf)
13. [TOST](#13-tost)
14. [BA (Bland-Altman)](#14-ba-bland-altman)
15. [NRMSE](#15-nrmse)
16. [KLD](#16-kld)
17. [DTW](#17-dtw)
18. [CDT (Cohen's d)](#18-cdt-cohens-distance-test--cohens-d)
19. [NLL](#19-nll) ← **= f(RMSE) for deterministic algorithms**
20. [sMAE (Spectral MAE)](#20-smae-spectral-mae)
21. [ND](#21-nd)
22. [PFC](#22-pfc)

---

## 1. MAE

**Formula:**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

**What it measures**

Average absolute deviation between true and imputed values, in the same unit as y. Treats all errors equally regardless of magnitude. A value of 2 means the imputation is off by 2 units on average.

**Code**

```python
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
```

Uses `sklearn.metrics.mean_absolute_error`. **Correct.** Matches the canonical formula exactly.

**Reference implementations**
- sklearn `mean_absolute_error`: identical
- ImputeGAP `compute_mae`: identical
- PyPOTS `calc_mae`: identical

**Status: ✅ No issues**

---

## 2. RMSE

**Formula:**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}$$

**What it measures**

Like MAE but with squared errors, so large errors are penalised more heavily. Also in the unit of y. RMSE ≥ MAE always; the gap grows when errors are uneven, as sparse outliers inflate RMSE more than MAE.

**Code**

```python
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)
```

Uses `sklearn.metrics.root_mean_squared_error` (requires sklearn ≥ 1.4, as noted in requirements.txt). **Correct.**

**Reference implementations**
- sklearn ≥ 1.4 `root_mean_squared_error`: identical
- Pre-1.4 equivalent: `sqrt(mean_squared_error(...))`
- ImputeGAP `compute_rmse`: identical formula per series, but **aggregated differently** — see note below.

**Multi-series aggregation note:** for 2D input `(n_series, n_timesteps)`, `_apply_metric` computes RMSE per series and then averages those per-series RMSE values. ImputeGAP's `compute_rmse` instead pools all masked residuals across every series into one flat array and takes a single global RMSE: `sqrt(mean((input_data[nan_locations] - recov_data[nan_locations]) ** 2))`. Because `sqrt` is nonlinear, `mean(sqrt(x_i))` ≠ `sqrt(mean(x_i))` in general, so the two approaches can diverge whenever per-series error magnitudes differ (e.g. `random_spikes` on `mcar_20pct`: ours = 3.9457 vs. ImputeGAP's pooled value = 4.2095). They agree exactly when the per-series RMSE is constant across series (e.g. `constant_offset`, where both give 4.0). MAE is unaffected by this, since the mean is linear and mean-of-means equals the global mean. This is not a bug — per-series averaging gives every series equal weight regardless of its scale, whereas ImputeGAP's pooled RMSE gives every *point* equal weight — but the two numbers are not directly comparable, so note which convention is used when citing RMSE against ImputeGAP-reported values.

**Status: ✅ No issues** (see aggregation note above for cross-tool comparisons)

---

## 3. MSE

**Formula:**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

**What it measures**

RMSE before the square root. Values are in y², making them hard to interpret directly. Mathematically MSE = RMSE², so they encode identical information and produce the same algorithm ranking. Its main use is as a loss function during training.

**Code**

```python
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
```

**Correct.** Note from the metric selection spreadsheet: redundant with RMSE for benchmarking purposes.

**Status: ✅ No issues**

---

## 4. MRE

**Formula:**

$$\text{MRE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \frac{\left| y_i - \hat{y}_i \right|}{\left| y_i \right|}, \qquad \mathcal{M} = \{ i : y_i \neq 0 \}$$

**What it measures**

Relative error: how large the error is as a fraction of the true value. Unitless. A value of 0.05 corresponds to a 5% average error. Useful when the magnitude of y varies across datasets.

**Code**

```python
def mre(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))
```

**Correct, with a deliberate deviation:** zero-valued entries are excluded. The denominator is count(y_true ≠ 0), not n. This matches the PyPOTS `calc_mre` behaviour and avoids division by zero.

**Z-score normalisation note:** z-score normalisation is generally applied before imputation (TODO: doublecheck). On z-score normalised data the series is centred at zero, meaning many `y_true[i]` values will be close to (but not exactly) zero. The `!= 0` mask does not catch these near-zero entries, and their small denominators can inflate MRE arbitrarily, making an otherwise reasonable imputation appear poor. This is a fundamental incompatibility between MRE and zero-mean data, not a bug. The implementation follows the PyPOTS convention, and MRE should be interpreted with caution or excluded from reporting when z-score normalised data is used. Metrics that handle zero-mean data correctly include RMSE, MAE, NRMSE (with std normalisation), and sMAPE.

**Percentage form:** Several papers (MPIN, ReCTSi, CATSI) report MRE multiplied by 100 and label it "MRE%". The implementation returns the raw fraction (e.g. 0.05), not the percentage (5%). This is not a bug, but when comparing results against paper tables, the form used should be verified.

**Reference implementations**
- PyPOTS `calc_mre`: same zero-exclusion strategy

**Status: ✅ No issues** (zero-exclusion is intentional; MRE unreliable on z-score normalised data; note percentage vs. fraction form when comparing to papers)

---

## 5. sMAPE

**Formula:**

$$\text{sMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{\left| y_i - \hat{y}_i \right|}{0.5 \left( \left| y_i \right| + \left| \hat{y}_i \right| \right)}$$

**What it measures**

sMAPE addresses two known problems with standard MAPE. First, MAPE explodes when y_true[i] ≈ 0. Second, MAPE is asymmetric: over-imputing and under-imputing give different errors for the same absolute deviation. sMAPE uses the average of |y_true| and |y_pred| as denominator, bounding the result between 0% and 200% and treating both directions equally. This makes it suitable for z-score normalised data where standard MAPE would be undefined.

**Code**

```python
def smape(y_true, y_pred):
    denom = 0.5 * (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)
```

**Correct.**

**Reference implementations**
- sklearn has no sMAPE; the formula matches the standard definition exactly.
- sktime `MeanAbsolutePercentageError(symmetric=True)`: identical.

**Status: ✅ No issues**

---

## 6. WD

**Formula:**

$$\text{WD}(p, q) = \inf_{\gamma \in \Gamma(p,q)} \mathbb{E}_{(x,y) \sim \gamma} \left[ \left| x - y \right| \right]$$

For 1D distributions this simplifies exactly to:

$$\text{WD}(p, q) = \int_0^1 \left| F_p^{-1}(t) - F_q^{-1}(t) \right| dt$$

where $F^{-1}$ is the quantile function (inverse CDF).

**What it measures**

The minimum "work" (mass × distance) needed to transform one distribution into the other. WD is the only metric in this set that is sensitive to distributional shift without being sensitive to temporal structure. The shuffled synthetic scenario scores WD ≈ 0 precisely because the value distribution is unchanged regardless of temporal order.

**Code**

```python
def wd(y_true, y_pred):
    return wasserstein_distance(y_true, y_pred)
```

Uses `scipy.stats.wasserstein_distance`, which computes the exact 1-Wasserstein distance between two 1D empirical distributions. **Correct.**

**Reference implementations**
- scipy `wasserstein_distance`: this is the reference implementation.

**Status: ✅ No issues**

---

## 7. Pearson

**Formula:**

$$r = \frac{\sum_{i=1}^{n}(y_i - \bar{y})\,(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2 \cdot \sum_{i=1}^{n}(\hat{y}_i - \bar{\hat{y}})^2}}$$

Range: $[-1, 1]$, higher is better.

**What it measures**

Linear correlation between the two series. r = 1 means a perfect linear relationship, not necessarily y_pred = y_true: a constant offset or amplitude scaling also yields r = 1. r = 0 indicates no linear dependency. This is why the constant_offset and amplitude_scaled synthetic scenarios both score Pearson = 1 despite having nonzero errors.

**Code**

```python
def pearson(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]
```

`scipy.stats.pearsonr` returns `(correlation, p_value)`. The `[0]` correctly extracts the correlation. The zero-std guard is correct. **Correct.**

**Comparison with ImputeGAP `compute_correlation`:** both use `scipy.stats.pearsonr` on the masked positions only. The implementations are equivalent. The one difference is in the constant-array guard: ImputeGAP returns `np.nan` when either input is constant, whereas this implementation returns `0.0`. Both choices are reasonable: `np.nan` is mathematically precise (the correlation is genuinely undefined), while `0.0` follows the convention that a constant series has no correlation with anything. For consistency with ImputeGAP, consider returning `np.nan` instead.

**Reference implementations**
- scipy `pearsonr`: this is the reference implementation.
- ImputeGAP `compute_correlation`: equivalent formula, minor difference in constant-array handling noted above.

**Status: ✅ No issues**

---

## 8. JSD

**Formula:**

$$\text{JSD}(p \| q) = \frac{1}{2} D_{\text{KL}}(p \| m) + \frac{1}{2} D_{\text{KL}}(q \| m), \qquad m = \frac{p + q}{2}$$

Range: $[0, \ln 2]$ (base-$e$) or $[0, 1]$ (base-2), lower is better.

**What it measures**

A symmetric, bounded measure of how different two probability distributions are. Unlike KLD it is always defined and symmetric. Applied here to histogram-approximated distributions of the time series values.

**Code**

```python
def jsd(y_true, y_pred):
    bins = max(10, int(np.sqrt(len(y_true))))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    p, _ = np.histogram(y_true, bins=bins, range=(lo, hi))
    q, _ = np.histogram(y_pred, bins=bins, range=(lo, hi))
    p = p.astype(float) + 1e-10
    q = q.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q) ** 2)
```

`scipy.spatial.distance.jensenshannon` returns the Jensen-Shannon *distance* (√JSD). Squaring the result recovers the divergence, which is what is reported here and what papers typically refer to when citing JSD.

**Reference implementations**
- scipy `jensenshannon(p, q)**2`: what the code uses.

**Status: ✅ No issues**

---

## 9. MI

**Formula:**

$$\text{MI}(X; Y) = \sum_{x} \sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)}$$

Range: $[0, \infty)$, higher is better (more shared information).

**What it measures**

The amount of information (in nats, for log base e) that knowing Y gives about X. Unlike Pearson, it captures nonlinear statistical dependency. Two series can have Pearson = 0 but MI > 0 if they are nonlinearly related.

**Code**

```python
def mi(y_true, y_pred):
    bins = max(10, int(np.sqrt(len(y_true))))
    ...
    return float(mutual_info_score(
        np.digitize(y_true, edges),
        np.digitize(y_pred, edges),
    ))
```

Uses histogram binning to discretise both series, then `sklearn.metrics.mutual_info_score` on the bin labels. This is a standard approximation with two caveats. First, the result depends on the number of bins (√n rule): more bins produce lower MI due to a sparser joint histogram. The √n rule is a reasonable default but not universal. Second, `mutual_info_score` returns nats (log base e). Implementations using log base 2 (bits) will differ by a factor of 1/ln(2) ≈ 1.44.

**Comparison with ImputeGAP `compute_mi`:** both use `sklearn.metrics.mutual_info_score` with histogram binning and digitize. Two differences are worth noting. First, ImputeGAP uses a fixed 10 bins for both series; this implementation uses `max(10, √n)`, which adapts slightly to series length. Second, and more substantially, ImputeGAP computes bin edges independently for each series using the range of that series alone, while this implementation uses a shared range `[min(both), max(both)]` for both. Shared bin edges ensure that the same bin label refers to the same value interval in both series, which is the standard requirement for a valid joint histogram. ImputeGAP's approach is a reasonable and widely used convention, and values will be close in practice; the shared-range approach is noted here as the choice made in this implementation.

**Reference implementations**
- sklearn `mutual_info_score`: what the code uses.
- ImputeGAP `compute_mi`: equivalent approach, differences in bin count and edge strategy noted above.
- For continuous data: sklearn `mutual_info_regression` uses k-nearest neighbours and avoids binning. This is more accurate for continuous time series data and worth considering as an alternative.

**Status: ✅ No issues** (binning approximation is standard; bin count and edge strategy noted above)

---

## 10. R²

**Formula:**

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Range: $(-\infty, 1]$, higher is better. $R^2 = 1$ means perfect prediction; $R^2 = 0$ means no better than the mean; $R^2 < 0$ means worse than predicting the mean.

**What it measures**

The fraction of variance in y_true explained by y_pred. Unlike RMSE, R² is scale-free: it compares the algorithm to a mean-prediction baseline, answering "is this imputer better than doing nothing?"

**Code**

```python
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
```

`sklearn.metrics.r2_score`. **Correct.**

**Status: ✅ No issues**

---

## 11. CRPS

**⚠️ For deterministic algorithms, CRPS is mathematically identical to MAE.**

**Formula (for a probabilistic forecast $F$):**

$$\text{CRPS}(F, y) = \mathbb{E}_F |X - y| - \frac{1}{2} \mathbb{E}_F |X - X'|$$

For a deterministic forecast $f$ (a Dirac delta $\delta_f$), this collapses to:

$$\text{CRPS}(\delta_f, y) = |f - y| \quad \Longrightarrow \quad \frac{1}{n}\sum_{i=1}^n \text{CRPS}(\delta_{f_i}, y_i) = \text{MAE}$$

**What it measures**

CRPS is a proper scoring rule that evaluates both the accuracy and calibration of a probabilistic forecast. It rewards forecasters for providing well-spread distributions that also centre on the truth. For any algorithm that outputs point estimates (all ImputeGAP algorithms), CRPS = MAE. It provides additional information only for probabilistic outputs such as BayOTIDE.

**Code**

```python
def crps(y_true, y_pred):
    forecasts = y_pred[:, np.newaxis] if y_pred.ndim == 1 else y_pred
    return float(np.mean(ps.crps_ensemble(y_true, forecasts)))
```

When `y_pred` is 1D (a point estimate), the function creates a single-member ensemble and CRPS reduces to MAE. When `y_pred` is 2D of shape `(n_timesteps, n_samples)`, the full predictive distribution is evaluated and CRPS becomes genuinely informative.

**When CRPS is meaningful:** only for probabilistic algorithms that expose posterior samples (e.g. BayOTIDE, CSDI, GP-VAE, PRISTI). For all deterministic algorithms in ImputeGAP (CDRec, IterativeSVD, SoftImpute, SVT, SPIRIT, GROUSE, STMVL, and all statistics baselines), CRPS = MAE by definition. Keeping it in the metric set is still useful as an empirical demonstration of this equivalence.

**From the papers (BayOTIDE):** *"CRPS used only for BayOTIDE and probabilistic baselines; the primary indicator of uncertainty quantification quality."*

**TODO: check ImputeGAP output format for probabilistic algorithms** to confirm whether they expose posterior samples, in what shape, and whether the pipeline (`load_data`, `compute_all_scores`) needs to handle the mixed 1D/2D case.

**Reference implementations**
- `properscoring.crps_ensemble`: the reference for ensemble CRPS.

**Status: ✅ Implementation correct for both point estimates and sample arrays. Meaningful only for probabilistic algorithms; see TODO above.**

---

## 12. ACF

**Formula:**

$$\text{ACF}(k) = \frac{\text{Cov}(y_t,\, y_{t+k})}{\text{Var}(y_t)}$$

The ACF distance metric used in this implementation:

$$\text{ACF\_diff} = \frac{1}{K+1} \sum_{k=0}^{K} \left| \text{ACF}_y(k) - \text{ACF}_{\hat{y}}(k) \right|, \qquad K = \min(40,\, \lfloor n/4 \rfloor)$$

**What it measures**

The ACF of a series describes how correlated a value is with past values at lag k. This metric computes how much the imputed series distorts the autocorrelation structure of the ground truth. It is sensitive to oversmoothing (which increases low-lag autocorrelation) and to shuffling (which destroys all autocorrelation).

**Code**

```python
def acf(y_true, y_pred):
    nlags = min(40, len(y_true) // 4)
    acf_true = _acf(y_true, nlags=nlags, fft=True)
    acf_pred = _acf(y_pred, nlags=nlags, fft=True)
    return float(np.mean(np.abs(acf_true - acf_pred)))
```

`statsmodels.tsa.stattools.acf` with `fft=True`. Returns an array of shape (nlags+1,) starting at lag 0 (always 1.0 for any series). **Correct.** The lag-0 value cancels out harmlessly. Note: `acf` can return values slightly outside [-1, 1] for short series due to FFT numerical noise; this is a known statsmodels issue and does not affect the metric materially.

**Status: ✅ No issues**

---

## 13. TOST

**Formula:**

Two one-sided paired $t$-tests:

$$H_{0,1}: \mu_y - \mu_{\hat{y}} \leq -\Delta \qquad H_{0,2}: \mu_y - \mu_{\hat{y}} \geq +\Delta$$

$$\text{TOST score} = \max(p_1,\, p_2)$$

Equivalence is declared if both $p$ values are below $\alpha$ (typically 0.05), i.e. $\max(p_1, p_2) < \alpha$. A lower score indicates stronger evidence of equivalence, since both one sided null hypotheses must be rejected to conclude that $-\Delta < \mu_y - \mu_{\hat{y}} < \Delta$.

**What it measures**

Whether the means of the two series are statistically equivalent within a margin Δ. This is a clinical statistics tool that answers "are these two measurement methods interchangeable?" rather than "which one is more accurate?"

**Code**

```python
def tost(y_true, y_pred, epsilon=None):
    if epsilon is None:
        epsilon = 0.1 * float(np.std(y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pg.tost(y_true, y_pred, bound=epsilon, paired=True)
    return float(result["pval"].max())
```

`pingouin.tost` with `paired=True` (correct: the two arrays are element aligned). Epsilon = 10% of std(y_true) is a reasonable data driven default. The `warnings.catch_warnings` block suppresses scipy's precision loss `RuntimeWarning`, which fires when the difference series is nearly constant (e.g. a perfect constant offset).

**Known limitations:**

1. **TOST only tests means.** An algorithm with a constant +4 offset has the same mean difference as one with random noise of the same mean. TOST cannot distinguish them, making it a weak test of imputation quality in general.

2. **The epsilon threshold is arbitrary.** Different datasets with different std values produce different thresholds. A Δ of 10% std is not universally meaningful; in medical contexts Δ is set by regulatory requirement.

3. **TOST gives low p-values (i.e. demonstrates equivalence) for noisy but unbiased algorithms.** An algorithm that imputes randomly with mean equal to the true mean will show statistical equivalence even though the individual imputed values are far from the truth, which is misleading.

**Status: ✅ Implementation correct. Ranked as "lower is better" in `metric_config.py`, since a lower TOST score reflects stronger evidence of equivalence. Use with caution: TOST is the weakest metric in this set for general benchmarking.**

---

## 14. BA (Bland-Altman)

**Formula:**

$$\text{bias} = \bar{d} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

$$\text{LoA} = 1.96 \cdot s_d, \qquad s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

The 95% limits of agreement are $[\bar{d} - \text{LoA},\; \bar{d} + \text{LoA}]$.

**What it measures**

A clinical measurement agreement method. The bias (mean_diff) captures systematic error: a positive value means the imputation consistently undershoots. LoA captures the spread: 95% of individual differences fall within ±LoA of the bias. A perfect algorithm has mean_diff = 0 and LoA as small as possible.

**Code**

```python
def ba(y_true, y_pred):
    diff = y_true - y_pred
    mean_diff = float(np.mean(diff))
    loa = float(1.96 * np.std(diff, ddof=1))
    return mean_diff, loa
```

`ddof=1` is correct (sample standard deviation). **Correct.** The ranking uses `|mean_diff|` as the scalar, which is sensible but discards the sign and ignores LoA. BA returns a tuple, handled specially throughout `generate_reports.py` via `to_scalar`, `format_value`, and the `ba` branch in `generate_metrics_report`. This is all correct.

**Status: ✅ No issues**

---

## 15. NRMSE

**Formula:**

$$\text{NRMSE} = \frac{\text{RMSE}}{\text{std}(y)}$$

**Normaliser choice:**

"nRMSE" is used with several different normalisers across the algorithm papers reviewed, and the metric selection spreadsheet itself defines it generically as "RMSE normalised by either the mean or standard deviation of the true values":

| Paper / Algorithm | Normaliser used |
|---|---|
| TRMF (Yu et al. 2016) | mean(y_true) |
| IterativeSVD (Troyanskaya et al. 2001) | mean of entire complete dataset |
| MissForest (Stekhoven & Bühlmann 2012) | std(y_true) |
| CATSI (DACMI challenge) | each variable's within-patient observed range (min-max-like, per variable) |

Mean(y_true) is used most often across these references, but it is unsuitable here for the same reason MRE is: if z-score normalisation is applied upstream (see section 4), `mean(y_true) ≈ 0` for the whole series, making `RMSE / mean(y_true)` blow up or flip sign unpredictably. The MRE section already identifies **NRMSE with std normalisation** as one of the metrics that handles zero-mean data correctly, so std normalisation (the MissForest definition) was chosen for consistency with that reasoning, even though it is not the most frequent choice in the papers.

**What it measures**

RMSE made dimensionless for cross-dataset or cross-variable comparison, expressed relative to the natural variability of the series. This is equal to the coefficient of variation of the residuals, and remains well-defined for zero-mean (z-score normalised) data, unlike mean or min-max normalisation.

**Code**

```python
def nrmse(y_true, y_pred):
    denom = float(np.std(y_true))
    if denom == 0:
        return 0.0
    return float(root_mean_squared_error(y_true, y_pred) / denom)
```

The `denom == 0` guard handles the degenerate case of a constant series (std = 0), returning 0.0 since RMSE is also 0 in that case.

**Status: ✅ No issues. Std normalisation chosen explicitly and documented; consistent with the z-score handling discussed in section 4 (MRE).**

---

## 16. KLD

**Formula:**

$$D_{\text{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

**Asymmetric:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$. Undefined if $q(x) = 0$ where $p(x) > 0$.

**What it measures**

How much extra information is needed to encode samples from p using a code optimised for q. KLD = 0 means p = q. The implementation approximates it via histograms.

**Code**

```python
def kld(y_true, y_pred):
    ...
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))
```

`scipy.stats.entropy(p, q)` computes KL(p||q). The ε = 1e-10 smoothing avoids the undefined case. **Correct.**

**Note:** KLD is asymmetric and undefined when distributions do not overlap; JSD was specifically designed to fix both problems. KLD is included because it ranked in the metric selection process, not because it is preferable to JSD. The results are expected to show that KLD and JSD rank algorithms very similarly, empirically demonstrating the redundancy.

**Status: ✅ No issues**

---

## 17. DTW

**Formula:**

$$\text{DTW}(x, y) = \sqrt{\min_{W \in \mathcal{W}} \sum_{(i,j) \in W} (x_i - y_j)^2}$$

where $\mathcal{W}$ is the set of all valid monotone warping paths through the $n \times m$ cost matrix, and the local cost is the squared difference (Euclidean), with the square root applied to the total accumulated cost of the optimal path.

**What it measures**

The minimum cost to align two series by allowing elastic time warping. Unlike pointwise metrics, a series that is merely phase-shifted can have a very low DTW distance even with a high RMSE. This makes DTW specifically useful for detecting the time_shift scenario and for evaluating whether the imputed series has the right shape but wrong timing.

**Code**

```python
def dtw(y_true, y_pred):
    return _dtw.distance(
        y_true.astype(np.float64),
        y_pred.astype(np.float64),
    )
```

`dtaidistance.dtw.distance` with C backend. **Correct.** Verified empirically: for two constant series offset by 1 (length n), `dtw.distance` returns `sqrt(n)` along the diagonal path, confirming the squared-difference local cost with a final square root, as in the formula above. This computes the full (unconstrained) DTW distance. DTW is O(n²) in time and memory. For 200-point synthetic series this is fine; for the multivariate ImputeGAP series (called per-series then averaged) it will be slower. No Sakoe-Chiba band is applied; with 200 points this is not an issue.

**From the papers (HKMF-T):** *"A method can have good RMSE but poor DTW if it recovers magnitudes but distorts timing."* This is a direct citation supporting the inclusion of DTW alongside RMSE.

**Status: ✅ No issues**

---

## 18. CDT (Cohen's Distance Test / Cohen's d)

**Formula:**

$$d = \frac{\bar{y} - \bar{\hat{y}}}{s_{\text{pooled}}}, \qquad s_{\text{pooled}} = \sqrt{\frac{s_y^2 + s_{\hat{y}}^2}{2}}$$

The implementation returns $|d|$. Unitless. $d = 0$ means no difference in means; $d = 0.8$ is conventionally considered "large."

**What it measures**

A standardised mean difference: how far apart the means of the two series are, relative to their pooled spread. This is conceptually similar to BA's `mean_diff` (section 14) and to TOST (section 13), but expressed as a dimensionless effect size rather than a raw difference or a hypothesis test, which makes it comparable across series with different scales.

**Code**

```python
def cdt(y_true, y_pred):
    pooled_std = np.sqrt(0.5 * (np.var(y_true, ddof=1) + np.var(y_pred, ddof=1)))
    if pooled_std == 0:
        return 0.0
    return float(abs(np.mean(y_true) - np.mean(y_pred)) / pooled_std)
```

`pooled_std` uses `ddof=1` (sample variance) on both arrays, following the standard Cohen's d definition. The `pooled_std == 0` guard handles the degenerate case of two constant series with equal values.

**Category:** moved from "Distributional" to "Statistical Agreement" in `metric_config.py`, alongside `pearson`, `mi`, `r2`, `tost`, and `ba`, since Cohen's d characterises agreement between the two series' means rather than their full distributions.

**Status: ✅ No issues. Implements Cohen's d as defined in the metric selection spreadsheet.**

---

## 19. NLL

**⚠️ For deterministic algorithms, NLL is a monotone transformation of RMSE.**

**Formula (Gaussian NLL):**

$$\text{NLL} = -\frac{1}{n}\sum_{i=1}^{n} \log \mathcal{N}(y_i;\, \hat{y}_i,\, \sigma^2) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \cdot \text{MSE}$$

where $\sigma^2 = \text{Var}(y - \hat{y})$ is estimated from residuals.

**What it measures**

How probable the true values are under the model's predictive distribution. For a probabilistic model with a learned $\sigma$ per timestep, this is a genuine probabilistic metric. When $\sigma$ is estimated from the residuals of a deterministic predictor, $\sigma = \text{RMSE}$, and NLL becomes:

$$\text{NLL} = \frac{1}{2}\log(2\pi \cdot \text{RMSE}^2) + \frac{1}{2}$$

This is a monotone function of RMSE, producing the same algorithm ranking at a different scale.

**Code**

```python
def nll(y_true, y_pred):
    if y_pred.ndim == 2:
        mu = np.mean(y_pred, axis=1)
        sigma = np.std(y_pred, axis=1)
        sigma_y = float(np.std(y_true))
        floor = sigma_y if sigma_y > 0 else 1.0
        sigma = np.where(sigma < 1e-6 * floor, floor, sigma)
        return float(-np.mean(norm.logpdf(y_true, loc=mu, scale=sigma)))

    residuals = y_true - y_pred
    sigma = float(np.std(residuals))
    ...
    return float(-np.mean(norm.logpdf(y_true, loc=y_pred, scale=sigma)))
```

**Mathematically correct** as a Gaussian NLL. For the 1D (deterministic) path, sigma is a single global value estimated from the residuals, so NLL is ranking-redundant with RMSE, for the same reason CRPS = MAE for deterministic algorithms. The guarding code (returning 0.0 when sigma < 1e-6 × sigma_y) is correct and prevents astronomical values from JSON rounding.

**Probabilistic case (same situation as CRPS, section 11):** if `y_pred` is a posterior sample matrix of shape `(n_timesteps, n_samples)`, a single global sigma estimated from residuals would discard the per-timestep predictive uncertainty entirely. The 2D branch instead estimates `mu` and `sigma` per timestep directly from the sample distribution, giving a genuine per-point Gaussian likelihood — analogous to how `crps_ensemble` evaluates the full per-timestep distribution rather than a single point.

**TODO:** check imputegap's output format for probabilistic algorithms to confirm whether they expose posterior samples and in what shape, then verify the pipeline (`load_data`, `compute_all_scores`) handles the mixed 1D/2D case — same open question as CRPS (section 11).

**From the papers (BayOTIDE):** The algorithm paper describes NLLK as *"how probable the true missing values are under the learned posterior distribution; the theoretically primary criterion for a Bayesian model."* It is reported only for probabilistic baselines, confirming that NLL is not meaningful for deterministic algorithms.

**Status: ⚠️ Correct. NLL = f(RMSE) for deterministic algorithms (document this, as with CRPS). Probabilistic (2D) path added; pending pipeline verification per the TODO above.**

---

## 20. sMAE (Spectral MAE)

**Formula:**

$$\text{S-MAE} = \frac{1}{B}\sum_{b=1}^{B} \left| \widetilde{S}_y(f_b) - \widetilde{S}_{\hat{y}}(f_b) \right|$$

where $\widetilde{S}(f_b)$ is the normalised Power Spectral Density (PSD) at frequency bin $f_b$, estimated via the Lomb-Scargle periodogram and normalised to sum to 1 across bins.

**What it measures**

MAE between the normalised PSDs of the true and imputed series. Each PSD is treated as a distribution over frequency bins (sums to 1), so S-MAE is independent of overall amplitude and instead captures how much the *relative frequency content* differs — e.g. an algorithm that smooths out a periodic component will shift power away from that frequency, which S-MAE detects even if its time-domain MAE/RMSE is low. From the LSCD paper: *"A model can achieve low MAE while distorting frequency structure; S-MAE captures this failure."*

Range: $[0, 2]$, with 0 meaning identical normalised PSDs.

**Code**

```python
def smae(y_true, y_pred, n_freqs=50):
    n = len(y_true)
    t = np.arange(n, dtype=np.float64)
    freqs = np.linspace(2 * np.pi / n, np.pi, n_freqs)

    psd_true = lombscargle(t, y_true - np.mean(y_true), freqs)
    psd_pred = lombscargle(t, y_pred - np.mean(y_pred), freqs)

    psd_true = psd_true / (psd_true.sum() + 1e-10)
    psd_pred = psd_pred / (psd_pred.sum() + 1e-10)

    return float(np.mean(np.abs(psd_true - psd_pred)))
```

`scipy.signal.lombscargle` computes the periodogram at `n_freqs=50` angular frequencies evenly spaced between $2\pi/n$ (lowest resolvable frequency) and $\pi$ (Nyquist for unit sampling). Both series are mean-centred before the periodogram (the Lomb-Scargle implementation does not subtract the mean itself, and a non-zero mean would otherwise dominate the spectrum as a spurious zero-frequency component). The `+ 1e-10` guard avoids division by zero for constant series, where the PSD is identically zero and S-MAE correctly evaluates to 0.

**FULL_SERIES_METRIC:** like ACF and DTW, S-MAE needs the entire series to estimate a meaningful spectrum — evaluating it on only the scattered masked positions would not yield a valid periodogram. Added to `FULL_SERIES_METRICS` in `metric_config.py` and moved from "Pointwise Error" to "Temporal / Shape" (alongside ACF and DTW), since it characterises frequency/shape rather than pointwise magnitude.

**Empirical sanity check:** identical series → 0; constant offset (+4) → ≈0 (offset removed by mean-centering); low-frequency vs. high-frequency sine of the same amplitude → 0.039; constant series vs. constant series → 0.

**Status: ✅ Implements the LSCD spectral S-MAE via Lomb-Scargle, distinct from MAE/MRE/ND/NRMSE which are all time-domain.**

---

## 21. ND

**Formula:**

$$\text{ND} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|}$$

**What it measures**

The aggregate absolute error normalised by the aggregate true magnitude. Like MRE but uses sums rather than means (the n cancels), giving more weight to timesteps where |y_true| is large. Used in TRMF for matrix data where scales vary across dimensions.

**Code**

```python
def nd(y_true, y_pred):
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)
```

**Correct.** For constant series (all values identical), ND = MRE exactly. They diverge for non-constant series. Verified: `ND([5,5,5,5], [5.2,4.7,5.5,4.9]) == MRE(...) == 0.055`.

Unlike MRE, ND does not have the z-score normalisation problem from section 4: `sum(|y_true|)` over a zero-mean series is still strictly positive (it sums absolute values, not signed ones), so the denominator does not collapse toward zero.

**Status: ✅ No issues**

---

## 22. PFC

**Formula (continuous adaptation as implemented):**

$$\text{PFC} = \frac{100}{n} \sum_{i=1}^{n} \mathbf{1}\!\left[ \frac{|y_i - \hat{y}_i|}{|y_i| + \varepsilon} \leq \tau \right]$$

where $\tau = 0.10$ (10% relative tolerance) and $\varepsilon = 10^{-10}$.

**Original categorical definition (MissForest):**

$$\text{PFC} = \frac{\text{number of wrongly classified missing entries}}{\text{total number of missing entries}}$$

**What it measures**

The percentage of imputed values that fall within a tolerance band around the true value. For continuous data, this gives the fraction of acceptably accurate imputations. Higher is better. The result is sensitive to the choice of τ.

**Code**

```python
def pfc(y_true, y_pred, tolerance=0.10):
    rel_err = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10)
    return float(np.mean(rel_err <= tolerance) * 100)
```

**Correct**, with the ε guard for near-zero values. Note: PFC is defined in the MissForest paper for categorical variables (proportion of misclassified categories). The implementation applies it to continuous data using relative error tolerance. This is a different interpretation; for continuous imputation benchmarking it is a reasonable adaptation, but it is not the standard PFC definition. This distinction should be explicit in the thesis.

PFC is also robust to the z-score case (section 4): when `y_true ≈ 0`, `rel_err = |y - ŷ| / (|y| + ε)` becomes a huge number rather than NaN/inf, which simply fails the `<= tolerance` test (contributes 0 to the mean) instead of corrupting the aggregate the way MRE's per-element ratio would.

**Status: ✅ Implementation correct for continuous use case. Note the deviation from the categorical definition.**

---

## Summary of Issues

| # | Metric | Issue | Priority |
|---|--------|-------|----------|
| 11 | CRPS | CRPS = MAE for all deterministic algorithms (expected; BayOTIDE paper confirms). TODO: verify ImputeGAP's output format for probabilistic algorithms and whether `load_data`/`compute_all_scores` handle mixed 1D/2D `y_pred`. | ⚠️ Medium |
| 19 | NLL | NLL = monotone function of RMSE for deterministic algorithms (expected; BayOTIDE paper confirms). Same probabilistic-pipeline TODO as CRPS — 2D (per-timestep) branch implemented, pending pipeline verification. | ⚠️ Medium |
| 9 | MI | Binning approximation; consider `mutual_info_regression` for continuous data. | ⚠️ Medium |
| 4 | MRE | Correct (matches PyPOTS); unreliable on z-score normalised data due to near-zero denominators. Papers report as MRE% (×100): verify form when comparing. | 🟢 Low |
| 13 | TOST | Correct; weakest metric in this set for general benchmarking (tests means only). | 🟢 Low |
| 22 | PFC | Applied to continuous data with relative-error tolerance; original definition is categorical-only. Note deviation in thesis. | 🟢 Low |

---

## Next Steps for Verification

1. **Resolve the CRPS/NLL probabilistic-pipeline TODO** (sections 11, 19): check ImputeGAP's output format for probabilistic algorithms (e.g. BayOTIDE, CSDI, GP-VAE, PRISTI) to confirm whether posterior samples are exposed and in what shape, then verify `load_data`/`compute_all_scores` handle the mixed 1D/2D `y_pred` case for both metrics.
