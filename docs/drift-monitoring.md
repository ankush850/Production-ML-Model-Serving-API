# Drift Monitoring

How feature drift is detected, what the numbers mean, and — importantly — **where this
implementation is weak**. Every figure below is real output from `src/training/drift.py`.

---

## Table of contents

- [What drift means here](#what-drift-means-here)
- [Reference statistics](#reference-statistics)
- [PSI](#psi-population-stability-index)
- [KS test](#ks-two-sample-test)
- [Categorical shift](#categorical-population-shift)
- [Thresholds and exit codes](#thresholds-and-exit-codes)
- [Measured behaviour: two real runs](#measured-behaviour-two-real-runs)
- [Known weaknesses](#known-weaknesses-read-this)
- [Using it as a retrain gate](#using-it-as-a-retrain-gate)

---

## What drift means here

**Covariate drift**: the distribution of *input features* in a new window differing from the
distribution the model was fitted on. The model is not re-evaluated — no labels are needed,
which is exactly why PSI/KS are used. Predictions silently degrade while every request still
returns HTTP 200.

Drift is measured against the **training window only**, not the whole dataset. `reference_stats.json`
is built by `build_reference_stats(train_df)` inside `train.py`, so it describes the data the
pipeline actually learned from. Comparing production against the *combined* data (a common
mistake) understates drift for anything the model never saw.

## Reference statistics

Written at training time into `reference_stats.json` alongside the model.

```json
{
  "numeric": {
    "n_line_items": {
      "quantiles": { "0.05": 1.0, "0.1": 2.0, "0.25": 6.0, "0.5": 15.0,
                     "0.75": 29.0, "0.9": 52.0, "0.95": 75.0 },
      "null_fraction": 0.0,
      "n": 30445,
      "bins":     [1.0, 2.0, 6.0, 15.0, 29.0, 52.0, 75.0],
      "expected": [0.0931, 0.0426, 0.1298, 0.2371, 0.2554, 0.1441, 0.0479, 0.0499]
    }
  },
  "categorical": {
    "country": { "United Kingdom": 0.9164, "Germany": 0.0180, "EIRE": 0.0154, "...": "..." }
  }
}
```

| Field | Purpose |
|---|---|
| `quantiles` (7 levels, 0.05–0.95) | Human-readable summary; also seeds the KS reference sample |
| `null_fraction` | Baseline missingness, so a change in null rate is itself a signal |
| `n` | Sample size the reference was built from |
| `bins` | **Unique** quantile edges — the PSI bin boundaries |
| `expected` | Share of training rows falling in each bin (sums to 1.0) |
| categorical value | Population share of each category in the training window |

`bins` and `expected` are the load-bearing fields, and their existence is a bug fix — see
[PSI](#psi-population-stability-index).

## PSI (Population Stability Index)

\[
\mathrm{PSI} = \sum_{i=1}^{N} (a_i - e_i)\,\ln\!\frac{a_i}{e_i}
\]

where \(e_i\) is the **expected** (training) share of bin \(i\) and \(a_i\) the **actual**
(new window) share. Shares are clipped to a floor of \(10^{-4}\) so an empty bin yields a large
but finite number rather than `-inf`/`nan`.

Properties worth internalising:
- **0 means identical distributions.** Not "small" — identical.
- **Asymmetric and unbounded above.** A category that vanishes entirely contributes
  \((0 - e)\ln(0/e)\), which the clip turns into a very large term. This is why `month` can
  report PSI ≈ 6.9 (see [measured runs](#measured-behaviour-two-real-runs)).
- **Not a probability.** "PSI 0.05" does not mean "5% of rows moved"; it is a divergence in nats-ish
  units. Compare against thresholds, not against intuition.

### Why bins are stored instead of recomputed — the bug this fixed

The original implementation derived PSI bins by calling `np.quantile` on the reference
*at check time*. For a low-cardinality integer feature this produces **repeated edges**:
`n_line_items` ∈ {1, 2, …} with 10 requested quantiles returns edges like `[1,1,2,2,3,3,…]`.
`np.digitize` against tied edges drops rows into the wrong (or empty) bins, and the clipping
floor then amplifies the empty bins into enormous PSI terms.

Measured result: **PSI = 0.577 for `n_line_items` when comparing the training data against its
own reference statistics.** A perfect, no-op check fired a drift alarm. On a daily schedule that
alerts every day, forever, for a feature that has not moved — which is how drift monitoring gets
turned off by an exhausted on-call engineer.

Fix: compute the edges once at training time with `np.unique` (collapsing ties), store the
edges **and** the resulting expected shares, and reuse them via `psi_from_bins()`. PSI on the
reference is now **0.0004** for that feature. `check_drift` falls back to the old
`psi_numeric()` path when `bins` is absent, so artifacts written before this fix still work.

> This is also why the model-versioning docs warn that artifacts must be regenerated after
> changing the statistics they carry: version `20261029T121219Z` predates the fix and has no
> `bins` key. Re-running training produced `20261029T145410Z` with correct statistics — ROC-AUC
> was **identical** (0.7716528247960346), confirming the fix touches only drift bookkeeping.

## KS (two-sample test)

A NumPy implementation of the two-sample Kolmogorov–Smirnov statistic — SciPy is deliberately
not a dependency.

\[
D = \sup_x |F_{\mathrm{actual}}(x) - F_{\mathrm{reference}}(x)|
\]

p-value from the asymptotic one-sided distribution, \(Q_{\mathrm{KS}}(\lambda)\) approximated by
the alternating series with \(\lambda = (\sqrt{n_{\mathrm{eff}} + 0.12} + 0.01/\sqrt{n_{\mathrm{eff}}})\,D\).

The reference side is **not the original training data** (that would require shipping millions of
values inside the artifact). It is a reconstruction: `n_eff = min(n_actual, 10_000)` points are
interpolated from the stored quantiles (`_reference_sample`). So the KS comparison is
*actual window vs. a piecewise-linear approximation of the training CDF*.

`d <= 0` short-circuits to `(0.0, 1.0)`, so identical inputs cannot produce a spurious
"significant" result.

## Categorical population shift

For each categorical feature, expected shares come from the reference and actual shares from
value counts in the new window. Categories present in one side only are unioned in with share
`0.0` (clipped as above), so **a brand-new market or a vanished market is detected as drift**.

An extra non-numeric signal is emitted when the category sets differ:

```json
"country": { "psi": 0.0181, "new_or_missing_categories": true }
```

`new_or_missing_categories` is informational and does **not** set `drifted` on its own — a new
sub-market with a 0.1% share is not an incident, and OneHotEncoder was fitted with
`handle_unknown="ignore"` so unseen categories degrade gracefully rather than crash. A large
*share* change is what crosses the PSI threshold.

## Thresholds and exit codes

| Constant | Value | Meaning |
|---|---|---|
| `PSI_THRESHOLD` | **0.2** | A feature is added to `drifted` when its PSI exceeds this |
| `KS_ALPHA` | **0.05** | Significance level, **but see the caveat below** |
| (KS effect-size floor) | `d > 0.1` | `ks_drift` requires both `p < 0.05` **and** `D > 0.1` |
| `DEFAULT_BINS` | 10 | Bin count for the legacy `psi()`/`psi_numeric()` path |

0.1–0.2 is the conventional PSI band: `< 0.1` ≈ no meaningful shift, `0.1–0.2` ≈ moderate,
`> 0.2` ≈ significant. This project alerts at 0.2 to keep false positives low, with the practical
consequence that slow, subtle erosion may go unflagged for a while.

CLI exit codes (`python -m src.training.drift`):

| Code | Meaning |
|---|---|
| `0` | Checked successfully, **no** drift |
| `1` | Drift detected (`has_drift: true`) |
| `2` | Usage/environment error (missing CSV, unloadable artifact) |

The report is printed as JSON to stdout; diagnostics go to stderr.

## Measured behaviour: two real runs

Both against LATEST (`20261029T145410Z`), whose reference is the **training window**
(30,445 invoices, before `2011-07-14T21:25:00`).

### Run A — reference window itself (control, should be clean)

```bash
python -m src.training.drift --data data/raw/online_retail.csv --source local
```

**Exit `0`**, `has_drift: false`, `drifted: []`.

| Feature | PSI | KS stat | Flagged |
|---|---|---|---|
| total_quantity | 0.0014 | 0.0525 | — |
| n_line_items | **0.0004** | 0.0528 | — |
| mean_unit_price | 0.0019 | 0.0526 | — |
| order_value | 0.0004 | 0.0525 | — |
| days_since_last_order | 0.0053 | 0.0657 | — |
| order_hour | 0.0003 | **0.1354** | `ks_drift: true` |
| country | 0.0010 | — | `new_or_missing_categories` |
| month | 0.0904 | — | `new_or_missing_categories` |
| day_of_week | 0.0002 | — | `new_or_missing_categories` |

PSI is ~0 everywhere — the fix works, and PSI correctly does **not** fire. Note `order_hour`
still raises `ks_drift` at PSI 0.0003; see [known weaknesses](#known-weaknesses-read-this).
`month` at 0.0904 is real: the full CSV includes the later window, whose seasonal mix genuinely
differs from the training window. Still below 0.2, so no alert. Correct behaviour.

### Run B — held-out future window (should show drift)

277,236 raw rows after the split cutoff.

**Exit `1`**, `has_drift: true`, `drifted: ['days_since_last_order', 'month']`.

| Feature | PSI | KS stat | Flagged |
|---|---|---|---|
| total_quantity | 0.0259 | 0.0694 | — |
| n_line_items | 0.0070 | 0.0620 | — |
| mean_unit_price | 0.0345 | 0.0930 | — |
| order_value | 0.0080 | 0.0603 | — |
| **days_since_last_order** | **0.7322** | 0.1892 | **drifted**, `ks_drift`, null_shift 0.1394 |
| order_hour | 0.0056 | 0.1403 | `ks_drift` |
| country | 0.0181 | — | `new_or_missing_categories` |
| **month** | **6.8994** | — | **drifted** |
| day_of_week | 0.0087 | — | — |

Both flagged features are **genuine, explainable, and expected** — not artefacts:

- **`month` (PSI 6.90)** — the training window covers Jul–mid-Dec 2011, so months 7–12 have
  ~0.17 share each and months 1–6 have **zero**. The later window fills January–June. A category
  that is genuinely empty on one side produces an enormous PSI through the clipping term. Real
  distribution change, correctly caught. This is also the single most important operational
  finding in the project: **the model was trained on six months and must never be assumed to
  generalise to the other six.**
- **`days_since_last_order` (PSI 0.73, null shift 0.14)** — nulls encode "first order", which
  only exists in the *earliest* data. A later window has far fewer first-orders-per-row, so the
  null fraction legitimately drops and the recency distribution shifts. Correct signal.

Basket-size and value features stayed below 0.04, and `country` PSI is 0.018 — the retail mix is
stable, the calendar is not. This is exactly the pattern a drift monitor should reveal.

## Known weaknesses — read this

Honest assessment of what this module does **not** do well. Do not present it as more reliable
than this.

1. **The KS p-value is uninformative at production scale.** In *both* runs above, every single
   `p_value` is `0.0` — including Run A, where PSI ≈ 0.0004 proves the distributions are
   effectively identical. With n in the tens of thousands, the KS test rejects the null at
   differences far below anything that matters. **The `D > 0.1` effect-size guard is what carries
   the signal; the p-value carries nothing.** Never alert on KS p alone.
   *Better:* drop the p-value from decision-making, keep `D`, and alert on `D > 0.1`; or compare
   against a subsample of the actual training data rather than a reconstruction.

2. **KS and PSI disagree on `order_hour`** (Run A: PSI 0.0003, `D = 0.1354`, `ks_drift: true`).
   Root cause is the quantile-interpolated reference sample: `order_hour` is discrete and tightly
   bounded (6–20), so a piecewise-linear CDF fitted through 7 quantiles does not reproduce it.
   The disagreement is a *measurement artefact of the reconstruction*, not evidence of drift.
   The report surfaces both so you can see the conflict rather than hiding it.

3. **PSI is unbounded, so the 0.2 threshold cannot express severity.** PSI 0.21 and PSI 6.90
   produce the same alert. Rank by PSI magnitude before acting; do not treat "1 feature drifted"
   and "1 feature at PSI 6.9" as the same page.

4. **Threshold is fixed in code**, not configurable per feature. `days_since_last_order`
   legitimately swings far more than `n_line_items`, so a single global 0.2 both under-alerts on
   stable features and over-alerts on calendar-driven ones. Per-feature thresholds stored in the
   artifact would be the natural extension.

5. **No label/concept drift.** This module cannot detect the relationship between features and
   outcome changing (inflation shifting what "reorder" means). That needs labels and periodic
   re-scoring against a holdout — genuinely out of scope here.

6. **Univariate only.** PSI/KS are computed per feature. Correlations can break while every
   marginal looks perfect. A multivariate check (e.g. drift in the model's score distribution)
   would be the next layer.

7. **Drift is measured on engineered features, so input drift can be masked.** If `Country`
   spelling changes, `build_features` absorbs it before PSI ever sees it — you would get a
   schema-violating *prediction* problem rather than a drift alert. Validate raw inputs
   (`src/data/validate.py`) as well.

## Using it as a retrain gate

Exit codes make this a CI/cron primitive:

```bash
python -m src.training.drift --data data/raw/latest_window.csv --source s3
if ($LASTEXITCODE -eq 1) {
  Write-Output "drift detected -> trigger retrain"
} elseif ($LASTEXITCODE -ne 0) {
  Write-Output "drift check itself failed"   # do NOT treat as healthy
}
```

Schedule it **on the same cadence you batch predictions** (daily is typical), and store the JSON
report per run so you can see a trend rather than a single snapshot — a PSI climbing
0.05 → 0.09 → 0.14 over a week is more actionable than one day's 0.14 crossing nothing.

Two rules for acting on it:

- **A non-zero exit from the *check* is not a clean result.** Exit `2` means the monitor itself
  failed; alert on it separately.
- **Drift is a trigger to investigate, not to retrain blindly.** Retrain on the drifted window
  only if you also re-verify the split and metrics; otherwise you are fitting to the anomaly and
  losing the ability to ever detect it again.

Related: [model-versioning.md](model-versioning.md) · [runbook.md](runbook.md#playbook-3-drift-alert)
