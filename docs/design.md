# Design decisions and statistical limits

## The decision comes before the metric

A weekly order must balance excess stock against insufficient stock. Under the
single-period loss `c_under × max(y − s, 0) + c_over × max(s − y, 0)`, the optimal
continuous target under the true predictive distribution is its quantile at
`c_under / (c_under + c_over)`. A 4:1 scenario therefore motivates the 80th percentile.
The project estimates that quantile and evaluates the result; it does not assume
that an estimated quantile is optimal in a real inventory system.

The experiment compares it with a four-week mean, a validation-tuned mean buffer,
a selected point model, and a separately validation-tuned point buffer. Buffer
search is fixed to 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, and 3. Quantile-model complexity
is selected by mean validation pinball loss across 0.1, 0.5, 0.8, and 0.9.

Each product-week starts with zero inventory; leftover inventory is charged an
overage cost and does not carry forward. There are no supplier lead times,
purchase budgets, ordering fees, perishability, or substitutions. The API's
inventory-position subtraction is a convenience for a single decision, not a
validated multi-period inventory policy.

## Preventing temporal leakage

`feature_row` accepts only the strictly prior history. Offline `supervised`
passes `units[:i]`, while the API requires history to end exactly seven days
before the forecast Monday. Both use the same function and feature names.
Features include 1/2/4/8-week lags, 4/8/13-week means, recent variability,
zero-sales fraction, a recent-trend ratio, annual lag, and known week-of-year.

The annual lag uses 52 prior weeks when available and the past eight-week mean
otherwise, with an explicit availability indicator. Product identity is a
categorical feature with a stable training-cohort index. Future prices, invoices,
customer identities, and target-week units are never features.

There is no random train/test split. Validation selects candidates before refit
on train + validation; calibration does not refit models; test does not change
models, cohort membership, multipliers, or calibration. Past test observations are
valid inputs for later one-step forecasts. A regression test changes every target
at and after a cutoff and verifies that features through that cutoff stay identical.

## Why several losses are compared

Poisson boosting models a nonnegative conditional mean. Squared loss on `log1p(y)`
reduces the influence of large observations; `expm1(prediction)` is **not an
unbiased estimate of the arithmetic conditional mean**. That distinction matters
when a retailer faces asymmetric costs. WAPE/MAE selection rewards lower absolute
error, which need not minimize shortage cost. The observed model has this tradeoff.

The last-week and four-week-mean baselines need no learned parameters. The annual
baseline reveals how product lifecycle changes can make last year's sales poor
guidance. Only two tree complexities per loss are searched. All validation scores
are retained; test performance is never used to revise the search.

## Quantiles and empirical calibration

Four separately fitted quantile models may cross. Predictions are sorted across
quantile levels before validation, calibration, or test evaluation. Central raw
bounds are the rearranged 10th and 90th percentiles.

For each calibration row, set `scale = max(past_8_week_mean, 1)` and score
`max(q10 − y, y − q90, 0) / scale`. The correction is the order statistic at
`ceil((n + 1) × 0.8)` among the 450 calibration scores. New bounds subtract/add
that correction times the new row's scale, clipping the lower bound at zero.
The extra zero prevents shrinking the raw interval. This wrapper is an adaptation
of conformalized quantile regression, not a claim of a new algorithm.

Because product-weeks are dependent and sales can shift over time, exchangeability
is not defensible here. Coverage is assessed empirically, together with width and
weekly variation. Calibration adjusts the displayed interval; it does **not**
change the raw 80th-percentile stock target or establish an 80% future fill rate.

## What is being measured

- WAPE weights absolute errors by the aggregate scale of observed sales; it is
  not an average percentage error per product. Individual zero-sales rows are safe.
- Forecast bias is the mean signed error in units, revealing underprediction.
- Interval coverage is the proportion of product-weeks within the bounds.
- Fill rate is fulfilled units / total recorded units. The fraction of weeks
  without shortages is a different service metric, also retained.
- All inventory targets are rounded upward to whole units before evaluating costs.

The cost sensitivity file recomputes the costs of **the same frozen targets** at
ratios 1:1, 2:1, 4:1, and 8:1. It does not claim those targets are reoptimized for
each ratio. The uncertainty calculation resamples three-week contiguous blocks
with all products together, preserving some serial and cross-product dependence.
The resulting bootstrap interval is descriptive given the short test history.

## Engineering boundaries

The model artifact includes the cohort, feature schema, training/calibration dates,
parameters, source/panel hashes, and dependency versions. Loading requires a
compatible schema and the same scikit-learn major/minor version. Missing artifacts
fail startup. Artifacts are trusted local joblib files; the API never accepts them
from requests. A history mean ratio outside [0.5, 2] is a heuristic flag, not a
statistical drift test or automatic retraining trigger.

The core test suite covers contamination by future labels, batch/serving feature
parity, gaps/duplicates, training-only cohort selection, cleaning reconciliation,
finite-sample calibration rank, stock rounding/cost identities, model persistence,
and API validation. The workflow also executes the entire real-panel experiment
and notebook in a fresh environment. No live deployment or throughput SLA is claimed.

## Next experiments

1. Add product-level diagnostics and seasonal rolling validation across several
   years; preserve a fresh holdout before searching new models.
2. Obtain stock availability, promotion, and lead-time data to separate recorded
   sales from demand and evaluate a multi-period inventory policy.
3. Compare cost-specific quantiles and dependence-aware calibration approaches
   under more seasons and retailers.
4. Evaluate smaller-volume and new products explicitly rather than extending
   conclusions from the active top-50 cohort.
