# Delta To Best Analysis

This document describes the `delta_to_best` analysis implemented in `learning-dynamics/src/nnrun/analyses/delta_to_best.py`.

The purpose of the analysis is to measure, for every model and every epoch, how far that model is from the best model in the same dataset at that epoch.

It is a trajectory analysis. Unlike a pure final-score analysis, it keeps the whole training curve and asks:

- how quickly does a model approach the best model
- whether it improves steadily or stalls
- which models finish close to the best and which stay far away

## Core Idea

At each epoch, the analysis identifies the best observed value across models in the dataset and compares every model to that reference.

The delta is always defined as:

- distance behind the best model at that epoch

So a smaller delta is always better.

That means:

- for accuracy metrics, the best model has the largest value
- for loss metrics, the best model has the smallest value

The code handles both cases correctly with the `higher_is_better` flag.

## How Delta To Best Is Computed

For one dataset, one metric, and one epoch:

Assuming a metric where higher values are better, let \(v_{m,t}\) be the metric value of model \(m\) at epoch \(t\), and let the epoch-wise best value be

$$
b_t = \max_m v_{m,t}
$$

Then the delta-to-best value is

$$
\Delta_{m,t} = b_t - v_{m,t}
$$

Equivalently:

- `delta_to_best = best(epoch) - model_value(epoch)`

Interpretation:

- `delta_to_best = 0`
  - the model is best at that epoch
- small positive delta
  - the model is close to the best
- large delta
  - the model is far behind the best

This makes the metric easy to interpret consistently:

- lower is always better

## What The Analysis Does

For each dataset:

1. Build a `models x epochs` matrix for the chosen metric.
2. Compute the best value at each epoch.
3. Compute `delta_to_best` for every model and epoch.
4. Convert the result to a long table.
5. Fit a linear trend per model on `delta_to_best` over epoch.
6. Build additional per-model summary features from the delta curve.
7. Optionally generate diagnostic plots.

So the analysis has three layers:

- raw delta trajectories
- linear trend summaries
- richer curve-shape features

## Why This Analysis Is Useful

This analysis is useful when you care about relative progress rather than only absolute metric values.

It helps answer questions like:

- Is a model catching up to the best model or falling behind?
- Are some models consistently strong across the whole trajectory?
- Do some models start weak but approach the frontier later?
- How stable is a model near the end of training?

It is especially useful because the reference is dynamic:

- the “best” model can change over time
- the analysis therefore measures distance to the current frontier, not just distance to the final winner

## What It Brings

The analysis brings a standardized relative-performance view.

Instead of looking at raw accuracy or loss curves only, it rewrites them into:

- “how far behind the best am I?”

This is valuable because:

- it normalizes the interpretation across epochs
- it highlights models that remain competitive even if not best
- it separates final closeness from improvement rate
- it makes different models easier to compare on one common scale

## Main Output Tables

The analysis writes three main tables.

### 1. `delta_to_best_all`

This is the raw long-form table.

Columns:

- `dataset_name`
- `epoch`
- `model_name`
- `value`
- `best_value`
- `delta_to_best`

What it tells you:

- the actual metric value of each model at each epoch
- the best observed value at that epoch
- how far each model is behind the best

This is the most direct representation of the analysis.

### 2. `delta_to_best_lr`

This table summarizes each model’s delta curve with a linear regression.

The code fits:

- `delta_to_best ~ slope * epoch + intercept`

Stored columns include:

- `dataset_name`
- `model_name`
- `n_points`
- `slope`
- `intercept`
- `r2`
- `epoch_min`
- `epoch_max`
- `delta_mean`
- `delta_final`
- `se_slope`
- `t_stat`
- `p_value`
- `df`

How to interpret it:

- negative `slope`
  - the model is getting closer to the best over time
- positive `slope`
  - the model is drifting farther away from the best
- low `delta_final`
  - the model ends close to the best
- high `r2`
  - a straight-line trend explains the delta curve relatively well

This table is useful for ranking models by:

- final closeness
- rate of improvement
- smoothness of trend

### 3. `delta_to_best_features`

This is a richer per-model summary built by `build_delta_features(...)`.

Columns include:

- `dataset_name`
- `model_name`
- `n_points`
- `k`
- `start_mean_k`
- `end_mean_k`
- `improvement_k`
- `end_mad_k` or `end_std_k`
- `auc_mean`
- `theil_sen_slope`
- `spearman_rho`
- `spearman_p`
- `z_outlier_score`
- `iso_outlier`
- `iso_score`

These are curve-shape features, not just a single fitted line.

## What The Derived Features Mean

### `start_mean_k`

Mean delta over the first `k` epochs.

Interpretation:

- how far behind the best the model is at the start

### `end_mean_k`

Mean delta over the last `k` epochs.

Interpretation:

- how far behind the best the model is near the end
- smaller values mean better final competitiveness

### `improvement_k`

Defined as:

- `start_mean_k - end_mean_k`

Interpretation:

- positive value
  - the model improved relative to the best
- negative value
  - the model became worse relative to the best

### `end_mad_k` or `end_std_k`

Measures dispersion of the final `k` delta values.

If `use_mad = true`:

- `end_mad_k` is used
- this is a robust stability measure

Otherwise:

- `end_std_k` is used

Interpretation:

- small value
  - stable late training behavior
- large value
  - unstable late behavior

### `auc_mean`

Average delta over the whole epoch range, based on trapezoidal integration.

Interpretation:

- low value
  - the model stayed close to the best for most of training
- high value
  - the model spent much of training far from the best

### `theil_sen_slope`

Robust trend estimate of delta over time.

Interpretation:

- negative
  - catching up
- positive
  - falling behind

This is more robust to outliers than ordinary least squares.

### `spearman_rho`

Spearman correlation between epoch and delta.

Interpretation:

- negative
  - deltas generally decrease over time
- positive
  - deltas generally increase over time

This captures monotonic trend rather than linear trend.

### `z_outlier_score`

Euclidean distance in z-score space using selected summary features.

Interpretation:

- large value
  - the model looks unusual compared with others in the same dataset

### `iso_outlier` and `iso_score`

Outlier flags from Isolation Forest on the delta feature space.

Interpretation:

- `iso_outlier = true`
  - the model was flagged as anomalous in feature space

These features help identify unusual learning trajectories.

## Which Plots Are Generated

The analysis can generate up to three plot families when `plot.enabled = true`.

### 1. Per-Model Delta-To-Best Fit Grid

File name pattern:

- `{dataset}_delta_to_best_fit_{metric}_top{top_n}_{select_by}`

What it shows:

- one panel per selected model
- scatter of `delta_to_best` over epoch
- a fitted linear regression line

The title of each panel includes:

- final rank within the selected dataset
- final delta
- slope
- `r2`
- `p_value`

How models are selected:

- `select_by: final_delta`
  - models ending closest to the best
- `select_by: slope`
  - models with the most negative slopes first
- `select_by: r2`
  - models with the strongest linear fit

How to read it:

- downward fitted line
  - model is catching up
- flat line near zero
  - model stays close to the best throughout
- high line that stays high
  - model remains far behind

This is the most detailed visual output.

### 2. End vs Improvement Scatter

File name pattern:

- `{dataset}_end_vs_improvement`

What it shows:

- x-axis: `end_mean_k`
- y-axis: `improvement_k`
- color: late-stage stability (`end_mad_k` or `end_std_k`)

How to read it:

- left side
  - models that end close to the best
- high values on y-axis
  - models that improved a lot
- color scale
  - indicates whether the end of training is stable or noisy

What you can learn:

- which models both improved strongly and finished near the frontier
- which models finished well but were unstable
- which models improved a lot but still remained far from the best

### 3. Feature Distribution Plots

File name pattern:

- `{dataset}_feature_distributions`

What it shows:

- histograms of selected delta features

By default it includes:

- `end_mean_k`
- `improvement_k`
- `auc_mean`
- `end_mad_k` or `end_std_k`
- `z_outlier_score`
- `iso_score`

Each histogram title includes:

- sample skewness
- sample size

What you can learn:

- whether the dataset has a long tail of poor models
- whether improvement is symmetric or concentrated in a few models
- whether outlier behavior is common or rare

## What You Can Learn From The Analysis

This analysis can help you answer:

- Which models consistently stay close to the best?
- Which models improve most over time?
- Which models are stable at the end?
- Which models follow unusual trajectories?
- Is the dataset dominated by a few strong models or does performance remain competitive across many models?

Typical interpretation patterns:

- low `end_mean_k` and high `improvement_k`
  - strong recovery and strong finish
- low `end_mean_k` and low dispersion
  - stable near-best model
- high `auc_mean`
  - weak relative performance through most of training
- large outlier score
  - unusual training trajectory worth inspecting

## Important Limitations

This analysis is descriptive, not causal.

It has several important limitations:

- the reference “best model” is dataset-internal and epoch-specific
- if the best model changes abruptly, delta curves can reflect that change too
- linear regression is only a rough summary and may miss nonlinear trajectories
- outlier flags depend on the chosen feature set and contamination level
- results are metric-specific

Also:

- a model can have a strong negative slope but still finish far from the best
- a model can have low final delta without much improvement if it was strong from the start

So it is best to interpret the outputs together rather than relying on one number alone.

## Main Config Knobs

Important configuration options:

- `metric`
  - metric to analyze, for example `test_acc`
- `higher_is_better`
  - controls how the best value is selected each epoch
- `epoch_col`
  - source dataframe epoch column
- `model_col`
  - source dataframe model identifier column
- `epoch_range`
  - optional restriction on considered epochs
- `window_k`
  - number of early and late epochs used for start/end summary features
- `use_mad`
  - whether to use robust late-stage dispersion instead of standard deviation
- `isoforest_contamination`
  - outlier fraction used for Isolation Forest

Plot config:

- `plot.enabled`
- `plot.top_n`
- `plot.select_by`
- `plot.ncols`
- `plot.figsize`
- `plot.dpi`
- `plot.use_log`

## Summary

The `delta_to_best` analysis converts raw training curves into a relative-performance representation:

- how far each model is from the best model at each epoch

From that, it derives:

- raw delta trajectories
- linear trend summaries
- interpretable curve features
- outlier diagnostics

This makes it a useful analysis for understanding:

- competitiveness during training
- rate of catching up
- final closeness to the frontier
- unusual model trajectories
