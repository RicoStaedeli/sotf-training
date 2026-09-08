# Ranking Flips Analysis

This document describes the `ranking_flips` analysis implemented in `learning-dynamics/src/nnrun/analyses/ranking_flips.py`.

The purpose of the analysis is to measure how unstable model rankings are over training.

It asks questions like:

- how often do models swap relative order between epochs
- how many models change rank from one epoch to the next
- how similar are consecutive rankings
- how early does the ranking start to resemble the final ranking

This is a ranking-dynamics analysis, not a raw-metric analysis.

## Core Idea

For each dataset and each selected metric, the analysis builds a `models x epochs` matrix and converts the raw metric values into model rankings over time.

It then compares rankings between epochs in several ways:

- pairwise cross-overs between model pairs
- number of models whose rank changed
- Kendall tau agreement between rankings
- Kendall tau agreement with the final ranking

Together, these quantities describe whether the ordering of models is stable or still reshuffling.

## What The Analysis Does

For each dataset and each configured metric:

1. Build a matrix of model values over epochs.
2. Rank models at every epoch.
3. Compute pairwise flip counts between consecutive epochs.
4. Compute rank-change counts between consecutive epochs.
5. Compute Kendall tau between consecutive epochs.
6. Compute Kendall tau between each epoch and the final epoch.
7. Merge the main switching signals into a per-metric switch table.
8. Generate panel plots over metrics for that dataset.

The output is therefore both:

- tabular
- visual

and it is metric-specific.

## Why This Analysis Is Useful

This analysis is useful when you want to know whether intermediate rankings are trustworthy.

It helps answer questions like:

- Are the best-looking models early in training also the best later?
- Is ranking still unstable at epoch 10, 20, or 50?
- Do losses stabilize earlier than accuracies?
- Do many models remain competitive, or does the ordering settle quickly?

This matters for:

- early stopping
- model selection during training
- checkpoint budgeting
- understanding how quickly ranking-based conclusions become reliable

## Rank Convention

The analysis uses:

- rank `1` = best

This is handled correctly for both metric types:

- accuracy-like metrics: higher is better
- loss-like metrics: lower is better

The metric configuration therefore needs the correct `higher_is_better` setting.

## The Main Quantities

### 1. Pairwise Flips

This is the strictest notion of a ranking cross-over.

For each pair of models `(i, j)` and each consecutive epoch transition `t -> t+1`, the analysis checks whether:

- model `i` was ahead of model `j` at `t`
- but behind model `j` at `t+1`

If so, that pair flipped.

The analysis stores two versions:

- `pairwise_flips_without_smoothening`
- `pairwise_flips_with_smoothening`

Interpretation:

- high pairwise flip count
  - many model pairs reversed order
  - ranking is unstable
- low pairwise flip count
  - ordering is mostly preserved

This is the most direct measure of actual pairwise cross-over structure.

### 2. Rank Changes

This is a coarser measure than pairwise flips.

For each consecutive epoch transition, the analysis counts how many individual models changed rank.

Stored columns from the rank-change table include:

- `models_changed_rank`
- `models_changed_rank_with_smoothening`
- `smoothening_threshold_rank_change`

Interpretation:

- high values
  - many models moved in rank from one epoch to the next
- low values
  - rankings are more stable

This measure is easier to read than pairwise flips, but it is less strict:

- many small local changes can increase this count even if pairwise structure changes only modestly

### 3. Kendall Tau Between Consecutive Epochs

For each consecutive epoch pair, the analysis computes Kendall’s tau-b between the two ranking vectors.

Stored columns:

- `from_epoch`
- `to_epoch`
- `kendall_tau`
- `kendall_p_value`

Interpretation:

- tau near `1`
  - ranking is very similar between the two epochs
- tau near `0`
  - weak agreement
- tau below `0`
  - strong reversal structure

This is a global summary of rank similarity, rather than a direct count of flips.

### 4. Kendall Tau To Final Epoch

For each epoch `t`, the analysis compares the ranking at `t` to the ranking at the final available epoch.

Stored columns:

- `epoch`
- `final_epoch`
- `kendall_tau_to_final`
- `kendall_p_value_to_final`

Interpretation:

- high early tau to final
  - the ranking is already close to its final form
- low early tau to final
  - the final ranking is not yet visible
- a steadily increasing curve
  - rankings are gradually converging toward the final ordering

This is often the most useful quantity if you care about when ranking becomes predictive of the end result.

## The Role Of `smoothening_threshold`

The analysis supports a threshold parameter `smoothening_threshold`.

This threshold is used in:

- pairwise flip counting
- rank-change counting

The idea is to avoid treating very tiny differences as meaningful flips.

Interpretation:

- without smoothening
  - every sign change counts, even if values were extremely close
- with smoothening
  - only sufficiently separated comparisons count

Why this matters:

- near-ties can create noisy flips
- smoothening makes the analysis more conservative

So the smoothened signals are often better when you want robust ranking instability rather than sensitivity to tiny fluctuations.

## What The Analysis Brings

This analysis brings a structured view of ranking stability over time.

It helps separate several different phenomena:

- local reshuffling from one epoch to the next
- overall agreement of rankings
- convergence toward final ranking
- noise due to near-ties

That makes it more informative than simply plotting raw validation curves.

## What You Can Learn From It

This analysis can teach you:

- whether rankings stabilize early or late
- whether some metrics are much noisier than others
- whether the final ordering is already visible far before training ends
- whether rank changes are driven by many actual cross-overs or mostly small perturbations

Interpretation examples:

- high pairwise flips and low Kendall tau
  - ranking is actively unstable
- low pairwise flips and high Kendall tau
  - ordering is already stable
- low tau-to-final early, high late
  - final ranking emerges gradually
- big difference between without-smoothening and with-smoothening versions
  - many apparent flips are probably near-tie noise

## Which Tables Are Generated

The implementation writes per-dataset tables.

### 1. `{dataset}_all_switches_`

This is the main merged per-metric switch table.

It contains one row per consecutive epoch transition and metric, and merges:

- rank-change information
- pairwise flip counts
- average metric movement across models

Columns typically include:

- `from_epoch`
- `to_epoch`
- `models_changed_rank`
- `models_changed_rank_with_smoothening`
- `pairwise_flips_without_smoothening`
- `pairwise_flips_with_smoothening`
- `smoothening_threshold_pairwise_flip`
- `avg_from`
- `avg_to`
- `avg_delta`
- `metric`
- `n_models`

What it tells you:

- how much the ranking changed
- how many actual cross-overs happened
- whether the average metric also moved strongly in that transition

This is the most important table for downstream analyses.

### 2. `{dataset}_kendall_tau_step__{metric}`

This table stores Kendall tau between consecutive epochs for one metric.

What it tells you:

- how similar each epoch ranking is to the immediately previous one

### 3. `{dataset}_kendall_tau_to_final__{metric}`

This table stores Kendall tau between each epoch and the final epoch for one metric.

What it tells you:

- how early the ranking starts to resemble its final form

## Which Plots Are Generated

The current implementation creates panel plots per dataset, with one subplot per metric.

### 1. Pairwise Flips Panel Without Smoothening

File name pattern:

- `{dataset}_pairwise_flips_without_smoothening`

What it shows:

- y-axis: `pairwise_flips_without_smoothening`
- x-axis: `to_epoch`

Interpretation:

- peaks indicate epochs where many model pairs swapped order

### 2. Pairwise Flips Panel With Smoothening

File name pattern:

- `{dataset}_pairwise_flips_with_smoothening`

This plot is only produced if `smoothening_threshold` is provided.

What it shows:

- y-axis: `pairwise_flips_with_smoothening`
- x-axis: `to_epoch`

Interpretation:

- a cleaner, more conservative version of flip activity

### 3. Rank Switches Panel

File name pattern:

- `{dataset}_rank_switches`

What it shows:

- `models_changed_rank` as the without-smoothening curve
- `models_changed_rank_with_smoothening` as the smoothened curve if available

Interpretation:

- how many models changed rank at each transition
- how much of that remains after smoothening

### 4. Kendall Tau Panel

File name pattern:

- `{dataset}_kendall_tau`

What it shows:

- Kendall tau between consecutive epochs for each metric

Interpretation:

- values near `1` mean the ordering barely changed
- dips indicate local ranking instability

### 5. Kendall Tau To Final Panel

File name pattern:

- `{dataset}_kendall_tau_to_final`

What it shows:

- Kendall tau between each epoch and the final epoch

Interpretation:

- how early the final ranking becomes visible

## A Small Plotting Detail

For plots only, the transition:

- `0 -> 1`

is dropped by `_drop_epoch0_to1_for_plot(...)`.

This means:

- the tables still contain that transition
- but the plots omit it

The reason is that the first training step can create a large, visually distracting jump.

## Important Limitations

This analysis studies relative ranking, not absolute model quality.

Important limitations:

- rank-based summaries ignore the magnitude of metric differences
- many rank changes can occur even if actual metric differences are tiny
- results depend on the choice of metric
- results depend on the smoothening threshold when smoothening is used
- consecutive-epoch instability does not automatically mean the final ranking is unpredictable

So it is best to read:

- pairwise flips
- rank changes
- Kendall tau
- tau-to-final

together rather than in isolation.

## Main Config Knobs

Important configuration options:

- `metrics`
  - list of metrics to analyze, each optionally with `higher_is_better`
- `epoch_col`
  - epoch column in the source dataframe
- `model_col`
  - model identifier column
- `epoch_range`
  - optional restriction on considered epochs
- `smoothening_threshold`
  - threshold used for smoothened flip and rank-change calculations
- `png_dpi`
  - output resolution

Panel layout options:

- `panel.nrows`
- `panel.ncols`
- `panel.figsize`

## Typical Questions This Analysis Can Answer

- At what epoch do model rankings become stable?
- Are early rankings reliable enough for pruning or early stopping?
- Do losses stabilize earlier than accuracies?
- Are observed ranking changes mostly meaningful or mostly near-tie noise?
- Which datasets have the most volatile training competition between models?

## Summary

The `ranking_flips` analysis is a ranking-stability analysis over training.

It measures:

- actual pairwise cross-overs
- number of models that change rank
- rank similarity between consecutive epochs
- rank similarity to the final epoch

This makes it a useful tool for understanding:

- how noisy rankings are
- when the ranking starts to settle
- whether early ranking-based decisions are trustworthy
