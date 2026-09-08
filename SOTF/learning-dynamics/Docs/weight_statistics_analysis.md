# Weight Statistics Analysis

This document describes the `weight_analysis` implementation in `learning-dynamics/src/nnrun/analyses/weight_statistics.py`.

This analysis is the reporting layer for checkpoint-derived weight statistics. It does not define the statistics themselves. Instead, it:

- loads or computes the cached weight-statistics table
- selects metric combinations to analyze
- aggregates those statistics over epochs
- generates dataset-level plots and tables

If you want the meaning of each individual weight statistic, see:

- `learning-dynamics/Docs/weight_statistics_reference.md`

## Purpose

The purpose of the analysis is to answer questions like:

- how does a specific weight statistic evolve over training
- how variable is that statistic across models in the same dataset
- which layers follow similar or different trajectories
- do certain weight statistics shift systematically over epochs

It is therefore a descriptive analysis of weight-space dynamics.

## What The Analysis Does

For each dataset:

1. Determine which epochs are available from the source dataframe.
2. Load or compute the cached weight-statistics long table.
3. Optionally stop after cache generation only.
4. Filter the long table to the selected metric family, metric, and scope.
5. Resolve one or many `(metric_family, metric, scope)` combinations.
6. For each resolved combination:
   - aggregate model-level values over epoch
   - aggregate layer-level values over epoch
   - generate one or more plots
   - save the corresponding tables

So the analysis is organized around:

- dataset
- metric family
- metric
- scope

## The Input Table

The analysis operates on a cached long-form dataframe created by `compute_metrics_df_long_for_dataset(...)`.

That table contains rows like:

- one metric value
- for one model
- at one epoch
- for one layer or for the model-level `__model__` row

Important columns include:

- `model_id`
- `epoch`
- `layer_name`
- `layer_order`
- `layer_type`
- `metric_family`
- `metric`
- `scope`
- `value`

This means the analysis can work uniformly across:

- norms
- distribution statistics
- matrix statistics
- sharpness proxies
- distance metrics

## Why This Analysis Is Useful

This analysis is useful when you want to understand how parameter-space properties evolve during training.

It helps answer questions like:

- Do layer norms grow steadily or plateau?
- Are biases becoming more dispersed over time?
- Do certain layers become unusually large or unstable?
- Are changes concentrated in a few layers or spread across the model?

It is especially useful because it can compare:

- model-level statistics
- individual layers
- all layers together

## Metric Selection Modes

The analysis supports several ways to select what to plot.

### 1. Single Combination

You can select one explicit combination such as:

- `metric_family: "norm"`
- `metric: "l2"`
- `scope: "weight_only"`

This produces plots and tables for that single combination only.

### 2. All Metrics

You can select:

- `metric_family: "all"`
- `metric: "all"`
- `scope: "all"`

In this mode, the analysis discovers all available combinations in the cached table and processes each of them.

This is the broadest scan mode.

### 3. Unterthiner Mode

The analysis also supports:

- `metric_family: "unterthiner"`

This is a convenience mode that selects the subset corresponding to the Unterthiner-style statistics currently implemented:

- `distribution/*`
- `norm/l1`
- `norm/l2`

with the relevant scopes such as:

- `weight_only`
- `bias_only`
- `full_model` for the full-vector distribution statistics

This is useful if you want to focus on that paper-specific subset without manually enumerating metrics.

## Scope Semantics In The Analysis

The same metric name can appear under different scopes.

Common scopes include:

- `weight_only`
- `bias_only`
- `combined`
- `full_model`
- `default`

This matters because different scopes answer different questions.

Examples:

- `norm | l2 | weight_only`
  - how large the weights are
- `distribution | mean | bias_only`
  - how the bias distribution shifts
- `distribution | variance | full_model`
  - how dispersed the entire flattened parameter vector is

Important detail:

- not every scope has both layer-level and model-level rows

For example:

- some combinations exist only at layer level
- some exist only at model level

That is why the analysis can warn that no model-level rows exist for a given combination while still producing layer-level plots.

## Cache Behavior

The analysis caches the full long-form weight-statistics table per dataset and split.

Cache path pattern:

- `learning-dynamics/outputs/cache/weight_metrics/complete_ws_tables/{dataset}_{split}_all_metrics_long.parquet`

This is useful because computing all checkpoint-derived weight statistics can be expensive.

### Cache-Only Mode

If:

- `cache_only: true`

the analysis will:

- load or compute the cache
- return the cached long table as a report table
- skip all plotting and aggregation

This is useful if you only want the raw statistics for downstream work.

## Epoch Handling

The analysis separates:

- source dataframe epoch column
- weight-statistics table epoch column

Important config keys:

- `source_epoch_col`
  - used to infer available epochs from the original report dataframe
- `epoch_col`
  - used inside the cached weight-statistics table, usually `epoch`

This separation matters because the source dataframe often uses:

- `training_iteration`

while the cached weight table uses:

- `epoch`

## Main Aggregations

The analysis uses two main aggregation strategies.

### 1. Model-Level Aggregation Over Epoch

If model-level rows exist for a combination, the analysis groups by epoch and computes:

- `mean`
- `std`
- `median`
- `min`
- `max`
- `q10`
- `q25`
- `q75`
- `q90`
- `count`

Interpretation:

- mean and median
  - central tendency across models
- percentile bands
  - spread across models
- count
  - how many valid model values contributed

### 2. Layer-Level Aggregation Over Epoch

For layer-level rows, the analysis groups by:

- epoch
- layer
- optionally additional grouping columns

and computes an aggregated value, currently:

- mean over models

This produces one trajectory per layer.

## Which Tables Are Generated

The exact set depends on the selected combinations, but the main patterns are:

### 1. Resolved Combination Table

Per dataset:

- `{dataset}_{split}_resolved_metric_combinations`

This tells you exactly which `(metric_family, metric, scope)` combinations were analyzed.

### 2. Raw Cache Table In Cache-Only Mode

Per dataset:

- `{dataset}_{split}_all_metrics_long`

This is the full cached long-form statistics table.

### 3. Model-Level Epoch Statistics

Per dataset and combination:

- `{dataset}_{split}_{metric_family}_model_{metric}_{scope}_epoch_stats`

This table contains the epoch-wise summary statistics for the model-level `__model__` rows.

### 4. Single-Layer Epoch Statistics

Per dataset, layer, and combination:

- `{dataset}_{split}_{metric_family}_{layer_name}_{metric}_{scope}_epoch_stats`

This is the summary table for one selected layer.

### 5. All-Layer Mean-Long Table

Per dataset and combination:

- `{dataset}_{split}_{metric_family}_all_layers_{metric}_{scope}_mean_long`

This contains one row per epoch and layer, with the mean value across models.

## Which Plots Are Generated

The exact set depends on config flags, but there are three main plot families.

### 1. Single Metric Over Dataset

This is generated both for:

- model-level rows
- one selected layer

It shows:

- x-axis: epoch
- solid line: mean across models
- dashed line: median across models
- darker band: 25th to 75th percentile
- lighter band: 10th to 90th percentile

There are two common title patterns:

- `{dataset} [split] | {metric_family} | model | {metric} | {scope} over dataset`
- `{dataset} [split] | {metric_family} | {layer_name} | {metric} | {scope} over dataset`

What it is good for:

- seeing central tendency and variability across models
- comparing mean and median
- spotting skewed or highly variable epochs

### 2. All-Layer Line Plot

File name pattern:

- `{dataset}_{split}_{metric_family}_all_layers_{metric}_{scope}`

It shows:

- one line per layer
- x-axis: epoch
- y-axis: mean value across models for that layer

What it is good for:

- comparing how layers differ over time
- seeing whether certain layers dominate the change
- identifying layers with especially large or unstable values

### 3. Layer-Epoch Heatmap

File name pattern:

- `{dataset}_{split}_{metric_family}_heatmap_{metric}_{scope}`

It shows:

- rows: layers
- columns: epochs
- color: mean metric value

What it is good for:

- quickly seeing where in the network and when in training the statistic is large or small
- identifying layer-specific phases of change
- spotting blocks of similar behavior across layers

## How To Interpret The Main Plots

### Single-Metric Aggregate Plot

Use this when you want to answer:

- how does this statistic evolve across models over time

Interpretation patterns:

- rising mean and median
  - the statistic is increasing for most models
- wide percentile bands
  - strong between-model heterogeneity
- mean far from median
  - skewed distribution across models
- disappearing percentile band
  - very little spread or too few valid models at that epoch

### All-Layer Line Plot

Use this when you want to answer:

- which layers drive the statistic

Interpretation patterns:

- one layer consistently above others
  - that layer dominates the scale of the statistic
- many parallel lines
  - layers move together
- crossings between layers
  - relative layer importance changes over training

### Heatmap

Use this when you want to answer:

- where and when does the statistic concentrate

Interpretation patterns:

- vertical bands
  - many layers change at the same epoch
- horizontal bands
  - one layer stays consistently distinct
- isolated hotspots
  - transient layer-specific events

## Grouped Analysis

The analysis also supports:

- `groupby_cols`

If those columns exist in the cached table, the model-level and single-metric plots can be stratified by group.

Examples might include:

- optimizer
- initialization type
- dropout

When grouping is used:

- each group gets its own line in the aggregate plot
- the heatmap can be restricted to a chosen group selection

This is useful when you want to compare weight-space dynamics across hyperparameter subgroups.

## What You Can Learn From This Analysis

This analysis can help you learn:

- whether parameter norms or distributions systematically grow or shrink
- which layers are most dynamic
- whether weight-space changes are concentrated in a few components
- whether models in the same dataset follow similar or diverse weight-statistic trajectories
- whether a paper-specific subset, such as the Unterthiner statistics, behaves in a consistent way

Typical questions it can answer:

- Do early layers or late layers change more?
- Are biases stable while weights shift?
- Do model-level norms plateau while layer-level distributions continue to move?
- Which statistics are most heterogeneous across models?

## Important Limitations

This is a descriptive analysis.

Important limitations:

- it does not directly relate weight statistics to functional performance
- it does not infer causality
- it can only analyze statistics that were computed and cached
- not every statistic exists at every scope or aggregation level

Also:

- model-level and layer-level views answer different questions
- a visually strong trend does not necessarily imply functional importance
- some combinations may produce warnings simply because that representation does not exist structurally

## Main Config Knobs

Important configuration options:

- `cache_dir`
  - where the cached weight-statistics parquet files live
- `dataset_split`
  - which split to compute from the checkpoint dataset
- `force_recompute`
  - whether to ignore the cache and recompute
- `skip_missing_cache`
  - whether to skip datasets whose cached weight-statistics table is missing instead of computing it
- `cache_only`
  - build/load the cache without plotting
- `metric_family`
  - explicit family, `all`, or `unterthiner`
- `metric`
  - explicit metric or `all`
- `scope`
  - explicit scope or `all`
- `epoch_col`
  - epoch column inside the cached weight-statistics table
- `source_epoch_col`
  - epoch column in the original report dataframe
- `epoch_range`
  - optional epoch filtering for plotting
- `layer_name`
  - which single layer to plot; if omitted, the first layer is used
- `groupby_cols`
  - optional subgroup columns for stratified aggregation
- `include_single_metric_plot`
- `include_all_layer_plot`
- `include_heatmap`
- `heatmap_group_selection`
- `png_dpi`

## Summary

The `weight_analysis` module is the reporting and visualization layer for checkpoint-derived weight statistics.

It provides:

- cached long-form tables of all computed weight statistics
- model-level epoch summaries
- single-layer summaries
- all-layer trajectory plots
- layer-epoch heatmaps

This makes it useful for understanding:

- how weight statistics evolve during training
- how models differ in weight space
- which layers and statistics are most dynamic
- how broad or narrow those changes are across a dataset
