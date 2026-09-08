# Clustering Analysis

This document describes the clustering analysis implemented in `learning-dynamics/src/nnrun/analyses/clustering.py`.

The goal of the analysis is to group models by the shape of their training curves.

Instead of asking only:

- which model is best at the end

it asks:

- which models behave similarly over the whole training trajectory

This is useful for identifying typical learning patterns inside a model zoo.

## Core Idea

For one chosen metric, the analysis:

1. builds a `models x epochs` matrix
2. extracts summary features from each model’s curve
3. clusters models in that feature space with KMeans
4. plots one representative curve summary per cluster

So the clustering is not done directly on the raw curve vectors. It is done on derived curve features.

## What The Analysis Does

For each dataset:

1. Build a metric matrix using the selected metric and epoch range.
2. Extract per-model curve features.
3. Select a subset of those features for clustering.
4. Standardize the features.
5. Cluster models with KMeans.
6. Save the cluster assignment together with the curve features.
7. Plot median cluster curves with interquartile ranges.

The output therefore gives both:

- a table of cluster assignments
- a visualization of how the clusters differ over training

## Why This Analysis Is Useful

This analysis is useful when final score alone is not enough.

It helps answer questions like:

- Are there distinct types of training behavior in the dataset?
- Do some models learn fast and plateau early while others improve slowly?
- Are there unstable models, smooth models, or late-improving models?
- Are poor final models poor from the start, or do they fail later?

It is especially useful for:

- understanding heterogeneity in a hyperparameter sweep
- detecting typical trajectory families
- comparing datasets by how diverse their learning behavior is

## The Input To Clustering

The analysis clusters one metric at a time.

Typical examples:

- `validation_acc`
- `test_acc`
- `validation_loss`

It first converts the dataframe into a matrix:

- rows = models
- columns = epochs
- values = metric values

If there are fewer than:

- 2 models
- or 2 epochs

the analysis skips that dataset.

## Curve Features Used For Clustering

The feature extraction is done by `extract_curve_features(...)` in `learning-dynamics/src/nnrun/compute/features.py`.

It computes several per-model summaries of the training curve. The clustering code then uses the following default subset:

- `final_val`
- `auc_norm`
- `peak_drop`
- `std_delta`
- `early_slope`
- `late_slope`
- `t95_epoch`
- `avg_regret`

These are the features actually used by `cluster_models_by_features(...)`.

### `final_val`

Final metric value of the curve.

What it captures:

- how strong the model ends

### `auc_norm`

Area under the curve, normalized by epoch span.

What it captures:

- average performance over training
- not just the final point

### `peak_drop`

Difference between the peak value and the final value.

What it captures:

- whether the model peaked and then degraded
- useful especially when curves overshoot

### `std_delta`

Standard deviation of epoch-to-epoch changes.

What it captures:

- how volatile the learning trajectory is

### `early_slope`

Slope between the first epoch and an early epoch window.

What it captures:

- how quickly the model improves at the beginning

### `late_slope`

Slope over the last part of training.

What it captures:

- whether the model is still improving late
- whether it has plateaued or deteriorated

### `t95_epoch`

First epoch where the curve reaches 95% of its final value.

What it captures:

- how quickly the model approaches its final level

### `avg_regret`

Average gap between the final value and the whole trajectory.

What it captures:

- how much performance was “left on the table” during training

## How Clustering Is Performed

The clustering helper is implemented in `learning-dynamics/src/nnrun/compute/clusters.py`.

The process is:

1. keep the selected feature columns
2. drop models with `NaN` in any clustering feature
3. standardize all used features with `StandardScaler`
4. run `KMeans`

This matters because:

- clustering should not be dominated by one feature’s scale
- models with incomplete feature vectors cannot be clustered reliably

## Choosing The Number Of Clusters

The config parameter `k` controls the number of clusters.

Two modes are supported:

### Fixed `k`

If `k > 0`, that exact number of clusters is used.

Example:

- `k: 5`

### Automatic `k`

If `k == 0`, the code selects `k` automatically by maximizing silhouette score over a range.

Default search range:

- `k_min = 2`
- `k_max = 10`

The chosen `k` is the one with the best valid silhouette score.

Interpretation:

- higher silhouette
  - cleaner separation between clusters
- lower silhouette
  - clusters overlap more strongly

This makes `k=0` useful when you want the code to choose a reasonable cluster count for each dataset.

## What The Analysis Brings

This analysis brings a compact summary of curve diversity.

Instead of manually scanning hundreds of training curves, it groups models into a few typical trajectory families.

That helps you:

- see whether the dataset contains clearly distinct learning behaviors
- understand whether good and bad models fail in different ways
- reduce visual complexity
- study characteristic training archetypes

## What You Can Learn From It

This analysis can teach you:

- whether there are fast learners and slow learners
- whether some clusters plateau early
- whether some clusters are unstable or noisy
- whether final performance differences are already visible early
- whether different curve shapes lead to similar final performance

Typical interpretation patterns:

- one cluster rises fast and plateaus high
  - strong early learners
- one cluster rises slowly but steadily
  - late improvers
- one cluster is noisy with wide IQR
  - unstable training behavior
- one cluster ends low with weak slopes
  - consistently poor models

## Which Tables Are Generated

Per dataset, the analysis writes two tables.

### 1. `{dataset}_cluster_curve_features_{metric}_{tag}`

This is the main final table.

It contains:

- all extracted curve features
- the assigned cluster label

What it tells you:

- which cluster each model belongs to
- how the feature values differ across clusters

This is the most useful table for downstream inspection.

### 2. `{dataset}_cluster_features_used_{metric}_{tag}`

This table contains the exact feature matrix that was actually used for clustering.

Important detail:

- models with `NaN` in the selected clustering features are dropped before clustering

So this table shows the cleaned feature subset on which clustering was performed.

## Which Plot Is Generated

The implementation currently generates one figure per dataset.

### Cluster Median Curves With IQR

File name pattern:

- `{dataset}_clusters_{metric}_{tag}`

What it shows:

- one curve per cluster
- line = median curve of models in that cluster
- shaded band = interquartile range

The plot is generated by `plot_cluster_median_curves(...)`.

How to read it:

- separated median lines
  - clusters represent meaningfully different trajectory families
- narrow IQR
  - models inside the cluster are internally consistent
- wide IQR
  - the cluster is more heterogeneous

This is the main visual summary of the clustering result.

## Interpretation Guidance

When reading the cluster plot, ask:

- Do clusters differ mainly in final level?
- Do they differ in early slope?
- Do they differ in stability or noise?
- Are some clusters only temporarily strong and then collapse?

You should also compare the plot with the feature table:

- the plot shows the shape
- the table shows the numerical reasons for the grouping

## Important Limitations

This analysis is exploratory.

Important limitations:

- KMeans imposes roughly centroid-based cluster structure
- results depend on the chosen feature set
- results depend on the selected metric
- different `k` choices can change the interpretation substantially
- dropped-NaN models are excluded from clustering

Also:

- clustering summarizes trajectories, it does not explain why they differ
- good-looking clusters do not imply causal groups

So the safest interpretation is:

- use clustering to discover trajectory families
- then inspect those clusters with the feature tables and original metadata

## Main Config Knobs

Important configuration options:

- `metric`
  - metric whose curves are clustered
- `epoch_col`
  - source dataframe epoch column
- `model_col`
  - source dataframe model identifier column
- `epoch_range`
  - optional restriction on considered epochs
- `k`
  - number of clusters, or `0` for silhouette-based auto-selection
- `tag`
  - string appended to output names
- `png_dpi`
  - figure resolution

## Typical Questions This Analysis Can Answer

- Are there a few recurring training behaviors in this dataset?
- Do strong final models all look similar, or do they come from multiple trajectory families?
- Are poor models uniformly poor, or are there several distinct failure modes?
- Does the dataset naturally separate into stable and unstable training groups?
- Is automatic `k` choosing a small number of clear trajectory classes or a more fragmented structure?

## Summary

The clustering analysis groups models by training-curve shape using extracted curve features and KMeans.

It provides:

- cluster assignments
- the feature table used for clustering
- a median-curve plot with IQR per cluster

This makes it a useful exploratory tool for understanding:

- diversity of training behavior
- typical learning archetypes
- how model trajectories differ beyond final score alone
