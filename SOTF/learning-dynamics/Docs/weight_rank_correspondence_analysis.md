# Weight Rank Correspondence Analysis

This document describes the `weight_rank_correspondence` analysis implemented in `learning-dynamics/src/nnrun/analyses/weight_rank_correspondence.py`.

The current implementation is a flip-based correspondence analysis.

Its purpose is to answer a specific question:

- when model pairs cross over in weight-statistic space, are those the same cross-overs that happen in functional performance space

In other words, it compares ranking instability induced by weight statistics to ranking instability induced by a functional metric such as:

- `test_acc`
- `test_loss`

## Core Idea

For each weight statistic, the analysis builds a trajectory matrix over:

- models
- epochs

It then counts pairwise cross-overs in that statistic exactly the same way `ranking_flips` counts cross-overs for functional metrics.

After that, it compares the weight-space flips to the functional flips in two ways:

1. aggregate correspondence
   - do the counts rise and fall together over epochs
2. same-pair correspondence
   - are they literally the same model pairs that cross over

The second question is the stricter and more important one.

## What The Analysis Does

For each dataset:

1. Load or generate the functional `ranking_flips` table for the chosen functional metric.
2. Build the functional metric matrix over models and epochs.
3. Load or compute the cached weight-statistics table.
4. Discover the selected weight-statistic combinations.
5. For each `(metric_family, metric, scope)` combination:
   - build a weight-statistic trajectory matrix
   - count pairwise flips between consecutive epochs
   - align the model identities between weight space and functional space
   - compute overlap of the exact flip pairs
   - summarize correspondence scores
6. Aggregate all metrics into summary tables and plots.

So the analysis is centered on:

- one functional metric
- many weight statistics
- pairwise ranking cross-overs

## Why This Analysis Is Useful

This analysis is useful when you want to know whether changes in weight space are actually tracking meaningful functional ranking changes.

It helps answer questions like:

- Which weight statistics best reflect functional rank reshuffling?
- Do some weight statistics flip at the same epochs as `test_acc` or `test_loss`?
- Are the same model pairs changing order in weight space and performance space?
- Which weight statistics are merely noisy, and which ones capture functionally relevant structure?

This is stronger than simple correlation between a weight statistic and performance rank, because it asks about the actual pairwise cross-over structure.

## The Two Sides Of The Comparison

### Functional Side

The functional side comes from:

- `ranking_flips.py`

using one selected metric such as:

- `test_acc`
- `test_loss`

The analysis looks for:

- `learning-dynamics/outputs/reports/tables/ranking_flips/{dataset}_all_switches_.parquet`

If the file does not exist, or the requested functional metric is missing, it calls `run_ranking_flips(...)` to generate it.

So the functional side is always based on the same pairwise-flip logic as the main `ranking_flips` analysis.

### Weight Side

The weight side comes from:

- `weight_statistics.py`

For each selected weight metric combination, the analysis builds a trajectory matrix:

- rows = models
- columns = epochs
- values = weight statistic

It then counts pairwise flips across consecutive epochs using the same `count_pairwise_flips(...)` logic.

## How Weight Trajectories Are Built

For each weight-statistic combination:

- if model-level `__model__` rows exist and `include_model_row = true`
  - use those directly
- otherwise
  - average the layer values within each model and epoch

This is important because some weight statistics exist naturally at model level, while others only exist per layer.

## What Counts As A Cross-Over

For one epoch transition `t -> t+1`, a model pair `(i, j)` is considered a flip if:

- model `i` was above model `j` at `t`
- but below model `j` at `t+1`

This is checked separately for:

- the weight statistic
- the functional metric

The analysis can optionally use:

- `pairwise_flip_tau`

to ignore very small differences in the weight statistic before counting a flip.

## Aggregate Correspondence vs Same-Pair Correspondence

The analysis computes two different kinds of correspondence.

### 1. Aggregate Flip Correspondence

This asks:

- do the number of weight-space flips and the number of functional flips move together across epochs

Stored summary values include:

- `pearson_pairwise_flips`
- `spearman_pairwise_flips`
- `pearson_pairwise_flip_fraction`
- `spearman_pairwise_flip_fraction`

Interpretation:

- high correlation
  - epochs with many functional flips also tend to have many weight-space flips

This is useful, but not sufficient.

Two signals can correlate in count even if they involve completely different model pairs.

### 2. Same-Pair Correspondence

This asks:

- are the actual model pairs that flip in weight space the same ones that flip in functional space

This is the more meaningful correspondence test.

For each epoch transition, the analysis computes:

- `n_weight_flip_pairs`
- `n_functional_flip_pairs`
- `n_overlap_flip_pairs`
- `overlap_jaccard`
- `overlap_precision_weight_vs_functional`
- `overlap_recall_weight_vs_functional`
- `same_pair_overlap_pct_of_weight`
- `same_pair_overlap_pct_of_functional`

## What The Overlap Metrics Mean

### `overlap_jaccard`

This is:

- size of intersection divided by size of union

Interpretation:

- `1.0`
  - exactly the same model pairs flipped on both sides
- `0.0`
  - no shared flip pairs

This is the strictest symmetric overlap measure and is the main summary score used to rank metrics.

### `same_pair_overlap_pct_of_functional`

This asks:

- among all functional flip pairs, what percent are also weight-space flip pairs

Interpretation:

- high value
  - the weight statistic captures much of the functional cross-over structure

This is functional-side recall.

### `same_pair_overlap_pct_of_weight`

This asks:

- among all weight-space flip pairs, what percent are also functional flip pairs

Interpretation:

- high value
  - the weight statistic is precise and not generating many extra irrelevant flips

This is weight-side precision.

## Why This Analysis Brings More Than Simple Correlation

A simple correlation analysis asks:

- do larger weight statistics tend to appear in better or worse models

This analysis asks something stricter:

- when the relative order of models changes in weight space, does the relative order also change in functional space

That makes it useful for studying whether a weight statistic reflects:

- the actual competitive ordering of models
- not just a broad monotonic trend

## Model Alignment Between Weight And Functional Space

The analysis must align models across two different sources:

- functional dataframe model identifiers
- cached weight-statistics model identifiers

It tries two alignment strategies:

1. canonicalized basename
2. extracted trial token

It chooses the version with the best overlap of:

- common models
- common epochs

This alignment step is important. Without it, same-pair overlap would be meaningless.

## Which Tables Are Generated

The exact set depends on the selected metric combinations, but the main outputs are:

### 1. Functional Flip Table

Per dataset:

- `{dataset}_functional_pairwise_flips__{functional_metric}`

This is the prepared functional flip series used for comparison.

### 2. Weight Trajectory Matrix

Per metric combination:

- `{dataset}_weight_trajectory_matrix__{metric_id}`

This stores the model-by-epoch matrix of the weight statistic.

### 3. Weight Pairwise Flips

Per metric combination:

- `{dataset}_weight_pairwise_flips__{metric_id}`

This stores the weight-space pairwise flip counts across epoch transitions.

### 4. Weight-Functional Same-Pair Overlap

Per metric combination:

- `{dataset}_weight_functional_flip_overlap__{metric_id}`

This is the most important detailed table.

It stores, per epoch transition:

- weight flip counts
- functional flip counts
- exact overlap count
- Jaccard overlap
- overlap percentages

### 5. Metric Summary Table

Per dataset:

- `{dataset}_metric_flip_summary`

This ranks the weight statistics by summary correspondence quantities.

Important columns include:

- `mean_pairwise_flips`
- `mean_pairwise_flip_fraction`
- `mean_overlap_flip_pairs`
- `mean_overlap_jaccard`
- `mean_same_pair_overlap_pct_of_functional`
- `mean_same_pair_overlap_pct_of_weight`
- correlation-based summary columns

### 6. Long Tables Over Metrics

Per dataset:

- `{dataset}_metric_pairwise_flips_long`
- `{dataset}_metric_pairwise_flip_overlap_long`

These are useful for heatmaps and downstream analysis across many metrics.

## Which Plots Are Generated

The analysis generates both summary plots and detailed per-metric plots.

### 1. Top Weight Statistics By Mean Same-Pair Jaccard

File name pattern:

- `{dataset}_metric_scan_top_mean_pairwise_flips`

What it shows:

- a bar chart of the top weight statistics ranked by `mean_overlap_jaccard`

Interpretation:

- higher bar
  - the weight statistic more consistently matches functional same-pair flips

This is the main ranking plot.

### 2. Top Weight Statistics By Mean % Of Weight Pairs Matched

File name pattern:

- `{dataset}_metric_scan_top_mean_pairwise_flip_fraction`

What it shows:

- a bar chart of `mean_same_pair_overlap_pct_of_weight`

Interpretation:

- how precise the weight-statistic flips are relative to the functional flips

### 3. Heatmap Of Mean % Of Functional Pairs Matched

File name pattern:

- `{dataset}_metric_scan_heatmap_mean_pairwise_flip_fraction`

What it shows:

- rows = metric family
- columns = metric
- value = `mean_same_pair_overlap_pct_of_functional`

Interpretation:

- which families and specific metrics recover the functional flip structure best on average

### 4. Per-Epoch Heatmap Of Functional Pair Match Percentage

File name pattern:

- `{dataset}_metric_pairwise_flip_count_heatmap`

What it shows:

- rows = weight metrics
- columns = epoch transitions
- value = `% of functional pairs matched`

Interpretation:

- when each weight statistic tracks functional flips well or poorly

### 5. Detailed Per-Metric Flip Count Plot

File name pattern:

- `{dataset}_weight_pairwise_flips__{metric_id}`

What it shows:

- weight-space pairwise flip count
- functional pairwise flip count
- same-pair overlap count
- Jaccard overlap on a second axis

How to read it:

- aligned peaks in weight and functional flips
  - similar instability timing
- high same-pair overlap and high Jaccard
  - strong structural correspondence
- many weight flips but low overlap
  - noisy or irrelevant weight-space reshuffling

### 6. Detailed Per-Metric Same-Pair Overlap Percentage Plot

File name pattern:

- `{dataset}_weight_functional_overlap_pct__{metric_id}`

What it shows:

- `% of {functional_metric} flip pairs matched`
- `% of weight flip pairs matched`

How to read it:

- high functional-side percentage
  - the weight statistic captures much of the functional cross-over structure
- high weight-side percentage
  - most weight-space flips are functionally relevant

## What You Can Learn From This Analysis

This analysis can help answer:

- Which weight statistics track functional rank instability best?
- Do certain weight metrics anticipate or reflect functional reshuffling?
- Are some weight statistics sensitive but noisy, while others are precise?
- Which metrics capture the same model-pair competition that matters functionally?

Typical interpretation patterns:

- high Jaccard, high functional overlap percentage
  - strong candidate weight statistic for tracking functional structure
- high flip-count correlation, low same-pair overlap
  - timing matches, but the actual model pairs do not
- high weight-side precision, low functional recall
  - the weight statistic is selective but misses many functional flips
- high functional recall, low weight-side precision
  - the weight statistic captures many functional flips but also adds many extra ones

## Important Limitations

This analysis is more demanding than simple correlation, but it still has limitations.

Important ones are:

- it depends on correct model alignment between the two data sources
- it only measures cross-over correspondence, not causal influence
- results depend on the chosen functional metric
- results depend on the chosen weight statistic scope and aggregation
- same-pair overlap can be low even when overall instability timing is similar

Also:

- if a metric has too few valid models or epochs, it is skipped
- some weight statistics may exist only in a subset of scopes or aggregation levels

## Main Config Knobs

Important configuration options:

- `functional_metric`
  - the functional metric to compare against
- `functional_higher_is_better`
  - direction for the functional metric
- `metric_family`
  - specific family or `all`
- `metric`
  - specific metric or `all`
- `scope`
  - specific scope or `all`
- `include_model_row`
  - whether to prefer model-level `__model__` rows when available
- `pairwise_flip_tau`
  - optional threshold for weight-space flip counting
- `drop_epoch_zero`
  - whether to exclude epoch 0
- `epoch_col`
  - epoch column in the source dataframe
- `model_col`
  - model identifier column in the source dataframe
- `epoch_range`
  - optional epoch restriction for the functional matrix
- `dataset_split`
  - which checkpoint split to use for weight statistics
- `cache_dir`
  - location of the cached weight-statistics tables
- `ranking_flips_tables_dir`
  - location of the functional ranking-flips tables
- `force_recompute`
  - whether to recompute the cached weight-statistics table
- `skip_missing_cache`
  - whether to skip datasets whose cached weight-statistics table is missing instead of computing it
- `top_k_metrics`
  - number of top metrics shown in summary bar plots
- `show_progress`
  - whether to show dataset progress with `tqdm`

## Notes On Legacy Config Keys

The file still contains some older helper functions related to:

- epoch-wise association with rank
- event alignment
- lagged correlations

and some older config keys are still present in the sample YAML, such as:

- `detailed_top_k`
- `crossover_quantile`
- `min_changed_rank`
- `min_event_separation`
- `align_window`
- `pre_window`
- `post_window`
- `max_lag`

The current main implementation in `run_weight_rank_correspondence(...)` is centered on the flip-overlap analysis described above.

So for practical use, the most relevant outputs are the flip, overlap, and summary correspondence plots and tables.

## Summary

The `weight_rank_correspondence` analysis compares weight-space ranking cross-overs to functional ranking cross-overs.

It does this by:

- building weight-statistic trajectories
- counting pairwise flips over epochs
- loading functional flips from `ranking_flips`
- checking whether the same model pairs cross over in both spaces

This makes it a strong tool for asking:

- which weight statistics capture the functionally meaningful ordering of models
- and whether those statistics match not just aggregate instability, but the exact same pairwise cross-over events
