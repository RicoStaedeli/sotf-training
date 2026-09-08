# Late Bloomers Analysis

This document describes the late-bloomers analysis implemented in `learning-dynamics/src/nnrun/analyses/late_bloomers.py`.

The purpose of the analysis is to identify models that look weak at a chosen intermediate epoch but end up among the best models by the end of training.

In other words, it answers:

- which models start in the lower part of the ranking and still finish in the top part later on

## Core Idea

A late bloomer is defined by two rank conditions:

1. At a chosen middle epoch `epoch_mid`, the model is in the bottom fraction of the ranking.
2. At the final available epoch, the model is in the top fraction of the ranking.

The code uses the rank convention:

- rank `1` = best

This is handled correctly for both:

- accuracy metrics, where higher is better
- loss metrics, where lower is better

So the analysis is about rank movement over training, not about raw metric differences alone.

## What The Analysis Does

For each dataset and each configured metric:

1. Build a `models x epochs` metric matrix.
2. Rank models at every epoch.
3. Read the rank at `epoch_mid`.
4. Read the rank at the last available epoch.
5. Mark models as late bloomers if they move:
   - from the bottom `bottom_frac_mid` at `epoch_mid`
   - into the top `top_frac_end` at the final epoch

It then produces:

- a detailed table of all identified late bloomers
- a summary table with counts per dataset and metric
- an optional summary-table figure

## Why This Analysis Is Useful

This analysis is useful when you care about model trajectories rather than just final outcome.

It helps answer questions like:

- Are some models slow starters that become very strong later?
- Do early rankings underestimate eventual winners?
- Is a dataset or metric especially prone to late improvement?
- Are we discarding promising models too early during training?

It is useful for:

- studying training dynamics
- early stopping decisions
- checkpoint selection strategy
- understanding whether intermediate performance is a reliable indicator of final quality

## How The Ranking Rule Works

The analysis uses percent-based thresholds.

Let:

- `n_models` be the number of models in a dataset
- `bottom_frac_mid` be the fraction considered “bottom” at the middle epoch
- `top_frac_end` be the fraction considered “top” at the end

Then the code defines:

- the bottom threshold at `epoch_mid`
- the top threshold at the final epoch

Operationally:

- a model is considered bad enough at `epoch_mid` if its rank is near the bottom according to `bottom_frac_mid`
- a model is considered good enough at the end if its rank is within the best `top_frac_end`

Example:

- if `n_models = 100`
- `bottom_frac_mid = 0.25`
- `top_frac_end = 0.25`

then a late bloomer is:

- in the bottom 25 models at `epoch_mid`
- and in the top 25 models at the final epoch

This is a strict definition. A model must satisfy both conditions.

## What The Analysis Brings

The analysis gives a focused way to detect surprising improvement.

Instead of asking:

- which models are best at the end

it asks:

- which models looked weak at a meaningful intermediate time but still recovered strongly

That brings several benefits:

- it highlights models whose potential is not visible early
- it reveals instability in intermediate rankings
- it shows whether “bad early” really means “bad final”
- it helps compare datasets by how much reordering happens over time

## What You Can Learn From It

This analysis can teach you:

- whether early performance is predictive of final performance
- whether some metrics have many slow starters
- whether a specific dataset has many models that recover late
- whether training should be allowed to run longer before making decisions

Interpretation examples:

- many late bloomers
  - early ranks are not reliable
  - aggressive early stopping may be risky
- very few late bloomers
  - early poor performers usually stay poor
  - early ranking is more stable and informative
- late bloomers for accuracy but not for loss
  - different metrics may reflect different parts of learning dynamics

## Data Selection Strategy

The analysis uses one fixed intermediate epoch and one final epoch.

Specifically:

- `epoch_mid` comes from the config
- `epoch_end` is the maximum available epoch in the metric matrix

If `epoch_range` is set, only epochs inside that range are considered before `epoch_end` is determined.

This means the analysis compares:

- one chosen “mid-training” snapshot
- one final available snapshot

It does not inspect all intermediate rank changes in between.

## Which Metrics It Supports

The analysis accepts a `metrics` list in the config.

Each metric entry can specify:

- `name`
- `higher_is_better`

Typical examples:

- `train_loss`
- `validation_loss`
- `test_loss`
- `train_acc`
- `validation_acc`
- `test_acc`

This matters because the ranking direction must be correct:

- for accuracy, larger values should get better ranks
- for loss, smaller values should get better ranks

## Which Tables Are Generated

The implementation currently writes two tables.

### 1. `late_bloomers_all`

This is the detailed table of all identified late-bloomer models.

Columns include:

- `dataset_name`
- `model_name`
- `metric`
- `rank_epoch_mid`
- `rank_epoch_end`
- `value_epoch_mid`
- `value_epoch_end`
- `epoch_mid`
- `epoch_end`

What this table tells you:

- exactly which models qualified
- how large their rank improvement was
- how their raw metric value changed between the two checkpoints

This is the most useful table if you want to inspect specific models.

### 2. `late_bloomers_summary`

This is the per-dataset, per-metric summary.

Columns include:

- `dataset_name`
- `metric`
- `n_models`
- `late_bloomers`
- `bottom_frac_mid`
- `top_frac_end`
- `epoch_mid`
- `epoch_end`
- `reason`

What this table tells you:

- how many late bloomers were found in each dataset and metric
- whether the analysis succeeded normally or failed because data was missing

The `reason` column is especially useful for debugging. Examples:

- `ok`
- `no data or missing epoch_mid`
- `no models`

## Which Plots Are Generated

The current implementation generates one optional figure.

### Summary Table Figure

File name pattern:

- `late_bloomers__summary_table__mid{epoch_mid}__last_epoch`

What it is:

- a rendered table figure showing the summary results

By default it shows columns such as:

- `dataset_name`
- `metric`
- `n_models`
- `late_bloomers`

How to read it:

- each row is one dataset-metric combination
- the `late_bloomers` column tells you how many models satisfied the definition
- comparing rows across metrics tells you whether the phenomenon is metric-specific
- comparing rows across datasets tells you whether some datasets are more rank-unstable than others

This is not a curve or scatter plot. It is a formatted summary table intended for PDF and PNG output.

## How To Interpret The Output

### If `late_bloomers` Is Large

This means a substantial number of models:

- were ranked poorly at `epoch_mid`
- but still ended in the top group later

Interpretation:

- early ranking is not stable
- models need more training to reveal their final quality
- early stopping based on mid-training rank may discard good models

### If `late_bloomers` Is Near Zero

This means few or no models make that transition.

Interpretation:

- mid-training ranking is already close to final ranking
- poor early performers rarely recover enough to become top models

### If Results Differ Strongly Across Metrics

This means different metrics reflect different aspects of progress.

Example:

- a model may look poor under one metric at `epoch_mid` but not under another
- or it may improve in accuracy rankings more dramatically than in loss rankings

So it is important to interpret late bloomers in the context of the chosen metric.

## Important Limitations

This analysis is intentionally simple and strict.

Important limitations:

- it compares only two time points: `epoch_mid` and final epoch
- it does not measure continuous improvement over all epochs
- it does not quantify how much a model improved, only whether it crossed the chosen thresholds
- results depend strongly on the chosen `epoch_mid`
- results also depend on the chosen fractions `bottom_frac_mid` and `top_frac_end`

This means:

- changing the thresholds can change the count substantially
- the analysis is best used comparatively, not as an absolute truth

## Main Config Knobs

Important configuration options:

- `metrics`
  - list of metrics to analyze, each optionally with `higher_is_better`
- `epoch_mid`
  - the epoch at which the model must still be in the bottom group
- `bottom_frac_mid`
  - bottom fraction used at `epoch_mid`
- `top_frac_end`
  - top fraction used at the final epoch
- `epoch_range`
  - optional restriction on which epochs are considered
- `plot_summary_table`
  - whether to render the summary table as a figure
- `summary_columns`
  - which columns to show in the summary table figure
- `summary_max_rows`
  - maximum number of rows displayed in the figure
- `png_dpi`
  - output resolution for the figure

## Typical Questions This Analysis Can Answer

- Are there genuine slow starters in this model zoo?
- Is early stopping likely to remove future top performers?
- Which datasets have the most surprising recoveries?
- Which metrics are most vulnerable to early misranking?
- Is rank at epoch 15 already meaningful, or still noisy?

## Summary

The late-bloomers analysis is a simple rank-based screen for delayed success.

It identifies models that:

- are in the lower part of the ranking at a chosen middle epoch
- but end in the upper part of the ranking at the final epoch

This helps you understand:

- whether early rankings are trustworthy
- whether good final models sometimes emerge late
- how much patience is needed before judging model quality
