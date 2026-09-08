# Weight Early Signal Analysis

This document describes the `weight_early_signal` analysis implemented in `learning-dynamics/src/nnrun/analyses/weight_early_signal.py`.

The purpose of the analysis is to test whether early weight statistics contain useful signal about what will happen later in training.

In particular, it asks two related questions:

- can an early weight statistic help identify future late bloomers
- can an early weight statistic help explain how much a model will still improve

So unlike `weight_rank_correspondence`, which studies cross-over agreement, this analysis is explicitly predictive in spirit:

- take a weight statistic at an early checkpoint
- compare it to a future training outcome

That future outcome can be:

- a binary late-bloomer label
- a continuous future-delta target

## Core Idea

For one chosen functional metric such as:

- `test_acc`
- `test_loss`

the analysis first builds a functional trajectory matrix over:

- models
- epochs

Then it constructs future targets from that functional metric.

After that, for each weight-statistic combination:

- `metric_family`
- `metric`
- `scope`

it takes the weight-statistic value at an early epoch and measures how informative it is about the future target.

So the analysis is centered on:

- one functional metric
- many weight statistics
- many early prediction epochs
- future outcome prediction

## What The Analysis Does

For each dataset:

1. Build a `models x epochs` matrix for the chosen functional metric.
2. Choose one intermediate epoch `epoch_mid` and one final epoch `epoch_end`.
3. Define future targets from the functional metric.
4. Load or compute the cached weight-statistics table.
5. Discover the selected weight-statistic combinations.
6. For each `(metric_family, metric, scope)` combination:
   - build one weight-statistic series per model and epoch
   - choose the model-level row if available and allowed
   - otherwise average layer values within each model and epoch
7. For each selected `prediction_epoch`:
   - take the weight statistic at that epoch
   - align model identities between weight space and functional space
   - compare the early weight statistic to the future targets
8. Save summary tables and optional plots.

So each row in the summary tables corresponds to:

- one dataset
- one weight-statistic combination
- one early prediction epoch

## Why This Analysis Is Useful

This analysis is useful when the real goal is not just describing weight-space behavior, but making a training decision.

It helps answer questions like:

- Are some weight statistics already informative very early?
- Are there weight statistics that flag slow starters which later recover?
- Which statistics are predictive only late, and which are useful early?
- Do some statistics correlate with later improvement even after controlling for early functional performance?

This makes it useful for:

- early stopping research
- checkpoint triage
- candidate filtering
- identifying slow-but-promising models

## The Two Future Targets

The analysis studies two different future targets derived from the same functional metric.

### 1. Late Bloomer Target

This is a binary target.

A model is marked as a late bloomer if:

1. it is in the bottom fraction at `epoch_mid`
2. it is in the top fraction at `epoch_end`

The fractions are controlled by:

- `bottom_frac_mid`
- `top_frac_end`

This definition matches the logic used in the dedicated late-bloomers analysis.

### 2. Future Delta Target

This is a continuous target.

For one prediction epoch `t`, the future delta is:

- for accuracy-like metrics where higher is better:
  - `future_delta = value_final - value_t`
- for loss-like metrics where lower is better:
  - `future_delta = value_t - value_final`

So larger `future_delta` always means:

- more room for later improvement from that early checkpoint to the end

This makes the target direction consistent across accuracy and loss.

## How The Functional Side Is Built

The analysis builds a functional metric matrix:

- rows = models
- columns = epochs
- values = the selected functional metric

The config provides:

- `functional_metric`
- `higher_is_better`

The matrix is then used for two things:

1. ranking models at `epoch_mid` and `epoch_end`
2. computing future delta from each `prediction_epoch` to `epoch_end`

If `drop_epoch_zero = true`, epoch `0` is removed before target construction.

If `epoch_range` is set, only epochs inside that range are considered.

## How `epoch_mid` And `prediction_epoch` Work

The analysis uses two different epoch concepts:

### `epoch_mid`

This is the epoch used to define the late-bloomer label.

If the configured `epoch_mid` is not present exactly, the code uses the nearest earlier available epoch.

If no `epoch_mid` is configured, the code uses the middle available epoch.

### `prediction_epoch`

This is the epoch where the weight statistic is observed.

The idea is:

- measure the weight statistic at `prediction_epoch`
- ask whether it tells you something about the future target

So `prediction_epoch` is the actual early observation time.

It is always chosen from available epochs at or before `epoch_mid`.

This ensures the analysis uses:

- earlier information
- to predict later behavior

## How Prediction Epochs Are Selected

Prediction epochs are selected in one of two ways.

### Explicit Selection

If `prediction_epochs` is provided in the config, only those epochs are used, as long as they exist and are not after `epoch_mid`.

### Automatic Selection

If `prediction_epochs` is empty or omitted, the analysis uses all available epochs up to `epoch_mid`.

Then two optional controls can reduce the number:

- `prediction_stride`
  - keep every `k`th prediction epoch
- `max_prediction_epochs`
  - downsample to at most that many epochs

This is useful when you want:

- a dense early scan
- but not an overwhelming number of columns in the plots

## How Weight Statistics Are Built

The weight side comes from the cached weight-statistics table produced by `weight_statistics.py`.

For each selected weight-statistic combination:

- `metric_family`
- `metric`
- `scope`

the analysis builds one value per model and epoch.

Two cases are supported:

### Model Row Available

If model-level `__model__` rows exist and `include_model_row = true`:

- use those directly

This is the cleanest case because the statistic already exists naturally at model level.

### No Model Row Or Model Row Disabled

If no model-level row is used:

- average the layer values within each model and epoch

This gives one model-level summary per checkpoint even for statistics that were originally layer-level.

The summary table records this with:

- `weight_value_source = model_row`
- or `weight_value_source = layer_mean`

## Model Alignment Between Functional And Weight Space

The analysis must align model identities between:

- the functional dataframe
- the cached weight-statistics dataframe

It tries two alignment strategies:

1. `trial_token`
2. `model_key`

### `trial_token`

The code tries to extract an identifier of the form:

- `abc12_34567`

from the model path or name.

### `model_key`

The code also builds a canonicalized basename of the model identifier.

### How The Final Alignment Is Chosen

The analysis computes both alignments and keeps the one with more matched rows.

The summary tables record the chosen strategy in:

- `alignment_key`

This is important because a predictive result is only meaningful if the model identities match correctly.

## How Late-Bloomer Signal Is Calculated

For one dataset, one weight-statistic combination, and one `prediction_epoch`, the analysis takes:

- `x = weight statistic at prediction_epoch`
- `y = late_bloomer label`

It then computes several summary quantities.

### `pearson_weight_vs_late_bloomer`

This is the Pearson correlation between:

- the weight statistic
- the binary label encoded as `0` or `1`

This is effectively a point-biserial style association.

Interpretation:

- positive value
  - larger weight values tend to occur in future late bloomers
- negative value
  - smaller weight values tend to occur in future late bloomers

### `spearman_weight_vs_late_bloomer`

This is the Spearman rank correlation between:

- the weight statistic
- the binary late-bloomer label

Interpretation:

- monotonic association instead of linear association

### `partial_pearson_weight_vs_late_bloomer_given_early_metric`

This is a partial Pearson correlation that controls for:

- `functional_value_pred`

In other words, it asks:

- does the weight statistic still carry signal after accounting for the early functional metric itself

This matters because otherwise a weight statistic may look informative simply because it is redundant with the early functional score.

### `mannwhitney_u_p_value`

This compares the weight-statistic distribution between:

- future late bloomers
- non-bloomers

Interpretation:

- small p-value
  - the two groups are separated in distribution

### `auc_raw`

This is the raw area under the ROC curve using:

- score = weight statistic
- positive class = future late bloomer

The implementation computes this explicitly from ranks.

Let:

- $$s_i$$
  - the weight-statistic score for model `i`
- $$y_i \in {0,1}$$
  - the late-bloomer label for model `i`, where `1` means future late bloomer
- $$n_+ = \sum_i 1[y_i = 1]$$
  - number of positive examples
- $$n_- = \sum_i 1[y_i = 0]$$
  - number of negative examples

After dropping any rows where either `s_i` or `y_i` is not finite, the analysis:

1. ranks the scores `s_i` across all retained models, using average ranks for ties
2. sums the ranks of the positive examples:

$$
R_+ = \sum_{i: y_i = 1} {rank}(s_i)
$$

3. converts that rank sum into the Mann-Whitney `U` statistic:

$$
U = R_+ - \frac{n_+(n_+ + 1)}{2}
$$

4. normalizes by the total number of positive-negative pairs:

$$
\mathrm{auc\_raw} = \frac{U}{n_+ n_-}
$$

So the complete calculation is:

$$
\mathrm{auc\_raw}
= \frac{
\sum_{i: y_i = 1} {rank}(s_i) - \frac{n_+(n_+ + 1)}{2}
}{
n_+ n_-
}
$$

Equivalently, this is the probability that a randomly chosen late bloomer gets a larger score than a randomly chosen non-bloomer, counting ties as half:

$$
\mathrm{auc\_raw}
= P(s_+ > s_-) + \frac{1}{2} P(s_+ = s_-)
$$

Interpretation:

- `0.5`
  - no discrimination
- above `0.5`
  - larger weight values indicate future late bloomers
- below `0.5`
  - smaller weight values indicate future late bloomers

### `auc_best`

This is:

- `max(auc_raw, 1 - auc_raw)`

So it measures separation strength while ignoring sign direction.

Interpretation:

- `1.0`
  - perfect separation in either direction
- `0.5`
  - no separation

### `auc_best_direction`

This records which direction gave the best AUC:

- `+1`
  - larger values indicate future late bloomers
- `-1`
  - smaller values indicate future late bloomers

### Top-Quantile Lift

The analysis also measures a practical screening-style quantity.

It sorts models by the weight statistic in the best direction and takes the top fraction:

- `top_quantile`

Then it computes:

- `base_rate`
  - overall late-bloomer rate
- `top_rate`
  - late-bloomer rate inside the selected top fraction
- `top_rate_lift`
  - `top_rate / base_rate`

Interpretation:

- `top_rate_lift > 1`
  - the selected models are enriched for late bloomers

This is useful because it is closer to a real filtering decision than correlation alone.

## How Future-Delta Signal Is Calculated

For one dataset, one weight-statistic combination, and one `prediction_epoch`, the analysis takes:

- `x = weight statistic at prediction_epoch`
- `y = future_delta`

It then computes:

### `pearson_weight_vs_future_delta`

Linear correlation between the early weight statistic and later improvement.

Interpretation:

- positive value
  - larger weight values tend to predict more later improvement
- negative value
  - smaller weight values tend to predict more later improvement

### `spearman_weight_vs_future_delta`

Monotonic correlation between the early weight statistic and later improvement.

This is often the easier correlation to interpret because it is less sensitive to scale and outliers.

### `partial_pearson_weight_vs_future_delta_given_early_metric`

Partial Pearson correlation controlling for:

- the early functional metric value

So this asks:

- does the weight statistic add information beyond early performance

### Top-Quantile Gain

The analysis evaluates both directions:

- sort by `x`
- sort by `-x`

For each direction it takes the top fraction defined by:

- `top_quantile`

Then it computes:

- `base_mean`
  - average future delta across all aligned models
- `top_mean`
  - average future delta inside the selected top fraction
- `top_minus_base`
  - `top_mean - base_mean`

It keeps whichever direction gives the larger improvement and stores:

- `top_direction`
  - `+1` or `-1`

Interpretation:

- larger `top_minus_base`
  - the chosen direction of the weight statistic is useful for selecting models with more remaining improvement

## Which Tables Are Generated

The analysis currently writes three tables.

### 1. `weight_early_signal_targets`

This is a dataset-level summary of how targets were built.

Columns include:

- `dataset_name`
- `functional_metric`
- `higher_is_better`
- `reason`
- `n_models`
- `epoch_mid`
- `epoch_end`
- `n_prediction_epochs`
- `prediction_epochs`
- `late_bloomers`
- `bottom_frac_mid`
- `top_frac_end`

What this table tells you:

- which epochs were actually used
- how many models were available
- how many late bloomers exist
- whether a dataset was skipped and why

### 2. `weight_early_signal_late_bloomer_summary`

This is the main table for the binary late-bloomer target.

Columns include:

- `dataset_name`
- `functional_metric`
- `prediction_epoch`
- `epoch_mid`
- `epoch_end`
- `metric_family`
- `metric`
- `scope`
- `combo_label`
- `weight_value_source`
- `alignment_key`
- `n_aligned_models`
- `n_late_bloomers`
- `late_bloomer_rate`
- `pearson_weight_vs_late_bloomer`
- `spearman_weight_vs_late_bloomer`
- `partial_pearson_weight_vs_late_bloomer_given_early_metric`
- `mannwhitney_u_p_value`
- `auc_raw`
- `auc_best`
- `auc_best_direction`
- `top_n`
- `base_rate`
- `top_rate`
- `top_rate_lift`

What this table tells you:

- which weight statistics are predictive of future late blooming
- how early that signal appears
- whether the signal is positive or negative
- whether the signal is practically useful for enrichment

### 3. `weight_early_signal_delta_summary`

This is the main table for the continuous future-delta target.

Columns include:

- `dataset_name`
- `functional_metric`
- `prediction_epoch`
- `epoch_mid`
- `epoch_end`
- `metric_family`
- `metric`
- `scope`
- `combo_label`
- `weight_value_source`
- `alignment_key`
- `n_aligned_models`
- `pearson_weight_vs_future_delta`
- `spearman_weight_vs_future_delta`
- `partial_pearson_weight_vs_future_delta_given_early_metric`
- `top_n`
- `base_mean`
- `top_mean`
- `top_minus_base`
- `top_direction`

What this table tells you:

- which weight statistics are associated with later improvement
- whether the relation is positive or negative
- whether the selected tail of the weight statistic captures more-improving models

## Which Plots Are Generated

The current implementation writes trace plots and heatmaps.

Only the top metrics are plotted, controlled by:

- `plot.top_k_metrics`

This is done to keep the figures readable.

It can also write an optional AUC-detail plot, controlled by:

- `plot.include_auc_detail`
- `plot.auc_detail_top_k`

And an optional Mann-Whitney detail plot, controlled by:

- `plot.include_mannwhitney_detail`
- `plot.mannwhitney_detail_top_k`

## Trace Plots

The trace plots show one line per top weight-statistic combination.

The x-axis is:

- `prediction_epoch`

So each point means:

- use the weight statistic measured at this early checkpoint
- evaluate how informative it is about the future target

For the late-bloomer target, the analysis writes three trace plots:

- `auc_best`
- `pearson_weight_vs_late_bloomer`
- `spearman_weight_vs_late_bloomer`

For the future-delta target, it writes two trace plots:

- `pearson_weight_vs_future_delta`
- `spearman_weight_vs_future_delta`

### 1. Late-Bloomer AUC Trace Plot

The y-axis is:

- `auc_best`

The file name looks like:

- `{dataset}_weight_early_signal_late_bloomer_auc`

What this plot shows:

- how the discriminative power of each weight statistic changes across early epochs

How to read it:

- higher line
  - stronger separation between future late bloomers and non-bloomers
- strong performance far to the left
  - useful very early signal
- strong performance only near `epoch_mid`
  - predictive, but not very early

### 2. Late-Bloomer Correlation Trace Plots

The y-axis is either:

- `pearson_weight_vs_late_bloomer`
- `spearman_weight_vs_late_bloomer`

The file names look like:

- `{dataset}_weight_early_signal_late_bloomer_pearson`
- `{dataset}_weight_early_signal_late_bloomer_spearman`

What these plots show:

- whether larger or smaller weight values tend to be associated with the late-bloomer label
- how that association changes across early epochs

How to read them:

- positive value
  - larger weight values tend to be associated with future late bloomers
- negative value
  - smaller weight values tend to be associated with future late bloomers
- value near zero
  - weak association

These traces are signed, so they show direction directly, unlike `auc_best`, which keeps only separation strength.

### 3. Future-Delta Correlation Trace Plots

The y-axis is either:

- `pearson_weight_vs_future_delta`
- `spearman_weight_vs_future_delta`

The file names look like:

- `{dataset}_weight_early_signal_future_delta_pearson`
- `{dataset}_weight_early_signal_future_delta_spearman`

What this plot shows:

- how strongly each weight statistic correlates with later improvement across early epochs

How to read it:

- positive value
  - larger weight values predict more future improvement
- negative value
  - smaller weight values predict more future improvement
- magnitude near zero
  - weak signal
- large absolute value early
  - strong early monotonic relation

## Heatmaps

The heatmaps show the same information in a dense matrix form.

Rows are:

- top weight-statistic combinations

Columns are:

- `prediction_epoch`

Color shows:

- the selected score value

### 1. Late-Bloomer Heatmap

Three late-bloomer heatmaps are written.

The values are:

- `auc_best`
- `pearson_weight_vs_late_bloomer`
- `spearman_weight_vs_late_bloomer`

The file names look like:

- `{dataset}_weight_early_signal_late_bloomer_heatmap`
- `{dataset}_weight_early_signal_late_bloomer_pearson_heatmap`
- `{dataset}_weight_early_signal_late_bloomer_spearman_heatmap`

What this plot is good for:

- finding metrics that are consistently informative over many early epochs
- spotting whether a metric has one isolated peak or a broad early-signal window

How to read it:

- for `auc_best`
  - larger value means stronger separation in either direction
- for Pearson or Spearman heatmaps
  - red means positive association
  - blue means negative association
  - stronger color means larger absolute correlation

So:

- a strong red row in the correlation heatmaps
  - larger weight values tend to align with late bloomers
- a strong blue row
  - smaller weight values tend to align with late bloomers
- a bright row in the AUC heatmap but blue in the correlation heatmap
  - the metric separates well, but in the negative direction

### 2. Future-Delta Heatmap

Two future-delta heatmaps are written.

The values are:

- `pearson_weight_vs_future_delta`
- absolute Spearman correlation with future delta

The file names look like:

- `{dataset}_weight_early_signal_future_delta_pearson_heatmap`
- `{dataset}_weight_early_signal_future_delta_heatmap`

What this plot is good for:

- quickly identifying which metrics are strongly associated with later improvement
- comparing how that strength evolves over early epochs

How to read it:

- red
  - larger weight values tend to predict more future improvement
- blue
  - smaller weight values tend to predict more future improvement
- stronger color
  - stronger positive or negative relationship

## AUC Detail Plot

If `plot.include_auc_detail = true`, the analysis also writes a dedicated AUC explanation figure for the strongest late-bloomer examples in each dataset.

The file name looks like:

- `{dataset}_weight_early_signal_auc_detail_{k}`

Each figure has two panels.

### Left Panel: Rank View

This panel visualizes the exact quantities used by the rank-based AUC computation.

- x-axis
  - average rank of the weight statistic
- y-axis
  - raw weight-statistic value
- color
  - late bloomer vs non-bloomer

The annotation box reports:

- `R+`
  - sum of ranks for the late bloomers
- `U`
  - Mann-Whitney statistic
- `AUC_raw`
- `AUC_best`
- the best direction

How to read it:

- if late-bloomer points tend to occupy high ranks
  - `AUC_raw > 0.5`
- if late-bloomer points tend to occupy low ranks
  - `AUC_raw < 0.5`
- if the two colors are heavily mixed through the rank axis
  - AUC is near `0.5`

### Right Panel: ROC View

This panel shows the ROC curve for:

- the raw weight statistic
- the best-direction version used by `auc_best`

For one chosen `prediction_epoch`, the analysis treats the weight statistic as a classifier score for the binary target:

- positive class
  - future late bloomer
- negative class
  - non-bloomer

It then sweeps a decision threshold over the score and computes, at each threshold:

- true positive rate
  - fraction of late bloomers correctly selected
- false positive rate
  - fraction of non-bloomers incorrectly selected

Plotting:

- x-axis
  - false positive rate
- y-axis
  - true positive rate

So each curve shows the tradeoff between:

- catching more late bloomers
- accidentally selecting more non-bloomers

The ROC curve is another view of the same ranking information used by AUC:

- if late bloomers tend to receive better scores than non-bloomers
  - the curve bows toward the upper-left corner
- if the score is uninformative
  - the curve stays close to the diagonal
- if the score points in the wrong direction
  - the raw curve can fall below the useful orientation

How to read it:

- blue curve
  - discrimination from the raw score direction
- dashed green curve
  - the same score after flipping sign when that improves separation
- diagonal line
  - random baseline

Why there are two curves:

- `auc_raw` uses the weight statistic exactly as it is
- `auc_best` allows the analysis to reverse the sign when smaller values, rather than larger values, indicate future late bloomers

So:

- if the blue and green curves almost overlap
  - the raw score already points in the correct direction
- if the green curve is much better than the blue curve
  - the score is informative, but its useful direction is inverted

This figure is the most direct way to understand:

- why `auc_raw` is above or below `0.5`
- why `auc_best` may be high even when the raw direction is inverted

## Mann-Whitney Detail Plot

If `plot.include_mannwhitney_detail = true`, the analysis also writes a dedicated distribution plot for the strongest Mann-Whitney examples in each dataset.

The file name looks like:

- `{dataset}_weight_early_signal_mannwhitney_detail_{k}`

This plot is important because the Mann-Whitney p-value is only a scalar test summary.

By itself, the p-value does not show:

- whether the two groups are cleanly separated or only slightly shifted
- whether the result is driven by a few outliers
- how the direction of the effect relates to `auc_raw`

Each figure has two panels.

### Left Panel: Group Distributions

This panel shows:

- boxplots for late bloomers and non-bloomers
- overlaid individual model points

The annotation box reports:

- `MW p`
- `U`
- `AUC_raw`
- the two group medians

How to read it:

- clearly shifted boxes with limited overlap
  - typically small p-value and strong separation
- similar boxes with heavy overlap
  - typically larger p-value
- a visible shift but many points
  - p-value may be small even if the effect size is modest

### Right Panel: Empirical CDF

This panel shows the empirical cumulative distribution for both groups.

How to read it:

- one CDF consistently to the left or right of the other
  - broad stochastic dominance
- many crossings
  - weaker or less stable separation

This plot complements the AUC-detail plot:

- AUC emphasizes pairwise ranking discrimination
- Mann-Whitney emphasizes whether the group distributions differ overall

## Example

Suppose the analysis is run with:

- `functional_metric: test_acc`
- `epoch_mid: 15`
- `bottom_frac_mid: 0.30`
- `top_frac_end: 0.20`

and one of the plotted weight statistics is:

- `distance/distance_to_init_l2/default`

Then one point in the late-bloomer trace at:

- `prediction_epoch = 5`

means:

1. take the value of `distance_to_init_l2` at epoch `5`
2. align that value to the functional metric trajectories of the same models
3. label each model as late bloomer or non-bloomer based on:
   - rank at `epoch_mid = 15`
   - rank at `epoch_end`
4. compute `auc_best`

If the resulting value is:

- `auc_best = 0.82`

then the interpretation is:

- this weight statistic at epoch `5` separates future late bloomers from non-bloomers fairly well

If additionally:

- `auc_best_direction = -1`

then the meaning is:

- smaller values of that weight statistic are associated with future late bloomers

Similarly, for the future-delta trace at:

- `prediction_epoch = 5`

if:

- `spearman_weight_vs_future_delta = 0.38`

then the interpretation is:

- models with larger values of that weight statistic at epoch `5` tend to improve more by the end

## What The Analysis Brings

This analysis brings a practical early-signal view of weight statistics.

It moves beyond:

- descriptive weight-space trends
- or same-epoch correspondence

and asks:

- can a weight statistic observed early tell us something useful about the future

That is useful because the eventual decision problem is often:

- should this model keep training
- should this run be discarded
- are we throwing away slow starters too soon

## What You Can Learn From It

This analysis can teach you:

- whether any weight statistics are useful very early
- whether some signals are only visible later
- whether a signal is mostly redundant with early functional performance
- whether weight statistics are better at detecting late bloomers or later improvement
- whether a signal is strong but unstable across epochs

Typical interpretation patterns:

- high `auc_best` early and across many epochs
  - promising late-bloomer detector
- strong delta correlation but weak late-bloomer AUC
  - the statistic tracks continuous improvement better than rare binary recovery
- high raw correlation but weak partial correlation
  - the statistic may mostly duplicate the early functional metric
- strong heatmap only near `epoch_mid`
  - signal exists, but it may be too late for early triage

## Important Limitations

This analysis is still observational.

Important limitations:

- it does not prove causality
- it depends on correct model alignment between weight and functional sources
- late bloomers may be rare, so some AUC estimates can be noisy
- some signals may depend strongly on the chosen `epoch_mid`
- model-level averaging of layer statistics may hide important structure

Also:

- a strong correlation does not automatically imply a good decision rule
- `auc_best` hides direction, so the sign must be checked separately
- absolute Spearman heatmaps hide direction and should be interpreted together with the tables

## Main Config Knobs

Important configuration options include:

- `functional_metric`
- `higher_is_better`
- `epoch_mid`
- `bottom_frac_mid`
- `top_frac_end`
- `prediction_epochs`
- `prediction_stride`
- `max_prediction_epochs`
- `metric_family`
- `metric`
- `scope`
- `include_model_row`
- `top_quantile`
- `drop_epoch_zero`
- `plot.enabled`
- `plot.top_k_metrics`
- `plot.include_traces`
- `plot.include_heatmaps`
- `plot.include_auc_detail`
- `plot.auc_detail_top_k`
- `plot.include_mannwhitney_detail`
- `plot.mannwhitney_detail_top_k`

## Summary

The `weight_early_signal` analysis is a dataset-level early-prediction analysis for weight statistics.

It provides:

- future-target construction from a functional metric
- early-epoch screening of many weight statistics
- binary late-bloomer signal summaries
- continuous future-delta signal summaries
- trace plots across prediction epochs
- heatmaps of the strongest metrics

This makes it useful for understanding:

- whether early weight-space information can help predict future outcome
- which statistics are informative early enough to matter
- whether some signals are robust across epochs rather than isolated peaks
