# Hyper Parameter Analysis

This document describes the hyper-parameter analysis implemented in `learning-dynamics/src/nnrun/analyses/hyper_parameter.py`.

The goal of the analysis is to answer a practical question:

- which hyperparameters are associated with better final model performance in a dataset

It is a dataset-level analysis. It does not track training dynamics over time. Instead, it summarizes each model by its final available checkpoint and studies how the hyperparameters of that model relate to its final metric, such as `test_acc` or `test_loss`.

## What The Analysis Does

For each dataset:

1. Select the final available epoch for each model.
2. Detect hyperparameter columns from the dataframe.
3. Build a final table with one row per model and its hyperparameters.
4. Evaluate hyperparameters in three ways:
   - numeric hyperparameters with Spearman correlation
   - categorical hyperparameters with a Kruskal-Wallis test
   - joint multivariate contribution with a random forest and permutation importance

This gives both:

- simple univariate relationships
- a more joint, model-based estimate of feature importance

## Why This Analysis Is Useful

This analysis is useful when you want to understand the hyperparameter landscape of a model zoo.

It helps answer questions like:

- Does learning rate matter more than weight decay?
- Are certain nonlinearities or optimizers consistently better?
- Are the best models concentrated in a specific part of hyperparameter space?
- Are apparent effects still important when considered jointly with other hyperparameters?

It is especially useful for:

- hyperparameter sweeps
- seed sweeps with some configuration variation
- comparing which knobs matter most across datasets

## Data Selection Strategy

The analysis uses the final available epoch per model, not the full trajectory.

That means:

- each model contributes one final performance value
- hyperparameters are treated as static inputs
- the analysis focuses on final outcome, not training speed or intermediate behavior

If `epoch_range` is provided, the dataframe is first restricted to that range and then the latest remaining epoch per model is used.

This matters because the analysis answers:

- "Which hyperparameters are associated with better final performance within the selected epoch window?"

and not:

- "Which hyperparameters change the whole training curve?"

## Which Columns Count As Hyperparameters

Hyperparameter columns are discovered automatically by prefix.

By default, the analysis looks for columns starting with:

- `config.training::`
- `config.model::`
- `config.optim::`
- `config.seed`

Columns can also be selected through `hyper_prefixes` in the config.

Before analysis, the code drops non-informative hyperparameter columns:

- columns that are all missing
- columns that are constant within the dataset

This is important because many sweeps contain fixed settings that should not be interpreted as informative.

## The Three Analysis Components

### 1. Final Table Per Dataset

For each dataset, the analysis first creates a final summary table with:

- dataset name
- model id
- final epoch
- target metric
- all detected hyperparameters

This table is sorted by the target metric:

- descending if `higher_is_better = true`
- ascending otherwise

What this gives you:

- a clean one-row-per-model table
- an easy way to inspect what the best and worst models used
- a direct audit trail for the later statistical summaries

### 2. Numeric Hyperparameters: Spearman Correlation

Numeric hyperparameters are analyzed with Spearman rank correlation against the final target metric.

Why Spearman is used:

- it detects monotonic relationships
- it is less sensitive to nonlinear scaling than Pearson correlation
- it is robust when the relationship is not strictly linear

For each numeric hyperparameter, the analysis stores:

- `feature`
- `n`
- `spearman_rho`
- `abs_spearman_rho`
- `p_value`

How to interpret it:

- positive `spearman_rho`
  - larger hyperparameter values tend to go with larger metric values
- negative `spearman_rho`
  - larger hyperparameter values tend to go with smaller metric values
- large absolute value
  - stronger monotonic relationship
- small `p_value`
  - stronger evidence that the relationship is not random

What you can learn:

- whether larger learning rates tend to help or hurt
- whether weight decay has a systematic effect
- whether dropout strength tracks final performance
- whether some numeric settings matter much more than others

Important caveat:

- this is a univariate analysis
- a strong correlation does not prove the hyperparameter is independently important after controlling for others

### 3. Categorical Hyperparameters: Kruskal-Wallis Test

Categorical hyperparameters are analyzed with a Kruskal-Wallis test across groups.

Examples:

- optimizer type
- nonlinearity
- initialization type
- scheduler type

Why Kruskal-Wallis is used:

- it compares multiple groups
- it is nonparametric
- it does not assume normal distributions

Very small groups are filtered out first with `min_group_size`.

For each categorical hyperparameter, the analysis stores:

- `feature`
- `n`
- `n_levels`
- `kruskal_stat`
- `p_value`
- `best_level_mean`
- `worst_level_mean`
- `range_of_level_means`
- `level_means`

How to interpret it:

- small `p_value`
  - at least one category differs meaningfully from the others
- large `range_of_level_means`
  - the average performance gap between categories is large
- `level_means`
  - lets you see which category values tend to do best or worst

What you can learn:

- whether one optimizer consistently outperforms another
- whether one initialization scheme gives better final models
- whether some categorical choices barely matter at all

Important caveat:

- Kruskal-Wallis tells you that groups differ
- it does not by itself tell you which pairwise comparisons are significant

### 4. Joint Contribution: Random Forest + Permutation Importance

The final part uses all retained hyperparameters together in a supervised model.

Pipeline:

1. Numeric features:
   - median imputation
2. Categorical features:
   - most-frequent imputation
   - one-hot encoding
3. Regress final metric with a `RandomForestRegressor`
4. Measure permutation importance on the fitted data

Why this is useful:

- it evaluates hyperparameters jointly
- it can capture nonlinear effects
- it can capture interactions better than pure univariate summaries

For each feature, the analysis stores:

- `feature`
- `importance_mean`
- `importance_std`

How to interpret it:

- larger `importance_mean`
  - the feature matters more to predictive performance of the forest
- larger `importance_std`
  - the estimate is less stable across permutations

What you can learn:

- which hyperparameters still matter after considering the others jointly
- whether a feature that looked weak univariately becomes important in combination
- whether a feature that looked strong alone may actually be redundant

Important caveats:

- this is predictive importance, not causal importance
- correlated features can share importance
- importance is computed on the fitted data, not a held-out test set

## Which Plots Are Generated

The current implementation generates two plot types per dataset.

### 1. Numeric Hyperparameter Association Plot

File name pattern:

- `{dataset}_hyper_numeric_assoc_plot`

What it shows:

- a horizontal bar plot of the top numeric hyperparameters by absolute Spearman correlation
- the x-axis is `spearman_rho`

How to read it:

- bars to the right indicate a positive association with the final metric
- bars to the left indicate a negative association
- larger magnitude means a stronger monotonic association

What it is good for:

- quickly identifying the most influential numeric knobs in a dataset

### 2. Permutation Importance Plot

File name pattern:

- `{dataset}_hyper_permutation_importance_plot`

What it shows:

- a horizontal bar plot of the top features by permutation importance
- error bars show `importance_std`

How to read it:

- larger bars indicate features that the random forest relied on more
- wide error bars indicate less stable importance estimates

What it is good for:

- seeing which hyperparameters matter most jointly rather than one by one

## Which Tables Are Generated

Per dataset:

- `hyper_final_table__{dataset}`
- `{dataset}_hyper_numeric_assoc`
- `{dataset}_hyper_categorical_assoc`
- `{dataset}_hyper_permutation_importance`

Across all datasets:

- `hyper_final_table_all`
- `hyper_numeric_assoc_all`
- `hyper_categorical_assoc_all`
- `hyper_permutation_importance_all`

These tables are often more informative than the plots because they preserve:

- exact values
- p-values
- counts
- category-level means
- importance uncertainty

## What You Can Learn In Practice

This analysis is particularly helpful for answering:

- Which hyperparameters are worth tuning further?
- Which hyperparameters appear mostly irrelevant in this model zoo?
- Are the same hyperparameters important across datasets or does importance shift?
- Are the strongest effects numeric or categorical?
- Do univariate and multivariate views agree?

Typical workflow:

1. Check the final table to see what the best models used.
2. Look at numeric correlations for simple monotonic effects.
3. Look at categorical tables for discrete choices like optimizer or nonlinearity.
4. Use permutation importance to judge joint relevance.

## Important Limitations

This analysis is descriptive and predictive, not causal.

That means:

- a hyperparameter can look important because it is correlated with another one
- a feature can look weak in isolation but matter jointly
- a feature can look strong only because the sweep sampled it unevenly

Other limitations:

- only the final available epoch is used
- no interaction plots are generated by default
- no pairwise post-hoc significance tests are run for categorical variables
- permutation importance is not cross-validated in the current implementation

So the safest interpretation is:

- use this analysis to identify patterns and candidates for follow-up
- do not treat the output as proof of causal effect

## Main Config Knobs

Important config options used by the implementation:

- `metric`
  - target metric to explain, for example `test_acc`
- `higher_is_better`
  - controls sorting direction of the final table
- `epoch_col`
  - source dataframe epoch column
- `model_col`
  - source dataframe model identifier column
- `epoch_range`
  - optional restriction before selecting the final epoch per model
- `png_dpi`
  - plot resolution
- `hyper_prefixes`
  - optional override for hyperparameter column discovery
- `rf_n_estimators`
  - number of trees in the random forest
- `rf_random_state`
  - random seed for the random forest and permutation importance
- `perm_n_repeats`
  - number of permutation repeats
- `min_group_size`
  - minimum category size for categorical testing

## Summary

The hyper-parameter analysis gives a structured way to understand which configuration choices are associated with model quality in a dataset.

It combines:

- direct inspection of the final models
- univariate numeric associations
- univariate categorical group differences
- multivariate predictive importance

Together, these outputs help you decide:

- which hyperparameters deserve attention
- which ones are likely secondary
- whether the signal is simple and monotonic or only visible jointly
