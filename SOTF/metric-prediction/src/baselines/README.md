# Learning-Curve Baselines

This directory contains learning-curve-only baselines used to compare learned
metric predictors against approaches that only see the observed accuracy curve.
The first implementation is
inspired by parametric learning-curve extrapolation methods: it fits simple
curve families to the observed prefix of a model's validation/test metric curve
and extrapolates that fitted curve to the requested future epoch.

The second implementation is an LCNet-style Bayesian neural-network predictor
following Klein et al. (2017): it trains a neural network across completed
learning curves and uses Monte Carlo dropout to approximate predictive
uncertainty. Both baselines intentionally avoid weight statistics, latent
checkpoint representations, and architecture features.

## Files

- `learning_curve_extrapolation.py`: core extrapolation logic and curve-family definitions.
- `run_learning_curve_extrapolation.py`: command-line runner for one dataset or a multi-dataset YAML.
- `configs/extrapolation.yaml`: default all-dataset configuration for the thesis metric-prediction experiments.
- `lcnet.py`: LCNet-style prefix-feature extraction and MC-dropout neural regressor.
- `run_lcnet_baseline.py`: command-line runner for the LCNet-style baseline.
- `configs/lcnet.yaml`: default all-dataset LCNet configuration.

## How To Run

Run all configured datasets and target modes:

```bash
uv run python metric-prediction/src/baselines/run_learning_curve_extrapolation.py -cn extrapolation.yaml
```

Run the LCNet-style Bayesian neural-network baseline:

```bash
uv run python metric-prediction/src/baselines/run_lcnet_baseline.py -cn lcnet.yaml
```

Inspect the resolved runs without fitting curves:

```bash
uv run python metric-prediction/src/baselines/run_learning_curve_extrapolation.py -cn extrapolation.yaml --dry-run
uv run python metric-prediction/src/baselines/run_lcnet_baseline.py -cn lcnet.yaml --dry-run
```

Run a single existing forecasting batch config:

```bash
uv run python metric-prediction/src/baselines/run_learning_curve_extrapolation.py \
  --batch-config metric-prediction/src/config/configs/experiments/forecasting/weight_statistics/final_value/xgboost/ds5.yaml \
  --min-history 4
```

## Prediction Task

For every eligible validation checkpoint, the runner builds the same forecasting
targets as the learned predictors:

- `final_value`: predict the model's final target value at its maximum available epoch.
- `horizon_value`: predict the target value `horizon` epochs after the current checkpoint.

The default YAML evaluates both:

```yaml
target_modes:
  - target_mode: final_value
    horizon: 5
    min_history: 4

  - target_mode: horizon_value
    horizon: 5
    min_history: 4
```

For `final_value`, `horizon` is stored for compatibility but the target epoch is
the model's `max_epoch`. For `horizon_value`, the target epoch is
`current_epoch + horizon`.

## Input Data

The default config points to pre-split weight-statistic `.pt` datasets:

```yaml
datasets:
  - name: ds1
    plot_name: cifar10_cnn_large_uniform
    dataset_path: metric-prediction/cache/weight_statistic_datasets/1_cifar10_uniform_large_full_all_metrics_long
```

For each `dataset_path`, the runner expects:

- `weight_statistics_train.pt`
- `weight_statistics_val.pt`

The runner uses these files only to recover metadata, target values, and the
same train/validation split used by the learned predictors. Feature matrices are
loaded through the existing dataset loader, but feature selection is set to zero
weight-statistic features:

```python
FeatureSelectionConfig(
    include_model_metrics=False,
    include_layer_metrics=False,
)
```

This keeps the baseline learning-curve-only.

## Observed History

For a validation checkpoint `(model_id, epoch)`, the baseline constructs the
observed prefix:

```text
all rows for the same model_id with epoch <= current epoch
```

Only the configured `target_column`, usually `test_acc`, is used. Rows with
non-finite epochs or target values are dropped. Duplicate epochs are reduced to
the first occurrence after stable epoch sorting.

The epoch axis is shifted before fitting:

```text
x = epoch - first_observed_epoch + 1
x_target = target_epoch - first_observed_epoch + 1
```

This avoids zero or negative values for logarithmic and power-law curves.

## Curve Families

For each checkpoint, the baseline fits a small ensemble of curve families with
`scipy.optimize.curve_fit`:

| Name | Formula | Minimum points |
| --- | --- | --- |
| `linear` | `a * x + b` | 2 |
| `log` | `a * log(x) + b` | 2 |
| `exponential` | `c - a * exp(-b * x)` | 3 |
| `power` | `c - a * x^(-b)` | 3 |

The actual minimum history is the maximum of the global `curve_min_points`
setting and the curve family's own minimum. The default `curve_min_points` is
`3`, and the all-dataset YAML uses `min_history: 4` so predictions are made only
after enough observed checkpoints are available.

## Fitting And Aggregation

Each curve family is fit independently to the observed prefix. Failed fits are
ignored for that checkpoint. A fit can fail because optimization does not
converge, parameters leave their bounds, or the prediction is non-finite.

The final prediction is:

```text
median(successful curve-family predictions)
```

The median is used instead of the mean to reduce the impact of one unstable
extrapolation.

## Bounds And Clipping

Predictions are clipped to dataset-level target bounds inferred from the full
train and validation target distribution.

For accuracy targets:

- lower bound is at least `0`
- upper bound is at most `1` when the observed scale looks like `[0, 1]`
- upper bound is at most `100` when the observed scale looks like percentages

For loss targets:

- lower bound is at least `0`

Otherwise, bounds are based on the observed target range plus a 10 percent
margin. The chosen bounds are written into `metrics.json` as:

```json
"target_lower_bound": ...,
"target_upper_bound": ...
```

## Fallback Behavior

The baseline falls back to the current observed target value when no robust
curve extrapolation is available. This is the same persistence behavior as a
current-value baseline, but it is used only as a failure mode.

Fallback reasons are:

- `fallback_too_few_points`: the observed prefix is shorter than `curve_min_points`.
- `fallback_invalid_target_epoch`: the target epoch cannot be represented on the shifted epoch axis.
- `fallback_fit_failed`: no curve family produced a finite prediction.

The validation prediction file records this per row:

```text
baseline_method
n_curve_predictions
target_epoch
```

`metrics.json` also records:

```json
"fallback_fraction": ...
```

This is the fraction of validation checkpoints where no curve family contributed
and the current target value was copied.

## LCNet-Style Baseline

The LCNet-style baseline learns across models rather than fitting one curve at a
time. For every eligible checkpoint, it converts the observed prefix into a
fixed-length feature vector:

- scalar progress features such as current epoch, target epoch, maximum epoch,
  progress fraction, and remaining epochs;
- prefix summary statistics such as current value, first value, mean, standard
  deviation, total gain, recent slope, and recent deltas;
- the most recent `max_history_points` observed values, epochs, value deltas,
  and mask indicators for missing lags.

Only rows with `epoch <= current epoch` are used for the prefix. Future
checkpoint values from the same model are never included in the features.

The regressor is a small MLP with dropout. Training uses the same train split as
the learned predictors. At inference time, dropout remains active for
`mc_samples` forward passes. The mean is used as the point prediction and the
sample standard deviation is written as `prediction_std`.

The default YAML uses:

```yaml
max_history_points: 8
hidden_layer_sizes: [128, 128]
dropout: 0.1
max_epochs: 120
patience: 15
mc_samples: 30
```

baseline and writes `training_history.json` in addition to the standard
artifacts.

## Outputs

For each dataset and target mode, outputs are written to:

```text
metric-prediction/outputs/baselines/learning_curve_extrapolation/<target_mode>/<dataset>/results/<experiment>/artifacts
```

LCNet outputs use the same layout under:

```text
metric-prediction/outputs/baselines/lcnet/<target_mode>/<dataset>/results/<experiment>/artifacts
```

Example:

```text
metric-prediction/outputs/baselines/learning_curve_extrapolation/horizon_value/ds1/results/ds1_learning_curve_horizon_value/artifacts
```

The output directory is stable and has no timestamp. Rerunning the same
configuration overwrites the previous artifacts for that run.

Each run writes:

- `metrics.json`: RMSE, MAE, R^2, sample counts, target settings, target bounds, and fallback fraction.
- `resolved_config.json`: fully resolved configuration for that run.
- `validation_predictions.parquet`: validation metadata, target, prediction, residual, curve method metadata.

The multi-run command additionally writes a timestamped CSV summary under:

```text
metric-prediction/outputs/baselines/learning_curve_extrapolation/summaries
```

The summary file is timestamped so full batch histories are not overwritten.

For LCNet, the multi-run summary is stable and overwritten on rerun:

```text
metric-prediction/outputs/baselines/lcnet/summaries/lcnet.csv
```

## Compatibility With Existing Evaluation

The artifacts are intentionally shaped like the forecasting experiment outputs.
The existing loader can read them:

```python
from pathlib import Path
from evaluation.forecasting import load_forecast_runs

df = load_forecast_runs(
    Path("metric-prediction/outputs/baselines/learning_curve_extrapolation")
)
```

The `model_type` resolves to:

```text
learning_curve_extrapolation
```

This makes the baseline usable in the same comparison and plotting scripts as
the learned predictors, once those scripts include the baseline output root.

## Differences from Klein et al. (2017)

The implementation in `lcnet.py` is inspired by the LC-BNN / SGHMC-LC model from Klein et al.
(2017, "Learning Curve Prediction with Bayesian Neural Networks", ICLR) but departs from it in
several design dimensions. The table below lists the paper's design choices alongside what was
actually implemented.

| Dimension | Paper (SGHMC-LC) | This implementation |
| --- | --- | --- |
| **Uncertainty source** | Bayesian posterior sampling via SGHMC or SGLD — M weight samples W¹…Wᴹ are drawn from p(W\|D); variance across forward passes gives epistemic uncertainty | Monte Carlo dropout — dropout stays active at inference for `mc_samples` forward passes; the empirical standard deviation approximates predictive uncertainty |
| **Network architecture** | Specialized learning-curve layer on top of shared hidden layers; jointly outputs (1) asymptotic value µ̂∞, (2) basis-function parameters Θ, (3) basis-function weights **w**, (4) heteroscedastic noise σ̂² | Plain MLP ending in a single scalar regression head; no basis-function layer, no asymptotic head, no noise head |
| **Parametric basis functions** | 5 curve families: vapor pressure, pow, log power, exponential, hill-3; the network predicts their parameters, and the weighted sum of the evaluated families is the curve prediction | None — the network maps directly from input features to a target value |
| **Input representation** | Raw tuple (x, t) where x ∈ ℝᵈ is the hyperparameter configuration vector and t ∈ (0, 1] is the normalized time step; one training row per (model, epoch) pair | Hand-crafted fixed-length prefix feature vector: scalar progress features (current epoch, target epoch, remaining epochs, progress fractions), prefix summary statistics (min, max, mean, std, slope, delta statistics), and the most recent `max_history_points` observed values with epoch and mask indicators |
| **Prediction task** | Predicts g(x, t) at any t jointly across all configurations; supports both extrapolation and predicting unobserved curves for new configurations | Predicts a single scalar target per checkpoint: either `horizon_value` (value at current_epoch + horizon) or `final_value` (value at max_epoch) |
| **Training algorithm** | MCMC — no explicit optimizer; SGHMC or SGLD move through weight space sampling from the posterior; no learning-rate scheduler or early stopping | Adam with ReduceLROnPlateau scheduler, gradient clipping, and early stopping on a held-out validation split |
| **Loss function** | Negative log-likelihood under a Gaussian observation model (used as part of the MCMC posterior) | Smooth L1 (Huber) loss |
| **Default architecture** | 3 hidden layers × 64 units × tanh activations | 2 hidden layers × 128 units × ReLU activations (configurable) |
| **Heteroscedastic noise** | Explicitly modelled: the network predicts a per-configuration noise σ̂² alongside the curve | Not modelled; MC-dropout std is used as a proxy for uncertainty but is not calibrated per-model |

### Rationale for the divergences

The original SGHMC/SGLD training is computationally expensive and difficult to integrate with the
existing PyTorch training infrastructure used by the other predictors in this project. MC dropout
provides a practical approximation to Bayesian uncertainty at a fraction of the cost and requires
only a standard Adam training loop.

The specialized basis-function layer is also omitted. The thesis task is a fixed-horizon regression
problem (predict a scalar at a known future epoch), not full learning-curve extrapolation to any t.
A plain MLP is sufficient to test whether the learning-curve prefix adds predictive signal over
weight-statistic features, which is the comparison goal.

The hand-crafted prefix feature vector replaces the raw (x, t) input because this implementation
does not have access to the hyperparameter configuration vector x. Models in the thesis model zoo
are characterized by their weight statistics, not their training hyperparameters. Prefix summary
features encode the same local information (trend, curvature, recency) without requiring
hyperparameter metadata.

## Important Limitations

This is a deterministic classical baseline, not a learned cross-model
performance predictor. It does not learn dataset-specific dynamics from other
models beyond using global target bounds. For each validation checkpoint, it
fits only the current model's observed curve prefix.

This means it is strongest when early learning curves are smooth and
informative. It can be weak when:

- the first few checkpoints are flat or noisy;
- the metric changes sharply after the observed prefix;
- the final target variance across models is very small;
- checkpoints are sparse enough that few points are available before the target.

Those limitations are useful for the thesis comparison: the baseline tests
whether weight statistics or latent representations add information beyond a
standard learning-curve-only extrapolation.
