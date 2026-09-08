# Metric Forecasting Overview

`metric-prediction` also contains a separate forecasting workflow for predicting future metric values from checkpoint trajectories. Unlike the standard metric-prediction pipeline, which treats each sample independently, forecasting builds sequence-aware samples from the same prepared dataset and predicts what happens later in training.

## Goal

The forecasting pipeline starts from a prepared wide-format parquet dataset and converts checkpoint histories into supervised regression samples.

Each sample represents a model state at epoch `t`:

- input: current weight-statistics features, optional lagged history, optional deltas, and optional time features
- target: a future metric derived from later checkpoints of the same model

The default target column is `test_acc`, but it is configurable through `data.target_column`.
You can also exclude whole trajectories whose final target value is too low with
`data.target_model_min_threshold`; for example, `0.1` removes models whose final
`test_acc` is `<= 0.1`.

Current forecasting backend:

- `xgboost`
- `mlp`

## Data Contract

Forecasting reuses the same prepared parquet artifact as `metric-prediction/runner.py`, but it requires enough metadata to reconstruct trajectories.

Required metadata columns:

- `model_id`
- `epoch`
- `path`

The prepared dataset must still contain:

- one row per checkpoint sample
- a unique `sample_idx`
- the selected target metric column
- feature columns derived from model-level and layer-level statistics

Forecasting also assumes there is only one row per `(model_id, epoch)` pair. Duplicate checkpoints for the same model and epoch are rejected.

## Forecast Sample Construction

Forecast samples are built per model trajectory after sorting rows by `model_id`, `epoch`, and `sample_idx`.

For each valid checkpoint, the design matrix can include:

- current features with suffix `__lag_0`
- lagged copies of features up to `sequence.max_lag`
- delta features such as `x_t - x_(t-k)` when `sequence.add_delta_features=true`
- time features when `sequence.include_time_features=true`:
  - `epoch`
  - `history_length`
  - `max_epoch`
  - `epochs_remaining`
  - `epoch_gap_prev`
  - `progress_fraction`
- optionally the current target value when `sequence.include_current_target=true`

Set `sequence.include_time_features=false` to remove those automatically added
time/progress columns from the forecast model input.

Rows are filtered out when they do not satisfy the sequence settings, for example:

- not enough history for `sequence.min_history`
- missing future target for the selected horizon
- terminal checkpoints when `sequence.include_terminal_samples=false`

## Target Modes

The forecasting pipeline supports two target constructions:

- `final_value`: predict the final value of the chosen metric for the trajectory
- `horizon_value`: predict the value `h` checkpoints ahead using `sequence.horizon`

## Split Strategy

Forecasting uses grouped train-validation splitting by `model_id`, not a plain row-wise split.

That prevents leakage across checkpoints from the same trajectory. A model is assigned entirely to either the training split or the validation split.

Default split settings:

- validation split: `0.2`
- random seed: `42`

## Training Flow

The main entry point is `metric-prediction/runner.py`, which uses the unified Hydra config tree under `metric_prediction/config/configs`.

The experiment flow is:

1. Load the prepared dataset and apply the configured feature selection.
2. Build a forecasting dataset from checkpoint trajectories.
3. Split rows by `model_id` with grouped validation sampling.
4. Impute missing values using the configured fill strategy.
5. Train the configured regressor on the sequence-derived feature matrix.
6. Evaluate validation predictions with `rmse`, `mae`, and `r2`.
7. Write run artifacts and resolved configuration.

Default sequence settings:

- target mode: `final_value`
- horizon: `5`
- max lag: `3`
- min history: `1`
- add delta features: `true`
- include time features: `true`
- include current target: `false`
- include terminal samples: `false`

## Outputs

Hydra writes each single run to:

```text
metric-prediction/outputs_forecasting/YYYY-MM-DD/<experiment_name>/HH-MM-SS-ffffff/
```

For sequential batch runs, use `metric-predict-batch` with `mode: forecasting` in `common_overrides`.
When `batch_name` is set, runs are grouped under:

```text
metric-prediction/outputs_forecasting/batch_experiments/<batch_name>/results/<experiment_name>/
```

Typical artifacts include:

- `metrics.json`
- `resolved_config.json`
- `used_features.json`
- `validation_predictions.parquet`
- `forecast_feature_matrix.parquet`
- `feature_importances.parquet`
- `model_bundle.pkl`

The metrics payload also records forecasting-specific context such as:

- `target_mode`
- `target_column`
- `horizon`
- `max_lag`
- `min_history`
- number of training and validation models

## Typical Commands

Predict the final value of `test_acc` from checkpoint history:

```bash
uv run metric-predict-forecast \
  experiment_name=forecast_final_value \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  sequence.target_mode=final_value \
  sequence.max_lag=3
```

Predict the metric value five checkpoints ahead:

```bash
uv run metric-predict-forecast \
  experiment_name=forecast_tplus5 \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  sequence.max_lag=3
```

Restrict forecasting to one feature family:

```bash
uv run metric-predict-forecast \
  experiment_name=forecast_distribution_only \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  'features.metric_families=[distribution]' \
  sequence.target_mode=final_value
```

Use the MLP backend instead of XGBoost:

```bash
uv run metric-predict-forecast \
  experiment_name=forecast_mlp \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  model=mlp \
  model.params.hidden_layer_sizes='[256,128]' \
  model.params.max_iter=300
```

## Summary

In practice, metric forecasting is the trajectory-aware extension of the metric-prediction package. It reuses the prepared weight-statistics dataset, converts checkpoint histories into tabular sequence features, and trains a configurable regressor to estimate future metric behavior rather than only the value observed at the current checkpoint.
