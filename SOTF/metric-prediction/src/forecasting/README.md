# Forecasting

This package adds a separate forecasting workflow on top of the existing prepared
dataset used by `metric_prediction`.

It does not replace the current `metric-prediction/runner.py` pipeline. The original
checkpoint-level regression flow remains unchanged. Forecasting is implemented as
an additional path for predicting future metric values from checkpoint
trajectories.

## Goal

Given a model trajectory with checkpoints from epoch `0` to the last available
epoch, build supervised samples of the form:

- input: statistics observed up to checkpoint `t`
- target: a future metric derived from later checkpoints

The forecasting path now supports two families of models:

- tabular regressors such as XGBoost on sequence-derived wide features
- a layer-sequence Transformer encoder that reconstructs per-layer tokens from
  the prepared feature metadata and handles variable numbers of layers via
  masking and padding

## Modules

- `config.py`
  - dataclass configuration for forecasting runs
- `dataset.py`
  - converts the prepared checkpoint dataset into forecasting samples
- `split.py`
  - performs grouped train/validation splitting by `hyperparameter_key` when available, otherwise `model_id`
- `pipeline.py`
  - trains the configured forecasting model, evaluates it, and writes artifacts

## Data Construction

The forecasting dataset starts from the prepared wide parquet already used by the
main metric-prediction pipeline.

Each valid row in the forecasting dataset corresponds to one `(model_id, epoch)`
state. The row contains:

- current weight-stat features at epoch `t`
- lagged features from earlier checkpoints
- delta features such as `x_t - x_(t-1)`
- optional time features when `sequence.include_time_features=true`:
  - current epoch
  - history length
  - max epoch for the model
  - epochs remaining
  - epoch gap from the previous checkpoint
  - normalized progress

Set `sequence.include_time_features=false` to exclude these automatically added
time/progress columns from the model input while keeping the internal sequence
filtering logic unchanged.

## Supported Targets

`SequenceFeatureConfig.target_mode` supports:

- `final_value`
  - predict the final metric value of the model
- `horizon_value`
  - predict the metric value `h` checkpoints ahead

## Split Strategy

Forecasting uses grouped splitting by `hyperparameter_key` when present in metadata and falls back to `model_id` otherwise.

This is important. Checkpoints from the same model must not be split across train
and validation, otherwise the validation result is contaminated by trajectory
leakage.

## Entry Point

Forecasting uses the shared Hydra entry point and unified config tree.

The config tree lives in:

`metric-prediction/src/metric_prediction/config/configs/`

Run it with:

```bash
uv run metric-prediction/runner.py -cn config \
  mode=forecasting \
  experiment_name=forecast_distribution_tplus5 \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  features.metric_families=[distribution] \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  sequence.max_lag=3 \
  train.validation_size=0.2
```

The module launcher is equivalent:

```bash
uv run python -m metric_prediction_forecast \
  experiment_name=forecast_default \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  sequence.target_mode=final_value \
  sequence.max_lag=3 \
  train.validation_size=0.2
```

If you want to predict the metric value at checkpoint `t + 5`, use
`target_mode=horizon_value` with `horizon=5`:

```bash
uv run metric-predict-forecast \
  experiment_name=forecast_tplus5 \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  sequence.max_lag=3 \
  train.validation_size=0.2
```

Rows near the end of a trajectory without an available `t + 5` checkpoint are
dropped automatically.

You can also restrict forecasting to one feature family, for example only
`distribution` features:

```bash
uv run metric-predict-forecast \
  experiment_name=forecast_distribution_only \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  data.target_column=test_acc \
  'features.metric_families=[distribution]' \
  sequence.target_mode=final_value \
  sequence.max_lag=3 \
  train.validation_size=0.2
```

### Run Batch
Batches for forecasting are created under: ms-thesis/metric-prediction/experiment_batches_forecasting
batch_name defines the log directory where run results are saved.

Example:
```yaml
batch_name: Exp_2_Forecast_DS1_DS7

common_overrides:
  data.target_column: test_acc
  train.random_seed: 42
  train.save_predictions: true
  train.save_feature_matrix: false
  features.include_model_metrics: true
  features.include_layer_metrics: true
  features.metric_families: [distribution]

runs:
  - experiment_name: ds1_forecast_final_value_all_families
    overrides:
      data.parquet_path: metric-prediction/cache/prepared_datasets/1_cifar10_uniform_large_full_all_metrics_long__prepared_dataset.parquet
      sequence.target_mode: final_value
      sequence.max_lag: 3
      sequence.min_history: 1
```
Run with:
```shell
uv run metric-predict-batch metric-prediction/experiment_batches_forecasting/exp_1_forecast_final_value_ds1-ds7.yaml
```

When using `zsh`, quote Hydra overrides that contain brackets, such as
`'features.metric_families=[distribution]'`.

## Output

The forecasting pipeline writes artifacts under:

`metric-prediction/outputs_forecasting/<date>/<experiment_name>/<time>/`

Batch forecasting runs can also be launched from one YAML file. When
`batch_name` is set, each run is written to a stable experiment directory and
rerunning the same batch overwrites that run's previous artifacts:

`metric-prediction/outputs_forecasting/batch_experiments/<batch_name>/results/<experiment_name>/`

Typical artifacts:

- `metrics.json`
- `resolved_config.json`
- `used_features.json`
- `validation_predictions.parquet`
- `feature_importances.parquet`
- `model_bundle.pt` for PyTorch models, otherwise `model_bundle.pkl`

## Notes

- The existing `metric-prediction/runner.py` workflow is intentionally left untouched.
- The tabular forecasters still use the lagged wide feature matrix.
- The Transformer backend is most useful when `features.include_layer_metrics=true`.
