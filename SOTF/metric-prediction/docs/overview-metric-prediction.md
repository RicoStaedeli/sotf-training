# Metric Prediction Overview

`metric-prediction` is the training package for predicting functional model metrics from weight-statistics features. Its main use case is to estimate targets such as `test_acc`, `train_acc`, `test_loss`, or `train_loss` from precomputed statistics generated upstream in the thesis pipeline.

## Goal

The package turns a prepared parquet dataset into a supervised regression experiment. By default, it predicts `test_acc`, but the target metric is configurable through `data.target_column`.

Supported model backends:

- `ridge`
- `xgboost`
- `lightgbm`

Optional hyperparameter tuning is available through Ray Tune.

## Data Contract

Training expects a prepared wide-format parquet dataset, not the raw long-form weight-statistics table.

Expected properties of the prepared dataset:

- one row per sample, identified by `sample_idx`
- one target column such as `test_acc`
- metadata columns such as `model_id`, `path`, and `epoch`
- feature columns derived from model-level and layer-level weight statistics

If a raw long-form parquet is passed instead, the loader fails with a clear error and points to the expected prepared cache path.

Prepared datasets are generated from the long-form source with:

```bash
uv run python -m metric_prediction.scripts.data_preparation.generate_prepared_dataset \
  /absolute/path/to/full_all_metrics_long.parquet
```

## Training Flow

The main entry point is `metric-prediction/runner.py`, which is implemented with Hydra configuration.

The experiment flow is:

1. Load the prepared dataset and its optional feature-metadata sidecar.
2. Select features based on config filters such as metric family, scope, layer type, explicit names, or glob patterns.
3. Drop overly sparse or constant features.
4. Split the dataset into train and validation sets.
5. Impute missing values with the configured strategy (`median` by default).
6. Train the selected regressor.
7. Evaluate predictions on the validation split with `rmse`, `mae`, and `r2`.
8. Write artifacts such as metrics, predictions, model bundle, feature importance data, and SHAP outputs.

Default training settings:

- validation split: `0.2`
- shuffle: `true`
- random seed: `42`
- missing-value fill: `median`
- default model: `ridge`

## Feature Handling

Feature selection is driven by the `features` config group. The default configuration includes both model-level and layer-level metrics.

The package supports:

- filtering by metric families, metrics, scopes, and layer types
- explicit include and exclude lists
- glob-based include and exclude patterns
- optional sorting of feature columns
- logging of the resolved feature list

The final resolved training features are written to `used_features.json`.

## Outputs

Hydra writes each run to a dedicated output directory:

```text
metric-prediction/outputs/single_experiment/YYYY-MM-DD/<experiment_name>/HH-MM-SS-ffffff/
```

Inside the run's `artifacts/` directory, the package can write:

- `metrics.json`
- `resolved_model_config.json`
- `used_features.json`
- `model_bundle.pkl`
- `validation_predictions.parquet`
- `feature_matrix.parquet`
- `feature_importances.parquet`
- `ray_tune_summary.json`
- SHAP artifacts such as `shap_feature_contributions.parquet` and plots

## Typical Commands

Run a baseline ridge model:

```bash
uv run metric-prediction/runner.py \
  experiment_name=ridge_baseline \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet
```

Run XGBoost with tuning:

```bash
uv run metric-prediction/runner.py \
  experiment_name=xgboost_tuned \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  model=xgboost \
  tune=xgboost
```

Run a fixed sequence of experiments:

```bash
uv run metric-predict-batch metric-prediction/batches/example.yaml
```

## Summary

In practice, `metric-prediction` is the regression layer of the thesis workflow: preprocessing converts raw weight-statistics tables into a model-ready feature matrix, and this package trains, evaluates, and stores reproducible experiments that predict downstream quality metrics from those features.
