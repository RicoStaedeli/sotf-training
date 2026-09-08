# Metric Prediction

Framework package for predicting functional metrics such as `test_acc` from weight statistics produced by `nnrun.compute.weight_statistics_compute`.

## Entry points

This section is meant to be run directly from script files, not through
console-script aliases from the root `pyproject.toml`.

Train from an existing prepared parquet:

```bash
uv run metric-prediction/runner.py \
  experiment_name=ridge_baseline \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet
```

Run outputs now include the experiment name in the path, for example
`metric-prediction/outputs/single_experiment/2026-03-27/ridge_baseline/14-30-12-123456/`.
The default log file also uses the experiment name, for example `ridge_baseline.log`.

`metric-prediction/runner.py` now expects a prepared wide dataset. If `data.parquet_path` points to the raw long-form parquet, the loader looks for the corresponding prepared cache artifact in `data.partitioned.prepared_dataset_cache.dir` and errors clearly if it is missing.

Generate that prepared dataset from the long-form parquet with:

```bash
uv run python -m metric_prediction.scripts.data_preparation.generate_prepared_dataset \
  /absolute/path/to/cifar10_full_all_metrics_long.parquet
```

The cached prepared parquet stores the available metric columns directly, such as `test_acc`, `train_acc`, `test_loss`, and `train_loss`, rather than a generic `target` column.
The script writes partitioned tables to `./cache/partitioned_weight_statistic_datasets/` and the final prepared dataset to `./cache/prepared_datasets/`, relative to the directory where you start the script. It also writes a feature-metadata sidecar parquet next to the prepared dataset cache.
You can also point `metric-prediction/runner.py` directly at that prepared parquet artifact via `data.parquet_path`.

Merge pre-split latent representation datasets into one complete `.pt` file with:

```bash
uv run python -m metric_prediction.scripts.latent_representation.merge_latent_splits \
  metric-prediction/cache/latent_representation_datasets/resnet18_tinyimagenet_encoded_checkpoints
```

The script expects one `*_train.pt`, one `*_val.pt`, and one `*_test.pt` file in the directory.
By default it writes `<dataset>_latent_representations_complete.pt` next to the split files.
Pass `--add-split-column` if you want the merged file to retain a `split_name` entry for each row.

Enable Ray Tune for the XGBoost backend:

```bash
uv run metric-prediction/runner.py \
  experiment_name=xgboost_tuned \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  tune=xgboost
```

Or switch to the LightGBM backend:

```bash
uv run metric-prediction/runner.py \
  experiment_name=lightgbm_default \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  model=lightgbm
```

LightGBM also has a matching Ray Tune search space:

```bash
uv run metric-prediction/runner.py \
  experiment_name=lightgbm_tuned \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  model=lightgbm \
  tune=lightgbm
```

The default target column is `test_acc`, but `data.target_column` is configurable.

Use a simpler linear baseline instead of XGBoost:

```bash
uv run metric-prediction/runner.py \
  experiment_name=ridge_simple \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  model=ridge
```

## Batch runs

If you want to run a fixed list of experiments one after another, use
`metric-prediction/batch.py`.
Each batch entry gets its own `experiment_name`, so every run writes to a separate output and log path.
The batch runner always continues to the next run after a failure and reports all failed runs at the end.
If you set a top-level `batch_name`, all runs from that batch are grouped under
`metric-prediction/outputs/batch_experiments/<batch_name>/results/<experiment_name>/` instead of the default single-run directory.
The batch runner also copies the input batch YAML into `metric-prediction/outputs/batch_experiments/<batch_name>/batch_files/`
so later evaluation can use the exact batch definition that produced those runs.

Example batch config:

```yaml
batch_name: ds1-ds4

common_overrides:
  train.save_shap_artifacts: false

runs:
  - experiment_name: ridge_baseline
    overrides:
      data.parquet_path: /absolute/path/to/dataset_1__prepared_dataset.parquet
      model: ridge

  - experiment_name: lightgbm_tuned
    overrides:
      data.parquet_path: /absolute/path/to/dataset_2__prepared_dataset.parquet
      model: lightgbm
      tune: lightgbm
```

If `batch_name` is omitted, batch runs keep the default single-run output layout.

`common_overrides` and each run's `overrides` can be either:

- A YAML mapping with Hydra-style keys such as `data.parquet_path`, `train.save_shap_artifacts`, or `model`
- A YAML list of raw Hydra override strings if you need the exact CLI form

Run it with:

```bash
uv run metric-prediction/batch.py metric-prediction/batches/example.yaml
```

Preview the resolved runs without launching training:

```bash
uv run metric-prediction/batch.py metric-prediction/batches/example.yaml --dry-run
```

If you then evaluate a batch-specific directory such as `metric-prediction/outputs/batch_experiments/Exp_3_4`,
the evaluator automatically uses any `.yaml` or `.yml` files found in its `batch_files/` directory
unless you pass `--batch-file` explicitly.

Forecasting uses the same `metric-prediction/batch.py` command. Set `mode: forecasting`
in the batch YAML's `common_overrides`; when `output_root` is omitted, the runner
uses `metric-prediction/outputs_forecasting` automatically.
If you set `batch_name`, each run writes to
`metric-prediction/outputs_forecasting/batch_experiments/<batch_name>/results/<experiment_name>/`
and the batch YAML is copied to
`metric-prediction/outputs_forecasting/batch_experiments/<batch_name>/batch_files/`.

## Feature selection

By default, `metric-prediction/runner.py` now uses both model-level statistics and layerwise statistics. Model-level rows remain separate per metric, and layerwise rows also stay separate per metric-layer pair. The default layer feature name includes the metric, scope, `layers` row kind, and sanitized layer name, for example `distribution__percentile_25__weight_only__layers__module_list_0`.

You can restrict the final resolved feature set explicitly:

```bash
uv run metric-prediction/runner.py \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  features.include_feature_names='[matrix__fro_norm_mean__default____model__,norm__l2__combined____model__]'
```

You can also select or remove features by glob pattern:

```bash
uv run metric-prediction/runner.py \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet \
  features.include_feature_patterns='[*fro_norm*,*sharp_proxy*]' \
  features.exclude_feature_patterns='[*max*]'
```

Feature-name includes/excludes are applied when the prepared dataset is loaded for training. The preprocessing script now uses fixed feature-selection settings rather than exposing separate CLI flags.

The final fitted feature list is written to `used_features.json` in the run artifacts directory.
By default, the run also writes `shap_feature_contributions.parquet` plus `shap_feature_contributions.png`, `shap_beeswarm.png`, and `shap_heatmap.png` based on the validation split. Use `train.save_shap_artifacts=false` to skip SHAP generation, or tune the runtime with `train.shap_max_samples` and `train.shap_max_display`.

If you want to disable layer features entirely, set `features.include_layer_metrics=false`. If you want to collapse layerwise rows into one feature per layer, set `features.layer_feature_mode=aggregate_per_layer` and use `features.layer_feature_name_parts` with `features.layer_value_aggregation`.
