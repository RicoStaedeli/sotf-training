# Prediction Experiment Plans

This directory contains Hydra experiment plans for static prediction tasks.
Plans predict `test_acc` from prepared weight-statistic feature matrices at
epoch 50, covering all 12 datasets with Ridge, LightGBM, and XGBoost.

Run a plan with:

```bash
uv run metric-prediction/runner.py -cn experiments/prediction/exp_1_distribution_epoch50
```

Preview the generated runs without training:

```bash
uv run metric-prediction/runner.py -cn experiments/prediction/exp_1_distribution_epoch50 dry_run=true
```

## Plans

| Plan | Runs | Description |
| --- | ---: | --- |
| `exp_1_distribution_epoch50.yaml` | 36 | Epoch-50 `test_acc` prediction for DS1-DS12 using **distribution features only** (model-wise + layer-wise) with Ridge, LightGBM, and XGBoost. |
| `exp_2_all_features_epoch50.yaml` | 36 | Epoch-50 `test_acc` prediction for DS1-DS12 using **all feature families** (model-wise + layer-wise) with Ridge, LightGBM, and XGBoost. |

## Common Settings

Both plans use:

- `mode: prediction`
- `output_root: metric-prediction/outputs_prediction`
- `train.save_shap_artifacts: true`
- `data.epoch_filter: [50]`
- `features.include_model_metrics: true`
- `features.include_layer_metrics: true`
- prepared parquet inputs from `metric-prediction/cache/prepared_datasets/`

## Dataset Shorthand

Experiment names use `dsN` prefixes that correspond to prepared datasets under
`metric-prediction/cache/prepared_datasets/`.

| ID | Dataset |
| --- | --- |
| `ds1` | CIFAR-10 uniform large |
| `ds2` | CIFAR-10 large hyperparameter random seeds |
| `ds3` | CIFAR-10 large hyperparameter fixed seeds |
| `ds4` | CIFAR-10 ResNet-18 Kaiming |
| `ds5` | CIFAR-10 ResNet-18 width sweep |
| `ds6` | CIFAR-10 ResNet-50 width sweep |
| `ds7` | CIFAR-10 ViT |
| `ds8` | STL-10 uniform large |
| `ds9` | STL-10 large hyperparameter random seeds |
| `ds10` | STL-10 large hyperparameter fixed seeds |
| `ds11` | TinyImageNet ResNet-18 Kaiming |
| `ds12` | Phase ResNet-18 TinyImageNet |
