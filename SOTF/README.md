# Master Thesis of Rico Staedeli

A comprehensive pipeline for analyzing neural network model zoos through weight statistics extraction, metric prediction, and population-based model selection.

## Project Overview

This monorepo implements a three-stage pipeline:
1. **Weight Statistics Extraction** → produces long-form parquet datasets with weight statistics per model, epoch, and metric
2. **Metric Prediction** → trains regressors (Ridge, XGBoost, LightGBM, MLP) to predict metrics from weight statistics, including time-series forecasting
3. **Survival of the Fittest Selection** → simulates a population-based model selection process using one of three predictor backends

## Monorepo Structure

Three sub-packages under a single `pyproject.toml`, each with source code under `<subpackage>/src/`:

- **`learning-dynamics/`** — Extracts weight statistics from model zoo checkpoints; uses Hydra for config management
- **`metric-prediction/`** — Trains regressors to predict metrics from weight statistics; includes time-series forecasting module with Ray Tune support
- **`sotf-selection/`** — Simulates model selection on a fixed model zoo using predictor backends
- **`tex_master_thesis/`** — LaTeX thesis document with references

The root `pyproject.toml` defines shared dependencies. This repo is **not** installed as one Python package because sections contain overlapping top-level modules (`evaluation`, `utils`, `config`, `data`). Use direct script paths from the repository root.

## Environment Setup

**Requirements:** Python 3.12, UV package manager

```bash
# Install dependencies without editable install (avoids distutils issues)
uv sync --no-install-project

# For development with editable install:
uv pip install -e .
```

### Environment Notes
- On Linux `x86_64`, pins `torch==2.7.1` to PyTorch's official `cu118` index for CUDA 11.8 compatibility with older NVIDIA drivers (470.xx+)
- All data is in **Parquet** format
- All configuration uses **Hydra** with structured configs

## Quick Start

### Learning Dynamics (`nnrun`)

Extracts weight statistics from model zoo checkpoints and generates analyses and plots. See [learning-dynamics/README.md](learning-dynamics/README.md) for detailed documentation.

#### Configuration Files

- **[datasets_template.yaml](learning-dynamics/configs/datasets_template.yaml)** — Template for configuring dataset sources; specify model zoo directories and their types (Seed, Hyp-rand, Hyp-fix, ViT)
- **[datasets_dgx.yaml](learning-dynamics/configs/datasets_dgx.yaml)** — DGX server dataset configuration with absolute paths to production model zoos
- **[datasets_plots.yaml](learning-dynamics/configs/datasets_plots.yaml)** — Dataset configuration for plot generation runs
- **[model_zoo_informations.json](learning-dynamics/configs/model_zoo_informations.json)** — Metadata about all available model zoos (sizes, architectures, hyperparameters)
- **[reports/plot_configs.yaml](learning-dynamics/configs/reports/plot_configs.yaml)** — Main plot configuration file; specifies which analyses to generate (weight statistics, ranking flips, clustering, etc.)
- **[reports/plot_configs_ws.yaml](learning-dynamics/configs/reports/plot_configs_ws.yaml)** — Weight-statistics-only plot configuration
- **[reports/plot_configs_test.yaml](learning-dynamics/configs/reports/plot_configs_test.yaml)** — Minimal test configuration for quick validation

#### Usage

```bash
# Cache a dataset as parquet from raw checkpoints
./.venv/bin/python learning-dynamics/nnrun.py cache \
  --datasets learning-dynamics/configs/datasets_template.yaml --dataset ds1

# Generate plots from a YAML plot config
./.venv/bin/python learning-dynamics/nnrun.py report \
  --config learning-dynamics/configs/reports/plot_configs.yaml
```

#### Documentation

Per-analysis deep dives in [learning-dynamics/Docs/](learning-dynamics/Docs/):
- [weight_statistics_analysis.md](learning-dynamics/Docs/weight_statistics_analysis.md) — Overview of weight statistics metrics and extraction
- [weight_statistics_reference.md](learning-dynamics/Docs/weight_statistics_reference.md) — Reference guide for all weight statistic types
- [weight_rank_correspondence_analysis.md](learning-dynamics/Docs/weight_rank_correspondence_analysis.md) — Analysis of rank stability across epochs
- [weight_early_signal_analysis.md](learning-dynamics/Docs/weight_early_signal_analysis.md) — How early (epoch-1) weights predict final performance
- [ranking_flips_analysis.md](learning-dynamics/Docs/ranking_flips_analysis.md) — When and why model rankings change
- [delta_to_best_analysis.md](learning-dynamics/Docs/delta_to_best_analysis.md) — Gap between each model and best performer
- [clustering_analysis.md](learning-dynamics/Docs/clustering_analysis.md) — Grouping models by weight space similarity
- [late_bloomers_analysis.md](learning-dynamics/Docs/late_bloomers_analysis.md) — Models that improve significantly later in training
- [hyper_parameter_analysis.md](learning-dynamics/Docs/hyper_parameter_analysis.md) — Hyperparameter sensitivity in model zoos

### Metric Prediction

Trains regressors (Ridge, XGBoost, LightGBM, MLP) to predict metrics from weight statistics. Includes standard regression and time-series forecasting workflows. See [metric-prediction/README.md](metric-prediction/README.md) for detailed documentation.

#### Documentation

- [overview-metric-prediction.md](metric-prediction/docs/overview-metric-prediction.md) — Predicting static metrics from weight statistics; model backends, data contract, and configuration
- [overview-metric-forecasting.md](metric-prediction/docs/overview-metric-forecasting.md) — Time-series forecasting to predict future metric values; sequence building and trajectory reconstruction

#### Data Preparation

Prepare a wide dataset from raw long-form parquet:
```bash
uv run python -m metric_prediction.scripts.data_preparation.generate_prepared_dataset \
  /absolute/path/to/cifar10_full_all_metrics_long.parquet
# Writes to ./cache/prepared_datasets/ and ./cache/partitioned_weight_statistic_datasets/
```

#### Configuration Groups

**Prediction configs** (`metric_prediction/config/configs_prediction/`):
- **data/** — Dataset paths, target column, train/val/test splits
- **features/** — Feature patterns to include/exclude, layer metrics (see [Feature Selection](#feature-selection-metric-prediction) below)
- **model/** — Model backend selection: `ridge`, `xgboost`, `lightgbm`, `mlp`
- **train/** — Training hyperparameters (learning rate, batch size, epochs)
- **tune/** — Ray Tune hyperparameter search config: `xgboost`, `lightgbm`
- **logging/** — Log levels and artifact saving (SHAP, features lists)

**Forecasting configs** (`metric_prediction/config/configs_forecasting/`):
- Same as above, plus **sequence/** — Sequence construction (context window, target horizon, history lags, delta features)

#### Training

Train a model on prepared data:
```bash
uv run metric-predict \
  experiment_name=ridge_baseline \
  data.parquet_path=/absolute/path/to/weight_metrics__prepared_dataset.parquet

# Switch model backend: model=ridge | model=lightgbm (default is XGBoost)
# Enable Ray Tune: tune=xgboost | tune=lightgbm
# Change target: data.target_column=train_acc
```

Batch runs (multiple experiments):
```bash
uv run metric-predict-batch metric-prediction/batches/example.yaml
uv run metric-predict-batch metric-prediction/batches/example.yaml --dry-run
```

#### Forecasting (Time-Series)

Predict future metric values from checkpoint trajectories:
```bash
uv run metric-predict-forecast experiment_name=<name> data.parquet_path=<path>
uv run metric-predict-forecast-batch <batch_config>.yaml
uv run metric-predict-forecast-evaluate <outputs_dir>
```

**Output paths:**
- Single experiment: `metric-prediction/outputs/single_experiment/<date>/<experiment_name>/<time>/`
- Batch experiments: `metric-prediction/outputs/batch_experiments/<batch_name>/results/<experiment_name>/<time>/`

**Key artifacts:**
- `model_bundle.pkl` — Fitted preprocessor + regressor (used by sotf-select)
- `used_features.json` — Exact feature list used in training
- `shap_feature_contributions.parquet` + SHAP plots (disable with `train.save_shap_artifacts=false`)

### SOTF Selection

Simulates a "Survival of the Fittest" population-based model selection process on a fixed model zoo. See [sotf-selection/README.md](sotf-selection/README.md) for detailed documentation.

#### How It Works

The simulation operates on prepared checkpoint datasets (one row per `(model_id, epoch)`) using predictions to iteratively select the best models:

1. **Initialize** — Start from all models in the zoo
2. **Decision round** — At a given epoch, predict which models will perform best at `epoch + horizon`
3. **Selection** — Keep only the top fraction (e.g., 50%) of models
4. **Repeat** — Move to the next decision epoch and repeat until the minimum population size is reached

Example configuration:
- Decision start epoch: `10`
- Decision interval: `10`
- Prediction horizon: `5`
- Survival rate: `0.5`

→ At epoch 10, predict epoch 15 performance, remove bottom 50%; repeat at epoch 20, etc.

#### Configuration Groups

**Simulation configs** (`sotf_selection/config/`):
- **data/** — Prepared dataset path, target column, train/val/test splits
- **features/** — Feature patterns (same selection as metric-prediction)
- **sequence/** — Sequence construction for forecasting (context window, horizon, history)
- **selection/** — Selection parameters: start epoch, decision interval, survival rate, minimum population
- **predictor/** — Predictor backend: `naive`, `forecast_model`, `xgboost`
- **logging/** — Log levels and output artifacts

#### Predictor Modes

Three predictor backends are supported:

**`predictor=naive`** — Baseline using current metric value
```bash
uv run sotf-select \
  experiment_name=naive_test \
  data.parquet_path=/path/to/prepared_dataset.parquet \
  data.target_column=test_acc \
  predictor=naive \
  predictor.strategy=current_target
```

**`predictor=forecast_model`** — Uses pre-fitted model from metric-prediction
```bash
uv run sotf-select \
  predictor=forecast_model \
  predictor.model_bundle_path=/path/to/model_bundle.pkl \
  data.parquet_path=/path/to/prepared_dataset.parquet
```
⚠️ Horizon and target must match the training bundle.

**`predictor=xgboost`** — Online XGBoost refitting at each decision epoch
```bash
uv run sotf-select \
  predictor=xgboost \
  data.parquet_path=/path/to/prepared_dataset.parquet
```
Uses only observable history (causal constraint).

#### Complete Example

```bash
uv run sotf-select \
  experiment_name=naive_test \
  data.parquet_path=/path/to/prepared_dataset.parquet \
  data.target_column=test_acc \
  predictor=naive \
  predictor.strategy=current_target \
  sequence.target_mode=horizon_value \
  sequence.horizon=5 \
  selection.start_epoch=10 \
  selection.interval=10 \
  selection.survival_rate=0.5
```

If console script is unavailable:
```bash
PYTHONPATH=sotf-selection/src:metric-prediction/src uv run python -m sotf_selection ...
```

**Output path:** `sotf-selection/outputs/<date>/<experiment_name>/<time>/`

**Key artifacts:**
- `round_details.parquet` — Per-round survival statistics
- `round_summary.parquet` — Summary of each selection round
- `final_survivors.parquet` — Models remaining after all selection rounds

## Running Tests

```bash
pytest learning-dynamics/tests/
pytest metric-prediction/tests/
pytest sotf-selection/tests/
```

## Key Architecture Details

### Data Flow

1. **Weight Statistics Extraction** — `nnrun` extracts weight statistics from model zoo checkpoints
   - Output: Long-form parquet (one row per `(model_id, epoch, metric, stat)`)

2. **Data Preparation** — `generate_prepared_dataset` pivots to wide-form parquet
   - Output: Wide-form parquet (one row per `(model_id, epoch)`) + feature metadata sidecar
   - Used by: metric-prediction and sotf-selection

3. **Metric Prediction** — Train regressors or forecasting models on wide-form data
   - `metric-predict` — Static metric prediction (one output per sample)
   - `metric-predict-forecast` — Time-series forecasting (predicts future metric values)
   - Output: `model_bundle.pkl` (fitted preprocessor + regressor) + `used_features.json` + SHAP artifacts

4. **Model Selection** — `sotf-select` uses wide-form data and optional `model_bundle.pkl`
   - Loads saved forecasting model or trains online XGBoost
   - Iteratively removes underperforming models based on predictions
   - Output: `round_details.parquet`, `round_summary.parquet`, `final_survivors.parquet`

### Configuration Hierarchy

All sections use **Hydra** for configuration management with a common pattern:
- Base config: `config.yaml` (entry point)
- Config groups: subdirectories with YAML files for different concerns
- Overrides: Command-line arguments override config files
- Snapshots: `resolved_config.json` saved in each run for reproducibility

### Monorepo Rationale

The three sub-packages are **not** installed as one package because they each define overlapping top-level modules (`evaluation`, `utils`, `config`, `data`). Instead:
- Install dependencies once via root `pyproject.toml`: `uv sync`
- Run scripts from workspace root with direct paths: `uv run <subpackage>/script.py`
- Chain modules via `PYTHONPATH` when needed: `PYTHONPATH=sotf-selection/src:metric-prediction/src`

## Documentation Index

### Section READMEs
- [learning-dynamics/README.md](learning-dynamics/README.md) — Setup and workflow for weight statistics extraction
- [metric-prediction/README.md](metric-prediction/README.md) — Metric prediction and forecasting pipeline
- [metric-prediction/SANE-Documentation.md](metric-prediction/SANE-Documentation.md) — Structured notes on configuration and workflows
- [sotf-selection/README.md](sotf-selection/README.md) — SOTF selection simulation details

### Learning Dynamics Analysis Docs
- [weight_statistics_analysis.md](learning-dynamics/Docs/weight_statistics_analysis.md) — Weight statistics metrics
- [weight_statistics_reference.md](learning-dynamics/Docs/weight_statistics_reference.md) — Complete reference guide
- [weight_rank_correspondence_analysis.md](learning-dynamics/Docs/weight_rank_correspondence_analysis.md) — Rank stability
- [weight_early_signal_analysis.md](learning-dynamics/Docs/weight_early_signal_analysis.md) — Early epoch predictivity
- [ranking_flips_analysis.md](learning-dynamics/Docs/ranking_flips_analysis.md) — Ranking changes over time
- [delta_to_best_analysis.md](learning-dynamics/Docs/delta_to_best_analysis.md) — Performance gaps
- [clustering_analysis.md](learning-dynamics/Docs/clustering_analysis.md) — Weight space clustering
- [late_bloomers_analysis.md](learning-dynamics/Docs/late_bloomers_analysis.md) — Late-training improvements
- [hyper_parameter_analysis.md](learning-dynamics/Docs/hyper_parameter_analysis.md) — Hyperparameter sensitivity

### Metric Prediction Overviews
- [overview-metric-prediction.md](metric-prediction/docs/overview-metric-prediction.md) — Static metric prediction framework
- [overview-metric-forecasting.md](metric-prediction/docs/overview-metric-forecasting.md) — Time-series forecasting framework

### Configuration References
- [learning-dynamics/configs/](learning-dynamics/configs/) — Dataset and report configurations
- [metric-prediction/src/config/](metric-prediction/src/config/) — Prediction and forecasting config groups
- [sotf-selection/src/config/](sotf-selection/src/config/) — Selection simulation config groups
