# Learning Dynamics in Model Zoos

This directory contains all relevant code to reproduce the learning-dynamics analysis on multiple model zoos of neural networks.

## Install

Run all commands from the repo root `/ms-thesis`. No editable install is required (this avoids Python 3.12 `distutils` build issues).

```bash
uv sync --no-install-project
./.venv/bin/python learning-dynamics/nnrun.py --help
```

For development with an editable install:
```bash
uv pip install -e .
```

## Directory Structure

```
learning-dynamics/
├── configs/
│   ├── datasets_template.yaml       # Dataset source config (raw checkpoints)
│   ├── datasets_dgx.yaml            # Dataset config for DGX server
│   ├── datasets_plots.yaml          # Dataset config for plot runs
│   ├── model_zoo_informations.json  # Metadata about all model zoos
│   └── reports/
│       ├── plot_configs.yaml        # Main plot config (all analyses)
│       ├── plot_configs_ws.yaml     # Weight-statistics-only config
│       └── plot_configs_test.yaml   # Minimal config for testing
├── Docs/                            # Per-analysis documentation
├── Notebooks/                       # Exploratory notebooks
├── outputs/
│   ├── cache/                       # Cached parquet datasets
│   ├── logs/
│   └── reports/
│       ├── PDF/
│       ├── PNG/
│       └── tables/
├── src/nnrun/                       # Package source
│   ├── cli.py                       # Entry point (cache / report subcommands)
│   └── ...
└── tests/
```

## Workflow

### 1. Cache a Dataset

Converts raw checkpoint directories into a long-form `.parquet` file for fast reuse.

```bash
./.venv/bin/python learning-dynamics/nnrun.py cache \
  --datasets learning-dynamics/configs/datasets_template.yaml \
  --dataset ds1
```

Cached files are written to `learning-dynamics/outputs/cache/` as `long__<dataset_name>.parquet`.

The datasets config (`datasets_template.yaml`) maps named keys to checkpoint directories:

```yaml
datasets:
  ds1:
    root_dir: data/tune_zoo_cifar10_small_hyperparameter_10_random_seeds
    dstype: Hyp-rand
```

### 2. Generate Plots (Report)

Plots are driven by a YAML config. Run from the repo root:

```bash
./.venv/bin/python learning-dynamics/nnrun.py report \
  --config learning-dynamics/configs/reports/plot_configs.yaml
```

Outputs are written to:
- `learning-dynamics/outputs/reports/PDF/`
- `learning-dynamics/outputs/reports/PNG/`
- `learning-dynamics/outputs/reports/tables/`

## Plot Config Structure

`plot_configs.yaml` has three top-level sections:

| Key | Description |
|-----|-------------|
| `datasets` | List of named datasets to load from the parquet cache |
| `analyses` | List of analysis blocks to run (see below) |
| `output_*_dir` | Output directories for PDF, PNG, and tables |

Global column aliases:

```yaml
epoch_col: training_iteration
model_col: model_name
metric_aliases:
  validation_acc: [validation_acc, val_acc]
  validation_loss: [validation_loss, val_loss]
```

## Available Analyses

Each entry in `analyses` is identified by its `name` field. Multiple entries with the same name are allowed (e.g., one per metric).

### `acc_curves`

Plots per-model accuracy curves across epochs as boxplots.

```yaml
- name: acc_curves
  metric: test_acc          # train_acc | test_acc | validation_acc
  epoch_range: []           # [] = all epochs
  png_dpi: 200
  avg_curves: false
  boxplot_curves: true
  boxplot_overview:
    enabled: true
    single_page:
      enabled: true
      ncols: 4
    paged:
      enabled: true
      nrows: 2
      ncols: 2
```

### `ranking_flips`

Counts how often model rankings reverse between consecutive epochs across multiple metrics.

```yaml
- name: ranking_flips
  metrics:
    - {name: test_acc, higher_is_better: true}
    - {name: test_loss, higher_is_better: false}
    # ...
  epoch_range: []
  smoothening_threshold: 0.02
  mode: one_page_per_dataset   # or overlay_by_dataset
  panel:
    nrows: 2
    ncols: 3
    figsize: [12, 7]
```

See [Docs/ranking_flips_analysis.md](Docs/ranking_flips_analysis.md).

### `clustering`

Clusters models by their metric trajectory using k-means (k=0 auto-selects k).

```yaml
- name: clustering
  tag: valacc_k5            # unique label for this run
  metric: validation_acc
  epoch_range: []
  k: 5                      # 0 = auto-select
```

See [Docs/clustering_analysis.md](Docs/clustering_analysis.md).

### `late_bloomers`

Identifies models that rank in the bottom fraction at a mid-epoch but recover to the top fraction by the end.

```yaml
- name: late_bloomers
  metrics:
    - {name: test_acc, higher_is_better: true}
    # ...
  epoch_mid: 15
  bottom_frac_mid: 0.3
  top_frac_end: 0.3
  epoch_range: []
  plot_summary_table: true
  summary_columns: [dataset_name, metric, n_models, late_bloomers]
  summary_max_rows: 80
  png_dpi: 200
```

See [Docs/late_bloomers_analysis.md](Docs/late_bloomers_analysis.md).

### `delta_to_best`

Computes each model's gap to the best model over time, with optional MAD integration.

```yaml
- name: delta_to_best
  metric: test_acc
  higher_is_better: true
  epoch_range: []
  window_k: 5               # epochs at start/end for mean computation
  use_mad: true
  plot:
    enabled: true
    selection_mode: rank_bands   # rank_bands | top_n
    best_n: 4
    middle_n: 4
    worst_n: 4
    top_n: 12
    ncols: 4
    figsize: [12, 9]
    dpi: 200
    use_log: false
```

See [Docs/delta_to_best_analysis.md](Docs/delta_to_best_analysis.md).

### `hyper_analysis`

Fits a Random Forest on hyperparameters to predict final metric, then computes permutation feature importances.

```yaml
- name: hyper_analysis
  metric: test_acc
  higher_is_better: true
  epoch_range: []
  png_dpi: 200
  top_n_table: 30
  rf_n_jobs: 2
  perm_n_jobs: 2
  cv_splits: 3
  perm_repeats: 5
  rf_n_estimators: 200
```

See [Docs/hyper_parameter_analysis.md](Docs/hyper_parameter_analysis.md).

### `weight_rank_correspondence`

Correlates weight statistics with model rankings. Requires a pre-computed weight metrics cache.

```yaml
- name: weight_rank_correspondence
  cache_dir: "learning-dynamics/outputs/cache/weight_metrics"
  dataset_split: "full"
  force_recompute: false
  skip_missing_cache: true
  drop_epoch_zero: true
  rank_metric: "test_acc"
  higher_is_better: true
  metric_family: "all"
  metric: "all"
  scope: "all"
  include_model_row: true
  top_k_metrics: 15
  detailed_top_k: 5
  crossover_quantile: 0.9
  min_changed_rank: 5
  min_event_separation: 2
  align_window: 4
  pre_window: 2
  post_window: 2
  max_lag: 8
```

See [Docs/weight_rank_correspondence_analysis.md](Docs/weight_rank_correspondence_analysis.md).

### `weight_early_signal`

Tests whether weight statistics computed at early epochs can predict final model quality (late-bloomer-style split).

```yaml
- name: weight_early_signal
  functional_metric: test_acc
  higher_is_better: true
  use_partitioned_cache: true
  partition_batch_size: 250000
  epoch_mid: 15
  bottom_frac_mid: 0.30
  top_frac_end: 0.30
  prediction_epochs: []     # auto-selects <= epoch_mid when empty
  metric_family: all
  metric: all
  scope: all
  include_model_row: true
  top_quantile: 0.20
  plot:
    enabled: true
    top_k_metrics: 10
    include_auc_detail: true
    auc_detail_top_k: 1
    include_mannwhitney_detail: true
    mannwhitney_detail_top_k: 1
```

See [Docs/weight_early_signal_analysis.md](Docs/weight_early_signal_analysis.md).

### `weight_analysis` (disabled by default)

Computes and plots all weight statistics directly. Commented out in `plot_configs.yaml`; use `plot_configs_ws.yaml` instead.

See [Docs/weight_statistics_analysis.md](Docs/weight_statistics_analysis.md) and [Docs/weight_statistics_reference.md](Docs/weight_statistics_reference.md).

## Running Tests

```bash
pytest learning-dynamics/tests/
```
