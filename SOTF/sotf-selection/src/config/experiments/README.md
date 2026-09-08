# SOTF Experiment Configs

## Directory layout

```
experiments/
  final_value_prediction/
    se{N}_sr{N}_i{N}/        # simulation params: start_epoch / survival_rate / interval
      ds{N}.yaml             # one file per dataset
  horizon_value_prediction/
    se{N}_sr{N}_i{N}/
      ds{N}.yaml
  archive/                   # one-off or superseded configs
```

The folder path encodes the condition; the file encodes the dataset.  Every
file inside a single `se*/` folder shares the same `batch_name`, so all
datasets for that condition land in one output directory tree.

## Running a single config

Pass the config path relative to this `experiments/` directory:

```bash
uv run sotf-select \
  --config-dir sotf-selection/src/config \
  --config-name experiments/final_value_prediction/se10_sr50_i5/ds1
```

Or run as a batch (all predictors for one dataset × condition):

```bash
uv run sotf-select-batch \
  sotf-selection/src/config/experiments/final_value_prediction/se10_sr50_i5/ds1.yaml
```

## Output structure

Batch outputs land under a stable per-condition directory (no timestamp):

```
sotf-selection/outputs/batch_experiments/<batch_name>/
  batch_files/          ← copy of the YAML that produced this batch
  results/
    <predictor>/
      <target_column>/
        <target_mode>/
          dataset_<id>_<slug>/
            <experiment_name>/
```

Because `batch_name` is shared across datasets of the same condition, all
datasets are co-located under one `batch_name/results/` tree, making
cross-dataset comparison straightforward.

## Naming conventions

| Folder segment | Meaning | Example |
|---|---|---|
| `final_value_prediction/` | `sequence.target_mode = final_value` | — |
| `horizon_value_prediction/` | `sequence.target_mode = horizon_value` | — |
| `se{N}` | `selection.start_epoch` | `se10` = start epoch 10 |
| `sr{N}` | `selection.survival_rate × 100` | `sr50` = 0.5 |
| `i{N}` | `selection.interval` | `i5` = interval 5 |

## batch_name convention

```
{target_mode_short}_se{start_epoch}_sr{survival_rate*100}_i{interval}
```

Examples: `final_value_se10_sr50_i5`, `horizon5_se25_sr50_i5`

## Active conditions

| Condition dir | batch_name | target_mode | start_epoch | survival_rate | interval | Datasets |
|---|---|---|---|---|---|---|
| `final_value_prediction/se10_sr50_i5/` | `final_value_se10_sr50_i5` | final_value | 10 | 0.5 | 5 | ds1–5, 8–10, 12 |
| `final_value_prediction/se25_sr50_i5/` | `final_value_se25_sr50_i5` | final_value | 25 | 0.5 | 5 | ds1–5, 8–10, 12 |
| `horizon_value_prediction/se10_sr50_i5/` | `horizon5_se10_sr50_i5` | horizon_value | 10 | 0.5 | 5 | ds1–5, 8–10, 12 |
| `horizon_value_prediction/se25_sr50_i5/` | `horizon5_se25_sr50_i5` | horizon_value | 25 | 0.5 | 5 | ds1–5, 8–10, 12 |

## Adding a new condition

1. Create `<target_mode>_prediction/se{N}_sr{N}_i{N}/`
2. Copy an existing dataset YAML into that folder and rename to `ds{N}.yaml`
3. Update `batch_name`, `common_overrides`, and any predictor paths
4. All datasets sharing the condition should have the same `batch_name`
