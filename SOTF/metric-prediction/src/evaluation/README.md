# Evaluation

`metric-prediction/evaluate.py` is the main entry point for evaluation.

Use it with one batch:

```shell
uv run metric-prediction/evaluate.py /path/to/batch-or-output-dir
```

Or with multiple batches of the same kind:

```shell
uv run metric-prediction/evaluate.py /path/to/batch-a /path/to/batch-b /path/to/batch-c
```

This now uses the normal evaluator for the detected kind and merges the runs
from all provided batch roots before generating the usual plots and tables.
There is no separate comparison-YAML workflow for this anymore.

You can force a backend with:

- `--kind prediction`
- `--kind forecasting`
- `--kind dataset-overview`

All provided directories must resolve to the same evaluation kind.

Every unified evaluation run gets a unique evaluation ID. Outputs are written
under:

- `metric-prediction/outputs_evaluation/prediction/`
- `metric-prediction/outputs_evaluation/forecasting/`

Each run is stored in:

```text
metric-prediction/outputs_evaluation/<kind>/<batch-name(s)>_<evaluation-id>/
```

The evaluation ID is stamped into every generated PNG plot, and
`metric-prediction/outputs_evaluation/<kind>/README.md` is extended with a
table row linking the evaluation output and the underlying batch roots.

## Direct Backend Commands

The unified entry point is the default. The direct backends still exist and can
be useful for focused debugging.

### Prediction

```shell
PYTHONPATH=metric-prediction/src uv run python -m evaluation.prediction
```

To scan a different experiment directory, pass `--outputs-root`. If
`--output-dir` is not set, the evaluation report is written below that same
root:

```shell
PYTHONPATH=metric-prediction/src uv run python -m evaluation.prediction \
  --outputs-root /path/to/experiment-outputs
```

If that outputs root contains batch YAML files copied by
`metric-predict-batch`, the evaluator uses them automatically from
`batch_files/`, and it scans run artifacts from `results/` when that directory
exists. You can still override that with one or more explicit `--batch-file`
arguments.

### Forecasting

```shell
PYTHONPATH=metric-prediction/src uv run python -m evaluation.forecasting \
  /path/to/forecast-batch-root
```

You can point this either at the batch root, for example
`metric-prediction/outputs_forecasting/batch_experiments/<batch_name>`, or
directly at its `results/` directory.

The direct backend now also supports merging multiple batches with repeated
flags:

```shell
PYTHONPATH=metric-prediction/src uv run python -m evaluation.forecasting \
  /path/to/batch-a \
  --extra-results-root /path/to/batch-b \
  --extra-results-root /path/to/batch-c
```

If `--output-dir` is not set, the direct backend writes to
`<passed-path>/evaluation/`.

### Dataset Overview

Generates a focused overview for a **single dataset**: R² and RMSE heatmaps
where rows are experiment names and columns are forecasting modes
(`final_value`, `horizon_value`). This makes it easy to compare all
experiments for one dataset at a glance.

**Quick start — scan all experiments automatically:**

```shell
uv run metric-prediction/evaluate.py --kind dataset-overview --dataset ds5
```

This defaults to scanning `metric-prediction/outputs/forecasting` recursively,
so every experiment across all feature sets and model types is included without
specifying individual batch roots.

**With an explicit root or multiple roots:**

```shell
uv run metric-prediction/evaluate.py \
  metric-prediction/outputs/forecasting/latent_averaged \
  metric-prediction/outputs/forecasting/weight_statistics \
  --kind dataset-overview --dataset ds5
```

**Dataset filter** (`--dataset`) accepts:

- Short names: `ds5`, `ds14`
- Full dataset names: `5_cifar10_Resnet18_width_2-256`
- Substrings (resolved unambiguously): `cifar10_Resnet18`

If only one dataset is present in the scanned runs, `--dataset` can be omitted.

**Outputs** (written to `metric-prediction/outputs/evaluation/dataset-overview/`):

| File | Description |
| --- | --- |
| `13_final_value_r2_rmse_overview.png` | R² and RMSE heatmaps side by side for all `final_value` experiments |
| `14_horizon_value_r2_rmse_overview.png` | R² and RMSE heatmaps side by side for all `horizon_value` experiments |
| `15_r2_per_epoch_final_value.png` | R² per decision epoch — one line per experiment (`final_value`) |
| `16_rmse_per_epoch_final_value.png` | RMSE per decision epoch — one line per experiment (`final_value`) |
| `17_r2_per_epoch_horizon_value.png` | R² per decision epoch — one line per experiment (`horizon_value`) |
| `18_rmse_per_epoch_horizon_value.png` | RMSE per decision epoch — one line per experiment (`horizon_value`) |

Plots 13–14: experiments sorted best-first by R², R² ordering shared across both panels.
Plots 15–18: line plots coloured by model type (ridge / xgboost / lightgbm / mlp).

**Direct backend:**

```shell
PYTHONPATH=metric-prediction/src uv run python -m evaluation.dataset_overview \
  --dataset ds5
```

Passing `results_dir` is optional; it defaults to
`metric-prediction/outputs/forecasting`.
