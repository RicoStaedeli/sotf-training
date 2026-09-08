# SOTF Evaluation

The SOTF evaluation tooling lives in `sotf-selection/src/evaluation`.

## Main Commands

Run commands from the repository root.

Evaluate one experiment:

```bash
uv run sotf-selection/evaluate.py \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10
```

If you want to call the source module directly instead of the wrapper script:

```bash
PYTHONPATH=sotf-selection/src uv run python -m evaluation \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10
```

Evaluate one experiment against a baseline:

```bash
uv run sotf-selection/evaluate.py \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10 \
  --baseline-root sotf-selection/outputs/batch_experiments/baseline_naive_h5_sr05_i10
```

Compare predictors from separate experiment roots:

```bash
uv run sotf-selection/evaluate.py \
  --experiment-path forecast_model=sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_s15_i5 \
  --experiment-path naive=sotf-selection/outputs/batch_experiments/baseline_naive_h5_sr05_s15_i5 \
  --experiment-path xgboost=sotf-selection/outputs/batch_experiments/online_xgboost_h5_sr05_s15_i5_obs_metrics \
  --output-dir sotf-selection/outputs/evaluation/forecast_exp5_h5_sr15_i5
```

Write outputs to a custom directory:

```bash
uv run sotf-selection/evaluate.py \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10 \
  --output-dir sotf-selection/outputs/evaluation/forecast_exp5_h5_sr05_i10
```

## Useful Filters

```bash
uv run sotf-selection/evaluate.py \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10 \
  --predictor forecast_model \
  --target-column test_acc \
  --target-mode horizon_value \
  --horizon 5 \
  --start-epoch 10 \
  --interval 10 \
  --survival-rate 0.5
```

Supported predictor aliases are normalized by the evaluator. For example, `forecast`, `forecast-model`, and `forecasting_model` are treated as `forecast_model`, while `baseline` is treated as `naive`.

## Outputs

By default, evaluation artifacts are written to `sotf-selection/outputs/evaluation/<name>_<evaluation-id>` when the evaluated runs live under `sotf-selection/outputs/...`. Use `--output-dir` to override this location.

Main outputs:

- `summary.md`: human-readable evaluation report.
- `latest_runs.csv`: latest run selected for each predictor/dataset/configuration key.
- `round_metrics.csv`: round-by-round decision, prediction, and ranking metrics.
- `dataset_summary.csv`: one summary row per evaluated dataset and predictor.
- `predictor_summary.csv`: aggregate metrics per predictor.
- `predictor_comparison.csv`: comparison against the baseline predictor when possible.
- `01_*_dataset_overview.png`: dataset-level overview plots per predictor.
- `02_*_round_quality_heatmaps.png`: round-level decision quality heatmaps.
- `03_*_per_dataset_round_quality.png`: per-dataset round quality plots.
- `04_*_prediction_quality_heatmaps.png`: prediction `R2` and RMSE heatmaps.
- `11_best_survivor_gap_comparison.png`: comparison plot for the final survivor gap when multiple predictors are available.
- `11_best_survivor_gap_by_experiment.png`: a single experiment-level comparison plot showing the best-survivor gap per experiment and predictor.

## Selection Count Plots

Render kept-versus-killed model counts:

```bash
uv run sotf-selection/selection-counts.py \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10
```

Module fallback:

```bash
PYTHONPATH=sotf-selection/src uv run python -m evaluation.selection_counts \
  sotf-selection/outputs/batch_experiments/forecast_exp5_h5_sr05_i10
```

By default, outputs are written to `sotf-selection/outputs/evaluation/selection_counts/<experiment-name>` when the evaluated runs live under `sotf-selection/outputs/...`.

- `selection_counts_overview_all_datasets.png`: small-multiple overview across all selected model-zoo datasets.
- `selection_counts_<dataset>.png`: one dataset-specific stacked count plot per dataset.
- `selection_counts.csv`: the plotted counts in tabular form.
