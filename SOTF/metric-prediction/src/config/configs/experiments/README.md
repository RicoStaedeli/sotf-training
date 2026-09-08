# Experiment Plans

This directory contains Hydra-native experiment plans. Use these files instead
of the old `experiment_batches_*` folders.

## Run A Plan

```bash
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_1
uv run metric-prediction/runner.py -cn experiments/prediction/exp_1a
```

Preview the generated per-run commands without training:

```bash
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_1 dry_run=true
```

## Layout

- `forecasting/`: forecasting and latent-forecasting experiment plans.
- `prediction/`: static prediction and latent-prediction experiment plans.

## Available Plans

Forecasting:

```bash
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_1
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_2
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_3
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_4
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_5
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_6
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_7
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_8
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_9
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_10
uv run metric-prediction/runner.py -cn experiments/forecasting/exp_latent
```

Prediction:

```bash
uv run metric-prediction/runner.py -cn experiments/prediction/exp_1a
uv run metric-prediction/runner.py -cn experiments/prediction/exp_1b
uv run metric-prediction/runner.py -cn experiments/prediction/exp_2a
uv run metric-prediction/runner.py -cn experiments/prediction/exp_2b
uv run metric-prediction/runner.py -cn experiments/prediction/exp_3a
uv run metric-prediction/runner.py -cn experiments/prediction/exp_3b
uv run metric-prediction/runner.py -cn experiments/prediction/exp_4a
uv run metric-prediction/runner.py -cn experiments/prediction/exp_4b
uv run metric-prediction/runner.py -cn experiments/prediction/exp_5
uv run metric-prediction/runner.py -cn experiments/prediction/exp_6
uv run metric-prediction/runner.py -cn experiments/prediction/exp_7
uv run metric-prediction/runner.py -cn experiments/prediction/exp_test
uv run metric-prediction/runner.py -cn experiments/prediction/test_cfg
```

Each plan has the same simple structure:

```yaml
batch_name: Example_Batch
output_root: metric-prediction/outputs_forecasting
dry_run: false

evaluation:
  kind: forecasting

common_overrides:
  mode: forecasting
  train.random_seed: 42

runs:
  - experiment_name: example_run
    overrides:
      model: xgboost
```

`common_overrides` apply to every run. Each item in `runs` adds or replaces
settings for one concrete experiment.

Batch outputs use stable per-experiment directories:
`<output_root>/batch_experiments/<batch_name>/results/<experiment_name>/`.
Rerunning the same experiment overwrites that directory.

The optional `evaluation.kind` field tells `metric-predict-evaluate` which
batch-local evaluator to use. Cross-batch comparisons should live in separate
configs under `evaluation/comparisons/`, because they are analysis artifacts
rather than properties of a single experiment batch.
