# Forecasting Experiment Plans

This directory contains Hydra experiment plans for checkpoint-sequence
forecasting. Plans are organised by **feature type** (top-level directory)
and then by **target mode** (`final_value/` or `horizon/` sub-directory).

## Directory layout

```
forecasting/
  weight_statistics/
    final_value/   exp_1–5  — weight-statistic XGBoost/MLP, final-value target
    horizon/       exp_1–5  — same, horizon-value target (horizon=5)
  latent_averaged/
    ds{N}.yaml     — averaged per-epoch latent embeddings (covers both target modes)
  latent_windows/
    final_value/   ds{N}    — LSTM on sliding-window latent sequences, final-value
    horizon/       ds{N}    — same, horizon-value
  latent_tokens/
    final_value/   ds{N}    — LSTM on token-level latent sequences, final-value
    horizon/
      ds{N}.yaml            — LSTM, horizon-value
      transformer/ds{N}    — hierarchical transformer, horizon-value (ds1,4,5,12)
  archive/         test.yaml and other one-off files
```

## Running a plan

```bash
uv run metric-prediction/runner.py \
  -cn experiments/forecasting/weight_statistics/final_value/exp_4

uv run metric-prediction/runner.py \
  -cn experiments/forecasting/latent_windows/horizon/ds12

# Preview without training
uv run metric-prediction/runner.py \
  -cn experiments/forecasting/latent_tokens/horizon/ds1 dry_run=true
```

## Plans

### weight_statistics/

| Plan | Target | Description |
| --- | --- | --- |
| `final_value/exp_1` | final | DS1-DS7, model+layer-wise, all families, max_lag=3 |
| `final_value/exp_2` | final | DS1-DS12, model-wise distribution, no lag |
| `final_value/exp_3` | final | DS1-DS12, model-wise distribution, lag=3 |
| `final_value/exp_4` | final | DS1-DS12, distribution, lag=3 + delta + time features |
| `final_value/exp_5` | final | DS1-DS12, model+layer distribution, lag=3 + delta + time |
| `horizon/exp_1–5` | horizon 5 | Mirror of above with `target_mode=horizon_value` |

### latent_averaged/

Each `ds{N}.yaml` file runs both `final_value` and `horizon_value` targets in a
single batch (XGBoost, Ridge, MLP sweep). Split into separate runs via the
`experiment_name` suffix (`_final_value` vs `_horizon_5`).

### latent_windows/

LSTM trained on sliding-window sequences of latent embeddings.

| Target | Datasets |
| --- | --- |
| `final_value/` | ds1–5, 8–10, 12 |
| `horizon/` | ds1–5, 8–10, 12 |

### latent_tokens/

LSTM and transformer trained on token-level latent sequences.

| Plan | Target | Datasets |
| --- | --- | --- |
| `final_value/ds{N}` | final | ds1–5, 8–11, 12 |
| `horizon/ds{N}` | horizon 5 | ds1–5, 8–10, 12 |
| `horizon/transformer/ds{N}` | horizon 5 | ds1, 4, 5, 12 |

## Dataset shorthand

| ID | Dataset |
| --- | --- |
| `ds1` | CIFAR-10 uniform large |
| `ds2` | CIFAR-10 large hyperparameter random seeds |
| `ds3` | CIFAR-10 large hyperparameter fixed seeds |
| `ds4` | CIFAR-10 ResNet-18 Kaiming |
| `ds5` | CIFAR-10 ResNet-18 width sweep |
| `ds8` | STL-10 uniform large |
| `ds9` | STL-10 large hyperparameter random seeds |
| `ds10` | STL-10 large hyperparameter fixed seeds |
| `ds11` | TinyImageNet ResNet-18 Kaiming |
| `ds12` | TinyImageNet phase ResNet-18 |
