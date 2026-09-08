# Nowcasting Future Performance for Population Pruning in Survival-of-the-Fittest Training

This repository contains the full implementation for **Survival-of-the-Fittest training (SOTF training)**: a population-level resource-allocation method that repeatedly ranks actively-training neural networks by their *predicted future performance* and stops lower-ranked runs early, so that compute is concentrated on the candidates most likely to end up on top.

The codebase is organized into two parts that together cover the whole pipeline described in the paper:

- **[`SANE/`](SANE)** — the Sequential Autoencoder for Neural Embeddings, used to learn structured latent representations of neural-network checkpoints.
- **[`SOTF/`](SOTF)** — the analysis, prediction, and selection pipeline: extracting weight statistics and learning-curve features from checkpointed model zoos, training the performance nowcasters (weight-statistic, SANE-latent, and learning-curve predictor families), and running the retrospective SOTF training simulation.

## Paper Summary

Selecting a neural network from many candidate runs is expensive because each is typically trained to completion, and stopping runs early is risky when learning curves cross and early leaders do not ultimately perform best. We ask whether intermediate checkpoint weights reveal how a model's performance will develop.

Across nine image-classification model zoos — CNNs, width-varying ResNet-18s, and Vision Transformers on CIFAR-10, STL-10, and Tiny ImageNet — we:

1. **Characterize ranking dynamics**: quantify pairwise rank cross-overs and rank-correlation instability throughout training.
2. **Train performance nowcasters**: predict test accuracy five epochs ahead or at the end of training, from observed accuracy/loss histories, engineered weight statistics, and learned SANE checkpoint representations.
3. **Run SOTF training retrospectively**: on held-out complete trajectories, repeatedly rank active runs by predicted future performance, prune the weakest fraction, and measure whether the best-performing models survive and how much training compute is avoided.

Checkpoint weights contribute signal beyond learning curves, but their utility depends on forecast horizon and population: compact weight statistics are strongest for short-range prediction, structured SANE representations are more effective for final-performance extrapolation, and combining checkpoint and performance features is usually best or close to best. These gains matter most in heterogeneous ResNet-18 and Vision Transformer populations, where rankings change substantially; in homogeneous CNN populations, candidates finish similarly and additional modeling offers little benefit. Relative to current-accuracy pruning, the best checkpoint predictor reduces the aggregate mean best-retained-model gap by approximately 79% for final-value nowcasting and 22% for horizon-5 nowcasting.

## Pipeline Overview

```
Model Zoo (checkpointed training runs)
        │
        ├──► Weight Statistics Extraction  (SOTF/learning-dynamics)
        │
        ├──► SANE Checkpoint Representations  (SANE)
        │
        ▼
Metric Prediction / Nowcasting  (SOTF/metric-prediction)
   predicts test accuracy at horizon-5 or final epoch,
   from performance history, weight statistics, and/or SANE latents
        │
        ▼
Survival-of-the-Fittest Training  (SOTF/sotf-selection)
   ranks the active population by nowcasted performance,
   prunes the weakest fraction, repeats until one model remains
```

1. **Weight statistics extraction** turns raw model-zoo checkpoints into long-form parquet datasets of weight statistics, accuracy, and loss per model and epoch.
2. **SANE** independently learns a transformer-based autoencoder over checkpoint parameter tensors, producing contextualized latent-token representations for each checkpoint.
3. **Metric prediction** trains nowcasting predictors (ridge, MLP, XGBoost, LSTM, hierarchical Transformer) on performance histories, weight statistics, and/or SANE latents to forecast test accuracy at a fixed horizon or at the end of training.
4. **SOTF selection** replays held-out trajectories, using a fixed, pre-trained predictor to rank and prune the population at regular decision intervals, and reports the best-retained-model gap, oracle top-K recall, and compute savings.

## Repository Layout

```
SANE/                       Sequential Autoencoder for Neural Embeddings
├── src/sane/                model, trainer, runner, orchestrator, data, generative, downstream
├── config/                  Hydra configs for SANE training runs
├── scripts/                 Utility and analysis scripts
├── tests/                   Unit tests
└── run.py                   Entry point for SANE training

SOTF/                        Model-zoo analysis, nowcasting, and SOTF training simulation
├── learning-dynamics/        Weight statistics extraction and ranking-dynamics analysis (Hydra-based)
├── metric-prediction/        Nowcasting predictors and time-series forecasting (Ray Tune-enabled)
└── sotf-selection/           Survival-of-the-Fittest training simulation
```

Each subdirectory has its own README with setup and usage instructions:

- [SANE/README.md](SANE/README.md) — SANE architecture, trainers/runners/orchestrator abstractions, and training usage
- [SANE/SANE_DOCUMENTATION.md](SANE/SANE_DOCUMENTATION.md) — extended SANE documentation
- [SOTF/README.md](SOTF/README.md) — full pipeline usage: weight statistics extraction, metric prediction/forecasting, and SOTF selection
- [SOTF/learning-dynamics/README.md](SOTF/learning-dynamics/README.md)
- [SOTF/metric-prediction/README.md](SOTF/metric-prediction/README.md)
- [SOTF/sotf-selection/README.md](SOTF/sotf-selection/README.md)

## Getting Started

Both parts use [`uv`](https://docs.astral.sh/uv/) for dependency management and Python ≥ 3.12.

```bash
# SANE — checkpoint representation learning
cd SANE
uv sync
uv run python run.py

# SOTF — weight statistics, nowcasting, and SOTF training simulation
cd SOTF
uv sync --no-install-project
uv run <subpackage>/<script>.py ...
```

See the section READMEs linked above for full configuration options, data preparation steps, and example commands.

## Model Zoo Corpus

Experiments span nine checkpointed image-classification model zoos:

| Dataset name | Image data | Architecture | Population | Models | Epochs |
|---|---|---|---|---|---|
| cifar10_cnn_large_uniform | CIFAR-10 | CNN (large) | Uniform | 1000 | 50 |
| cifar10_cnn_large_hyp_random_seeds | CIFAR-10 | CNN (large) | Hyp. random | 1280 | 50 |
| cifar10_cnn_large_hyp_fixed_seeds | CIFAR-10 | CNN (large) | Hyp. fixed | 1280 | 50 |
| stl10_cnn_large_uniform | STL-10 | CNN (large) | Uniform | 1000 | 50 |
| stl10_cnn_large_hyp_random_seeds | STL-10 | CNN (large) | Hyp. random | 2560 | 50 |
| stl10_cnn_large_hyp_fixed_seeds | STL-10 | CNN (large) | Hyp. fixed | 2560 | 50 |
| tinyimagenet_resnet18_phase_transitions | Tiny ImageNet | ResNet-18 (width) | Phase transitions | 192 | 100 |
| cifar10_resnet18_width_2-256 | CIFAR-10 | ResNet-18 (width) | Phase transitions | 192 | 150 |
| cifar10_vit | CIFAR-10 | ViT | Phase transitions | 147 | 300 |

## Citation

If you use this code, please cite the paper:

## License

SANE is released under the MIT License (see [SANE/LICENSE](SANE/LICENSE)).
