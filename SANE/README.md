# SANE - Sequential Autoencoder for Neural Embeddings

Internal repository for a refactored SANE codebase.

[![Unit tests](https://github.com/HSG-AIML/internal-sane/actions/workflows/pytest.yml/badge.svg)](https://github.com/HSG-AIML/internal-sane/actions/workflows/pytest.yml)

## Code Structure

* `data/` — Neural network weights as input data for weight space models and SANE.
    * `checkpoint/` — Everything needed to manage checkpoints (saved model states).
    * `datasets/` — PyTorch datasets that handle models in checkpoint or tokenized format.
* `loss/` — Loss functions used in the context of SANE.
* `model/` — The SANE autoencoder and its components, cf. the corresponding [README.md](./src/sane/model/README.md).
* `trainer/` — `SANETrainer` classes, which handle the top-level logic of configuring, loading and running an entire SANE experiment. Understanding how SANE works will usually start with the `SANETrainer` class.
* `runner/` — Execution backends (`LocalRunner`, `RayRunner`) that own the training loop, checkpointing schedule, grid search, and metrics logging.
* `logging/` — Metrics logging backends (`JsonLogger`, `TensorBoardLogger`) behind a common protocol.
* `orchestrator.py` — Wires config, runner, and trainer together. The main entry point for experiments.
* `generative/` — Flow matching model and sampling.
* `downstream/` — Evaluation benchmarks.
* `utils/` — Various utility functions.

## Architecture Overview

The codebase is organized around three core abstractions that separate **what** to train from **how** to execute it:

```
run.py  (entry point — Hydra config composition)
  └─► Orchestrator  (sequences stages, resolves trainer/runner, saves config)
        └─► Runner  (owns training loop, checkpointing, grid search, logging)
              └─► Trainer  (pure PyTorch — model setup, forward/backward, optimizer step)
```

| Component | Responsibility | Does NOT own |
|---|---|---|
| **Trainer** | Model, data, optimizer, loss, forward/backward pass | Logging, checkpointing schedule, grid search, Ray |
| **Runner** | Training loop, checkpoint schedule, metrics logging, grid search | Pipeline sequencing, stage caching |
| **Orchestrator** | Resolving trainer/runner from config, wiring everything together, launching the run | Training mechanics, execution backend details |



---

## Trainers

Trainers contain the core ML logic. They are plain Python classes — no base class inheritance required, just adherence to the `Trainer` protocol (via structural subtyping, read more about it [here](https://typing.python.org/en/latest/spec/protocol.html)):

```python
class Trainer(Protocol):
    epoch: int
    def setup(self, config: dict[str, Any]) -> None: ...
    def step(self) -> dict[str, Any]: ...
    def save_checkpoint(self, checkpoint_dir: str) -> None: ...
    def load_checkpoint(self, checkpoint_dir: str) -> None: ...
```

- **`setup(config)`** — Called once by the runner after construction. Initializes model, data loaders, optimizer, loss, etc.
- **`step()`** — Runs one epoch of training (and optionally validation). Returns a `dict` of metrics (e.g. `{"loss": 0.42, "val_loss": 0.51}`).
- **`save_checkpoint` / `load_checkpoint`** — Persist and restore full training state to/from a directory.

### Available trainers

| Class | Location | Purpose |
|---|---|---|
| `SANETrainer` | `sane/trainer/trainer.py` | Base autoencoder trainer (reconstruction) |
| `SANEContrastiveTrainer` | `sane/trainer/contrastive_trainer.py` | Adds contrastive views and augmentation |
| `SANEDummyTrainer` | `sane/trainer/dummy_trainer.py` | Synthetic data for testing |
| `FlowMatchingTrainer` | `sane/generative/` | Flow matching on pre-encoded latents |

### Writing a new trainer

Implement the four protocol methods. The trainer should never import `ray`, `wandb`, or any logging library — just return metrics from `step()`.

```python
class MyTrainer:
    def __init__(self) -> None:
        self.epoch = 0

    def setup(self, config: dict[str, Any]) -> None:
        self.model = build_model(config["model"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        # ...

    def step(self) -> dict[str, Any]:
        self.epoch += 1
        loss = train_one_epoch(self.model, self.optimizer, self.train_loader)
        return {"loss": loss}

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        torch.save(self.model.state_dict(), Path(checkpoint_dir) / "model.pt")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        self.model.load_state_dict(torch.load(Path(checkpoint_dir) / "model.pt"))
```
### Metrics reporting

- **Return flat dicts from `step()`** — Keys like `"loss"`, `"val_loss"`, `"lr"` work well. Avoid nested dicts; logging backends expect `dict[str, scalar]`.
- **Use consistent metric names** — The runner picks the best grid search result by the `"loss"` key. Include it in every `step()` return value.
- **Keep trainers logging-free** — Never import `wandb`, `tensorboard`, or `sane.logging` inside a trainer. The runner handles all metric forwarding.


## Runners

Runners own the training loop and everything _about_ execution: checkpointing schedule, metrics logging, and grid search expansion. They implement the `Runner` protocol:

```python
class Runner(Protocol):
    def run(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
    ) -> RunResult: ...
```

A `RunResult` is returned with the best checkpoint path, the resolved config, and final metrics:

```python
@dataclass
class RunResult:
    checkpoint_path: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    from_cache: bool = False
```

### `LocalRunner`

Runs training in the current process. Supports grid search by expanding grid specs sequentially and returning the best result by loss. If no logger is provided, it creates a default `CompositeLogger` with JSON + TensorBoard backends.

### `RayRunner`

Wraps the trainer as a Ray Tune function trainable and delegates to `tune.Tuner`. Grid specs (`{"grid": [v1, v2]}`) are automatically converted to `tune.grid_search(...)`. Ray is initialized/shut down around each run.

### Switching runners

The runner is selected via the `runner` key in the config. Set `runner: "local"` or `runner: "ray"` — no code changes needed. You can also pass `--debug` to `run.py` to force local execution regardless of config:

```bash
python run.py --debug            # forces LocalRunner
python run.py runner=ray         # uses RayRunner
```

## Orchestrator

The `Orchestrator` is the glue between config, runner, and trainer. It:

1. Reads the config and resolves the trainer class and runner implementation.
2. Creates the storage directory and saves the full config as `config.json`.
3. Delegates to `runner.run(...)` and returns the `RunResult`.

There will be different orchestrators for distinct usecases, e.g. for training an autoencoder, multi-staged pipelines where multiple trainers are trained like in the case of flow matching.
If your specific pipeline needs multiple trainers, the orchestrator defines how they interact.

```python
orchestrator = Orchestrator(config=config, force_local=False)
result = orchestrator.run()
print(result.checkpoint_path)
```


# Running the Code

## Basic usage

```bash
python run.py                            # default config (config/1_cifar10_uniform_large.yaml)
python run.py -cn 7_cifar10_ViT          # use a different root config by name
python run.py runner=ray                 # overwrite the hydra config runner specifically
python run.py --debug                    # force local execution, will use the local runner without any config change
```

Additionally, you can use hydra to set the log level for individual modules:
```bash
python run.py log_level=INFO hydra.job_logging.loggers.sane.model.level=DEBUG
```

With that, you can set the global log level to ERROR and keep the files of interest in DEBUG.

## Grid search

Define parameter grids in YAML using the `{grid: [...]}` syntax:

```yaml
loss:
  class: "MSEReconstructionLoss"
  normalize: {grid: ['token', 'batch', null]}

optimizer:
  class: "AdamW"
  lr: {grid: [1e-4, 1e-3]}
```

This runs `3 × 2 = 6` trials. With `runner: "local"`, trials run sequentially. With `runner: "ray"`, they run in parallel via Ray Tune.


## Unit Tests

```bash
uv run pytest                  # standard tests
uv run pytest --runslow        # include slow tests
```

Some tests require resources not available in GitHub Actions and will be skipped automatically.
