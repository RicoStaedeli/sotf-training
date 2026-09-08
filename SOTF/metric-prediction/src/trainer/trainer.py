from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
import os
import random
from typing import Any

import numpy as np
import torch

from utils.wandb_tracking import WandbRun, WandbSettings

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    mse: float
    rmse: float
    mae: float


@dataclass(slots=True)
class TorchRegressionTrainer:
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 200
    patience: int = 20
    early_stopping: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    min_learning_rate: float = 1.0e-5
    gradient_clip_norm: float | None = 1.0
    loss: str = "smooth_l1"
    device: str = "auto"
    random_state: int = 42
    deterministic: bool = False
    disable_tf32: bool = False
    wandb_settings: WandbSettings | None = None

    def fit(self, model: Any, train_loader: Any, validation_loader: Any | None = None) -> list[dict[str, float]]:
        """Fit.

        Args:
            model (Any): Model to process.
            train_loader (Any): Value used by the function.
            validation_loader (Any | None): Any | None value used by the function.

        Returns:
            list[dict[str, float]]: Result of the function.
        """
        prepare_torch_runtime(
            seed=self.random_state,
            deterministic=self.deterministic,
            disable_tf32=self.disable_tf32,
        )
        device = resolve_device(self.device)
        model.to(device)
        log.info("Resolved training device: %s", device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.scheduler_factor),
            patience=int(self.scheduler_patience),
            min_lr=float(self.min_learning_rate),
        )
        loss_fn = build_loss(self.loss)

        best_state = deepcopy(model.state_dict())
        best_metric = float("inf")
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        with WandbRun(self.wandb_settings or WandbSettings()) as wandb_run:
            for epoch in range(int(self.max_epochs)):
                train_metrics = self._run_epoch(
                    model=model,
                    loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=device,
                )
                validation_metrics = train_metrics
                if validation_loader is not None:
                    validation_metrics = self._run_epoch(
                        model=model,
                        loader=validation_loader,
                        loss_fn=loss_fn,
                        optimizer=None,
                        device=device,
                    )

                scheduler.step(validation_metrics.loss)
                current_lr = float(optimizer.param_groups[0]["lr"])

                if validation_metrics.loss + 1.0e-6 < best_metric:
                    best_metric = float(validation_metrics.loss)
                    best_state = deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    improved = True
                else:
                    epochs_without_improvement += 1
                    improved = False

                epoch_record = {
                    "epoch": float(epoch + 1),
                    "train_loss": train_metrics.loss,
                    "train_mse": train_metrics.mse,
                    "train_rmse": train_metrics.rmse,
                    "train_mae": train_metrics.mae,
                    "validation_loss": validation_metrics.loss,
                    "validation_mse": validation_metrics.mse,
                    "validation_rmse": validation_metrics.rmse,
                    "validation_mae": validation_metrics.mae,
                    "learning_rate": current_lr,
                }
                history.append(epoch_record)
                wandb_run.log(epoch_record, step=int(epoch + 1))
                log.info(
                    (
                        "Epoch %d/%d | "
                        "train_loss=%.6f train_mse=%.6f train_rmse=%.6f train_mae=%.6f | "
                        "val_loss=%.6f val_mse=%.6f val_rmse=%.6f val_mae=%.6f | "
                        "lr=%.6g patience=%d/%d%s"
                    ),
                    int(epoch + 1),
                    int(self.max_epochs),
                    train_metrics.loss,
                    train_metrics.mse,
                    train_metrics.rmse,
                    train_metrics.mae,
                    validation_metrics.loss,
                    validation_metrics.mse,
                    validation_metrics.rmse,
                    validation_metrics.mae,
                    current_lr,
                    int(epochs_without_improvement),
                    int(self.patience),
                    " *" if improved else "",
                )

                if self.early_stopping and epochs_without_improvement >= int(self.patience):
                    log.info("Early stopping at epoch %d | best_val_loss=%.6f", epoch + 1, best_metric)
                    break

        model.load_state_dict(best_state)
        return history

    def _run_epoch(
        self,
        *,
        model: Any,
        loader: Any,
        loss_fn: Any,
        optimizer: Any | None,
        device: Any,
    ) -> EpochMetrics:
        """Run epoch.

        Args:
            model (Any): Model to process.
            loader (Any): Value used by the function.
            loss_fn (Any): Value used by the function.
            optimizer (Any | None): Any | None value used by the function.
            device (Any): Value used by the function.

        Returns:
            EpochMetrics: Result of the function.
        """
        is_training = optimizer is not None
        model.train(is_training)
        total_loss = 0.0
        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_examples = 0

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for features, targets in loader:
                features = move_batch_to_device(features, device)
                targets = targets.to(device)

                if is_training:
                    optimizer.zero_grad(set_to_none=True)

                predictions = model(features)
                loss = loss_fn(predictions, targets)

                if is_training:
                    loss.backward()
                    if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=float(self.gradient_clip_norm),
                        )
                    optimizer.step()

                batch_size = batch_size_from_features(features)
                total_loss += float(loss.detach().cpu().item()) * batch_size
                errors = (predictions.detach() - targets.detach()).reshape(batch_size, -1)
                total_squared_error += float(torch.sum(errors.square()).cpu().item())
                total_absolute_error += float(torch.sum(errors.abs()).cpu().item())
                total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError("Cannot train or evaluate on an empty DataLoader.")
        mse = total_squared_error / total_examples
        return EpochMetrics(
            loss=total_loss / total_examples,
            mse=mse,
            rmse=math.sqrt(mse),
            mae=total_absolute_error / total_examples,
        )


def build_loss(name: str) -> Any:
    """Build loss.

    Args:
        name (str): Name used by the function.

    Returns:
        Any: Result of the function.
    """
    normalized = str(name).lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized == "smooth_l1":
        return torch.nn.SmoothL1Loss(beta=0.5)
    raise ValueError(f"Unsupported Torch regression loss '{name}'.")


def resolve_device(device: str) -> Any:
    """Resolve device.

    Args:
        device (str): str value used by the function.

    Returns:
        Any: Result of the function.
    """
    normalized = str(device).lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def move_batch_to_device(features: Any, device: Any) -> Any:
    """Move tensor or nested tensor batch features to the selected device."""
    if isinstance(features, torch.Tensor):
        return features.to(device)
    if isinstance(features, dict):
        return {key: move_batch_to_device(value, device) for key, value in features.items()}
    if isinstance(features, tuple):
        return tuple(move_batch_to_device(value, device) for value in features)
    if isinstance(features, list):
        return [move_batch_to_device(value, device) for value in features]
    return features


def batch_size_from_features(features: Any) -> int:
    """Resolve batch size for tensor or nested tensor batch features."""
    if isinstance(features, torch.Tensor):
        return int(features.shape[0])
    if isinstance(features, dict):
        for key in ("sequences", "features", "scalar_features", "lengths"):
            value = features.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        for value in features.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
    if isinstance(features, (tuple, list)):
        for value in features:
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
    raise ValueError("Could not determine batch size from DataLoader features.")


def prepare_torch_runtime(
    *,
    seed: int,
    deterministic: bool = False,
    disable_tf32: bool = False,
) -> None:
    """Prepare PyTorch runtime settings for reproducible training."""
    set_seed(seed)
    configure_torch_runtime(
        deterministic=deterministic,
        disable_tf32=disable_tf32,
    )


def set_seed(seed: int) -> None:
    """Set random seed.

    Args:
        seed (int): int value used by the function.

    Returns:
        None: None.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_torch_runtime(
    *,
    deterministic: bool = False,
    disable_tf32: bool = False,
) -> None:
    """Configure backend flags that affect reproducibility across devices."""
    if deterministic and torch.cuda.is_available():
        # Required by some deterministic CUDA kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = not bool(disable_tf32)

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not bool(disable_tf32)

    torch.use_deterministic_algorithms(bool(deterministic))
