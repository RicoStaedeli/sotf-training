from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from data.scalers import (
    StandardScalerState,
    TargetScalerState,
    apply_standard_scaler,
    apply_target_scaler,
    fit_standard_scaler,
    fit_target_scaler,
    invert_target_scaler,
)
from data.torch_datasets import build_regression_dataloaders
from trainer.trainer import TorchRegressionTrainer, prepare_torch_runtime, resolve_device
from utils.wandb_tracking import WandbSettings


_DEFAULT_HIDDEN_LAYER_SIZES = (128, 128)
_DEFAULT_ACTIVATION = "relu"
_DEFAULT_DROPOUT = 0.1
_DEFAULT_USE_LAYER_NORM = False
_DEFAULT_BATCH_SIZE = 512
_DEFAULT_LEARNING_RATE = 1.0e-3
_DEFAULT_WEIGHT_DECAY = 1.0e-4
_DEFAULT_MAX_EPOCHS = 120
_DEFAULT_PATIENCE = 15
_DEFAULT_VALIDATION_FRACTION = 0.15
_DEFAULT_EARLY_STOPPING = True
_DEFAULT_SCHEDULER_FACTOR = 0.5
_DEFAULT_SCHEDULER_PATIENCE = 8
_DEFAULT_MIN_LEARNING_RATE = 1.0e-5
_DEFAULT_GRADIENT_CLIP_NORM = 1.0
_DEFAULT_SCALE_FEATURES = True
_DEFAULT_SCALE_TARGET = True
_DEFAULT_RANDOM_STATE = 42
_DEFAULT_DEVICE = "auto"
_DEFAULT_LOSS = "smooth_l1"
_DEFAULT_MC_SAMPLES = 30
_DEFAULT_DETERMINISTIC = False
_DEFAULT_DISABLE_TF32 = False


@dataclass(frozen=True, slots=True)
class LCNetPrediction:
    mean: np.ndarray
    std: np.ndarray


@dataclass(slots=True)
class LCNetConfig:
    hidden_layer_sizes: tuple[int, ...] = _DEFAULT_HIDDEN_LAYER_SIZES
    activation: str = _DEFAULT_ACTIVATION
    dropout: float = _DEFAULT_DROPOUT
    use_layer_norm: bool = _DEFAULT_USE_LAYER_NORM
    batch_size: int | str = _DEFAULT_BATCH_SIZE
    learning_rate: float = _DEFAULT_LEARNING_RATE
    weight_decay: float = _DEFAULT_WEIGHT_DECAY
    max_epochs: int = _DEFAULT_MAX_EPOCHS
    patience: int = _DEFAULT_PATIENCE
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION
    early_stopping: bool = _DEFAULT_EARLY_STOPPING
    scheduler_factor: float = _DEFAULT_SCHEDULER_FACTOR
    scheduler_patience: int = _DEFAULT_SCHEDULER_PATIENCE
    min_learning_rate: float = _DEFAULT_MIN_LEARNING_RATE
    gradient_clip_norm: float | None = _DEFAULT_GRADIENT_CLIP_NORM
    scale_features: bool = _DEFAULT_SCALE_FEATURES
    scale_target: bool = _DEFAULT_SCALE_TARGET
    random_state: int = _DEFAULT_RANDOM_STATE
    device: str = _DEFAULT_DEVICE
    loss: str = _DEFAULT_LOSS
    mc_samples: int = _DEFAULT_MC_SAMPLES
    deterministic: bool = _DEFAULT_DETERMINISTIC
    disable_tf32: bool = _DEFAULT_DISABLE_TF32


class LCNetNetwork(torch.nn.Module):
    """Small fully connected network used as an LCNet-style curve predictor."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: str,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = int(input_dim)
        activation_layer = _activation(activation)
        for hidden_dim in hidden_layer_sizes:
            hidden_dim = int(hidden_dim)
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            if bool(use_layer_norm):
                layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class LCNetRegressor:
    """Bayesian neural-network learning-curve predictor with MC dropout.

    The original LCNet baseline predicts learning-curve values from partially
    observed curves using a Bayesian neural network. This implementation keeps
    the same comparison surface for the thesis experiments: it only consumes
    learning-curve prefix and progress features, trains an MLP across completed
    training curves, and estimates predictive uncertainty with Monte Carlo
    dropout at inference time.
    """

    def __init__(self, **params: Any) -> None:
        self.config = _parse_config(params)
        self._device = resolve_device(self.config.device)
        self._network: LCNetNetwork | None = None
        self._feature_scaler: StandardScalerState | None = None
        self._target_scaler = TargetScalerState(mean=0.0, scale=1.0)
        self.training_history_: list[dict[str, float]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LCNetRegressor":
        X_array = X.to_numpy(dtype=np.float32, copy=True)
        y_array = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        if X_array.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, received shape {X_array.shape}.")
        if len(X_array) < 2:
            raise ValueError("LCNetRegressor requires at least two training samples.")

        self._feature_scaler = fit_standard_scaler(X_array, enabled=self.config.scale_features)
        self._target_scaler = fit_target_scaler(y_array, enabled=self.config.scale_target)
        X_scaled = apply_standard_scaler(X_array, self._feature_scaler)
        y_scaled = apply_target_scaler(y_array, self._target_scaler)

        loaders = build_regression_dataloaders(
            X_scaled,
            y_scaled,
            batch_size=self.config.batch_size,
            validation_fraction=self.config.validation_fraction,
            random_state=self.config.random_state,
        )
        prepare_torch_runtime(
            seed=self.config.random_state,
            deterministic=self.config.deterministic,
            disable_tf32=self.config.disable_tf32,
        )
        self._network = LCNetNetwork(
            input_dim=X_scaled.shape[1],
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            dropout=self.config.dropout,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self._device)
        trainer = TorchRegressionTrainer(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_epochs=self.config.max_epochs,
            patience=self.config.patience,
            early_stopping=self.config.early_stopping,
            scheduler_factor=self.config.scheduler_factor,
            scheduler_patience=self.config.scheduler_patience,
            min_learning_rate=self.config.min_learning_rate,
            gradient_clip_norm=self.config.gradient_clip_norm,
            loss=self.config.loss,
            device=self.config.device,
            random_state=self.config.random_state,
            deterministic=self.config.deterministic,
            disable_tf32=self.config.disable_tf32,
            wandb_settings=WandbSettings(),
        )
        self.training_history_ = trainer.fit(self._network, loaders.train_loader, loaders.validation_loader)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_distribution(X).mean

    def predict_distribution(self, X: pd.DataFrame) -> LCNetPrediction:
        if self._network is None or self._feature_scaler is None:
            raise RuntimeError("LCNetRegressor.predict_distribution called before fit.")

        X_array = X.to_numpy(dtype=np.float32, copy=True)
        X_scaled = apply_standard_scaler(X_array, self._feature_scaler)
        X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32, device=self._device)

        n_samples = max(1, int(self.config.mc_samples))
        samples: list[np.ndarray] = []
        torch.manual_seed(int(self.config.random_state))
        self._network.train()
        with torch.no_grad():
            for _ in range(n_samples):
                prediction = self._network(X_tensor).detach().cpu().numpy().reshape(-1, 1)
                restored = invert_target_scaler(prediction, self._target_scaler)
                samples.append(restored.reshape(-1).astype("float64", copy=False))
        stacked = np.stack(samples, axis=0)
        if n_samples == 1:
            std = np.zeros(stacked.shape[1], dtype="float64")
        else:
            std = stacked.std(axis=0, ddof=1).astype("float64", copy=False)
        mean = stacked.mean(axis=0).astype("float64", copy=False)
        return LCNetPrediction(mean=mean, std=std)


def build_lcnet_features(
    *,
    full_metadata: pd.DataFrame,
    forecast_metadata: pd.DataFrame,
    target_column: str,
    target_mode: str,
    horizon: int,
    max_history_points: int = 8,
) -> pd.DataFrame:
    """Build fixed-length learning-curve prefix features for LCNet."""
    if max_history_points <= 0:
        raise ValueError("max_history_points must be positive.")
    required = {"model_id", "epoch", target_column}
    missing = sorted(required.difference(full_metadata.columns))
    if missing:
        raise ValueError("full_metadata is missing required columns: " + ", ".join(missing))
    required_forecast = {"model_id", "epoch", "current_target", "max_epoch", "history_length", "progress_fraction"}
    missing_forecast = sorted(required_forecast.difference(forecast_metadata.columns))
    if missing_forecast:
        raise ValueError("forecast_metadata is missing required columns: " + ", ".join(missing_forecast))

    curve_frame = _curve_frame(full_metadata, target_column=target_column)
    curves_by_model = {
        str(model_id): model_curve
        for model_id, model_curve in curve_frame.groupby("model_id", sort=False)
    }

    records: list[dict[str, float]] = []
    for _, row in forecast_metadata.iterrows():
        model_id = str(row["model_id"])
        current_epoch = float(row["epoch"])
        model_curve = curves_by_model.get(model_id, curve_frame.iloc[0:0])
        observed = model_curve[model_curve["epoch"] <= current_epoch]
        records.append(
            _prefix_features(
                observed_epochs=observed["epoch"].to_numpy(dtype="float64"),
                observed_values=observed[target_column].to_numpy(dtype="float64"),
                row=row,
                target_mode=target_mode,
                horizon=horizon,
                max_history_points=max_history_points,
            )
        )

    features = pd.DataFrame(records, index=forecast_metadata.index)
    features.index.name = forecast_metadata.index.name or "sample_idx"
    return features.astype("float64")


def _prefix_features(
    *,
    observed_epochs: np.ndarray,
    observed_values: np.ndarray,
    row: pd.Series,
    target_mode: str,
    horizon: int,
    max_history_points: int,
) -> dict[str, float]:
    epochs, values = _clean_history(observed_epochs, observed_values)
    current_epoch = float(row["epoch"])
    max_epoch = float(row["max_epoch"])
    current_target = float(row["current_target"])
    target_epoch = _target_epoch(row, target_mode=target_mode, horizon=horizon)

    if values.size == 0:
        epochs = np.asarray([current_epoch], dtype="float64")
        values = np.asarray([current_target], dtype="float64")

    diffs = np.diff(values)
    epoch_diffs = np.diff(epochs)
    safe_epoch_span = max(float(epochs[-1] - epochs[0]), 1.0)
    safe_max_epoch = max(max_epoch, 1.0)
    safe_target_gap = max(float(target_epoch - current_epoch), 0.0)

    features: dict[str, float] = {
        "epoch": current_epoch,
        "max_epoch": max_epoch,
        "target_epoch": target_epoch,
        "progress_fraction": _safe_divide(current_epoch, safe_max_epoch),
        "target_progress_fraction": _safe_divide(target_epoch, safe_max_epoch),
        "epochs_remaining": max(max_epoch - current_epoch, 0.0),
        "target_epochs_remaining": safe_target_gap,
        "history_length": float(row["history_length"]),
        "observed_points": float(values.size),
        "current_target": current_target,
        "first_target": float(values[0]),
        "prefix_min": float(np.min(values)),
        "prefix_max": float(np.max(values)),
        "prefix_mean": float(np.mean(values)),
        "prefix_std": float(np.std(values)) if values.size > 1 else 0.0,
        "total_gain": float(values[-1] - values[0]),
        "prefix_slope": float((values[-1] - values[0]) / safe_epoch_span),
        "last_delta": float(diffs[-1]) if diffs.size else 0.0,
        "mean_delta": float(np.mean(diffs)) if diffs.size else 0.0,
        "std_delta": float(np.std(diffs)) if diffs.size > 1 else 0.0,
        "mean_epoch_gap": float(np.mean(epoch_diffs)) if epoch_diffs.size else 0.0,
    }

    recent_values = values[::-1]
    recent_epochs = epochs[::-1]
    for lag in range(max_history_points):
        if lag < recent_values.size:
            value = float(recent_values[lag])
            epoch = float(recent_epochs[lag])
            mask = 1.0
        else:
            value = current_target
            epoch = current_epoch
            mask = 0.0
        features[f"value_lag_{lag}"] = value
        features[f"epoch_lag_{lag}"] = epoch
        features[f"value_delta_lag_{lag}"] = current_target - value
        features[f"lag_mask_{lag}"] = mask
    return features


def _curve_frame(full_metadata: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    frame = full_metadata.loc[:, ["model_id", "epoch", target_column]].copy()
    frame["model_id"] = frame["model_id"].astype(str)
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=["model_id", "epoch", target_column])
    return frame.sort_values(["model_id", "epoch"], kind="stable")


def _target_epoch(row: pd.Series, *, target_mode: str, horizon: int) -> float:
    mode = str(target_mode).lower()
    if mode == "final_value":
        return float(row["max_epoch"])
    if mode == "horizon_value":
        return float(row["epoch"]) + float(horizon)
    raise ValueError(f"Unsupported target_mode={target_mode!r}.")


def _clean_history(epochs: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(epochs, dtype="float64")
    y = np.asarray(values, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return x, y
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_epochs, unique_indices = np.unique(x, return_index=True)
    return unique_epochs.astype("float64"), y[unique_indices].astype("float64")


def _parse_config(params: Mapping[str, Any]) -> LCNetConfig:
    raw = dict(params)
    hidden_layer_sizes = _parse_hidden_layers(raw.pop("hidden_layer_sizes", _DEFAULT_HIDDEN_LAYER_SIZES))
    max_epochs = raw.pop("max_epochs", raw.pop("max_iter", _DEFAULT_MAX_EPOCHS))
    patience = raw.pop("patience", raw.pop("n_iter_no_change", _DEFAULT_PATIENCE))
    config = LCNetConfig(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=str(raw.pop("activation", _DEFAULT_ACTIVATION)),
        dropout=float(raw.pop("dropout", _DEFAULT_DROPOUT)),
        use_layer_norm=bool(raw.pop("use_layer_norm", _DEFAULT_USE_LAYER_NORM)),
        batch_size=raw.pop("batch_size", _DEFAULT_BATCH_SIZE),
        learning_rate=float(raw.pop("learning_rate", _DEFAULT_LEARNING_RATE)),
        weight_decay=float(raw.pop("weight_decay", _DEFAULT_WEIGHT_DECAY)),
        max_epochs=int(max_epochs),
        patience=int(patience),
        validation_fraction=float(raw.pop("validation_fraction", _DEFAULT_VALIDATION_FRACTION)),
        early_stopping=bool(raw.pop("early_stopping", _DEFAULT_EARLY_STOPPING)),
        scheduler_factor=float(raw.pop("scheduler_factor", _DEFAULT_SCHEDULER_FACTOR)),
        scheduler_patience=int(raw.pop("scheduler_patience", _DEFAULT_SCHEDULER_PATIENCE)),
        min_learning_rate=float(raw.pop("min_learning_rate", _DEFAULT_MIN_LEARNING_RATE)),
        gradient_clip_norm=_optional_float(raw.pop("gradient_clip_norm", _DEFAULT_GRADIENT_CLIP_NORM)),
        scale_features=bool(raw.pop("scale_features", _DEFAULT_SCALE_FEATURES)),
        scale_target=bool(raw.pop("scale_target", _DEFAULT_SCALE_TARGET)),
        random_state=int(raw.pop("random_state", _DEFAULT_RANDOM_STATE)),
        device=str(raw.pop("device", _DEFAULT_DEVICE)),
        loss=str(raw.pop("loss", _DEFAULT_LOSS)),
        mc_samples=int(raw.pop("mc_samples", _DEFAULT_MC_SAMPLES)),
        deterministic=bool(raw.pop("deterministic", _DEFAULT_DETERMINISTIC)),
        disable_tf32=bool(raw.pop("disable_tf32", _DEFAULT_DISABLE_TF32)),
    )
    solver = raw.pop("solver", None)
    if solver is not None and str(solver).lower() != "adam":
        raise ValueError(f"LCNetRegressor only supports solver='adam', received '{solver}'.")
    if raw:
        raise ValueError(f"Unsupported LCNetRegressor parameters: {', '.join(sorted(raw))}.")
    return config


def _parse_hidden_layers(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        layers = tuple(int(item) for item in value)
    else:
        layers = (int(value),)
    if not layers:
        raise ValueError("LCNetRegressor requires at least one hidden layer.")
    return layers


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _activation(name: str) -> Any:
    activations = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
    }
    normalized = str(name).lower()
    if normalized not in activations:
        raise ValueError(f"Unsupported LCNetRegressor activation '{name}'.")
    return activations[normalized]
