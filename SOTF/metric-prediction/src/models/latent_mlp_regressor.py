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

from .base import RegressorProtocol


_DEFAULT_HIDDEN_LAYER_SIZES = (64, 32)
_DEFAULT_ACTIVATION = "gelu"
_DEFAULT_DROPOUT = 0.0
_DEFAULT_USE_LAYER_NORM = True
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_LEARNING_RATE = 1.0e-3
_DEFAULT_WEIGHT_DECAY = 1.0e-3
_DEFAULT_MAX_EPOCHS = 1000
_DEFAULT_PATIENCE = 60
_DEFAULT_VALIDATION_FRACTION = 0.15
_DEFAULT_EARLY_STOPPING = True
_DEFAULT_SCHEDULER_FACTOR = 0.5
_DEFAULT_SCHEDULER_PATIENCE = 15
_DEFAULT_MIN_LEARNING_RATE = 1.0e-5
_DEFAULT_GRADIENT_CLIP_NORM = 1.0
_DEFAULT_SCALE_FEATURES = True
_DEFAULT_SCALE_TARGET = True
_DEFAULT_RANDOM_STATE = 42
_DEFAULT_DEVICE = "auto"
_DEFAULT_LOSS = "smooth_l1"


@dataclass(slots=True)
class LatentMLPConfig:
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
    deterministic: bool = False
    disable_tf32: bool = False
    wandb_project: str | None = None
    wandb_group: str | None = None
    wandb_api_key_file: str | None = None
    wandb_run_name: str | None = None
    wandb_config: dict[str, Any] | None = None


class LatentMLPNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: str,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        """Initialize instance.

        Args:
            input_dim (int): int value used by the function.
            hidden_layer_sizes (Sequence[int]): Sequence[int] value used by the function.
            activation (str): str value used by the function.
            dropout (float): float value used by the function.
            use_layer_norm (bool): bool value used by the function.

        Returns:
            None: None.
        """
        super().__init__()
        nn = torch.nn
        layers: list[Any] = []
        current_dim = int(input_dim)
        activation_layer = _activation(activation)

        for hidden_dim in hidden_layer_sizes:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            features (torch.Tensor): torch.Tensor value used by the function.

        Returns:
            torch.Tensor: Result of the function.
        """
        return self.network(features)


class LatentMLPRegressor(RegressorProtocol):
    def __init__(self, **params: Any) -> None:
        """Initialize instance.

        Args:
            **params (Any): Value used by the function.

        Returns:
            None: None.
        """
        self.config = _parse_config(params)
        self._device = resolve_device(self.config.device)
        self._network: LatentMLPNetwork | None = None
        self._feature_scaler: StandardScalerState | None = None
        self._target_scaler = TargetScalerState(mean=0.0, scale=1.0)
        self.training_history_: list[dict[str, float]] = []
        self.feature_importances_: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Getstate.

        Returns:
            dict[str, Any]: Result of the function.
        """
        state = dict(self.__dict__)
        state["_device"] = str(self._device)
        state["_network_state"] = None
        if self._network is not None:
            state["_network_state"] = {
                key: value.detach().cpu()
                for key, value in self._network.state_dict().items()
            }
        state.pop("_network", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Setstate.

        Args:
            state (Mapping[str, Any]): Mapping[str, Any] value used by the function.

        Returns:
            None: None.
        """
        self.__dict__.update(dict(state))
        self._device = resolve_device(self.config.device)
        network_state = self.__dict__.pop("_network_state", None)
        self._network = None
        if network_state is not None and self._feature_scaler is not None:
            self._network = self._build_network(input_dim=len(self._feature_scaler.mean))
            self._network.load_state_dict(network_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LatentMLPRegressor":
        """Fit.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.
            y (pd.Series): pd.Series value used by the function.

        Returns:
            "LatentMLPRegressor": Result of the function.
        """
        X_array = X.to_numpy(dtype=np.float32, copy=True)
        y_array = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        if X_array.ndim != 2:
            raise ValueError(f"Expected a 2D latent feature matrix, received shape {X_array.shape}.")
        if len(X_array) < 2:
            raise ValueError("LatentMLPRegressor requires at least two training samples.")

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
        self._network = self._build_network(input_dim=X_scaled.shape[1])
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
            wandb_settings=WandbSettings(
                project=self.config.wandb_project,
                group=self.config.wandb_group,
                api_key_file=self.config.wandb_api_key_file,
                run_name=self.config.wandb_run_name,
                config=self.config.wandb_config or {},
            ),
        )
        self.training_history_ = trainer.fit(self._network, loaders.train_loader, loaders.validation_loader)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.

        Returns:
            np.ndarray: Result of the function.
        """
        if self._network is None or self._feature_scaler is None:
            raise RuntimeError("LatentMLPRegressor.predict called before fit.")

        X_array = X.to_numpy(dtype=np.float32, copy=True)
        X_scaled = apply_standard_scaler(X_array, self._feature_scaler)
        X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32, device=self._device)

        self._network.eval()
        with torch.no_grad():
            predictions = self._network(X_tensor).detach().cpu().numpy().reshape(-1, 1)
        restored = invert_target_scaler(predictions, self._target_scaler)
        return restored.reshape(-1).astype("float64", copy=False)

    def _build_network(self, *, input_dim: int) -> LatentMLPNetwork:
        """Build network.

        Args:
            input_dim (int): int value used by the function.

        Returns:
            LatentMLPNetwork: Result of the function.
        """
        return LatentMLPNetwork(
            input_dim=input_dim,
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            dropout=self.config.dropout,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self._device)


def build_latent_mlp_regressor(params: Mapping[str, Any]) -> RegressorProtocol:
    """Build latent mlp regressor.

    Args:
        params (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        RegressorProtocol: Result of the function.
    """
    return LatentMLPRegressor(**dict(params))


def _parse_config(params: Mapping[str, Any]) -> LatentMLPConfig:
    """Parse config.

    Args:
        params (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        LatentMLPConfig: Result of the function.
    """
    raw = dict(params)
    hidden_layer_sizes = _parse_hidden_layers(raw.pop("hidden_layer_sizes", _DEFAULT_HIDDEN_LAYER_SIZES))

    learning_rate = raw.pop("learning_rate", _DEFAULT_LEARNING_RATE)
    learning_rate_init = raw.pop("learning_rate_init", None)
    if learning_rate_init is not None:
        learning_rate = learning_rate_init

    weight_decay = raw.pop("weight_decay", _DEFAULT_WEIGHT_DECAY)
    alpha = raw.pop("alpha", None)
    if alpha is not None:
        weight_decay = alpha

    max_epochs = raw.pop("max_epochs", _DEFAULT_MAX_EPOCHS)
    max_iter = raw.pop("max_iter", None)
    if max_iter is not None:
        max_epochs = max_iter

    patience = raw.pop("patience", _DEFAULT_PATIENCE)
    n_iter_no_change = raw.pop("n_iter_no_change", None)
    if n_iter_no_change is not None:
        patience = n_iter_no_change

    solver = raw.pop("solver", None)
    if solver is not None and str(solver).lower() != "adam":
        raise ValueError(f"LatentMLPRegressor only supports solver='adam', received '{solver}'.")

    raw.pop("learning_rate_schedule", None)
    raw.pop("learning_rate_mode", None)

    config = LatentMLPConfig(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=str(raw.pop("activation", _DEFAULT_ACTIVATION)),
        dropout=float(raw.pop("dropout", _DEFAULT_DROPOUT)),
        use_layer_norm=bool(raw.pop("use_layer_norm", _DEFAULT_USE_LAYER_NORM)),
        batch_size=raw.pop("batch_size", _DEFAULT_BATCH_SIZE),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
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
        deterministic=bool(raw.pop("deterministic", False)),
        disable_tf32=bool(raw.pop("disable_tf32", False)),
        wandb_project=_optional_str(raw.pop("wandb_project", None)),
        wandb_group=_optional_str(raw.pop("wandb_group", None)),
        wandb_api_key_file=_optional_str(raw.pop("wandb_api_key_file", None)),
        wandb_run_name=_optional_str(raw.pop("wandb_run_name", None)),
        wandb_config=_optional_dict(raw.pop("wandb_config", None)),
    )
    if raw:
        raise ValueError(f"Unsupported LatentMLPRegressor parameters: {', '.join(sorted(raw))}.")
    return config


def _parse_hidden_layers(value: Any) -> tuple[int, ...]:
    """Parse hidden layers.

    Args:
        value (Any): Value to process.

    Returns:
        tuple[int, ...]: Result of the function.
    """
    if isinstance(value, (list, tuple)):
        layers = tuple(int(item) for item in value)
    else:
        layers = (int(value),)
    if not layers:
        raise ValueError("LatentMLPRegressor requires at least one hidden layer.")
    return layers


def _optional_float(value: Any) -> float | None:
    """Optional float.

    Args:
        value (Any): Value to process.

    Returns:
        float | None: Result of the function.
    """
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    """Optional str.

    Args:
        value (Any): Value to process.

    Returns:
        str | None: Result of the function.
    """
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    """Optional dict.

    Args:
        value (Any): Value to process.

    Returns:
        dict[str, Any] | None: Result of the function.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("wandb_config must be a mapping when provided.")


def _activation(name: str) -> Any:
    """Activation.

    Args:
        name (str): Name used by the function.

    Returns:
        Any: Result of the function.
    """
    activations = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    normalized = str(name).lower()
    if normalized not in activations:
        raise ValueError(f"Unsupported LatentMLPRegressor activation '{name}'.")
    return activations[normalized]
