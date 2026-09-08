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
from data.torch_datasets import (
    build_window_sequence_dataloaders,
    extract_window_sequence_features,
    infer_sequence_input_dim,
)
from forecasting.dataset import WINDOW_SEQUENCE_COLUMN
from trainer.trainer import TorchRegressionTrainer, prepare_torch_runtime, resolve_device
from utils.wandb_tracking import WandbSettings

from .base import RegressorProtocol


@dataclass(slots=True)
class LatentWindowLSTMConfig:
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0
    head_hidden_layer_sizes: tuple[int, ...] = (64,)
    activation: str = "gelu"
    sequence_column: str = WINDOW_SEQUENCE_COLUMN
    batch_size: int | str = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-3
    max_epochs: int = 500
    patience: int = 40
    validation_fraction: float = 0.15
    early_stopping: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15
    min_learning_rate: float = 1.0e-5
    gradient_clip_norm: float | None = 1.0
    scale_features: bool = True
    scale_target: bool = True
    random_state: int = 42
    device: str = "auto"
    loss: str = "smooth_l1"
    deterministic: bool = False
    disable_tf32: bool = False
    wandb_project: str | None = None
    wandb_group: str | None = None
    wandb_api_key_file: str | None = None
    wandb_run_name: str | None = None
    wandb_config: dict[str, Any] | None = None


class LatentWindowLSTMNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        scalar_dim: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        head_hidden_layer_sizes: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        encoded_dim = int(hidden_size) * (2 if bool(bidirectional) else 1)
        head_input_dim = encoded_dim + int(scalar_dim)
        layers: list[torch.nn.Module] = []
        current_dim = head_input_dim
        activation_layer = _activation(activation)
        for hidden_dim in head_hidden_layer_sizes:
            hidden_dim = int(hidden_dim)
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            layers.append(activation_layer())
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, 1))
        self.head = torch.nn.Sequential(*layers)

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        sequences = features["sequences"]
        lengths = features["lengths"]

        # Fast path: when every sample in the batch already has the same sequence
        # length, we can skip packing entirely and run the dense LSTM directly.
        if bool(torch.all(lengths == lengths[0]).item()) and int(lengths[0].item()) == int(sequences.shape[1]):
            _, (hidden, _) = self.lstm(sequences)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sequences,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            encoded = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            encoded = hidden[-1]

        scalar_features = features.get("scalar_features")
        if scalar_features is not None and scalar_features.shape[1] > 0:
            encoded = torch.cat([encoded, scalar_features], dim=1)
        return self.head(encoded)


class LatentWindowLSTMRegressor(RegressorProtocol):
    def __init__(self, **params: Any) -> None:
        self.config = _parse_config(params)
        self._device = resolve_device(self.config.device)
        self._network: LatentWindowLSTMNetwork | None = None
        self._sequence_scaler: StandardScalerState | None = None
        self._scalar_scaler: StandardScalerState | None = None
        self._target_scaler = TargetScalerState(mean=0.0, scale=1.0)
        self._input_dim: int | None = None
        self._scalar_dim: int | None = None
        self.training_history_: list[dict[str, float]] = []
        self.feature_importances_: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
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
        self.__dict__.update(dict(state))
        self._device = resolve_device("auto")
        network_state = self.__dict__.pop("_network_state", None)
        self._network = None
        if network_state is not None and self._input_dim is not None and self._scalar_dim is not None:
            self._network = self._build_network(input_dim=self._input_dim, scalar_dim=self._scalar_dim)
            self._network.load_state_dict(network_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LatentWindowLSTMRegressor":
        sequences, scalar_features = extract_window_sequence_features(
            X,
            sequence_column=self.config.sequence_column,
        )
        y_array = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        if len(sequences) < 2:
            raise ValueError("LatentWindowLSTMRegressor requires at least two training samples.")

        self._input_dim = infer_sequence_input_dim(sequences)
        self._scalar_dim = int(scalar_features.shape[1])
        self._sequence_scaler = fit_standard_scaler(
            np.concatenate(sequences, axis=0),
            enabled=self.config.scale_features,
        )
        self._scalar_scaler = fit_standard_scaler(scalar_features, enabled=self.config.scale_features)
        self._target_scaler = fit_target_scaler(y_array, enabled=self.config.scale_target)

        scaled_sequences = [apply_standard_scaler(sequence, self._sequence_scaler) for sequence in sequences]
        scaled_scalar_features = apply_standard_scaler(scalar_features, self._scalar_scaler)
        scaled_targets = apply_target_scaler(y_array, self._target_scaler)

        loaders = build_window_sequence_dataloaders(
            scaled_sequences,
            scaled_scalar_features,
            scaled_targets,
            batch_size=self.config.batch_size,
            validation_fraction=self.config.validation_fraction,
            random_state=self.config.random_state,
            shuffle=True,
            split_validation=True,
        )
        prepare_torch_runtime(
            seed=self.config.random_state,
            deterministic=self.config.deterministic,
            disable_tf32=self.config.disable_tf32,
        )
        self._network = self._build_network(input_dim=self._input_dim, scalar_dim=self._scalar_dim)
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
        if self._network is None or self._sequence_scaler is None or self._scalar_scaler is None:
            raise RuntimeError("LatentWindowLSTMRegressor.predict called before fit.")

        sequences, scalar_features = extract_window_sequence_features(
            X,
            sequence_column=self.config.sequence_column,
        )
        input_dim = infer_sequence_input_dim(sequences)
        if input_dim != self._input_dim:
            raise ValueError(f"Expected window embedding dimension {self._input_dim}, received {input_dim}.")
        if int(scalar_features.shape[1]) != self._scalar_dim:
            raise ValueError(f"Expected {self._scalar_dim} scalar feature columns, received {scalar_features.shape[1]}.")

        scaled_sequences = [apply_standard_scaler(sequence, self._sequence_scaler) for sequence in sequences]
        scaled_scalar_features = apply_standard_scaler(scalar_features, self._scalar_scaler)
        dummy_targets = np.zeros((len(sequences), 1), dtype=np.float32)
        loaders = build_window_sequence_dataloaders(
            scaled_sequences,
            scaled_scalar_features,
            dummy_targets,
            batch_size=self.config.batch_size,
            validation_fraction=self.config.validation_fraction,
            random_state=self.config.random_state,
            shuffle=False,
            split_validation=False,
        )

        self._network.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in loaders.train_loader:
                features = {key: value.to(self._device) for key, value in features.items()}
                predictions = self._network(features).detach().cpu().numpy().reshape(-1, 1)
                batches.append(predictions)
        restored = invert_target_scaler(np.concatenate(batches, axis=0), self._target_scaler)
        return restored.reshape(-1).astype("float64", copy=False)

    def _build_network(self, *, input_dim: int, scalar_dim: int) -> LatentWindowLSTMNetwork:
        return LatentWindowLSTMNetwork(
            input_dim=input_dim,
            scalar_dim=scalar_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
            head_hidden_layer_sizes=self.config.head_hidden_layer_sizes,
            activation=self.config.activation,
        ).to(self._device)


def build_latent_window_lstm_regressor(params: Mapping[str, Any]) -> RegressorProtocol:
    return LatentWindowLSTMRegressor(**dict(params))


def _parse_config(params: Mapping[str, Any]) -> LatentWindowLSTMConfig:
    raw = dict(params)
    hidden_size = raw.pop("hidden_size", raw.pop("lstm_hidden_size", 128))
    head_hidden = raw.pop("head_hidden_layer_sizes", raw.pop("hidden_layer_sizes", (64,)))

    learning_rate = raw.pop("learning_rate", 1.0e-3)
    learning_rate_init = raw.pop("learning_rate_init", None)
    if learning_rate_init is not None:
        learning_rate = learning_rate_init

    weight_decay = raw.pop("weight_decay", 1.0e-3)
    alpha = raw.pop("alpha", None)
    if alpha is not None:
        weight_decay = alpha

    max_epochs = raw.pop("max_epochs", 500)
    max_iter = raw.pop("max_iter", None)
    if max_iter is not None:
        max_epochs = max_iter

    patience = raw.pop("patience", 40)
    n_iter_no_change = raw.pop("n_iter_no_change", None)
    if n_iter_no_change is not None:
        patience = n_iter_no_change

    raw.pop("learning_rate_schedule", None)
    raw.pop("learning_rate_mode", None)
    solver = raw.pop("solver", None)
    if solver is not None and str(solver).lower() != "adam":
        raise ValueError(f"LatentWindowLSTMRegressor only supports solver='adam', received '{solver}'.")

    config = LatentWindowLSTMConfig(
        hidden_size=int(hidden_size),
        num_layers=int(raw.pop("num_layers", 1)),
        bidirectional=bool(raw.pop("bidirectional", False)),
        dropout=float(raw.pop("dropout", 0.0)),
        head_hidden_layer_sizes=_parse_hidden_layers(head_hidden),
        activation=str(raw.pop("activation", "gelu")),
        sequence_column=str(raw.pop("sequence_column", WINDOW_SEQUENCE_COLUMN)),
        batch_size=raw.pop("batch_size", 16),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        max_epochs=int(max_epochs),
        patience=int(patience),
        validation_fraction=float(raw.pop("validation_fraction", 0.15)),
        early_stopping=bool(raw.pop("early_stopping", True)),
        scheduler_factor=float(raw.pop("scheduler_factor", 0.5)),
        scheduler_patience=int(raw.pop("scheduler_patience", 15)),
        min_learning_rate=float(raw.pop("min_learning_rate", 1.0e-5)),
        gradient_clip_norm=_optional_float(raw.pop("gradient_clip_norm", 1.0)),
        scale_features=bool(raw.pop("scale_features", True)),
        scale_target=bool(raw.pop("scale_target", True)),
        random_state=int(raw.pop("random_state", 42)),
        device=str(raw.pop("device", "auto")),
        loss=str(raw.pop("loss", "smooth_l1")),
        deterministic=bool(raw.pop("deterministic", False)),
        disable_tf32=bool(raw.pop("disable_tf32", False)),
        wandb_project=_optional_str(raw.pop("wandb_project", None)),
        wandb_group=_optional_str(raw.pop("wandb_group", None)),
        wandb_api_key_file=_optional_str(raw.pop("wandb_api_key_file", None)),
        wandb_run_name=_optional_str(raw.pop("wandb_run_name", None)),
        wandb_config=_optional_dict(raw.pop("wandb_config", None)),
    )
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if raw:
        raise ValueError(f"Unsupported LatentWindowLSTMRegressor parameters: {', '.join(sorted(raw))}.")
    return config


def _parse_hidden_layers(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        layers = tuple(int(item) for item in value)
    else:
        layers = (int(value),)
    if any(layer <= 0 for layer in layers):
        raise ValueError("All hidden layer sizes must be positive.")
    return layers


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("wandb_config must be a mapping when provided.")


def _activation(name: str) -> Any:
    activations = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    normalized = str(name).lower()
    if normalized not in activations:
        raise ValueError(f"Unsupported LatentWindowLSTMRegressor activation '{name}'.")
    return activations[normalized]
