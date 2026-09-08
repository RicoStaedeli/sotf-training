from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LatentHierarchicalTransformerConfig:
    """Configuration for the hierarchical transformer regressor.

    Architecture overview
    ---------------------
    Stage 1 – Local encoder (skipped when ``local_window_size == 1``):
        Input tokens are grouped into non-overlapping chunks of
        ``local_window_size``.  A shared ``TransformerEncoder`` processes each
        chunk independently and the output is mean-pooled to produce one
        *window embedding* per chunk.

    Stage 2 – Global encoder:
        A learnable ``[CLS]`` token is prepended to the window embeddings.
        A second ``TransformerEncoder`` with a proper padding mask aggregates
        the sequence.  The ``[CLS]`` output is passed to an MLP head.

    Parameters
    ----------
    d_model:
        Internal transformer dimension.  Input embeddings are projected to
        this size before the first stage.
    nhead:
        Number of attention heads (must divide ``d_model``).
    num_local_layers:
        Depth of the local encoder.  Ignored when ``local_window_size == 1``.
    num_global_layers:
        Depth of the global encoder.
    local_window_size:
        Chunk size for stage 1.  Use ``1`` for a flat (single-stage) transformer
        – the full sequence is fed directly into the global encoder.
    dim_feedforward:
        Feed-forward hidden size inside each transformer encoder layer.
    dropout:
        Dropout applied inside attention, FFN, and the regression head.
    head_hidden_layer_sizes:
        Hidden layer widths of the MLP regression head.
    activation:
        Activation function used in the regression head.
    use_positional_encoding:
        When ``True``, sinusoidal positional encodings are added before each
        transformer stage.
    sequence_column:
        DataFrame column that holds the variable-length sequence arrays.
    """

    # --- architecture ---
    d_model: int = 128
    nhead: int = 4
    num_local_layers: int = 2
    num_global_layers: int = 2
    local_window_size: int = 1
    dim_feedforward: int = 256
    dropout: float = 0.1
    head_hidden_layer_sizes: tuple[int, ...] = (64,)
    activation: str = "gelu"
    use_positional_encoding: bool = True
    
    # --- data ---
    sequence_column: str = WINDOW_SEQUENCE_COLUMN
    max_seq_len: int | None = None  # truncate sequences to this many tokens before attention
   
    # --- training ---
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
    
    # --- W&B ---
    wandb_project: str | None = None
    wandb_group: str | None = None
    wandb_api_key_file: str | None = None
    wandb_run_name: str | None = None
    wandb_config: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        # Number of even indices = ceil(d_model / 2).  Use this for div_term so
        # pe[:, 0::2] = sin(...) and pe[:, 1::2] = cos(...) both get the right
        # number of elements, for both even and odd d_model.
        n_even = (d_model + 1) // 2          # ceil(d_model / 2)
        n_odd  = d_model - n_even            # floor(d_model / 2)
        div_term = torch.exp(
            torch.arange(n_even, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)           # n_even slots
        pe[:, 1::2] = torch.cos(position * div_term[:n_odd])   # n_odd slots

        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x of shape [batch, seq_len, d_model]."""
        seq_len = x.shape[1]
        max_len = self.pe.shape[0]
        if seq_len > max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional encoding max_len {max_len}. "
                "Increase local_window_size to reduce the number of global positions, "
                "or set max_seq_len to truncate raw token sequences before processing."
            )
        x = x + self.pe[:seq_len].unsqueeze(0)  # broadcast over batch
        return self.drop(x)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class LatentHierarchicalTransformerNetwork(nn.Module):
    """Two-stage hierarchical transformer for variable-length latent sequences.

    Forward input (dict of tensors from the DataLoader):
        sequences:       [batch, max_seq_len, input_dim]  (zero-padded)
        lengths:         [batch]  (actual sequence lengths before padding)
        scalar_features: [batch, scalar_dim]  (optional side-channel, may be size-0)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        scalar_dim: int,
        d_model: int,
        nhead: int,
        num_local_layers: int,
        num_global_layers: int,
        local_window_size: int,
        dim_feedforward: int,
        dropout: float,
        head_hidden_layer_sizes: Sequence[int],
        activation: str,
        use_positional_encoding: bool,
    ) -> None:
        super().__init__()

        self.local_window_size = int(local_window_size)
        self.d_model = int(d_model)
        use_local_stage = self.local_window_size > 1 and int(num_local_layers) > 0

        # --- Input projection ---
        self.input_proj = nn.Linear(int(input_dim), int(d_model))

        # --- Stage 1: local encoder ---
        if use_local_stage:
            local_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,  # Pre-LN: more stable for small datasets
                activation="gelu",
            )
            self.local_encoder: nn.Module = nn.TransformerEncoder(
                local_layer,
                num_layers=int(num_local_layers),
                enable_nested_tensor=False,
            )
        else:
            self.local_encoder = nn.Identity()  # placeholder (never called)

        self._use_local_stage = use_local_stage

        # --- Positional encodings ---
        if bool(use_positional_encoding):
            # Local PE: reset position index for each window chunk
            self.local_pos_enc: nn.Module = _SinusoidalPositionalEncoding(
                int(d_model), dropout=float(dropout)
            )
            # Global PE: over the full sequence of window embeddings (+1 for CLS)
            self.global_pos_enc: nn.Module = _SinusoidalPositionalEncoding(
                int(d_model), dropout=float(dropout)
            )
        else:
            self.local_pos_enc = nn.Identity()
            self.global_pos_enc = nn.Identity()

        self._use_pos_enc = bool(use_positional_encoding)

        # --- [CLS] token ---
        self.cls_token = nn.Parameter(torch.empty(1, 1, int(d_model)))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Stage 2: global encoder ---
        global_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.global_encoder = nn.TransformerEncoder(
            global_layer,
            num_layers=int(num_global_layers),
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(int(d_model))

        # --- Regression head ---
        head_in = int(d_model) + int(scalar_dim)
        layers: list[nn.Module] = []
        cur = head_in
        act_cls = _activation_cls(activation)
        for hidden in head_hidden_layer_sizes:
            hidden = int(hidden)
            layers.append(nn.Linear(cur, hidden))
            layers.append(act_cls())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            cur = hidden
        layers.append(nn.Linear(cur, 1))
        self.head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute scalar predictions for a batch.

        Parameters
        ----------
        features:
            Dict with keys ``sequences``, ``lengths``, ``scalar_features``.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, 1]``.
        """
        sequences: torch.Tensor = features["sequences"]          # [B, T, D_in]
        lengths: torch.Tensor = features["lengths"]              # [B]
        scalar_features: torch.Tensor | None = features.get("scalar_features")

        batch_size, max_seq_len, _ = sequences.shape

        # --- Project to d_model ---
        x = self.input_proj(sequences)  # [B, T, d_model]

        if self._use_local_stage:
            x, window_mask = self._local_stage(x, lengths, batch_size, max_seq_len)
        else:
            # Flat mode: each token is directly a "window" for the global stage
            positions = torch.arange(max_seq_len, device=x.device).unsqueeze(0)  # [1, T]
            window_mask = positions >= lengths.unsqueeze(1)  # [B, T], True=pad

            if self._use_pos_enc:
                x = self.local_pos_enc(x)  # add positional encoding before global stage

        # --- Global stage ---
        x = self._global_stage(x, window_mask, batch_size)  # [B, d_model]

        # --- Head ---
        if scalar_features is not None and scalar_features.shape[1] > 0:
            x = torch.cat([x, scalar_features], dim=1)
        return self.head(x)  # [B, 1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_stage(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        batch_size: int,
        max_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply local transformer within non-overlapping windows.

        Parameters
        ----------
        x:
            Projected input of shape ``[B, T, d_model]``.
        lengths:
            Actual (unpadded) sequence lengths, shape ``[B]``.
        batch_size, max_seq_len:
            Dimensions of the current batch.

        Returns
        -------
        x_windows:
            Mean-pooled window embeddings, shape ``[B, n_windows, d_model]``.
        window_mask:
            Boolean padding mask for the global stage, shape ``[B, n_windows]``.
            ``True`` indicates a window that should be ignored.
        """
        lws = self.local_window_size

        # Pad sequence length to a multiple of lws
        remainder = max_seq_len % lws
        pad_len = (lws - remainder) % lws
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))  # pad along T
        padded_len = max_seq_len + pad_len
        n_windows = padded_len // lws

        # Reshape: [B, n_windows * lws, d_model] → [B * n_windows, lws, d_model]
        x = x.view(batch_size * n_windows, lws, self.d_model)

        # Optional local positional encoding (resets per chunk)
        if self._use_pos_enc:
            x = self.local_pos_enc(x)

        # Local self-attention
        x = self.local_encoder(x)  # [B * n_windows, lws, d_model]

        # Mean pool within each chunk → one embedding per window
        x = x.mean(dim=1)  # [B * n_windows, d_model]
        x = x.view(batch_size, n_windows, self.d_model)  # [B, n_windows, d_model]

        # Build the global-stage padding mask:
        # A window is "real" if it starts before the actual sequence end.
        # num_real_windows[i] = ceil(lengths[i] / lws)
        num_real_windows = lengths.float().div(lws).ceil().long()  # [B]
        win_positions = torch.arange(n_windows, device=x.device).unsqueeze(0)  # [1, n_windows]
        window_mask = win_positions >= num_real_windows.unsqueeze(1)  # [B, n_windows]

        return x, window_mask

    def _global_stage(
        self,
        x: torch.Tensor,
        window_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Prepend [CLS], run global encoder, return CLS representation.

        Parameters
        ----------
        x:
            Window embeddings of shape ``[B, n_windows, d_model]``.
        window_mask:
            Padding mask of shape ``[B, n_windows]``.  ``True`` = ignore.
        batch_size:
            Current batch size.

        Returns
        -------
        torch.Tensor
            CLS output after layer norm, shape ``[B, d_model]``.
        """
        # Prepend [CLS]: [B, 1 + n_windows, d_model]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Optional global positional encoding
        if self._use_pos_enc:
            x = self.global_pos_enc(x)

        # Build the full padding mask (CLS is never masked)
        cls_mask = window_mask.new_zeros(batch_size, 1)          # [B, 1] all-False
        full_mask = torch.cat([cls_mask, window_mask], dim=1)    # [B, 1 + n_windows]

        # Global self-attention over the CLS + windows sequence
        x = self.global_encoder(x, src_key_padding_mask=full_mask)  # [B, 1 + n_win, d_model]

        # Extract and normalise CLS token
        return self.final_norm(x[:, 0, :])  # [B, d_model]


# ---------------------------------------------------------------------------
# Regressor (scikit-learn-style interface)
# ---------------------------------------------------------------------------


class LatentHierarchicalTransformerRegressor:
    """Regressor wrapping :class:`LatentHierarchicalTransformerNetwork`.

    Follows the same interface as :class:`LatentWindowLSTMRegressor`:
    ``fit(X, y)`` / ``predict(X)`` where ``X`` is a DataFrame whose
    ``sequence_column`` holds variable-length 2-D numpy arrays.
    """

    def __init__(self, **params: Any) -> None:
        self.config = _parse_config(params)
        self._device = resolve_device(self.config.device)
        self._network: LatentHierarchicalTransformerNetwork | None = None
        self._sequence_scaler: StandardScalerState | None = None
        self._scalar_scaler: StandardScalerState | None = None
        self._target_scaler = TargetScalerState(mean=0.0, scale=1.0)
        self._input_dim: int | None = None
        self._scalar_dim: int | None = None
        self.training_history_: list[dict[str, float]] = []
        self.feature_importances_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Pickle serialisation (device-aware)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_device"] = str(self._device)
        state["_network_state"] = None
        if self._network is not None:
            state["_network_state"] = {
                k: v.detach().cpu() for k, v in self._network.state_dict().items()
            }
        state.pop("_network", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._device = resolve_device("auto")
        network_state = self.__dict__.pop("_network_state", None)
        self._network = None
        if network_state is not None and self._input_dim is not None and self._scalar_dim is not None:
            self._network = self._build_network(
                input_dim=self._input_dim, scalar_dim=self._scalar_dim
            )
            self._network.load_state_dict(network_state)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def _truncate_sequences(
        self, sequences: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Truncate each sequence to ``config.max_seq_len`` tokens (no-op if None)."""
        limit = self.config.max_seq_len
        if limit is None:
            return sequences
        return [s[:limit] for s in sequences]

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "LatentHierarchicalTransformerRegressor":
        """Fit the hierarchical transformer to (X, y).

        Parameters
        ----------
        X:
            DataFrame with at least one column ``sequence_column`` containing
            2-D float arrays of shape ``[seq_len, embedding_dim]``.  All other
            numeric columns are treated as scalar side-channel features.
        y:
            Regression target series.
        """
        sequences, scalar_features = extract_window_sequence_features(
            X, sequence_column=self.config.sequence_column
        )
        sequences = self._truncate_sequences(sequences)
        y_array = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        if len(sequences) < 2:
            raise ValueError(
                "LatentHierarchicalTransformerRegressor requires at least two training samples."
            )

        self._input_dim = infer_sequence_input_dim(sequences)
        self._scalar_dim = int(scalar_features.shape[1])

        # Scalers
        seq_sample = np.concatenate(sequences, axis=0) if self.config.scale_features else sequences[0][:1]
        self._sequence_scaler = fit_standard_scaler(seq_sample, enabled=self.config.scale_features)
        self._scalar_scaler = fit_standard_scaler(
            scalar_features, enabled=self.config.scale_features
        )
        self._target_scaler = fit_target_scaler(y_array, enabled=self.config.scale_target)

        scaled_sequences = [apply_standard_scaler(s, self._sequence_scaler) for s in sequences]
        scaled_scalars = apply_standard_scaler(scalar_features, self._scalar_scaler)
        scaled_targets = apply_target_scaler(y_array, self._target_scaler)

        loaders = build_window_sequence_dataloaders(
            scaled_sequences,
            scaled_scalars,
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
        self._network = self._build_network(
            input_dim=self._input_dim, scalar_dim=self._scalar_dim
        )

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
        self.training_history_ = trainer.fit(
            self._network, loaders.train_loader, loaders.validation_loader
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for X.

        Parameters
        ----------
        X:
            DataFrame in the same format as during ``fit``.

        Returns
        -------
        np.ndarray
            1-D float64 array of shape ``[n_samples]``.
        """
        if self._network is None or self._sequence_scaler is None or self._scalar_scaler is None:
            raise RuntimeError(
                "LatentHierarchicalTransformerRegressor.predict called before fit."
            )

        sequences, scalar_features = extract_window_sequence_features(
            X, sequence_column=self.config.sequence_column
        )
        sequences = self._truncate_sequences(sequences)
        in_dim = infer_sequence_input_dim(sequences)
        if in_dim != self._input_dim:
            raise ValueError(
                f"Expected embedding dimension {self._input_dim}, received {in_dim}."
            )
        if int(scalar_features.shape[1]) != self._scalar_dim:
            raise ValueError(
                f"Expected {self._scalar_dim} scalar feature columns, "
                f"received {scalar_features.shape[1]}."
            )

        scaled_sequences = [apply_standard_scaler(s, self._sequence_scaler) for s in sequences]
        scaled_scalars = apply_standard_scaler(scalar_features, self._scalar_scaler)
        dummy_targets = np.zeros((len(sequences), 1), dtype=np.float32)

        loaders = build_window_sequence_dataloaders(
            scaled_sequences,
            scaled_scalars,
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
            for batch_features, _ in loaders.train_loader:
                batch_features = {
                    k: v.to(self._device) for k, v in batch_features.items()
                }
                preds = self._network(batch_features).detach().cpu().numpy().reshape(-1, 1)
                batches.append(preds)

        restored = invert_target_scaler(np.concatenate(batches, axis=0), self._target_scaler)
        return restored.reshape(-1).astype("float64", copy=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_network(
        self, *, input_dim: int, scalar_dim: int
    ) -> LatentHierarchicalTransformerNetwork:
        return LatentHierarchicalTransformerNetwork(
            input_dim=input_dim,
            scalar_dim=scalar_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_local_layers=self.config.num_local_layers,
            num_global_layers=self.config.num_global_layers,
            local_window_size=self.config.local_window_size,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            head_hidden_layer_sizes=self.config.head_hidden_layer_sizes,
            activation=self.config.activation,
            use_positional_encoding=self.config.use_positional_encoding,
        ).to(self._device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_latent_hierarchical_transformer_regressor(
    params: Mapping[str, Any],
) -> LatentHierarchicalTransformerRegressor:
    """Factory function used by ``models.factory.create_regressor``."""
    return LatentHierarchicalTransformerRegressor(**dict(params))


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _parse_config(params: Mapping[str, Any]) -> LatentHierarchicalTransformerConfig:
    raw = dict(params)

    # Accept a few common aliases ------------------------------------------------
    learning_rate = raw.pop("learning_rate", 1.0e-3)
    lr_init = raw.pop("learning_rate_init", None)
    if lr_init is not None:
        learning_rate = lr_init

    weight_decay = raw.pop("weight_decay", 1.0e-3)
    alpha = raw.pop("alpha", None)
    if alpha is not None:
        weight_decay = alpha

    max_epochs = raw.pop("max_epochs", 500)
    max_iter = raw.pop("max_iter", None)
    if max_iter is not None:
        max_epochs = max_iter

    patience = raw.pop("patience", 40)
    n_no_change = raw.pop("n_iter_no_change", None)
    if n_no_change is not None:
        patience = n_no_change

    head_hidden = raw.pop(
        "head_hidden_layer_sizes", raw.pop("hidden_layer_sizes", (64,))
    )

    raw.pop("learning_rate_schedule", None)
    raw.pop("learning_rate_mode", None)
    solver = raw.pop("solver", None)
    if solver is not None and str(solver).lower() != "adam":
        raise ValueError(
            f"LatentHierarchicalTransformerRegressor only supports solver='adam', "
            f"received '{solver}'."
        )

    config = LatentHierarchicalTransformerConfig(
        d_model=int(raw.pop("d_model", 128)),
        nhead=int(raw.pop("nhead", 4)),
        num_local_layers=int(raw.pop("num_local_layers", 2)),
        num_global_layers=int(raw.pop("num_global_layers", 2)),
        local_window_size=int(raw.pop("local_window_size", 1)),
        dim_feedforward=int(raw.pop("dim_feedforward", 256)),
        dropout=float(raw.pop("dropout", 0.1)),
        head_hidden_layer_sizes=_parse_hidden_layers(head_hidden),
        activation=str(raw.pop("activation", "gelu")),
        use_positional_encoding=bool(raw.pop("use_positional_encoding", True)),
        sequence_column=str(raw.pop("sequence_column", WINDOW_SEQUENCE_COLUMN)),
        max_seq_len=_optional_int(raw.pop("max_seq_len", None)),
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

    # Validation
    if config.d_model <= 0:
        raise ValueError("d_model must be positive.")
    if config.nhead <= 0:
        raise ValueError("nhead must be positive.")
    if config.d_model % config.nhead != 0:
        raise ValueError(
            f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})."
        )
    if config.num_global_layers <= 0:
        raise ValueError("num_global_layers must be positive.")
    if config.local_window_size <= 0:
        raise ValueError("local_window_size must be >= 1.")

    if raw:
        raise ValueError(
            "Unsupported LatentHierarchicalTransformerRegressor parameters: "
            + ", ".join(sorted(raw))
            + "."
        )
    return config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_hidden_layers(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        layers = tuple(int(v) for v in value)
    else:
        layers = (int(value),)
    if any(v <= 0 for v in layers):
        raise ValueError("All head_hidden_layer_sizes must be positive.")
    return layers


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("wandb_config must be a mapping when provided.")


def _activation_cls(name: str) -> type:
    activations: dict[str, type] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(
            f"Unsupported LatentHierarchicalTransformerRegressor activation '{name}'. "
            f"Choose from: {', '.join(sorted(activations))}."
        )
    return activations[key]
