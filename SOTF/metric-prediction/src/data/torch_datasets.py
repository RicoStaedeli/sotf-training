from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class DataLoaderBundle:
    train_loader: Any
    validation_loader: Any | None


class TabularRegressionDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Initialize instance.

        Args:
            features (np.ndarray): np.ndarray value used by the function.
            targets (np.ndarray): np.ndarray value used by the function.

        Returns:
            None: None.
        """
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        """Return length.

        Returns:
            int: Result of the function.
        """
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return item at index.

        Args:
            index (int): Index value used by the function.

        Returns:
            tuple[Any, Any]: Result of the function.
        """
        return self.features[index], self.targets[index]


class WindowSequenceDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        scalar_features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Initialize instance.

        Args:
            sequences (Sequence[np.ndarray]): Sequence inputs.
            scalar_features (np.ndarray): Scalar side-channel features.
            targets (np.ndarray): Regression targets.

        Returns:
            None: None.
        """
        self.sequences = [torch.as_tensor(item, dtype=torch.float32) for item in sequences]
        self.scalar_features = torch.as_tensor(scalar_features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        """Return length.

        Returns:
            int: Result of the function.
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        """Return item at index.

        Args:
            index (int): Dataset index.

        Returns:
            tuple[dict[str, Tensor], Tensor]: Result of the function.
        """
        return (
            {
                "sequence": self.sequences[index],
                "scalar_features": self.scalar_features[index],
            },
            self.targets[index],
        )


def build_regression_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int | str,
    validation_fraction: float,
    random_state: int,
) -> DataLoaderBundle:
    """Build regression dataloaders.

    Args:
        features (np.ndarray): np.ndarray value used by the function.
        targets (np.ndarray): np.ndarray value used by the function.
        batch_size (int | str): Batch size used by the function.
        validation_fraction (float): float value used by the function.
        random_state (int): int value used by the function.

    Returns:
        DataLoaderBundle: Result of the function.
    """
    train_idx, validation_idx = split_train_validation_indices(
        num_rows=len(features),
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    resolved_batch_size = resolve_batch_size(
        batch_size, num_rows=len(train_idx))

    train_dataset = TabularRegressionDataset(
        features[train_idx], targets[train_idx])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=resolved_batch_size,
        shuffle=True,
    )

    validation_loader = None
    if validation_idx is not None:
        validation_dataset = TabularRegressionDataset(
            features[validation_idx], targets[validation_idx])
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
        )

    return DataLoaderBundle(train_loader=train_loader, validation_loader=validation_loader)


def build_window_sequence_dataloaders(
    sequences: Sequence[np.ndarray],
    scalar_features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int | str,
    validation_fraction: float,
    random_state: int,
    shuffle: bool,
    split_validation: bool,
) -> DataLoaderBundle:
    """Build dataloaders for variable-length sequence inputs with scalar side features.

    Args:
        sequences (Sequence[np.ndarray]): Window sequence inputs.
        scalar_features (np.ndarray): Scalar side-channel features.
        targets (np.ndarray): Regression targets.
        batch_size (int | str): Batch size used by the function.
        validation_fraction (float): Fraction for validation split.
        random_state (int): Random seed for split reproducibility.
        shuffle (bool): Whether to shuffle the training loader.
        split_validation (bool): Whether to create a validation loader.

    Returns:
        DataLoaderBundle: Result of the function.
    """
    if split_validation:
        train_idx, validation_idx = split_train_validation_indices(
            num_rows=len(sequences),
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
    else:
        train_idx = np.arange(len(sequences), dtype=np.int64)
        validation_idx = None

    resolved_batch_size = resolve_batch_size(batch_size, num_rows=len(train_idx))
    train_dataset = WindowSequenceDataset(
        [sequences[int(index)] for index in train_idx],
        scalar_features[train_idx],
        targets[train_idx],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=resolved_batch_size,
        shuffle=bool(shuffle),
        collate_fn=collate_window_sequences,
    )

    validation_loader = None
    if validation_idx is not None:
        validation_dataset = WindowSequenceDataset(
            [sequences[int(index)] for index in validation_idx],
            scalar_features[validation_idx],
            targets[validation_idx],
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            collate_fn=collate_window_sequences,
        )

    return DataLoaderBundle(train_loader=train_loader, validation_loader=validation_loader)


def collate_window_sequences(
    batch: Sequence[tuple[dict[str, Tensor], Tensor]],
) -> tuple[dict[str, Tensor], Tensor]:
    """Collate variable-length sequences into a padded batch.

    Args:
        batch (Sequence[tuple[dict[str, Tensor], Tensor]]): Batch items.

    Returns:
        tuple[dict[str, Tensor], Tensor]: Result of the function.
    """
    features, targets = zip(*batch)
    sequences = [item["sequence"] for item in features]
    lengths_list = [int(sequence.shape[0]) for sequence in sequences]
    lengths = torch.as_tensor(lengths_list, dtype=torch.long)
    if len(set(lengths_list)) == 1:
        padded = torch.stack(sequences, dim=0)
    else:
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    scalar_features = torch.stack([item["scalar_features"] for item in features], dim=0)
    return (
        {
            "sequences": padded,
            "lengths": lengths,
            "scalar_features": scalar_features,
        },
        torch.stack(list(targets), dim=0),
    )


def extract_window_sequence_features(
    X: pd.DataFrame,
    *,
    sequence_column: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract window sequence and scalar features from a dataframe.

    Args:
        X (pd.DataFrame): Input feature frame.
        sequence_column (str): Column holding variable-length sequences.

    Returns:
        tuple[list[np.ndarray], np.ndarray]: Result of the function.
    """
    if sequence_column not in X.columns:
        raise ValueError(f"Expected latent window sequence column '{sequence_column}'.")

    sequences = [_as_sequence_array(value, index=index) for index, value in X[sequence_column].items()]
    scalar_frame = X.drop(columns=[sequence_column])
    if scalar_frame.empty:
        scalar_features = np.zeros((len(X), 0), dtype=np.float32)
    else:
        scalar_features = scalar_frame.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(scalar_features).all():
            raise ValueError("Latent window scalar features contain NaN or infinite values.")
    return sequences, scalar_features


def infer_sequence_input_dim(sequences: Sequence[np.ndarray]) -> int:
    """Infer the shared embedding dimension for sequence inputs.

    Args:
        sequences (Sequence[np.ndarray]): Sequence inputs.

    Returns:
        int: Result of the function.
    """
    dims = {int(sequence.shape[1]) for sequence in sequences}
    if len(dims) != 1:
        raise ValueError(f"All window sequences must share one embedding dimension, got {sorted(dims)}.")
    return dims.pop()


def _as_sequence_array(value: Any, *, index: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"Expected window sequence at index {index!r} to be 2D, received shape {array.shape}.")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"Window sequence at index {index!r} must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"Window sequence at index {index!r} contains NaN or infinite values.")
    return array.astype(np.float32, copy=False)


def split_train_validation_indices(
    *,
    num_rows: int,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split train validation indices.

    Args:
        num_rows (int): int value used by the function.
        validation_fraction (float): float value used by the function.
        random_state (int): int value used by the function.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Result of the function.
    """
    indices = np.arange(num_rows, dtype=np.int64)
    if num_rows < 8 or validation_fraction <= 0.0:
        return indices, None

    validation_size = int(round(num_rows * validation_fraction))
    validation_size = min(max(validation_size, 1), num_rows - 1)
    if validation_size <= 0:
        return indices, None

    rng = np.random.default_rng(int(random_state))
    shuffled = rng.permutation(indices)
    validation_idx = np.sort(shuffled[:validation_size])
    train_idx = np.sort(shuffled[validation_size:])
    if len(train_idx) == 0 or len(validation_idx) == 0:
        return indices, None
    return train_idx, validation_idx


def resolve_batch_size(batch_size: int | str, *, num_rows: int) -> int:
    """Resolve batch size.

    Args:
        batch_size (int | str): Batch size used by the function.
        num_rows (int): int value used by the function.

    Returns:
        int: Result of the function.
    """
    if isinstance(batch_size, str):
        if batch_size.lower() != "auto":
            raise ValueError(f"Unsupported batch_size value '{batch_size}'.")
        return max(16, min(256, num_rows))

    resolved = int(batch_size)
    if resolved <= 0:
        raise ValueError("batch_size must be positive.")
    return min(resolved, num_rows)
