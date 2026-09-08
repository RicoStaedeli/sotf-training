from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig

from data.weightstatistic_dataset import WeightStatisticDataset

from .latent_dataset_loader import load_latent_dataset, load_latent_split_dataset
from .latent_forecast_dataset_loader import (
    load_latent_forecast_dataset,
    load_latent_forecast_split_datasets,
)
from .weightstatistic_dataset_loader import load_weightstatistic_dataset, load_weightstatistic_split_datasets
from .torch_datasets import DataLoaderBundle, build_regression_dataloaders


@dataclass(slots=True)
class DatasetFactory:
    """Entry point for loading metric-prediction datasets and Torch dataloaders."""

    def load(self, mode: str, data_cfg: DictConfig, feature_cfg: DictConfig | None = None) -> Any:
        """Load.

        Args:
            mode (str): str value used by the function.
            data_cfg (DictConfig): Configuration object used by the function.
            feature_cfg (DictConfig | None): Configuration for feature selection.

        Returns:
            Any: Result of the function.
        """
        normalized_mode = str(mode).lower()
        if normalized_mode in {"prediction", "forecasting"}:
            if feature_cfg is None:
                raise ValueError(f"mode='{mode}' requires a feature config.")
            if data_cfg.get("pt_train_path") and data_cfg.get("pt_val_path"):
                return load_weightstatistic_split_datasets(data_cfg, feature_cfg)
            return load_weightstatistic_dataset(data_cfg, feature_cfg)
        if normalized_mode == "latent":
            if data_cfg.get("pt_train_path") and data_cfg.get("pt_val_path"):
                return load_latent_split_dataset(data_cfg)
            return load_latent_dataset(data_cfg)
        if normalized_mode == "latent_forecast":
            if data_cfg.get("pt_train_path") and data_cfg.get("pt_val_path"):
                return load_latent_forecast_split_datasets(data_cfg)
            return load_latent_forecast_dataset(data_cfg)
        raise ValueError(f"Unsupported dataset mode '{mode}'.")

    def build_torch_dataloaders(
        self,
        prepared: WeightStatisticDataset,
        *,
        batch_size: int | str,
        validation_fraction: float,
        random_state: int,
    ) -> DataLoaderBundle:
        """Build torch dataloaders.

        Args:
            prepared (WeightStatisticDataset): WeightStatisticDataset value used by the function.
            batch_size (int | str): Batch size used by the function.
            validation_fraction (float): float value used by the function.
            random_state (int): int value used by the function.

        Returns:
            DataLoaderBundle: Result of the function.
        """
        features = prepared.features.to_numpy(dtype=np.float32, copy=True)
        targets = prepared.target.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        return build_regression_dataloaders(
            features,
            targets,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
