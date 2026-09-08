from __future__ import annotations

import unittest

from hydra import compose, initialize_config_module
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from metric_prediction.models import create_regressor

from data.latent_forecast_dataset_loader import load_latent_forecast_dataset
from forecasting.config import SequenceFeatureConfig
from forecasting.dataset import WINDOW_SEQUENCE_COLUMN, build_latent_window_forecast_dataset
from utils.config_utils import register_omegaconf_resolvers


class LatentWindowForecastingTests(unittest.TestCase):
    def test_windows_pt_loads_and_builds_horizon_targets(self) -> None:
        with _tmp_path() as tmp_path:
            pt_path = tmp_path / "latent_windows.pt"
            target_path = tmp_path / "targets.parquet"
            torch.save(
                {
                    "mode": "windows",
                    "model_ids": ["m1", "m1", "m2", "m2"],
                    "checkpoint_paths": ["a", "b", "c", "d"],
                    "epoch": [0, 1, 0, 1],
                    "embeddings": [
                        torch.ones(2, 4),
                        torch.ones(3, 4) * 2.0,
                        torch.ones(1, 4) * 3.0,
                        torch.ones(4, 4) * 4.0,
                    ],
                },
                pt_path,
            )
            pd.DataFrame(
                {
                    "model_id": ["m1", "m1", "m2", "m2"],
                    "epoch": [0, 1, 0, 1],
                    "test_acc": [0.1, 0.4, 0.2, 0.8],
                }
            ).to_parquet(target_path)

            prepared = load_latent_forecast_dataset(
                OmegaConf.create(
                    {
                        "pt_path": str(pt_path),
                        "target_parquet_path": str(target_path),
                        "target_column": "test_acc",
                        "id_normalize_basename": True,
                        "target_model_min_threshold": None,
                    }
                )
            )
            dataset = build_latent_window_forecast_dataset(
                prepared,
                SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    include_time_features=True,
                    include_current_target=True,
                ),
            )

        self.assertEqual(list(dataset.target), [0.4, 0.8])
        self.assertEqual(list(dataset.metadata["model_id"]), ["m1", "m2"])
        self.assertIn(WINDOW_SEQUENCE_COLUMN, dataset.features.columns)
        self.assertEqual(dataset.features.iloc[0][WINDOW_SEQUENCE_COLUMN].shape, (2, 4))
        self.assertIn("current_target", dataset.features.columns)

    def test_factory_builds_lstm_and_predicts_variable_length_sequences(self) -> None:
        rng = np.random.default_rng(17)
        sequences = [rng.normal(size=(2 + (index % 4), 5)).astype("float32") for index in range(24)]
        progress = np.linspace(0.0, 1.0, len(sequences), dtype=np.float32)
        X = pd.DataFrame(
            {
                WINDOW_SEQUENCE_COLUMN: sequences,
                "progress_fraction": progress,
            }
        )
        y = pd.Series(
            [
                float(sequence[:, 0].mean() + 0.5 * scalar)
                for sequence, scalar in zip(sequences, progress)
            ]
        )

        model = create_regressor(
            {
                "name": "latent_window_lstm",
                "params": {
                    "hidden_size": 8,
                    "head_hidden_layer_sizes": [8],
                    "batch_size": 6,
                    "max_epochs": 3,
                    "patience": 2,
                    "validation_fraction": 0.2,
                    "early_stopping": False,
                    "random_state": 17,
                    "device": "cpu",
                },
            }
        )

        fitted = model.fit(X, y)
        predictions = fitted.predict(X.iloc[:5])

        self.assertEqual(predictions.shape, (5,))
        self.assertEqual(predictions.dtype.kind, "f")
        self.assertTrue(np.isfinite(predictions).all())

    def test_latent_windows_mode_composes(self) -> None:
        register_omegaconf_resolvers()
        with initialize_config_module(config_module="metric_prediction.config.configs", version_base=None):
            cfg = compose(config_name="config", overrides=["mode=latent_windows_forecast"])

        self.assertEqual(cfg.mode, "latent_windows_forecast")
        self.assertEqual(cfg.model.name, "latent_window_lstm")
        self.assertEqual(cfg.sequence.target_mode, "final_value")


class _tmp_path:
    def __enter__(self):
        import tempfile
        from pathlib import Path

        self._tmp_dir = tempfile.TemporaryDirectory()
        return Path(self._tmp_dir.name)

    def __exit__(self, exc_type, exc, tb):
        self._tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
