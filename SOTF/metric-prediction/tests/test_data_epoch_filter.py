from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd
from omegaconf import OmegaConf

from data.weightstatistic_dataset_loader import _weightstatistic_dataset_from_frame


def _data_cfg(
    *,
    epoch_filter: list[int] | None = None,
    target_model_min_threshold: float | None = None,
):
    return OmegaConf.create(
        {
            "target_column": "test_acc",
            "metadata_columns": ["model_id", "path", "epoch"],
            "epoch_filter": epoch_filter or [],
            "target_model_min_threshold": target_model_min_threshold,
        }
    )


def _feature_cfg():
    return OmegaConf.create(
        {
            "include_model_metrics": True,
            "include_layer_metrics": True,
            "metric_families": [],
            "metrics": [],
            "scopes": [],
            "layer_types": [],
            "include_feature_names": [],
            "exclude_feature_names": [],
            "include_feature_patterns": [],
            "exclude_feature_patterns": [],
            "max_missing_fraction": 1.0,
            "drop_constant_features": False,
            "sort_columns": True,
        }
    )


def _prepared_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_idx": [10, 11, 12],
            "model_id": ["m1", "m1", "m2"],
            "path": ["p1", "p1", "p2"],
            "epoch": [1, 2, 2],
            "test_acc": [0.15, 0.35, 0.55],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [0.1, 0.2, 0.4],
        }
    )


class PreparedDatasetEpochFilterTests(unittest.TestCase):
    def test_epoch_filter_keeps_only_requested_epochs(self) -> None:
        prepared = _weightstatistic_dataset_from_frame(
            _prepared_frame(),
            parquet_path=Path("/tmp/example__prepared_dataset.parquet"),
            data_cfg=_data_cfg(epoch_filter=[2]),
            feature_cfg=_feature_cfg(),
        )

        self.assertEqual(prepared.features.index.tolist(), [11, 12])
        self.assertEqual(prepared.metadata["epoch"].tolist(), [2, 2])
        self.assertEqual(prepared.target.tolist(), [0.35, 0.55])

    def test_epoch_filter_errors_when_no_rows_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prepared dataset has no rows for data.epoch_filter=\\[99.0\\]"):
            _weightstatistic_dataset_from_frame(
                _prepared_frame(),
                parquet_path=Path("/tmp/example__prepared_dataset.parquet"),
                data_cfg=_data_cfg(epoch_filter=[99]),
                feature_cfg=_feature_cfg(),
            )

    def test_target_model_min_threshold_filters_entire_models_by_final_target(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_idx": [10, 11, 20, 21, 30, 31],
                "model_id": ["m1", "m1", "m2", "m2", "m3", "m3"],
                "path": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "epoch": [1, 2, 1, 2, 1, 2],
                "test_acc": [0.15, 0.35, 0.25, 0.10, 0.40, 0.55],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

        prepared = _weightstatistic_dataset_from_frame(
            frame,
            parquet_path=Path("/tmp/example__prepared_dataset.parquet"),
            data_cfg=_data_cfg(target_model_min_threshold=0.1),
            feature_cfg=_feature_cfg(),
        )

        self.assertEqual(prepared.features.index.tolist(), [10, 11, 30, 31])
        self.assertEqual(prepared.metadata["model_id"].tolist(), ["m1", "m1", "m3", "m3"])
        self.assertEqual(prepared.target.tolist(), [0.15, 0.35, 0.40, 0.55])

    def test_target_model_min_threshold_is_applied_before_epoch_filter(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_idx": [10, 11, 20, 21, 30, 31],
                "model_id": ["m1", "m1", "m2", "m2", "m3", "m3"],
                "path": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "epoch": [1, 2, 1, 2, 1, 2],
                "test_acc": [0.15, 0.35, 0.25, 0.10, 0.40, 0.55],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

        prepared = _weightstatistic_dataset_from_frame(
            frame,
            parquet_path=Path("/tmp/example__prepared_dataset.parquet"),
            data_cfg=_data_cfg(epoch_filter=[1], target_model_min_threshold=0.1),
            feature_cfg=_feature_cfg(),
        )

        self.assertEqual(prepared.features.index.tolist(), [10, 30])
        self.assertEqual(prepared.metadata["model_id"].tolist(), ["m1", "m3"])
        self.assertEqual(prepared.metadata["epoch"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
