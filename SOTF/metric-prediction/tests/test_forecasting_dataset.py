from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd

from data.dataset_splitter import split_dataset
from data.weightstatistic_dataset import WeightStatisticDataset
from forecasting.config import SequenceFeatureConfig
from forecasting.dataset import build_forecast_dataset, build_forecast_target_dataset


def _prepared_dataset() -> WeightStatisticDataset:
    features = pd.DataFrame(
        {
            "ws_stat_a": [1.0, 1.5, 2.0, 10.0, 11.0, 12.0],
            "ws_stat_b": [4.0, 4.5, 5.0, 40.0, 41.0, 42.0],
        },
        index=pd.Index([100, 101, 102, 200, 201, 202], name="sample_idx"),
    )
    target = pd.Series(
        [0.20, 0.40, 0.60, 0.10, 0.30, 0.80],
        index=features.index,
        name="test_acc",
    )
    metadata = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "epoch": [0, 1, 2, 0, 1, 2],
            "path": ["a", "a", "a", "b", "b", "b"],
            "hyperparameter_key": ["hp_a", "hp_a", "hp_a", "hp_b", "hp_b", "hp_b"],
        },
        index=features.index,
    )
    feature_metadata = pd.DataFrame(index=features.columns)
    feature_metadata.index.name = "feature"
    return WeightStatisticDataset(
        features=features,
        target=target,
        metadata=metadata,
        feature_metadata=feature_metadata,
    )


class ForecastDatasetTests(unittest.TestCase):
    def test_final_value_mode_builds_lags_and_excludes_terminal_samples(self) -> None:
        dataset = build_forecast_dataset(
            _prepared_dataset(),
            SequenceFeatureConfig(
                target_mode="final_value",
                max_lag=1,
                include_terminal_samples=False,
            ),
        )

        self.assertEqual(list(dataset.features.index), [100, 101, 200, 201])
        self.assertTrue("ws_stat_a__lag_0" in dataset.features.columns)
        self.assertTrue("ws_stat_a__lag_1" in dataset.features.columns)
        self.assertTrue("ws_stat_a__delta_1" in dataset.features.columns)
        self.assertTrue("epoch" in dataset.features.columns)
        self.assertTrue(np.isnan(dataset.features.loc[100, "ws_stat_a__lag_1"]))
        self.assertEqual(dataset.target.loc[100], 0.60)
        self.assertEqual(dataset.target.loc[201], 0.80)

    def test_include_time_features_false_excludes_time_columns_only(self) -> None:
        dataset = build_forecast_dataset(
            _prepared_dataset(),
            SequenceFeatureConfig(
                target_mode="final_value",
                max_lag=0,
                include_time_features=False,
                include_current_target=True,
                include_terminal_samples=False,
            ),
        )

        for column in (
            "epoch",
            "history_length",
            "max_epoch",
            "epochs_remaining",
            "epoch_gap_prev",
            "progress_fraction",
        ):
            self.assertNotIn(column, dataset.features.columns)
        self.assertIn("current_target", dataset.features.columns)

    def test_horizon_value_mode_uses_fixed_future_checkpoint(self) -> None:
        dataset = build_forecast_dataset(
            _prepared_dataset(),
            SequenceFeatureConfig(
                target_mode="horizon_value",
                horizon=1,
                max_lag=0,
                include_terminal_samples=True,
            ),
        )

        self.assertEqual(list(dataset.features.index), [100, 101, 200, 201])
        self.assertEqual(dataset.target.loc[100], 0.40)
        self.assertEqual(dataset.target.loc[101], 0.60)
        self.assertEqual(dataset.target.loc[200], 0.30)
        self.assertEqual(dataset.target.loc[201], 0.80)

    def test_target_only_builder_matches_full_forecast_rows(self) -> None:
        sequence_cfg = SequenceFeatureConfig(
            target_mode="horizon_value",
            horizon=1,
            max_lag=1,
            include_current_target=True,
            include_terminal_samples=False,
        )
        full_dataset = build_forecast_dataset(_prepared_dataset(), sequence_cfg)
        target_dataset = build_forecast_target_dataset(_prepared_dataset(), sequence_cfg)

        self.assertListEqual(list(target_dataset.target.index), list(full_dataset.target.index))
        pd.testing.assert_series_equal(target_dataset.target, full_dataset.target)
        pd.testing.assert_frame_equal(target_dataset.metadata, full_dataset.metadata)

    def test_zero_max_epoch_does_not_emit_progress_divide_warning(self) -> None:
        features = pd.DataFrame(
            {"ws_stat_a": [1.0], "ws_stat_b": [2.0]},
            index=pd.Index([100], name="sample_idx"),
        )
        target = pd.Series([0.2], index=features.index, name="test_acc")
        metadata = pd.DataFrame(
            {
                "model_id": ["m1"],
                "epoch": [0],
                "path": ["a"],
                "hyperparameter_key": ["hp_a"],
            },
            index=features.index,
        )
        prepared = WeightStatisticDataset(
            features=features,
            target=target,
            metadata=metadata,
            feature_metadata=pd.DataFrame(index=features.columns),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", RuntimeWarning)
            dataset = build_forecast_dataset(
                prepared,
                SequenceFeatureConfig(
                    target_mode="final_value",
                    max_lag=0,
                    include_terminal_samples=True,
                ),
            )

        self.assertFalse(caught)
        self.assertEqual(dataset.metadata.loc[100, "progress_fraction"], 0.0)

    def test_grouped_split_keeps_models_disjoint(self) -> None:
        dataset = build_forecast_dataset(
            _prepared_dataset(),
            SequenceFeatureConfig(
                target_mode="final_value",
                max_lag=0,
                include_terminal_samples=False,
            ),
        )

        split = split_dataset(dataset, validation_size=0.5, random_seed=7)

        train_models = set(split.metadata_train["model_id"].astype(str))
        val_models = set(split.metadata_val["model_id"].astype(str))
        self.assertTrue(train_models)
        self.assertTrue(val_models)
        self.assertTrue(train_models.isdisjoint(val_models))

    def test_grouped_split_prefers_hyperparameter_key(self) -> None:
        features = pd.DataFrame(
            {
                "ws_stat_a": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 10.0, 10.5, 11.0, 12.0, 12.5, 13.0],
            },
            index=pd.Index([100, 101, 102, 200, 201, 202, 300, 301, 302, 400, 401, 402], name="sample_idx"),
        )
        target = pd.Series(
            [0.20, 0.40, 0.60, 0.25, 0.45, 0.65, 0.10, 0.30, 0.80, 0.15, 0.35, 0.85],
            index=features.index,
            name="test_acc",
        )
        metadata = pd.DataFrame(
            {
                "model_id": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3", "m3", "m4", "m4", "m4"],
                "epoch": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "path": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
                "hyperparameter_key": ["hp_a", "hp_a", "hp_a", "hp_a", "hp_a", "hp_a", "hp_b", "hp_b", "hp_b", "hp_b", "hp_b", "hp_b"],
            },
            index=features.index,
        )
        prepared = WeightStatisticDataset(
            features=features,
            target=target,
            metadata=metadata,
            feature_metadata=pd.DataFrame(index=features.columns),
        )
        dataset = build_forecast_dataset(
            prepared,
            SequenceFeatureConfig(
                target_mode="final_value",
                max_lag=0,
                include_terminal_samples=False,
            ),
        )

        split = split_dataset(dataset, validation_size=0.5, random_seed=7)

        train_keys = set(split.metadata_train["hyperparameter_key"].astype(str))
        val_keys = set(split.metadata_val["hyperparameter_key"].astype(str))
        self.assertTrue(train_keys)
        self.assertTrue(val_keys)
        self.assertTrue(train_keys.isdisjoint(val_keys))


if __name__ == "__main__":
    unittest.main()
