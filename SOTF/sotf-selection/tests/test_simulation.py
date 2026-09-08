from __future__ import annotations

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from data.feature_preprocessing import FeaturePreprocessor
from forecasting.config import FeatureSelectionConfig, PreparedDatasetConfig, SequenceFeatureConfig
from utils.settings import (
    ForecastModelPredictorConfig,
    LoggingConfig,
    NaivePredictorConfig,
    OnlineLSTMPredictorConfig,
    OutputConfig,
    SelectionConfig,
    SimulationDataConfig,
    SimulationConfig,
    XGBoostPredictorConfig,
)
from simulation.runner import run_simulation


class ColumnSelectorModel:
    def __init__(self, column: str) -> None:
        """Init the function.

        Args:
            column (str): str value used by the function.

        Returns:
            None: None.
        """
        self.column = column

    def predict(self, X: pd.DataFrame):
        """Predict the function.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.

        Returns:
            Any: Result of the function.
        """
        return X[self.column].to_numpy(dtype="float64")


class WindowSequenceSelectorModel:
    def __init__(self, *, sequence_column: str = "window_sequence", scalar_column: str | None = None) -> None:
        self.sequence_column = sequence_column
        self.scalar_column = scalar_column

    def predict(self, X: pd.DataFrame):
        sequence_scores = X[self.sequence_column].map(
            lambda value: float(value[:, 0].mean())
        ).to_numpy(dtype="float64")
        if self.scalar_column is None:
            return sequence_scores
        scalar_scores = pd.to_numeric(X[self.scalar_column], errors="coerce").to_numpy(dtype="float64")
        return sequence_scores + scalar_scores


class TrainableWindowSequenceSelectorRegressor:
    def __init__(self, *, sequence_column: str = "window_sequence") -> None:
        self.sequence_column = sequence_column
        self.fit_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_columns_ = list(X.columns)
        self._assert_scalar_features_are_finite(X)
        self._assert_sequence_shapes(X)
        self.target_mean_ = float(pd.to_numeric(y, errors="coerce").mean())
        return self

    def predict(self, X: pd.DataFrame):
        self._assert_scalar_features_are_finite(X)
        self._assert_sequence_shapes(X)
        return X[self.sequence_column].map(lambda value: float(value[:, 0].mean())).to_numpy(dtype="float64")

    def _assert_scalar_features_are_finite(self, X: pd.DataFrame) -> None:
        scalar_frame = X.drop(columns=[self.sequence_column], errors="ignore")
        if scalar_frame.empty:
            return
        scalar_values = scalar_frame.to_numpy(dtype="float64", copy=True)
        if not pd.notna(scalar_values).all():
            raise AssertionError("Scalar latent-window features unexpectedly contain NaN values.")

    def _assert_sequence_shapes(self, X: pd.DataFrame) -> None:
        for value in X[self.sequence_column]:
            if value.ndim != 2:
                raise AssertionError(f"Expected 2D latent window sequence, received shape {value.shape}.")


def _write_dataset(path: str, rows: list[dict[str, object]]) -> None:
    """Write dataset.

    Args:
        path (str): Path to the input resource.
        rows (list[dict[str, object]]): list[dict[str, object]] value used by the function.

    Returns:
        None: None.
    """
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


class SimulationTests(unittest.TestCase):
    def test_naive_simulation_eliminates_bottom_half_each_round(self) -> None:
        """Verify that naive simulation eliminates bottom half each round.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/naive_dataset__prepared_dataset.parquet"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.50, 0.60, 0.80, 0.90], signal_scale=1.0),
                    *_trajectory("model_b", [0.45, 0.55, 0.70, 0.75], signal_scale=0.9),
                    *_trajectory("model_c", [0.40, 0.30, 0.20, 0.10], signal_scale=0.4),
                    *_trajectory("model_d", [0.35, 0.25, 0.15, 0.05], signal_scale=0.3),
                ],
            )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.5,
                    min_population=1,
                ),
                predictor=NaivePredictorConfig(strategy="current_target"),
                output=OutputConfig(
                    experiment_name="naive_simulation_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            round_details = pd.read_parquet(f"{result['artifacts_dir']}/round_details.parquet")

            self.assertEqual(summary["rounds_executed"], 2)
            self.assertEqual(summary["initial_population"], 4)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_a")
            self.assertEqual(summary["oracle_top_k_recall"], 1.0)
            self.assertEqual(summary["mean_survivor_rank_gap"], 0.0)
            final_survivors = pd.read_parquet(f"{result['artifacts_dir']}/final_survivors.parquet")

            round_one = round_details.loc[round_details["round_index"] == 1].copy()
            kept_round_one = set(round_one.loc[round_one["selected"], "model_id"].astype(str))
            self.assertEqual(kept_round_one, {"model_a", "model_b"})

            round_two = round_details.loc[round_details["round_index"] == 2].copy()
            kept_round_two = set(round_two.loc[round_two["selected"], "model_id"].astype(str))
            self.assertEqual(kept_round_two, {"model_a"})
            self.assertEqual(final_survivors.loc[0, "full_population_final_rank"], 1)
            self.assertEqual(final_survivors.loc[0, "survivor_rank"], 1)

    def test_forecast_model_bundle_can_drive_selection(self) -> None:
        """Verify that forecast model bundle can drive selection.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/bundle_dataset__prepared_dataset.parquet"
            bundle_path = f"{tmp_dir}/model_bundle.pkl"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.60, 0.62, 0.63], signal_scale=0.1),
                    *_trajectory("model_b", [0.55, 0.58, 0.90], signal_scale=1.0),
                    *_trajectory("model_c", [0.40, 0.35, 0.20], signal_scale=0.2),
                    *_trajectory("model_d", [0.30, 0.25, 0.10], signal_scale=0.15),
                ],
            )

            preprocessor = FeaturePreprocessor(strategy="median")
            preprocessor.fit(pd.DataFrame({"pred_signal__lag_0": [0.1, 0.5, 1.0]}))
            with open(bundle_path, "wb") as handle:
                pickle.dump(
                    {
                        "model": ColumnSelectorModel("pred_signal__lag_0"),
                        "preprocessor": preprocessor,
                        "feature_columns": ["pred_signal__lag_0"],
                        "config": {
                            "data": {"target_column": "test_acc"},
                            "features": {
                                "include_model_metrics": True,
                                "include_layer_metrics": True,
                                "metric_families": [],
                                "metrics": [],
                                "scopes": [],
                                "layer_types": [],
                                "include_feature_names": ["pred_signal"],
                                "exclude_feature_names": [],
                                "include_feature_patterns": [],
                                "exclude_feature_patterns": [],
                                "max_missing_fraction": 0.25,
                                "drop_constant_features": True,
                                "missing_value_fill": "median",
                                "sort_columns": True,
                                "log_feature_names": False,
                                "max_console_logged_features": 50,
                            },
                            "sequence": {
                                "target_mode": "horizon_value",
                                "horizon": 1,
                                "max_lag": 0,
                                "min_history": 1,
                                "add_delta_features": False,
                                "include_time_features": False,
                                "include_current_target": False,
                                "include_terminal_samples": False,
                            },
                        },
                    },
                    handle,
                )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(include_feature_names=["pred_signal"]),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.25,
                    min_population=1,
                ),
                predictor=ForecastModelPredictorConfig(model_bundle_path=bundle_path),
                output=OutputConfig(
                    experiment_name="forecast_bundle_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            final_survivors = pd.read_parquet(f"{result['artifacts_dir']}/final_survivors.parquet")

            self.assertEqual(summary["rounds_executed"], 1)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_b")
            self.assertEqual(summary["oracle_top_k_recall"], 1.0)
            self.assertEqual(summary["mean_survivor_rank_gap"], 0.0)
            self.assertEqual(list(final_survivors["model_id"]), ["model_b"])
            self.assertEqual(final_survivors.loc[0, "full_population_final_rank"], 1)
            self.assertEqual(final_survivors.loc[0, "survivor_rank"], 1)

    def test_torch_forecast_model_bundle_can_drive_selection(self) -> None:
        """Verify that torch forecast model bundle can drive selection.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/torch_bundle_dataset__prepared_dataset.parquet"
            bundle_path = f"{tmp_dir}/model_bundle.pt"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.60, 0.62, 0.63], signal_scale=0.1),
                    *_trajectory("model_b", [0.55, 0.58, 0.90], signal_scale=1.0),
                    *_trajectory("model_c", [0.40, 0.35, 0.20], signal_scale=0.2),
                    *_trajectory("model_d", [0.30, 0.25, 0.10], signal_scale=0.15),
                ],
            )
            torch.save(
                {
                    "model": ColumnSelectorModel("pred_signal__lag_0"),
                    "preprocessor": None,
                    "feature_columns": ["pred_signal__lag_0"],
                    "config": {
                        "data": {"target_column": "test_acc"},
                        "features": {
                            "include_model_metrics": True,
                            "include_layer_metrics": True,
                            "metric_families": [],
                            "metrics": [],
                            "scopes": [],
                            "layer_types": [],
                            "include_feature_names": ["pred_signal"],
                            "exclude_feature_names": [],
                            "include_feature_patterns": [],
                            "exclude_feature_patterns": [],
                            "max_missing_fraction": 0.25,
                            "drop_constant_features": True,
                            "missing_value_fill": "median",
                            "sort_columns": True,
                            "log_feature_names": False,
                            "max_console_logged_features": 50,
                        },
                        "sequence": {
                            "target_mode": "horizon_value",
                            "horizon": 1,
                            "max_lag": 0,
                            "min_history": 1,
                            "add_delta_features": False,
                            "include_time_features": False,
                            "include_current_target": False,
                            "include_terminal_samples": False,
                        },
                    },
                },
                bundle_path,
            )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(include_feature_names=["pred_signal"]),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.25,
                    min_population=1,
                ),
                predictor=ForecastModelPredictorConfig(model_bundle_path=bundle_path),
                output=OutputConfig(
                    experiment_name="torch_forecast_bundle_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]

            self.assertEqual(summary["rounds_executed"], 1)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_b")
            self.assertEqual(summary["oracle_top_k_recall"], 1.0)

    def test_forecast_model_bundle_without_preprocessor_can_drive_selection(self) -> None:
        """Verify that forecast model bundle without preprocessor can drive selection.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/bundle_dataset_no_preprocessor__prepared_dataset.parquet"
            bundle_path = f"{tmp_dir}/model_bundle_no_preprocessor.pkl"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.60, 0.62, 0.63], signal_scale=0.1),
                    *_trajectory("model_b", [0.55, 0.58, 0.90], signal_scale=1.0),
                    *_trajectory("model_c", [0.40, 0.35, 0.20], signal_scale=0.2),
                    *_trajectory("model_d", [0.30, 0.25, 0.10], signal_scale=0.15),
                ],
            )

            with open(bundle_path, "wb") as handle:
                pickle.dump(
                    {
                        "model": ColumnSelectorModel("pred_signal__lag_0"),
                        "preprocessor": None,
                        "feature_columns": ["pred_signal__lag_0"],
                        "config": {
                            "data": {"target_column": "test_acc"},
                            "features": {
                                "include_model_metrics": True,
                                "include_layer_metrics": True,
                                "metric_families": [],
                                "metrics": [],
                                "scopes": [],
                                "layer_types": [],
                                "include_feature_names": ["pred_signal"],
                                "exclude_feature_names": [],
                                "include_feature_patterns": [],
                                "exclude_feature_patterns": [],
                                "max_missing_fraction": 0.25,
                                "drop_constant_features": True,
                                "missing_value_fill": "median",
                                "sort_columns": True,
                                "log_feature_names": False,
                                "max_console_logged_features": 50,
                            },
                            "sequence": {
                                "target_mode": "horizon_value",
                                "horizon": 1,
                                "max_lag": 0,
                                "min_history": 1,
                                "add_delta_features": False,
                                "include_time_features": False,
                                "include_current_target": False,
                                "include_terminal_samples": False,
                            },
                        },
                    },
                    handle,
                )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(include_feature_names=["pred_signal"]),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.25,
                    min_population=1,
                ),
                predictor=ForecastModelPredictorConfig(model_bundle_path=bundle_path),
                output=OutputConfig(
                    experiment_name="forecast_bundle_no_preprocessor_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            self.assertEqual(summary["rounds_executed"], 1)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_b")

    def test_latent_forecast_model_bundle_can_drive_selection(self) -> None:
        """Verify that latent forecast model bundle can drive selection.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            target_path = f"{tmp_dir}/12_latent_targets__prepared_dataset.parquet"
            pt_path = f"{tmp_dir}/latent_representations.pt"
            bundle_path = f"{tmp_dir}/latent_model_bundle.pkl"
            _write_dataset(
                target_path,
                rows=[
                    {"model_id": "model_a", "epoch": 1, "test_acc": 0.60},
                    {"model_id": "model_a", "epoch": 2, "test_acc": 0.65},
                    {"model_id": "model_b", "epoch": 1, "test_acc": 0.50},
                    {"model_id": "model_b", "epoch": 2, "test_acc": 0.90},
                    {"model_id": "model_c", "epoch": 1, "test_acc": 0.40},
                    {"model_id": "model_c", "epoch": 2, "test_acc": 0.20},
                    {"model_id": "model_d", "epoch": 1, "test_acc": 0.30},
                    {"model_id": "model_d", "epoch": 2, "test_acc": 0.10},
                ],
            )
            torch.save(
                {
                    "model_ids": ["model_a", "model_a", "model_b", "model_b", "model_c", "model_c", "model_d", "model_d"],
                    "epochs": torch.tensor([1, 2, 1, 2, 1, 2, 1, 2]),
                    "embeddings": torch.tensor(
                        [
                            [0.2, 1.0],
                            [0.2, 1.0],
                            [1.0, 0.0],
                            [1.0, 0.0],
                            [0.1, 0.0],
                            [0.1, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                },
                pt_path,
            )
            with open(bundle_path, "wb") as handle:
                pickle.dump(
                    {
                        "model": ColumnSelectorModel("z_0__lag_0"),
                        "preprocessor": None,
                        "feature_columns": ["z_0__lag_0"],
                        "sequence_config": {
                            "target_mode": "horizon_value",
                            "horizon": 1,
                            "max_lag": 0,
                            "min_history": 1,
                            "add_delta_features": False,
                            "include_time_features": False,
                            "include_current_target": False,
                            "include_terminal_samples": False,
                        },
                    },
                    handle,
                )

            cfg = SimulationConfig(
                data=SimulationDataConfig(
                    source="latent_forecast",
                    parquet_path=target_path,
                    pt_path=pt_path,
                    target_column="test_acc",
                ),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.25,
                    min_population=1,
                ),
                predictor=ForecastModelPredictorConfig(model_bundle_path=bundle_path),
                output=OutputConfig(
                    experiment_name="latent_forecast_bundle_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            used_features = json.loads(Path(f"{result['artifacts_dir']}/used_features.json").read_text())

            self.assertEqual(summary["rounds_executed"], 1)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_b")
            self.assertEqual(used_features["features"], ["z_0__lag_0"])

    def test_latent_window_forecast_model_bundle_can_drive_selection(self) -> None:
        """Verify that latent window-sequence forecast bundles work in the simulation pipeline."""
        with TemporaryDirectory() as tmp_dir:
            target_path = f"{tmp_dir}/12_latent_window_targets__prepared_dataset.parquet"
            pt_path = f"{tmp_dir}/latent_windows.pt"
            bundle_path = f"{tmp_dir}/latent_window_model_bundle.pkl"
            _write_dataset(
                target_path,
                rows=[
                    {"model_id": "model_a", "epoch": 1, "test_acc": 0.60},
                    {"model_id": "model_a", "epoch": 2, "test_acc": 0.65},
                    {"model_id": "model_b", "epoch": 1, "test_acc": 0.50},
                    {"model_id": "model_b", "epoch": 2, "test_acc": 0.90},
                    {"model_id": "model_c", "epoch": 1, "test_acc": 0.40},
                    {"model_id": "model_c", "epoch": 2, "test_acc": 0.20},
                    {"model_id": "model_d", "epoch": 1, "test_acc": 0.30},
                    {"model_id": "model_d", "epoch": 2, "test_acc": 0.10},
                ],
            )
            torch.save(
                {
                    "model_ids": ["model_a", "model_a", "model_b", "model_b", "model_c", "model_c", "model_d", "model_d"],
                    "epochs": torch.tensor([1, 2, 1, 2, 1, 2, 1, 2]),
                    "embeddings": [
                        torch.tensor([[0.20, 1.0], [0.20, 0.0]], dtype=torch.float32),
                        torch.tensor([[0.20, 1.0], [0.20, 0.0]], dtype=torch.float32),
                        torch.tensor([[1.00, 0.0], [1.00, 0.0], [1.00, 0.0]], dtype=torch.float32),
                        torch.tensor([[1.00, 0.0], [1.00, 0.0], [1.00, 0.0]], dtype=torch.float32),
                        torch.tensor([[0.10, 0.0], [0.10, 0.0]], dtype=torch.float32),
                        torch.tensor([[0.10, 0.0], [0.10, 0.0]], dtype=torch.float32),
                        torch.tensor([[0.00, 0.0]], dtype=torch.float32),
                        torch.tensor([[0.00, 0.0]], dtype=torch.float32),
                    ],
                },
                pt_path,
            )
            with open(bundle_path, "wb") as handle:
                pickle.dump(
                    {
                        "model": WindowSequenceSelectorModel(),
                        "preprocessor": None,
                        "feature_columns": ["window_sequence"],
                        "sequence_config": {
                            "target_mode": "horizon_value",
                            "horizon": 1,
                            "max_lag": 0,
                            "min_history": 1,
                            "add_delta_features": False,
                            "include_time_features": False,
                            "include_current_target": False,
                            "include_terminal_samples": False,
                        },
                    },
                    handle,
                )

            cfg = SimulationConfig(
                data=SimulationDataConfig(
                    source="latent_forecast",
                    parquet_path=target_path,
                    pt_path=pt_path,
                    target_column="test_acc",
                ),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.25,
                    min_population=1,
                ),
                predictor=ForecastModelPredictorConfig(model_bundle_path=bundle_path),
                output=OutputConfig(
                    experiment_name="latent_window_forecast_bundle_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            used_features = json.loads(Path(f"{result['artifacts_dir']}/used_features.json").read_text())

            self.assertEqual(summary["rounds_executed"], 1)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_b")
            self.assertEqual(used_features["features"], ["window_sequence"])

    def test_prepared_dataset_is_aligned_to_latent_valid_forecast_population(self) -> None:
        """Verify that prepared-dataset runs are restricted to the latent-valid forecast population.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            split_pt_path = f"{tmp_dir}/weight_statistics_test.pt"
            torch.save(
                {
                    "feature_names": ["pred_signal"],
                    "features": torch.tensor(
                        [
                            [1.0],
                            [1.1],
                            [1.2],
                            [2.0],
                            [2.1],
                            [2.2],
                        ],
                        dtype=torch.float32,
                    ),
                    "model_ids": ["model_a", "model_a", "model_a", "model_c", "model_c", "model_c"],
                    "epochs": [0, 1, 2, 0, 1, 2],
                    "checkpoint_paths": [
                        "model_a_epoch0.ckpt",
                        "model_a_epoch1.ckpt",
                        "model_a_epoch2.ckpt",
                        "model_c_epoch0.ckpt",
                        "model_c_epoch1.ckpt",
                        "model_c_epoch2.ckpt",
                    ],
                    "target_name": "test_acc",
                    "targets": torch.tensor([0.40, 0.60, 0.80, 0.35, 0.65, 0.85], dtype=torch.float32),
                    "metadata_columns": ["sample_idx", "model_id", "path", "epoch", "hyperparameter_key"],
                    "metadata": {
                        "sample_idx": [0, 1, 2, 3, 4, 5],
                        "model_id": ["model_a", "model_a", "model_a", "model_c", "model_c", "model_c"],
                        "path": [
                            "model_a_epoch0.ckpt",
                            "model_a_epoch1.ckpt",
                            "model_a_epoch2.ckpt",
                            "model_c_epoch0.ckpt",
                            "model_c_epoch1.ckpt",
                            "model_c_epoch2.ckpt",
                        ],
                        "epoch": [0, 1, 2, 0, 1, 2],
                        "hyperparameter_key": ["hp"] * 6,
                    },
                },
                split_pt_path,
            )

            cfg = SimulationConfig(
                data=SimulationDataConfig(
                    pt_path=split_pt_path,
                    target_column="test_acc",
                ),
                features=FeatureSelectionConfig(),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    min_history=2,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=1.0,
                    min_population=1,
                ),
                predictor=NaivePredictorConfig(strategy="current_target"),
                output=OutputConfig(
                    experiment_name="aligned_prepared_population_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            round_details = pd.read_parquet(f"{result['artifacts_dir']}/round_details.parquet")

            self.assertEqual(summary["initial_population"], 2)
            self.assertEqual(set(round_details["model_id"].astype(str)), {"model_a", "model_c"})

    def test_prepared_dataset_rejects_model_ids_from_pt_path(self) -> None:
        """Verify that prepared-dataset runs reject the deprecated latent-ID subset config.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/deprecated_prepared_dataset__prepared_dataset.parquet"
            latent_ids_path = f"{tmp_dir}/latent_reference.pt"
            _write_dataset(
                dataset_path,
                rows=[*_trajectory("model_a", [0.40, 0.60, 0.80], signal_scale=1.0)],
            )
            torch.save({"model_ids": ["model_a"]}, latent_ids_path)

            cfg = SimulationConfig(
                data=SimulationDataConfig(
                    parquet_path=dataset_path,
                    target_column="test_acc",
                    model_ids_from_pt_path=latent_ids_path,
                ),
                features=FeatureSelectionConfig(),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=1.0,
                    min_population=1,
                ),
                predictor=NaivePredictorConfig(strategy="current_target"),
                output=OutputConfig(
                    experiment_name="deprecated_prepared_population_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            with self.assertRaisesRegex(ValueError, "data.model_ids_from_pt_path is not supported"):
                run_simulation(cfg)

    def test_latent_forecast_honors_model_ids_from_pt_path_subset(self) -> None:
        """Verify that latent-forecast runs honor a stricter model_ids_from_pt_path subset.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            target_path = f"{tmp_dir}/latent_targets__prepared_dataset.parquet"
            full_pt_path = f"{tmp_dir}/latent_full.pt"
            subset_pt_path = f"{tmp_dir}/latent_subset.pt"
            _write_dataset(
                target_path,
                rows=[
                    {"model_id": "model_a", "epoch": 0, "test_acc": 0.40},
                    {"model_id": "model_a", "epoch": 1, "test_acc": 0.60},
                    {"model_id": "model_a", "epoch": 2, "test_acc": 0.80},
                    {"model_id": "model_b", "epoch": 0, "test_acc": 0.45},
                    {"model_id": "model_b", "epoch": 1, "test_acc": 0.50},
                    {"model_id": "model_b", "epoch": 2, "test_acc": 0.55},
                    {"model_id": "model_c", "epoch": 0, "test_acc": 0.35},
                    {"model_id": "model_c", "epoch": 1, "test_acc": 0.65},
                    {"model_id": "model_c", "epoch": 2, "test_acc": 0.85},
                ],
            )
            torch.save(
                {
                    "model_ids": [
                        "model_a",
                        "model_a",
                        "model_a",
                        "model_b",
                        "model_b",
                        "model_b",
                        "model_c",
                        "model_c",
                        "model_c",
                    ],
                    "epochs": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                    "embeddings": torch.tensor(
                        [
                            [1.0, 0.0],
                            [1.1, 0.0],
                            [1.2, 0.0],
                            [0.5, 0.0],
                            [0.6, 0.0],
                            [0.7, 0.0],
                            [2.0, 0.0],
                            [2.1, 0.0],
                            [2.2, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                },
                full_pt_path,
            )
            torch.save(
                {
                    "model_ids": ["model_a", "model_a", "model_a", "model_c", "model_c", "model_c"],
                    "epochs": torch.tensor([0, 1, 2, 0, 1, 2]),
                    "embeddings": torch.tensor(
                        [
                            [1.0, 0.0],
                            [1.1, 0.0],
                            [1.2, 0.0],
                            [2.0, 0.0],
                            [2.1, 0.0],
                            [2.2, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                },
                subset_pt_path,
            )

            cfg = SimulationConfig(
                data=SimulationDataConfig(
                    source="latent_forecast",
                    parquet_path=target_path,
                    pt_path=full_pt_path,
                    target_column="test_acc",
                    model_ids_from_pt_path=subset_pt_path,
                ),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    min_history=1,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=0,
                    interval=1,
                    survival_rate=1.0,
                    min_population=1,
                ),
                predictor=NaivePredictorConfig(strategy="current_target"),
                output=OutputConfig(
                    experiment_name="latent_subset_alignment_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            round_details = pd.read_parquet(f"{result['artifacts_dir']}/round_details.parquet")

            self.assertEqual(summary["initial_population"], 2)
            self.assertEqual(set(round_details["model_id"].astype(str)), {"model_a", "model_c"})

    def test_summary_reports_rank_gap_when_survivor_is_not_oracle_top_model(self) -> None:
        """Verify that summary reports rank gap when survivor is not oracle top model.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/imperfect_selection__prepared_dataset.parquet"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.60, 0.80, 0.70], signal_scale=1.0),
                    *_trajectory("model_b", [0.50, 0.70, 0.90], signal_scale=1.0),
                    *_trajectory("model_c", [0.40, 0.30, 0.20], signal_scale=1.0),
                    *_trajectory("model_d", [0.35, 0.25, 0.15], signal_scale=1.0),
                ],
            )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.25,
                    min_population=1,
                ),
                predictor=NaivePredictorConfig(strategy="current_target"),
                output=OutputConfig(
                    experiment_name="imperfect_selection_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            final_survivors = pd.read_parquet(f"{result['artifacts_dir']}/final_survivors.parquet")

            self.assertEqual(summary["best_survivor_model_id"], "model_a")
            self.assertEqual(summary["oracle_top_k_recall"], 0.0)
            self.assertEqual(summary["mean_survivor_rank_gap"], 1.0)
            self.assertEqual(list(final_survivors["model_id"]), ["model_a"])
            self.assertEqual(final_survivors.loc[0, "survivor_rank"], 1)
            self.assertEqual(final_survivors.loc[0, "full_population_final_rank"], 2)

    def test_online_xgboost_retrains_causally_each_round(self) -> None:
        """Verify that online xgboost retrains causally each round.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/online_xgboost__prepared_dataset.parquet"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory_with_signals("model_a", [0.10, 0.20, 0.95, 0.96], [0.20, 0.95, 0.96, 0.00]),
                    *_trajectory_with_signals("model_b", [0.10, 0.30, 0.90, 0.91], [0.30, 0.90, 0.91, 0.00]),
                    *_trajectory_with_signals("model_c", [0.10, 0.40, 0.20, 0.10], [0.40, 0.20, 0.10, 0.00]),
                    *_trajectory_with_signals("model_d", [0.10, 0.35, 0.15, 0.05], [0.35, 0.15, 0.05, 0.00]),
                ],
            )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(include_feature_names=["pred_signal"]),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.5,
                    min_population=1,
                ),
                predictor=XGBoostPredictorConfig(
                    params={
                        "objective": "reg:squarederror",
                        "n_estimators": 20,
                        "max_depth": 2,
                        "learning_rate": 0.3,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "tree_method": "hist",
                        "n_jobs": 1,
                        "random_state": 0,
                    },
                    min_train_rows=1,
                    min_train_models=2,
                ),
                output=OutputConfig(
                    experiment_name="online_xgboost_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            result = run_simulation(cfg)
            summary = result["summary"]
            round_summary = pd.read_parquet(f"{result['artifacts_dir']}/round_summary.parquet")
            final_survivors = pd.read_parquet(f"{result['artifacts_dir']}/final_survivors.parquet")
            used_features = json.loads(
                Path(f"{result['artifacts_dir']}/used_features.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["rounds_executed"], 2)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_a")
            self.assertEqual(list(round_summary["fit_strategy"]), ["xgboost", "xgboost"])
            self.assertEqual(list(round_summary["training_rows"]), [4, 6])
            self.assertEqual(list(round_summary["training_models"]), [4, 4])
            self.assertEqual(list(final_survivors["model_id"]), ["model_a"])
            self.assertIn("pred_signal__lag_0", used_features["features"])
            self.assertIn("observed_metric__current_target", used_features["features"])
            self.assertIn("observed_metric__test_acc", used_features["features"])

    def test_online_xgboost_rejects_leakage_prone_observed_metric_columns(self) -> None:
        """Verify that online xgboost rejects leakage prone observed metric columns.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/online_xgboost_no_leakage__prepared_dataset.parquet"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.10, 0.20, 0.30], signal_scale=1.0),
                    *_trajectory("model_b", [0.15, 0.25, 0.35], signal_scale=1.0),
                ],
            )

            cfg = SimulationConfig(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(include_feature_names=["pred_signal"]),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.5,
                    min_population=1,
                ),
                predictor=XGBoostPredictorConfig(
                    min_train_rows=1,
                    min_train_models=1,
                    include_observed_metrics=True,
                    observed_metric_columns=["future_target"],
                ),
                output=OutputConfig(
                    experiment_name="online_xgboost_no_leakage_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            with self.assertRaisesRegex(ValueError, "leakage-prone columns"):
                run_simulation(cfg)

    def test_online_lstm_retrains_causally_each_round(self) -> None:
        """Verify that online latent-window LSTM retrains causally each round.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            target_path = f"{tmp_dir}/online_lstm_targets__prepared_dataset.parquet"
            pt_path = f"{tmp_dir}/online_lstm_windows.pt"
            _write_dataset(
                target_path,
                rows=[
                    {"model_id": "model_a", "epoch": 0, "test_acc": 0.10},
                    {"model_id": "model_a", "epoch": 1, "test_acc": 0.20},
                    {"model_id": "model_a", "epoch": 2, "test_acc": 0.95},
                    {"model_id": "model_a", "epoch": 3, "test_acc": 0.96},
                    {"model_id": "model_b", "epoch": 0, "test_acc": 0.10},
                    {"model_id": "model_b", "epoch": 1, "test_acc": 0.30},
                    {"model_id": "model_b", "epoch": 2, "test_acc": 0.90},
                    {"model_id": "model_b", "epoch": 3, "test_acc": 0.91},
                    {"model_id": "model_c", "epoch": 0, "test_acc": 0.10},
                    {"model_id": "model_c", "epoch": 1, "test_acc": 0.40},
                    {"model_id": "model_c", "epoch": 2, "test_acc": 0.20},
                    {"model_id": "model_c", "epoch": 3, "test_acc": 0.10},
                    {"model_id": "model_d", "epoch": 0, "test_acc": 0.10},
                    {"model_id": "model_d", "epoch": 1, "test_acc": 0.35},
                    {"model_id": "model_d", "epoch": 2, "test_acc": 0.15},
                    {"model_id": "model_d", "epoch": 3, "test_acc": 0.05},
                ],
            )
            torch.save(
                {
                    "model_ids": [
                        "model_a",
                        "model_a",
                        "model_a",
                        "model_a",
                        "model_b",
                        "model_b",
                        "model_b",
                        "model_b",
                        "model_c",
                        "model_c",
                        "model_c",
                        "model_c",
                        "model_d",
                        "model_d",
                        "model_d",
                        "model_d",
                    ],
                    "epochs": torch.tensor([0, 1, 2, 3] * 4),
                    "embeddings": [
                        torch.tensor([[0.95, 0.0], [0.95, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.95, 0.0], [0.95, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.95, 0.0], [0.95, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.95, 0.0], [0.95, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.90, 0.0], [0.90, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.90, 0.0], [0.90, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.90, 0.0], [0.90, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.90, 0.0], [0.90, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.20, 0.0], [0.20, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.20, 0.0], [0.20, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.20, 0.0], [0.20, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.20, 0.0], [0.20, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.15, 0.0], [0.15, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.15, 0.0], [0.15, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.15, 0.0], [0.15, 1.0]], dtype=torch.float32),
                        torch.tensor([[0.15, 0.0], [0.15, 1.0]], dtype=torch.float32),
                    ],
                },
                pt_path,
            )

            cfg = SimulationConfig(
                data=SimulationDataConfig(
                    source="latent_forecast",
                    parquet_path=target_path,
                    pt_path=pt_path,
                    target_column="test_acc",
                ),
                features=FeatureSelectionConfig(),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.5,
                    min_population=1,
                ),
                predictor=OnlineLSTMPredictorConfig(
                    params={
                        "hidden_size": 16,
                        "head_hidden_layer_sizes": [8],
                        "max_epochs": 5,
                        "patience": 2,
                    },
                    min_train_rows=2,
                    min_train_models=2,
                ),
                output=OutputConfig(
                    experiment_name="online_lstm_test",
                    output_root=tmp_dir,
                    logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                ),
            )

            build_calls: list[dict[str, object]] = []

            def _build_stub(model_cfg):
                build_calls.append(dict(model_cfg))
                self.assertEqual(model_cfg["name"], "latent_window_lstm")
                return TrainableWindowSequenceSelectorRegressor()

            with patch("models.predictors.create_regressor", side_effect=_build_stub):
                result = run_simulation(cfg)

            summary = result["summary"]
            round_summary = pd.read_parquet(f"{result['artifacts_dir']}/round_summary.parquet")
            final_survivors = pd.read_parquet(f"{result['artifacts_dir']}/final_survivors.parquet")
            used_features = json.loads(
                Path(f"{result['artifacts_dir']}/used_features.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["rounds_executed"], 2)
            self.assertEqual(summary["final_population"], 1)
            self.assertEqual(summary["best_survivor_model_id"], "model_a")
            self.assertEqual(list(round_summary["fit_strategy"]), ["latent_window_lstm", "latent_window_lstm"])
            self.assertEqual(list(round_summary["training_rows"]), [4, 6])
            self.assertEqual(list(round_summary["training_models"]), [4, 4])
            self.assertEqual(list(round_summary["metric_feature_count"]), [1, 1])
            self.assertEqual(list(final_survivors["model_id"]), ["model_a"])
            self.assertEqual(len(build_calls), 2)
            self.assertEqual(build_calls[0]["params"]["hidden_size"], 16)
            self.assertIn("window_sequence", used_features["features"])
            self.assertIn("observed_metric__current_target", used_features["features"])

    def test_missing_decision_epoch_models_are_retained_to_hit_target_population(self) -> None:
        """Verify that missing decision epoch models are retained to hit target population.

        Returns:
            None: None.
        """
        with TemporaryDirectory() as tmp_dir:
            dataset_path = f"{tmp_dir}/missing_epoch_alignment__prepared_dataset.parquet"
            bundle_path = f"{tmp_dir}/model_bundle.pkl"
            _write_dataset(
                dataset_path,
                rows=[
                    *_trajectory("model_a", [0.10, 0.95, 0.96, 0.97], signal_scale=0.95),
                    *_trajectory("model_b", [0.10, 0.90, 0.91, 0.92], signal_scale=0.05),
                    *_trajectory_with_explicit_epochs("model_c", [(0, 0.10), (1, 0.20), (3, 0.93)], signal_scale=10.0),
                    *_trajectory_with_explicit_epochs("model_d", [(0, 0.10), (1, 0.10), (3, 0.94)], signal_scale=9.0),
                ],
            )

            preprocessor = FeaturePreprocessor(strategy="median")
            preprocessor.fit(pd.DataFrame({"pred_signal__lag_0": [0.1, 0.5, 1.0]}))
            with open(bundle_path, "wb") as handle:
                pickle.dump(
                    {
                        "model": ColumnSelectorModel("pred_signal__lag_0"),
                        "preprocessor": preprocessor,
                        "feature_columns": ["pred_signal__lag_0"],
                        "config": {
                            "data": {"target_column": "test_acc"},
                            "features": {
                                "include_model_metrics": True,
                                "include_layer_metrics": True,
                                "metric_families": [],
                                "metrics": [],
                                "scopes": [],
                                "layer_types": [],
                                "include_feature_names": ["pred_signal"],
                                "exclude_feature_names": [],
                                "include_feature_patterns": [],
                                "exclude_feature_patterns": [],
                                "max_missing_fraction": 0.25,
                                "drop_constant_features": True,
                                "missing_value_fill": "median",
                                "sort_columns": True,
                                "log_feature_names": False,
                                "max_console_logged_features": 50,
                            },
                            "sequence": {
                                "target_mode": "horizon_value",
                                "horizon": 1,
                                "max_lag": 0,
                                "min_history": 1,
                                "add_delta_features": False,
                                "include_time_features": False,
                                "include_current_target": False,
                                "include_terminal_samples": False,
                            },
                        },
                    },
                    handle,
                )

            common_kwargs = dict(
                data=PreparedDatasetConfig(parquet_path=dataset_path, target_column="test_acc"),
                features=FeatureSelectionConfig(include_feature_names=["pred_signal"]),
                sequence=SequenceFeatureConfig(
                    target_mode="horizon_value",
                    horizon=1,
                    max_lag=0,
                    add_delta_features=False,
                    include_time_features=False,
                ),
                selection=SelectionConfig(
                    start_epoch=1,
                    interval=1,
                    survival_rate=0.75,
                    min_population=1,
                ),
            )

            naive_result = run_simulation(
                SimulationConfig(
                    predictor=NaivePredictorConfig(strategy="current_target"),
                    output=OutputConfig(
                        experiment_name="naive_missing_epoch_alignment",
                        output_root=tmp_dir,
                        logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                    ),
                    **common_kwargs,
                )
            )
            forecast_result = run_simulation(
                SimulationConfig(
                    predictor=ForecastModelPredictorConfig(model_bundle_path=bundle_path),
                    output=OutputConfig(
                        experiment_name="forecast_missing_epoch_alignment",
                        output_root=tmp_dir,
                        logging=LoggingConfig(level="INFO", save_to_file=False, filename="unused.log"),
                    ),
                    **common_kwargs,
                )
            )

            naive_round_details = pd.read_parquet(f"{naive_result['artifacts_dir']}/round_details.parquet")
            forecast_round_details = pd.read_parquet(f"{forecast_result['artifacts_dir']}/round_details.parquet")

            naive_round_two = naive_round_details.loc[naive_round_details["round_index"] == 2].copy()
            forecast_round_two = forecast_round_details.loc[forecast_round_details["round_index"] == 2].copy()

            self.assertEqual(int(naive_round_two["population_after"].iloc[0]), 3)
            self.assertEqual(int(forecast_round_two["population_after"].iloc[0]), 3)
            self.assertEqual(int(naive_round_two["selected"].sum()), 3)
            self.assertEqual(int(forecast_round_two["selected"].sum()), 3)

            kept_missing_forecast = forecast_round_two.loc[
                forecast_round_two["elimination_reason"] == "missing_decision_epoch_kept", "model_id"
            ].astype(str)
            self.assertEqual(set(kept_missing_forecast), {"model_c", "model_d"})


def _trajectory(model_id: str, targets: list[float], *, signal_scale: float) -> list[dict[str, object]]:
    """Trajectory the function.

    Args:
        model_id (str): str value used by the function.
        targets (list[float]): list[float] value used by the function.
        signal_scale (float): float value used by the function.

    Returns:
        list[dict[str, object]]: Result of the function.
    """
    rows: list[dict[str, object]] = []
    model_offset = 1000 * sum(ord(char) for char in model_id)
    for epoch, target in enumerate(targets):
        rows.append(
            {
                "sample_idx": model_offset + epoch,
                "model_id": model_id,
                "path": f"/tmp/{model_id}",
                "epoch": epoch,
                "hyperparameter_key": model_id,
                "test_acc": target,
                "pred_signal": target * signal_scale,
                "aux_feature": float(epoch),
            }
        )
    return rows


def _trajectory_with_explicit_epochs(
    model_id: str,
    epoch_targets: list[tuple[int, float]],
    *,
    signal_scale: float,
) -> list[dict[str, object]]:
    """Trajectory with explicit epochs.

    Args:
        model_id (str): str value used by the function.
        epoch_targets (list[tuple[int, float]]): list[tuple[int, float]] value used by the function.
        signal_scale (float): float value used by the function.

    Returns:
        list[dict[str, object]]: Result of the function.
    """
    rows: list[dict[str, object]] = []
    model_offset = 1000 * sum(ord(char) for char in model_id)
    for position, (epoch, target) in enumerate(epoch_targets):
        rows.append(
            {
                "sample_idx": model_offset + position,
                "model_id": model_id,
                "path": f"/tmp/{model_id}",
                "epoch": epoch,
                "hyperparameter_key": model_id,
                "test_acc": target,
                "pred_signal": target * signal_scale,
                "aux_feature": float(epoch),
            }
        )
    return rows


def _trajectory_with_signals(
    model_id: str,
    targets: list[float],
    signals: list[float],
) -> list[dict[str, object]]:
    """Trajectory with signals.

    Args:
        model_id (str): str value used by the function.
        targets (list[float]): list[float] value used by the function.
        signals (list[float]): list[float] value used by the function.

    Returns:
        list[dict[str, object]]: Result of the function.
    """
    rows: list[dict[str, object]] = []
    model_offset = 1000 * sum(ord(char) for char in model_id)
    for epoch, (target, signal) in enumerate(zip(targets, signals, strict=True)):
        rows.append(
            {
                "sample_idx": model_offset + epoch,
                "model_id": model_id,
                "path": f"/tmp/{model_id}",
                "epoch": epoch,
                "hyperparameter_key": model_id,
                "test_acc": target,
                "pred_signal": signal,
                "aux_feature": float(epoch),
            }
        )
    return rows


if __name__ == "__main__":
    unittest.main()
