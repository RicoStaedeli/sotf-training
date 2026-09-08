from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argparse import Namespace
from pathlib import Path

from baselines.learning_curve_extrapolation import (
    LearningCurveExtrapolator,
    infer_target_bounds,
    predict_forecast_targets,
)
from baselines.run_learning_curve_extrapolation import resolve_baseline_configs


def test_learning_curve_extrapolator_predicts_growth_above_current_value() -> None:
    extrapolator = LearningCurveExtrapolator(target_bounds=(0.0, 1.0), min_points=3)

    result = extrapolator.predict_one(
        np.asarray([1, 2, 3, 4], dtype="float64"),
        np.asarray([0.20, 0.36, 0.49, 0.59], dtype="float64"),
        target_epoch=8,
        fallback_value=0.59,
    )

    assert result.prediction > 0.59
    assert result.prediction <= 1.0
    assert result.n_curve_predictions >= 1


def test_learning_curve_extrapolator_falls_back_for_short_histories() -> None:
    extrapolator = LearningCurveExtrapolator(target_bounds=(0.0, 1.0), min_points=3)

    result = extrapolator.predict_one(
        np.asarray([1, 2], dtype="float64"),
        np.asarray([0.20, 0.36], dtype="float64"),
        target_epoch=8,
        fallback_value=0.36,
    )

    assert result.prediction == 0.36
    assert result.method == "fallback_too_few_points"
    assert result.n_curve_predictions == 0


def test_predict_forecast_targets_uses_only_observed_prefix() -> None:
    full_metadata = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m1", "m1"],
            "epoch": [1, 2, 3, 4, 5],
            "test_acc": [0.2, 0.35, 0.48, 0.58, 0.66],
        },
        index=pd.Index([10, 11, 12, 13, 14], name="sample_idx"),
    )
    validation_metadata = pd.DataFrame(
        {
            "model_id": ["m1"],
            "epoch": [3],
            "current_target": [0.48],
            "future_target": [0.66],
            "max_epoch": [5],
        },
        index=pd.Index([12], name="sample_idx"),
    )

    predictions = predict_forecast_targets(
        full_metadata=full_metadata,
        validation_metadata=validation_metadata,
        target_column="test_acc",
        target_mode="final_value",
        horizon=5,
        target_bounds=(0.0, 1.0),
        min_points=3,
    )

    assert list(predictions.index) == [12]
    assert predictions.loc[12, "prediction"] > 0.48
    assert predictions.loc[12, "target_epoch"] == 5


def test_infer_target_bounds_respects_accuracy_scale() -> None:
    assert infer_target_bounds(np.asarray([0.1, 0.8]), target_column="test_acc") == pytest.approx((0.03, 0.87))
    assert infer_target_bounds(np.asarray([10.0, 80.0]), target_column="test_acc") == pytest.approx((3.0, 87.0))


def test_config_name_expands_datasets_and_target_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "extrapolation.yaml"
    config_path.write_text(
        """
output_root: metric-prediction/outputs/baselines/test
defaults:
  target_column: test_acc
  curve_min_points: 3
target_modes:
  - target_mode: final_value
    horizon: 5
    min_history: 4
  - target_mode: horizon_value
    horizon: 5
    min_history: 4
datasets:
  - name: ds_a
    dataset_path: cache/ds_a
  - name: ds_b
    dataset_path: cache/ds_b
""",
        encoding="utf-8",
    )
    args = Namespace(
        config_name=str(config_path),
        batch_config=None,
        train_pt=None,
        val_pt=None,
        parquet=None,
        target_column=None,
        target_mode=None,
        horizon=None,
        min_history=None,
        epoch_filter=None,
        target_model_min_threshold=None,
        split_csv_path=None,
        validation_size=None,
        random_seed=None,
        experiment_name=None,
        output_root=Path("metric-prediction/outputs/baselines/learning_curve_extrapolation"),
        run_dir=None,
        curve_min_points=3,
    )

    configs = resolve_baseline_configs(args)

    assert len(configs) == 4
    assert configs[0]["experiment_name"] == "ds_a_learning_curve_final_value"
    assert configs[0]["pt_train_path"] == "cache/ds_a/weight_statistics_train.pt"
    assert configs[0]["pt_val_path"] == "cache/ds_a/weight_statistics_val.pt"
    assert configs[1]["target_mode"] == "horizon_value"
    assert configs[2]["batch_name"] == "ds_b"
