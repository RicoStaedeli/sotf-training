from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.lcnet import LCNetRegressor, build_lcnet_features
from baselines.run_lcnet_baseline import resolve_baseline_configs


def test_build_lcnet_features_uses_only_observed_prefix() -> None:
    full_metadata = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m1"],
            "epoch": [1, 2, 3, 4],
            "test_acc": [0.20, 0.35, 0.48, 0.60],
        },
        index=pd.Index([10, 11, 12, 13], name="sample_idx"),
    )
    forecast_metadata = pd.DataFrame(
        {
            "model_id": ["m1"],
            "epoch": [2],
            "current_target": [0.35],
            "max_epoch": [4],
            "history_length": [2],
            "progress_fraction": [0.5],
        },
        index=pd.Index([11], name="sample_idx"),
    )

    features = build_lcnet_features(
        full_metadata=full_metadata,
        forecast_metadata=forecast_metadata,
        target_column="test_acc",
        target_mode="final_value",
        horizon=5,
        max_history_points=3,
    )

    assert features.loc[11, "value_lag_0"] == 0.35
    assert features.loc[11, "value_lag_1"] == 0.20
    assert features.loc[11, "lag_mask_2"] == 0.0
    assert features.loc[11, "prefix_max"] == 0.35
    assert features.loc[11, "target_epoch"] == 4


def test_lcnet_regressor_predict_distribution_shapes() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(24, 5)), columns=[f"x{i}" for i in range(5)])
    y = pd.Series(0.2 + 0.1 * X["x0"] - 0.05 * X["x1"])
    model = LCNetRegressor(
        hidden_layer_sizes=[8],
        dropout=0.2,
        max_epochs=2,
        patience=1,
        batch_size=8,
        mc_samples=3,
        random_state=7,
        device="cpu",
    )

    model.fit(X, y)
    distribution = model.predict_distribution(X.iloc[:4])

    assert distribution.mean.shape == (4,)
    assert distribution.std.shape == (4,)
    assert np.all(np.isfinite(distribution.mean))
    assert np.all(distribution.std >= 0.0)


def test_lcnet_config_name_expands_datasets_and_target_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "lcnet.yaml"
    config_path.write_text(
        """
output_root: metric-prediction/outputs/baselines/test_lcnet
defaults:
  target_column: test_acc
  max_history_points: 6
  max_epochs: 2
  mc_samples: 3
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
        output_root=Path("metric-prediction/outputs/baselines/lcnet"),
        run_dir=None,
        max_history_points=None,
        hidden_layer_sizes=None,
        dropout=None,
        max_epochs=None,
        batch_size=None,
        learning_rate=None,
        weight_decay=None,
        patience=None,
        mc_samples=None,
        device=None,
    )

    configs = resolve_baseline_configs(args)

    assert len(configs) == 4
    assert configs[0]["experiment_name"] == "ds_a_lcnet_final_value"
    assert configs[0]["pt_train_path"] == "cache/ds_a/weight_statistics_train.pt"
    assert configs[0]["pt_val_path"] == "cache/ds_a/weight_statistics_val.pt"
    assert configs[0]["max_history_points"] == 6
    assert configs[1]["target_mode"] == "horizon_value"
    assert configs[2]["batch_name"] == "ds_b"
