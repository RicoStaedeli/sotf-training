from __future__ import annotations

from datetime import datetime

import pytest

from runner.batch import copy_batch_config, load_batch_config
from runner.batch_runner import (
    BatchConfig,
    ScheduledExperiment,
    build_run_dir,
    build_run_overrides,
)
from utils.config_utils import sanitize_path_component


def test_sanitize_path_component_normalizes_for_paths() -> None:
    assert sanitize_path_component(" LightGBM / tuned run #1 ") == "LightGBM_tuned_run_1"


def test_build_run_overrides_includes_experiment_name_override() -> None:
    cfg = BatchConfig(
        batch_name=None,
        output_root="metric-prediction/outputs",
        common_overrides=("data.parquet_path=/tmp/dataset.parquet",),
        runs=(),
    )
    run = ScheduledExperiment(
        experiment_name="ridge_baseline",
        overrides=("model=ridge",),
    )

    assert build_run_overrides(cfg, run=run) == (
        "experiment_name=ridge_baseline",
        "data.parquet_path=/tmp/dataset.parquet",
        "model=ridge",
    )


def test_build_run_dir_uses_batch_output_root() -> None:
    cfg = BatchConfig(
        batch_name="ds1-ds4",
        output_root="metric-prediction/outputs",
        common_overrides=(),
        runs=(),
    )
    run = ScheduledExperiment(
        experiment_name="ridge_baseline",
        overrides=("model=ridge",),
    )

    run_dir = build_run_dir(
        cfg,
        run=run,
        run_cfg={"mode": "prediction", "experiment_name": "ridge_baseline"},
        now=datetime(2026, 5, 1, 12, 30, 45),
    )

    assert run_dir.as_posix() == "metric-prediction/outputs/batch_experiments/ds1-ds4/results/ridge_baseline"


def test_build_run_overrides_rejects_manual_experiment_name_override() -> None:
    cfg = BatchConfig(batch_name=None, output_root="metric-prediction/outputs", common_overrides=(), runs=())
    run = ScheduledExperiment(experiment_name="ridge_baseline", overrides=("experiment_name=other",))

    with pytest.raises(ValueError, match="experiment_name"):
        build_run_overrides(cfg, run=run)


def test_build_run_overrides_rejects_manual_hydra_run_dir_override() -> None:
    cfg = BatchConfig(batch_name="ds1-ds4", output_root="metric-prediction/outputs", common_overrides=(), runs=())
    run = ScheduledExperiment(experiment_name="ridge_baseline", overrides=("hydra.run.dir=somewhere/else",))

    with pytest.raises(ValueError, match="hydra.run.dir"):
        build_run_overrides(cfg, run=run)


def test_load_batch_config_reads_common_and_run_overrides(tmp_path) -> None:
    batch_config_path = tmp_path / "batch.yaml"
    batch_config_path.write_text(
        "\n".join(
            [
                "common_overrides:",
                "  train.save_shap_artifacts: false",
                "runs:",
                "  - experiment_name: ridge_baseline",
                "    overrides:",
                "      data.parquet_path: /tmp/data.parquet",
                "      model: ridge",
                "  - experiment_name: lightgbm_tuned",
                "    overrides:",
                "      data.parquet_path: /tmp/data_2.parquet",
                "      model: lightgbm",
                "      tune: lightgbm",
            ]
        ),
        encoding="utf-8",
    )

    config = load_batch_config(batch_config_path)

    assert config.common_overrides == ("train.save_shap_artifacts=false",)
    assert [run.experiment_name for run in config.runs] == ["ridge_baseline", "lightgbm_tuned"]
    assert config.runs[0].overrides == ("data.parquet_path=/tmp/data.parquet", "model=ridge")
    assert config.runs[1].overrides == (
        "data.parquet_path=/tmp/data_2.parquet",
        "model=lightgbm",
        "tune=lightgbm",
    )


def test_load_batch_config_reads_optional_batch_name(tmp_path) -> None:
    batch_config_path = tmp_path / "batch.yaml"
    batch_config_path.write_text(
        "\n".join(
            [
                "batch_name: ds1-ds4",
                "runs:",
                "  - experiment_name: ridge_baseline",
            ]
        ),
        encoding="utf-8",
    )

    config = load_batch_config(batch_config_path)

    assert config.batch_name == "ds1-ds4"
    assert [run.experiment_name for run in config.runs] == ["ridge_baseline"]


def test_copy_batch_config_saves_yaml_in_batch_output_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "example.yaml"
    source_path.write_text("batch_name: Exp_3_4\nruns:\n  - experiment_name: ridge_baseline\n", encoding="utf-8")

    copied_path = copy_batch_config(source_path, "Exp_3_4")

    assert copied_path == (
        tmp_path / "metric-prediction" / "outputs" / "batch_experiments" / "Exp_3_4" / "batch_files" / "example.yaml"
    ).resolve()
    assert copied_path.read_text(encoding="utf-8") == source_path.read_text(encoding="utf-8")


def test_load_batch_config_infers_forecast_output_root_from_mode(tmp_path) -> None:
    batch_config_path = tmp_path / "forecast_batch.yaml"
    batch_config_path.write_text(
        "\n".join(
            [
                "batch_name: trajectory-study",
                "common_overrides:",
                "  mode: forecasting",
                "  train.save_predictions: false",
                "runs:",
                "  - experiment_name: forecast_final_value",
                "    overrides:",
                "      data.parquet_path: /tmp/data.parquet",
                "      sequence.target_mode: final_value",
            ]
        ),
        encoding="utf-8",
    )

    config = load_batch_config(batch_config_path)

    assert config.output_root.as_posix() == "metric-prediction/outputs_forecasting"
    assert config.common_overrides == ("mode=forecasting", "train.save_predictions=false")


def test_forecast_batch_config_uses_generic_run_dir() -> None:
    cfg = BatchConfig(
        batch_name="trajectory-study",
        output_root="metric-prediction/outputs_forecasting",
        common_overrides=("mode=forecasting",),
        runs=(),
    )
    run = ScheduledExperiment(
        experiment_name="forecast_baseline",
        overrides=("sequence.target_mode=horizon_value",),
    )

    run_dir = build_run_dir(
        cfg,
        run=run,
        run_cfg={"mode": "forecasting", "experiment_name": "forecast_baseline"},
        now=datetime(2026, 5, 1, 12, 30, 45),
    )

    assert (
        run_dir.as_posix()
        == "metric-prediction/outputs_forecasting/batch_experiments/trajectory-study/results/forecast_baseline"
    )


def test_copy_forecast_batch_config_uses_yaml_output_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "forecast_batch.yaml"
    source_path.write_text(
        "\n".join(
            [
                "batch_name: trajectory-study",
                "common_overrides:",
                "  mode: forecasting",
                "runs:",
                "  - experiment_name: forecast_baseline",
            ]
        ),
        encoding="utf-8",
    )

    copied_path = copy_batch_config(source_path, "trajectory-study")

    assert copied_path == (
        tmp_path
        / "metric-prediction"
        / "outputs_forecasting"
        / "batch_experiments"
        / "trajectory-study"
        / "batch_files"
        / "forecast_batch.yaml"
    ).resolve()
