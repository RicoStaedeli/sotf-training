from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from evaluation.forecasting import (
    build_best_experiments_by_model_type,
    build_model_type_metric_table,
    build_overall_metric_table,
    build_per_epoch_metric_table,
    heatmap_colormap,
    heatmap_value_limits,
    load_forecast_runs,
    main,
    render_heatmap,
    select_latest_runs,
)


def test_select_latest_forecast_runs_prefers_newest_run(tmp_path: Path) -> None:
    results_dir = tmp_path / "batch_results"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds4_forecast_final_value_all_families",
        run_name="10-00-00-000000",
        dataset_name="4_cifar10_resnet18_kaiming",
        target_mode="final_value",
        r2=0.10,
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds4_forecast_final_value_all_families",
        run_name="11-00-00-000000",
        dataset_name="4_cifar10_resnet18_kaiming",
        target_mode="final_value",
        r2=0.75,
    )

    latest = select_latest_runs(load_forecast_runs(results_dir))

    assert latest["r2"].tolist() == [0.75]
    assert latest["dataset_name"].tolist() == ["4_cifar10_resnet18_kaiming"]


def test_best_experiments_by_model_type_prefers_highest_r2(tmp_path: Path) -> None:
    results_dir = tmp_path / "batch_results"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_ridge_final_value_worse",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        model_name="ridge",
        r2=0.20,
        rmse=0.40,
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_ridge_final_value_best",
        run_name="11-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        model_name="ridge",
        r2=0.80,
        rmse=0.10,
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_xgboost_final_value_best",
        run_name="11-30-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        model_name="xgboost",
        r2=0.70,
        rmse=0.20,
    )

    latest = select_latest_runs(load_forecast_runs(results_dir))
    best = build_best_experiments_by_model_type(latest)

    winners = best.set_index("model_type")["experiment_name"].to_dict()
    assert winners == {
        "ridge": "ds1_ridge_final_value_best",
        "xgboost": "ds1_xgboost_final_value_best",
    }


def test_load_forecast_runs_reads_latent_model_config_without_resolved_config(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "batch_results" / "ds2_latent_xgb_final_value" / "10-00-00-000000" / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(
            {
                "r2": 0.90,
                "rmse": 0.05,
                "target_mode": "final_value",
                "target_column": "test_acc",
                "horizon": 5,
            }
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "resolved_model_config.json").write_text(
        json.dumps(
            {
                "name": "xgboost",
                "params": {
                    "wandb_config": {
                        "experiment_name": "ds2_latent_xgb_final_value",
                        "data": {
                            "target_column": "test_acc",
                            "target_parquet_path": (
                                "metric-prediction/cache/prepared_datasets/"
                                "2_cifar10_large_hyp_random_seeds_full_all_metrics_long__prepared_dataset.parquet"
                            ),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runs = load_forecast_runs(tmp_path / "batch_results")

    assert runs.loc[0, "model_type"] == "xgboost"
    assert runs.loc[0, "dataset_name"] == "2_cifar10_large_hyp_random_seeds"
    assert runs.loc[0, "horizon"] == 5


def test_load_forecast_runs_reads_dataset_name_from_hydra_config_for_artifacts_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch_results" / "ds2_latent_xgb_final_value"
    artifacts_dir = run_dir / "artifacts"
    hydra_dir = run_dir / ".hydra"
    artifacts_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(
            {
                "r2": 0.90,
                "rmse": 0.05,
                "target_mode": "final_value",
                "target_column": "test_acc",
                "horizon": 5,
            }
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "resolved_model_config.json").write_text(
        json.dumps(
            {
                "name": "xgboost",
                "params": {
                    "learning_rate": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    (hydra_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  target_parquet_path: metric-prediction/cache/prepared_datasets/2_cifar10_large_hyp_random_seeds_full_all_metrics_long__prepared_dataset.parquet",
                "  target_column: test_acc",
                "sequence:",
                "  target_mode: final_value",
                "model:",
                "  name: xgboost",
                "experiment_name: ds2_latent_xgb_final_value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    runs = load_forecast_runs(tmp_path / "batch_results")

    assert runs.loc[0, "experiment_name"] == "ds2_latent_xgb_final_value"
    assert runs.loc[0, "dataset_name"] == "2_cifar10_large_hyp_random_seeds"
    assert runs.loc[0, "target_mode"] == "final_value"


def test_build_forecast_heatmap_tables(tmp_path: Path) -> None:
    results_dir = tmp_path / "batch_results"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_forecast_final_value_all_families",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.60,
        prediction_rows=[
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
            {"epoch": 1, "target": 2.0, "prediction": 2.0},
            {"epoch": 2, "target": 1.0, "prediction": 1.0},
            {"epoch": 2, "target": 2.0, "prediction": 1.5},
        ],
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_forecast_horizon_5_all_families",
        run_name="10-05-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="horizon_value",
        r2=0.20,
        prediction_rows=[
            {"epoch": 1, "target": 0.0, "prediction": 0.0},
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
        ],
    )

    latest = select_latest_runs(load_forecast_runs(results_dir))
    overall = build_overall_metric_table(latest, metric_name="r2")
    final_value = build_per_epoch_metric_table(latest, target_mode="final_value", metric_name="r2")

    assert overall.loc["1_cifar10_uniform_large", "final_value"] == 0.60
    assert overall.loc["1_cifar10_uniform_large", "horizon_value"] == 0.20
    assert final_value.loc["1_cifar10_uniform_large", 1] == 1.0
    assert final_value.loc["1_cifar10_uniform_large", 2] == 0.5
    assert list(final_value.columns) == [1, 2]


def test_build_model_type_metric_table(tmp_path: Path) -> None:
    results_dir = tmp_path / "batch_results"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_ridge_final_value_best",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        model_name="ridge",
        r2=0.80,
        rmse=0.10,
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds2_xgboost_final_value_best",
        run_name="10-05-00-000000",
        dataset_name="2_cifar10_large_hyp_random_seeds",
        target_mode="final_value",
        model_name="xgboost",
        r2=0.70,
        rmse=0.20,
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds2_xgboost_horizon_5_best",
        run_name="10-10-00-000000",
        dataset_name="2_cifar10_large_hyp_random_seeds",
        target_mode="horizon_value",
        model_name="xgboost",
        r2=0.50,
        rmse=0.30,
    )

    latest = select_latest_runs(load_forecast_runs(results_dir))
    model_type_r2 = build_model_type_metric_table(latest, metric_name="r2")

    assert model_type_r2.loc["ridge", "final_value"] == 0.80
    assert model_type_r2.loc["xgboost", "final_value"] == 0.70
    assert model_type_r2.loc["xgboost", "horizon_value"] == 0.50
    assert list(model_type_r2.index) == ["ridge", "xgboost"]
    assert list(model_type_r2.columns) == ["final_value", "horizon_value"]


def test_build_per_epoch_r2_table_caps_epochs_at_50(tmp_path: Path) -> None:
    results_dir = tmp_path / "batch_results"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_forecast_final_value_all_families",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.60,
        prediction_rows=[
            {"epoch": 49, "target": 1.0, "prediction": 1.0},
            {"epoch": 49, "target": 2.0, "prediction": 2.0},
            {"epoch": 50, "target": 1.0, "prediction": 1.0},
            {"epoch": 50, "target": 2.0, "prediction": 2.0},
            {"epoch": 51, "target": 1.0, "prediction": 1.0},
            {"epoch": 51, "target": 2.0, "prediction": 2.0},
        ],
    )

    latest = select_latest_runs(load_forecast_runs(results_dir))
    final_value = build_per_epoch_metric_table(latest, target_mode="final_value", metric_name="r2")

    assert list(final_value.columns) == [49, 50]


def test_forecast_evaluation_main_writes_requested_outputs(tmp_path: Path) -> None:
    results_dir = tmp_path / "batch_results"
    output_dir = tmp_path / "evaluation"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_forecast_final_value_all_families",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.60,
        prediction_rows=[
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
            {"epoch": 1, "target": 2.0, "prediction": 2.0},
        ],
    )
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_forecast_horizon_5_all_families",
        run_name="10-05-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="horizon_value",
        r2=0.20,
        prediction_rows=[
            {"epoch": 1, "target": 0.0, "prediction": 0.0},
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
        ],
    )
    exit_code = main([str(results_dir), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "01_r2_by_dataset_and_mode.png").exists()
    assert (output_dir / "02_final_value_r2_by_dataset_and_epoch.png").exists()
    assert (output_dir / "03_horizon_value_r2_by_dataset_and_epoch.png").exists()
    assert (output_dir / "04_rmse_by_dataset_and_mode.png").exists()
    assert (output_dir / "05_final_value_rmse_by_dataset_and_epoch.png").exists()
    assert (output_dir / "06_horizon_value_rmse_by_dataset_and_epoch.png").exists()
    assert (output_dir / "07_r2_by_model_type_and_mode.png").exists()
    assert (output_dir / "08_rmse_by_model_type_and_mode.png").exists()
    assert (output_dir / "00_best_experiments_by_model_type.csv").exists()
    assert (output_dir / "00_best_experiments_by_model_type.md").exists()
    assert (output_dir / "00_model_type_summary.csv").exists()
    assert (output_dir / "01_r2_by_dataset_and_mode.csv").exists()
    assert (output_dir / "02_final_value_r2_by_dataset_and_epoch.csv").exists()
    assert (output_dir / "03_horizon_value_r2_by_dataset_and_epoch.csv").exists()
    assert (output_dir / "04_rmse_by_dataset_and_mode.csv").exists()
    assert (output_dir / "05_final_value_rmse_by_dataset_and_epoch.csv").exists()
    assert (output_dir / "06_horizon_value_rmse_by_dataset_and_epoch.csv").exists()
    assert (output_dir / "07_r2_by_model_type_and_mode.csv").exists()
    assert (output_dir / "08_rmse_by_model_type_and_mode.csv").exists()
    assert not (output_dir / "04_future_improvement_r2_by_dataset_and_epoch.csv").exists()


def test_forecast_evaluation_main_resolves_batch_root_results_dir(tmp_path: Path) -> None:
    batch_root = tmp_path / "trajectory-study"
    results_dir = batch_root / "results"
    _write_run(
        results_dir=results_dir,
        experiment_name="ds1_forecast_final_value_all_families",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.60,
        prediction_rows=[
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
            {"epoch": 1, "target": 2.0, "prediction": 2.0},
        ],
    )

    exit_code = main([str(batch_root)])

    assert exit_code == 0
    assert (batch_root / "evaluation" / "01_r2_by_dataset_and_mode.csv").exists()


def test_forecast_evaluation_main_merges_multiple_batch_roots(tmp_path: Path) -> None:
    first_batch = tmp_path / "batch_a"
    second_batch = tmp_path / "batch_b"
    output_dir = tmp_path / "evaluation"
    _write_run(
        results_dir=first_batch / "results",
        experiment_name="ds1_forecast_final_value_all_families",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.60,
        prediction_rows=[
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
            {"epoch": 1, "target": 2.0, "prediction": 2.0},
        ],
    )
    _write_run(
        results_dir=second_batch / "results",
        experiment_name="ds1_forecast_horizon_5_all_families",
        run_name="10-05-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="horizon_value",
        r2=0.20,
        prediction_rows=[
            {"epoch": 1, "target": 0.0, "prediction": 0.0},
            {"epoch": 1, "target": 1.0, "prediction": 1.0},
        ],
    )

    exit_code = main(
        [
            str(first_batch),
            "--extra-results-root",
            str(second_batch),
            "--output-dir",
            str(output_dir),
        ]
    )

    latest = pd.read_csv(output_dir / "latest_runs.csv")

    assert exit_code == 0
    assert set(latest["experiment_name"]) == {
        "ds1_forecast_final_value_all_families",
        "ds1_forecast_horizon_5_all_families",
    }
    assert (output_dir / "01_r2_by_dataset_and_mode.png").exists()


def test_select_latest_forecast_runs_keeps_same_experiment_name_from_different_roots(tmp_path: Path) -> None:
    first_results = tmp_path / "batch_a" / "results"
    second_results = tmp_path / "batch_b" / "results"
    _write_run(
        results_dir=first_results,
        experiment_name="shared_name",
        run_name="10-00-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.40,
    )
    _write_run(
        results_dir=second_results,
        experiment_name="shared_name",
        run_name="10-05-00-000000",
        dataset_name="1_cifar10_uniform_large",
        target_mode="final_value",
        r2=0.80,
    )

    runs = pd.concat([load_forecast_runs(first_results), load_forecast_runs(second_results)], ignore_index=True)
    latest = select_latest_runs(runs)

    assert latest["experiment_name"].tolist() == ["shared_name", "shared_name"]


def test_heatmap_colormap_reverses_scale_for_rmse() -> None:
    default_color = heatmap_colormap()(0.0)
    reversed_color = heatmap_colormap(reverse_scale=True)(0.0)

    assert np.allclose(default_color, matplotlib.colormaps["viridis"](0.0))
    assert np.allclose(reversed_color, matplotlib.colormaps["viridis_r"](0.0))


def test_heatmap_value_limits_use_fixed_ranges_for_r2_and_rmse() -> None:
    assert heatmap_value_limits("r2") == (0.0, 1.0)
    assert heatmap_value_limits("rmse") == (0.0, 0.3)
    assert heatmap_value_limits(None, clip_negative=True) == (0.0, None)


def test_render_heatmap_applies_fixed_metric_scales(monkeypatch, tmp_path: Path) -> None:
    captured: list[dict[str, float | None]] = []
    original_imshow = Axes.imshow

    def recording_imshow(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append({"vmin": kwargs.get("vmin"), "vmax": kwargs.get("vmax")})
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    table = pd.DataFrame([[0.2, 0.8]], index=["ds1"], columns=["a", "b"])

    render_heatmap(
        table,
        output_path=tmp_path / "r2.png",
        title="R2",
        x_label="x",
        y_label="y",
        colorbar_label="R2",
        metric_name="r2",
    )
    render_heatmap(
        table,
        output_path=tmp_path / "rmse.png",
        title="RMSE",
        x_label="x",
        y_label="y",
        colorbar_label="RMSE",
        metric_name="rmse",
    )

    assert captured[0] == {"vmin": 0.0, "vmax": 1.0}
    assert captured[1] == {"vmin": 0.0, "vmax": 0.3}


def _write_run(
    *,
    results_dir: Path,
    experiment_name: str,
    run_name: str,
    dataset_name: str,
    target_mode: str,
    r2: float,
    rmse: float = 0.10,
    model_name: str = "xgboost",
    prediction_rows: list[dict[str, float]] | None = None,
) -> None:
    run_dir = results_dir / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "r2": r2,
                "rmse": rmse,
                "target_mode": target_mode,
                "target_column": "test_acc",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "data": {
                    "parquet_path": f"metric-prediction/cache/prepared_datasets/{dataset_name}_full_all_metrics_long__prepared_dataset.parquet",
                    "target_column": "test_acc",
                },
                "output": {
                    "experiment_name": experiment_name,
                },
                "sequence": {
                    "target_mode": target_mode,
                },
                "model": {
                    "name": model_name,
                },
            }
        ),
        encoding="utf-8",
    )

    if prediction_rows is not None:
        pd.DataFrame(prediction_rows).to_parquet(run_dir / "validation_predictions.parquet", index=False)
