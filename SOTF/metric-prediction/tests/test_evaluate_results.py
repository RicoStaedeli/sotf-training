from __future__ import annotations

import json

import matplotlib
import pandas as pd

from evaluation.prediction import (
    add_rank_columns,
    build_dataset_metadata,
    build_completion_table,
    clip_negative_r2_for_plot,
    dataset_directory_name,
    default_report_dir,
    discover_batch_files,
    dataset_display_labels,
    format_dataset_display_label,
    heatmap_text_color,
    load_single_run,
    parse_experiment_name,
    resolve_outputs_root,
    resolve_batch_files,
    select_metric_completed_runs,
    with_plot_r2,
    with_target_metric,
    write_per_dataset_reports,
)


def test_parse_experiment_name_accepts_target_suffix() -> None:
    parts = parse_experiment_name("ds4_xgboost_model_and_layer_matrix_distance_sharpness_test_loss")

    assert parts.dataset_id == "ds4"
    assert parts.model == "xgboost"
    assert parts.scope == "model_and_layer"
    assert parts.family_combo == "matrix_distance_sharpness"
    assert parts.target_metric == "test_loss"


def test_parse_experiment_name_accepts_distribution_family() -> None:
    parts = parse_experiment_name("ds1_ridge_model_only_distribution_test_acc")

    assert parts.dataset_id == "ds1"
    assert parts.model == "ridge"
    assert parts.scope == "model_only"
    assert parts.family_combo == "distribution"
    assert parts.target_metric == "test_acc"


def test_parse_experiment_name_accepts_legacy_name() -> None:
    parts = parse_experiment_name("ds1_ridge_model_only_all_families")

    assert parts.dataset_id == "ds1"
    assert parts.model == "ridge"
    assert parts.scope == "model_only"
    assert parts.family_combo == "all_families"
    assert parts.target_metric is None


def test_default_report_dir_is_relative_to_outputs_root(tmp_path) -> None:
    outputs_root = tmp_path / "custom-outputs"

    assert default_report_dir(outputs_root) == outputs_root / "evaluation" / "ds1-ds7-overview"


def test_discover_batch_files_reads_yaml_files_from_outputs_root(tmp_path) -> None:
    outputs_root = tmp_path / "Exp_3_4"
    batch_files_dir = outputs_root / "batch_files"
    batch_files_dir.mkdir(parents=True)
    batch_a = batch_files_dir / "exp_3_unterthiner_ds1-ds4.yaml"
    batch_b = batch_files_dir / "exp_3_unterthiner_ds5-ds7.yml"
    batch_a.write_text("batch_name: Exp_3_4\n", encoding="utf-8")
    batch_b.write_text("batch_name: Exp_3_4\n", encoding="utf-8")
    (outputs_root / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    discovered = discover_batch_files(outputs_root)

    assert discovered == [batch_a.resolve(), batch_b.resolve()]


def test_resolve_batch_files_prefers_discovered_yaml_files(tmp_path) -> None:
    outputs_root = tmp_path / "Exp_3_4"
    batch_files_dir = outputs_root / "batch_files"
    batch_files_dir.mkdir(parents=True)
    batch_path = batch_files_dir / "example.yaml"
    batch_path.write_text("batch_name: Exp_3_4\n", encoding="utf-8")

    assert resolve_batch_files(outputs_root, None) == [batch_path.resolve()]


def test_resolve_outputs_root_redirects_legacy_batch_alias(tmp_path) -> None:
    legacy_root = tmp_path / "Exp_1_2"
    legacy_root.mkdir()
    canonical_root = tmp_path / "batch_experiments" / "Exp_1_2"
    (canonical_root / "batch_files").mkdir(parents=True)
    (canonical_root / "batch_files" / "example.yaml").write_text("batch_name: Exp_1_2\n", encoding="utf-8")

    assert resolve_outputs_root(legacy_root) == canonical_root.resolve()


def test_add_rank_columns_ranks_each_target_metric_separately() -> None:
    ranked = add_rank_columns(
        pd.DataFrame(
            [
                {"experiment_name": "acc_best", "target_metric": "test_acc", "dataset_id": "ds1", "r2": 0.90, "rmse": 0.10, "mae": 0.10},
                {"experiment_name": "acc_worse", "target_metric": "test_acc", "dataset_id": "ds1", "r2": 0.80, "rmse": 0.20, "mae": 0.20},
                {"experiment_name": "loss_best", "target_metric": "test_loss", "dataset_id": "ds1", "r2": 0.20, "rmse": 0.10, "mae": 0.10},
                {"experiment_name": "loss_worse", "target_metric": "test_loss", "dataset_id": "ds1", "r2": 0.10, "rmse": 0.20, "mae": 0.20},
            ]
        )
    )

    global_ranks = ranked.set_index("experiment_name")["r2_rank_global"].to_dict()
    assert global_ranks["acc_best"] == 1.0
    assert global_ranks["loss_best"] == 1.0
    assert global_ranks["acc_worse"] == 2.0
    assert global_ranks["loss_worse"] == 2.0


def test_load_single_run_prefers_metric_from_directory_name(tmp_path) -> None:
    run_dir = (
        tmp_path
        / "2026-03-28"
        / "ds1_ridge_model_only_all_families_test_acc"
        / "12-34-56-000000"
    )
    artifacts_dir = run_dir / "artifacts"
    hydra_dir = run_dir / ".hydra"
    artifacts_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)

    (artifacts_dir / "metrics.json").write_text(json.dumps({"r2": 0.5, "rmse": 0.2, "mae": 0.1}), encoding="utf-8")
    (hydra_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  target_column: test_loss",
                "  parquet_path: /tmp/1_dataset__prepared_dataset.parquet",
                "model:",
                "  name: ridge",
                "features:",
                "  include_model_metrics: true",
                "  include_layer_metrics: false",
            ]
        ),
        encoding="utf-8",
    )

    row = load_single_run(artifacts_dir / "metrics.json")

    assert row["target_metric"] == "test_acc"
    assert row["target_column"] == "test_acc"
    assert row["config_target_column"] == "test_loss"
    assert row["run_group"] == "2026-03-28"


def test_load_single_run_supports_named_batch_directories(tmp_path) -> None:
    run_dir = (
        tmp_path
        / "ds1-ds4"
        / "results"
        / "ds1_ridge_model_only_all_families_test_acc"
        / "12-34-56-000000"
    )
    artifacts_dir = run_dir / "artifacts"
    hydra_dir = run_dir / ".hydra"
    artifacts_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)

    (artifacts_dir / "metrics.json").write_text(json.dumps({"r2": 0.5, "rmse": 0.2, "mae": 0.1}), encoding="utf-8")
    (hydra_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  target_column: test_acc",
                "  parquet_path: /tmp/1_dataset__prepared_dataset.parquet",
                "model:",
                "  name: ridge",
            ]
        ),
        encoding="utf-8",
    )

    row = load_single_run(artifacts_dir / "metrics.json")

    assert row["run_group"] == "ds1-ds4"
    assert row["run_time"] == "12-34-56-000000"
    assert row["run_date"] == row["run_started_at"].date().isoformat()


def test_select_metric_completed_runs_keeps_completed_runs_when_metric_has_no_batch_entries() -> None:
    completed_df = pd.DataFrame(
        [
            {"experiment_name": "ds1_ridge_model_only_all_families_test_loss", "target_metric": "test_loss"},
        ]
    )
    expected_df = pd.DataFrame(columns=["experiment_name", "target_metric"])

    selected = select_metric_completed_runs(completed_df, expected_df)

    assert selected["experiment_name"].tolist() == ["ds1_ridge_model_only_all_families_test_loss"]


def test_build_completion_table_keeps_label_columns_without_expected_rows() -> None:
    completed_df = pd.DataFrame(
        [
            {
                "experiment_name": "ds1_ridge_model_only_all_families_test_loss",
                "dataset_id": "ds1",
                "dataset_number": 1,
                "model": "ridge",
                "model_label": "Ridge",
                "scope": "model_only",
                "scope_label": "Model only",
                "family_combo": "all_families",
                "family_label": "All families",
                "target_metric": "test_loss",
            }
        ]
    )

    completion_df = build_completion_table(pd.DataFrame(), completed_df)

    assert completion_df["model_label"].tolist() == ["Ridge"]
    assert completion_df["scope_label"].tolist() == ["Model only"]
    assert completion_df["family_label"].tolist() == ["All families"]
    assert completion_df["status"].tolist() == ["completed"]


def test_dataset_display_labels_prefers_full_dataset_names() -> None:
    df = pd.DataFrame(
        [
            {
                "dataset_id": "ds2",
                "dataset_number": 2,
                "dataset_label": "2_cifar10_structured_large_full_all_metrics_long",
            },
            {
                "dataset_id": "ds1",
                "dataset_number": 1,
                "dataset_label": "1_cifar10_uniform_large_full_all_metrics_long",
            },
            {
                "dataset_id": "ds1",
                "dataset_number": 1,
                "dataset_label": "1_cifar10_uniform_large_full_all_metrics_long",
            },
        ]
    )

    assert dataset_display_labels(df) == [
        "1_cifar10_uniform_large",
        "2_cifar10_structured_large",
    ]


def test_build_dataset_metadata_uses_display_names_for_directory_names() -> None:
    df = pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "dataset_number": 1,
                "dataset_label": "1_cifar10_uniform_large_full_all_metrics_long",
            },
            {
                "dataset_id": "ds2",
                "dataset_number": 2,
                "dataset_label": "2_cifar10_structured_large_full_all_metrics_long",
            },
        ]
    )

    metadata = build_dataset_metadata(df)

    assert metadata["dataset_display_label"].tolist() == [
        "1_cifar10_uniform_large",
        "2_cifar10_structured_large",
    ]
    assert metadata["dataset_dir_name"].tolist() == [
        "1_cifar10_uniform_large",
        "2_cifar10_structured_large",
    ]


def test_build_dataset_metadata_prefers_extracted_name_over_dataset_id() -> None:
    metadata = build_dataset_metadata(
        pd.DataFrame(
            [
                {"dataset_id": "ds1", "dataset_number": 1},
                {
                    "dataset_id": "ds1",
                    "dataset_number": 1,
                    "dataset_label": "1_cifar10_uniform_large_full_all_metrics_long",
                },
            ]
        )
    )

    assert metadata["dataset_display_label"].tolist() == ["1_cifar10_uniform_large"]


def test_dataset_directory_name_normalizes_invalid_path_separators() -> None:
    assert dataset_directory_name("dataset / split\\name") == "dataset_split_name"


def test_format_dataset_display_label_removes_readability_suffix() -> None:
    assert (
        format_dataset_display_label("1_cifar10_uniform_large_full_all_metrics_long")
        == "1_cifar10_uniform_large"
    )


def test_write_per_dataset_reports_copies_best_artifacts_to_extracted_dataset_dir(tmp_path) -> None:
    worse_artifacts_dir = tmp_path / "runs" / "worse" / "artifacts"
    best_artifacts_dir = tmp_path / "runs" / "best" / "artifacts"
    worse_artifacts_dir.mkdir(parents=True)
    best_artifacts_dir.mkdir(parents=True)
    (worse_artifacts_dir / "winner.txt").write_text("worse", encoding="utf-8")
    (best_artifacts_dir / "winner.txt").write_text("best", encoding="utf-8")

    completed_df = pd.DataFrame(
        [
            {
                "experiment_name": "exp_worse",
                "dataset_id": "ds1",
                "dataset_number": 1,
                "dataset_label": "1_cifar10_uniform_large_full_all_metrics_long",
                "model": "ridge",
                "model_label": "Ridge",
                "scope": "model_only",
                "scope_label": "Model only",
                "family_combo": "all_families",
                "family_label": "All families",
                "r2": 0.40,
                "rmse": 0.20,
                "mae": 0.10,
                "n_features": 8,
                "artifacts_dir": str(worse_artifacts_dir),
            },
            {
                "experiment_name": "exp_best",
                "dataset_id": "ds1",
                "dataset_number": 1,
                "dataset_label": "1_cifar10_uniform_large_full_all_metrics_long",
                "model": "xgboost",
                "model_label": "XGBoost",
                "scope": "model_and_layer",
                "scope_label": "Model + layer",
                "family_combo": "distribution",
                "family_label": "Distribution",
                "r2": 0.85,
                "rmse": 0.10,
                "mae": 0.05,
                "n_features": 12,
                "artifacts_dir": str(best_artifacts_dir),
            },
        ]
    )
    completion_df = completed_df[
        [
            "experiment_name",
            "dataset_id",
            "dataset_number",
            "model",
            "model_label",
            "scope",
            "scope_label",
            "family_combo",
            "family_label",
        ]
    ].copy()
    completion_df["status"] = "completed"

    output_dir = tmp_path / "evaluation" / "test_acc"
    write_per_dataset_reports(output_dir, completed_df, completion_df, top_n=5, target_metric="test_acc")

    dataset_dir = output_dir / "per_dataset" / "1_cifar10_uniform_large"
    assert dataset_dir.is_dir()
    assert not (output_dir / "per_dataset" / "ds1").exists()
    assert (dataset_dir / "artifacts" / "winner.txt").read_text(encoding="utf-8") == "best"


def test_heatmap_text_color_switches_to_black_for_bright_cells() -> None:
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    assert heatmap_text_color(0.95, cmap=matplotlib.cm.viridis, norm=norm) == "black"
    assert heatmap_text_color(0.1, cmap=matplotlib.cm.viridis, norm=norm) == "white"


def test_with_target_metric_appends_metric_to_plot_title() -> None:
    assert with_target_metric("Best available R2 per dataset and model", "test_loss") == (
        "Best available R2 per dataset and model (test_loss)"
    )


def test_with_plot_r2_uses_mathtext() -> None:
    assert with_plot_r2("Best available R2 per dataset and model") == "Best available $R^2$ per dataset and model"


def test_clip_negative_r2_for_plot_sets_negative_values_to_zero() -> None:
    series = pd.Series([-0.4, 0.0, 0.25])

    clipped = clip_negative_r2_for_plot(series)

    assert clipped.tolist() == [0.0, 0.0, 0.25]
    assert clip_negative_r2_for_plot(-0.2) == 0.0
