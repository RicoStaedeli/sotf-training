from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence, cast

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

_ZOO_INFO_PATH = Path(__file__).resolve().parents[3] / "learning-dynamics" / "configs" / "model_zoo_informations.json"


@lru_cache(maxsize=None)
def _load_zoo_plot_name_lookup() -> dict[str, str]:
    try:
        data = json.loads(_ZOO_INFO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        entry["zoo_name"]: entry["plot_name"]
        for entry in data
        if isinstance(entry, dict) and "zoo_name" in entry and "plot_name" in entry
    }


def _map_dataset_index(table: pd.DataFrame) -> pd.DataFrame:
    """Remap a table's index from zoo_name to plot_name."""
    lookup = _load_zoo_plot_name_lookup()
    table.index = pd.Index([lookup.get(str(n), str(n)) for n in table.index], name=table.index.name)
    return table


R2_PLOT_LABEL = r"$R^2$"
RMSE_PLOT_LABEL = "RMSE"
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FULL_TIMESTAMP_FORMATS = ("%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d-%H-%M-%S-%f")
FORECAST_MODE_ORDER = ("final_value", "horizon_value")
MODEL_TYPE_ORDER = ("ridge", "lightgbm", "xgboost", "mlp", "mlp_latent", "unknown")
MODEL_TYPE_COLORS: dict[str, str] = {
    "ridge": "#a8dadc",    # light teal — simplest baseline
    "lightgbm": "#f4a261", # warm orange
    "xgboost": "#1d3557",  # dark navy — strongest performer
    "mlp": "#2a9d8f",      # teal green
    "mlp_latent": "#457b9d",  # mid navy-blue
    "unknown": "#8d99ae",  # slate
}
MAX_PREDICTION_EPOCH = 50
BATCH_RESULTS_SUBDIR = "results"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv (Sequence[str] | None): Sequence[str] | None value used by the function.

    Returns:
        argparse.Namespace: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate metric-forecasting batch experiment results and render R^2/RMSE heatmaps.",
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing forecasting runs, or a batch root containing a results/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the evaluation heatmaps and CSV files will be written. Defaults to <results-dir>/evaluation.",
    )
    parser.add_argument(
        "--extra-results-root",
        dest="extra_results_roots",
        type=Path,
        action="append",
        default=None,
        help="Additional batch/output directories to merge into the same forecasting evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main.

    Args:
        argv (Sequence[str] | None): Sequence[str] | None value used by the function.

    Returns:
        int: Result of the function.
    """
    args = parse_args(argv)
    requested_root = args.results_dir.resolve()
    results_dir = resolve_results_dir(requested_root)
    output_dir = (args.output_dir if args.output_dir is not None else requested_root / "evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_runs = load_forecast_runs(results_dir)
    for extra_root in args.extra_results_roots or []:
        extra_results_dir = resolve_results_dir(extra_root.resolve())
        extra_runs = load_forecast_runs(extra_results_dir)
        if not extra_runs.empty:
            completed_runs = pd.concat([completed_runs, extra_runs], ignore_index=True)
            print(f"Merged {len(extra_runs)} forecasting runs from {extra_results_dir}")
    if completed_runs.empty:
        raise SystemExit(f"No completed forecasting runs with metrics.json found under {results_dir}")

    latest_runs = select_latest_runs(completed_runs)
    latest_runs.to_csv(output_dir / "latest_runs.csv", index=False)
    write_model_type_overview(output_dir, latest_runs)

    target_metrics = sorted(str(value) for value in latest_runs["target_metric"].dropna().unique())
    if not target_metrics:
        target_metrics = ["unknown"]

    multi_metric = len(target_metrics) > 1
    for target_metric in target_metrics:
        metric_runs = latest_runs[latest_runs["target_metric"] == target_metric].copy()
        if metric_runs.empty:
            metric_runs = latest_runs.copy()

        metric_output_dir = output_dir / sanitize_file_name(target_metric) if multi_metric else output_dir
        metric_output_dir.mkdir(parents=True, exist_ok=True)

        overall_table = build_overall_metric_table(metric_runs, metric_name="r2")
        overall_table.to_csv(metric_output_dir / "01_r2_by_dataset_and_mode.csv")
        render_heatmap(
            overall_table,
            output_path=metric_output_dir / "01_r2_by_dataset_and_mode.png",
            title=with_target_metric("Forecast " + R2_PLOT_LABEL + " by dataset and forecasting mode", target_metric),
            x_label="Forecasting mode",
            y_label="Dataset",
            colorbar_label=R2_PLOT_LABEL,
            metric_name="r2",
        )

        final_value_table = build_per_epoch_metric_table(metric_runs, target_mode="final_value", metric_name="r2")
        final_value_table.to_csv(metric_output_dir / "02_final_value_r2_by_dataset_and_epoch.csv")
        render_heatmap(
            final_value_table,
            output_path=metric_output_dir / "02_final_value_r2_by_dataset_and_epoch.png",
            title=with_target_metric("Final-value forecast " + R2_PLOT_LABEL + " by dataset and prediction epoch", target_metric),
            x_label="Prediction epoch",
            y_label="Dataset",
            colorbar_label=R2_PLOT_LABEL,
            metric_name="r2",
            clip_negative=True,
            annotate_values=False,
            compact=True,
            save_dpi=300,
        )

        horizon_value_table = build_per_epoch_metric_table(metric_runs, target_mode="horizon_value", metric_name="r2")
        horizon_value_table.to_csv(metric_output_dir / "03_horizon_value_r2_by_dataset_and_epoch.csv")
        render_heatmap(
            horizon_value_table,
            output_path=metric_output_dir / "03_horizon_value_r2_by_dataset_and_epoch.png",
            title=with_target_metric("Horizon-value forecast " + R2_PLOT_LABEL + " by dataset and prediction epoch", target_metric),
            x_label="Prediction epoch",
            y_label="Dataset",
            colorbar_label=R2_PLOT_LABEL,
            metric_name="r2",
            clip_negative=True,
            annotate_values=False,
            compact=True,
            save_dpi=300,
        )

        overall_rmse_table = build_overall_metric_table(metric_runs, metric_name="rmse")
        overall_rmse_table.to_csv(metric_output_dir / "04_rmse_by_dataset_and_mode.csv")
        render_heatmap(
            overall_rmse_table,
            output_path=metric_output_dir / "04_rmse_by_dataset_and_mode.png",
            title=with_target_metric("Forecast " + RMSE_PLOT_LABEL + " by dataset and forecasting mode", target_metric),
            x_label="Forecasting mode",
            y_label="Dataset",
            colorbar_label=RMSE_PLOT_LABEL,
            metric_name="rmse",
            reverse_scale=True,
        )

        final_value_rmse_table = build_per_epoch_metric_table(metric_runs, target_mode="final_value", metric_name="rmse")
        final_value_rmse_table.to_csv(metric_output_dir / "05_final_value_rmse_by_dataset_and_epoch.csv")
        render_heatmap(
            final_value_rmse_table,
            output_path=metric_output_dir / "05_final_value_rmse_by_dataset_and_epoch.png",
            title=with_target_metric("Final-value forecast " + RMSE_PLOT_LABEL + " by dataset and prediction epoch", target_metric),
            x_label="Prediction epoch",
            y_label="Dataset",
            colorbar_label=RMSE_PLOT_LABEL,
            metric_name="rmse",
            reverse_scale=True,
            annotate_values=False,
            compact=True,
            save_dpi=300,
        )

        horizon_value_rmse_table = build_per_epoch_metric_table(metric_runs, target_mode="horizon_value", metric_name="rmse")
        horizon_value_rmse_table.to_csv(metric_output_dir / "06_horizon_value_rmse_by_dataset_and_epoch.csv")
        render_heatmap(
            horizon_value_rmse_table,
            output_path=metric_output_dir / "06_horizon_value_rmse_by_dataset_and_epoch.png",
            title=with_target_metric("Horizon-value forecast " + RMSE_PLOT_LABEL + " by dataset and prediction epoch", target_metric),
            x_label="Prediction epoch",
            y_label="Dataset",
            colorbar_label=RMSE_PLOT_LABEL,
            metric_name="rmse",
            reverse_scale=True,
            annotate_values=False,
            compact=True,
            save_dpi=300,
        )

        model_type_r2_table = build_model_type_metric_table(metric_runs, metric_name="r2")
        model_type_r2_table.to_csv(metric_output_dir / "07_r2_by_model_type_and_mode.csv")
        render_heatmap(
            model_type_r2_table,
            output_path=metric_output_dir / "07_r2_by_model_type_and_mode.png",
            title=with_target_metric("Forecast " + R2_PLOT_LABEL + " by model type and forecasting mode", target_metric),
            x_label="Forecasting mode",
            y_label="Model type",
            colorbar_label=R2_PLOT_LABEL,
            metric_name="r2",
        )

        model_type_rmse_table = build_model_type_metric_table(metric_runs, metric_name="rmse")
        model_type_rmse_table.to_csv(metric_output_dir / "08_rmse_by_model_type_and_mode.csv")
        render_heatmap(
            model_type_rmse_table,
            output_path=metric_output_dir / "08_rmse_by_model_type_and_mode.png",
            title=with_target_metric("Forecast " + RMSE_PLOT_LABEL + " by model type and forecasting mode", target_metric),
            x_label="Forecasting mode",
            y_label="Model type",
            colorbar_label=RMSE_PLOT_LABEL,
            metric_name="rmse",
            reverse_scale=True,
        )

        render_experiment_performance_bar(
            metric_runs,
            output_path=metric_output_dir / "09_r2_per_experiment.png",
            title=with_target_metric("Forecast " + R2_PLOT_LABEL + " per experiment", target_metric),
            metric_name="r2",
            sort_ascending=True,
        )
        render_experiment_performance_bar(
            metric_runs,
            output_path=metric_output_dir / "10_rmse_per_experiment.png",
            title=with_target_metric("Forecast " + RMSE_PLOT_LABEL + " per experiment", target_metric),
            metric_name="rmse",
            sort_ascending=False,
        )

    print(f"Wrote forecasting evaluation heatmaps to {output_dir}")
    return 0


def resolve_results_dir(path: Path) -> Path:
    """Resolve results dir.

    Args:
        path (Path): Path to process.

    Returns:
        Path: Resolved path.
    """
    resolved = path.resolve()
    batch_results_dir = resolved / BATCH_RESULTS_SUBDIR
    if batch_results_dir.is_dir():
        return batch_results_dir.resolve()
    return resolved


def load_forecast_runs(results_dir: Path) -> pd.DataFrame:
    """Load forecast runs.

    Args:
        results_dir (Path): Path value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    rows: list[dict[str, Any]] = []
    source_results_root = results_dir.resolve()
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        row = load_single_run(metrics_path)
        row["source_results_root"] = str(source_results_root)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_single_run(metrics_path: Path) -> dict[str, Any]:
    """Load single run.

    Args:
        metrics_path (Path): Path value used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    artifacts_dir = metrics_path.parent
    run_dir = artifacts_dir.parent if artifacts_dir.name == "artifacts" else artifacts_dir
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = merge_config_mappings(
        load_hydra_config_if_exists(run_dir / ".hydra" / "config.yaml"),
        load_json_if_exists(artifacts_dir / "resolved_config.json"),
    )
    model_config = load_json_if_exists(artifacts_dir / "resolved_model_config.json")
    wandb_config = nested_get(model_config, "params", "wandb_config", default={})

    experiment_name = first_non_empty(
        nested_get(config, "output", "experiment_name"),
        nested_get(config, "experiment_name"),
        nested_get(wandb_config, "experiment_name"),
        experiment_name_from_run_dir(run_dir),
    )
    dataset_name = dataset_name_from_config(
        config,
        fallback=dataset_name_from_config(wandb_config, fallback=dataset_name_from_experiment_name(experiment_name)),
    )
    dataset_number = dataset_number_from_name(dataset_name)
    target_mode = first_non_empty(
        nested_get(config, "sequence", "target_mode"),
        nested_get(wandb_config, "sequence", "target_mode"),
        metrics.get("target_mode"),
        "unknown",
    )
    target_metric = first_non_empty(
        nested_get(config, "data", "target_column"),
        nested_get(wandb_config, "data", "target_column"),
        metrics.get("target_column"),
        "unknown",
    )
    model_type = first_non_empty(
        nested_get(config, "model", "name"),
        nested_get(model_config, "name"),
        nested_get(wandb_config, "model", "name"),
        model_type_from_experiment_name(experiment_name),
    )
    validation_predictions_path = artifacts_dir / "validation_predictions.parquet"

    return {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "run_dir": str(run_dir),
        "dataset_name": dataset_name,
        "dataset_number": dataset_number,
        "target_mode": target_mode,
        "target_metric": target_metric,
        "horizon": parse_optional_int(metrics.get("horizon", nested_get(config, "sequence", "horizon"))),
        "r2": float(metrics.get("r2", np.nan)),
        "rmse": float(metrics.get("rmse", np.nan)),
        "run_started_at": parse_run_started_at(run_dir, metrics_path),
        "validation_predictions_path": validation_predictions_path if validation_predictions_path.exists() else None,
    }


def select_latest_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Select latest runs.

    Args:
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if runs_df.empty:
        return runs_df.copy()

    subset = ["experiment_name"]
    if "source_results_root" in runs_df.columns:
        subset = ["source_results_root", "experiment_name"]

    latest = (
        runs_df.sort_values(["experiment_name", "run_started_at", "run_dir"])
        .drop_duplicates(subset=subset, keep="last")
        .sort_values(["dataset_number", "dataset_name", "target_mode", "experiment_name"], na_position="last")
        .reset_index(drop=True)
    )
    return latest


def build_overall_metric_table(runs_df: pd.DataFrame, *, metric_name: str) -> pd.DataFrame:
    """Build overall metric table.

    Args:
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.
        metric_name (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if runs_df.empty:
        return pd.DataFrame()

    grouped = (
        runs_df.groupby(["dataset_name", "dataset_number", "target_mode"], dropna=False, as_index=False)[metric_name].mean()
    )
    dataset_order = ordered_dataset_names(grouped)
    mode_order = ordered_modes(grouped["target_mode"].tolist())
    table = grouped.pivot(index="dataset_name", columns="target_mode", values=metric_name)
    table = table.reindex(index=dataset_order, columns=mode_order)
    table.index.name = "dataset"
    table.columns.name = "forecasting_mode"
    return _map_dataset_index(table)


def build_model_type_metric_table(runs_df: pd.DataFrame, *, metric_name: str) -> pd.DataFrame:
    """Build model type metric table.

    Args:
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.
        metric_name (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if runs_df.empty:
        return pd.DataFrame()

    grouped = runs_df.groupby(["model_type", "target_mode"], dropna=False, as_index=False)[metric_name].mean()
    model_type_order = ordered_model_types(grouped["model_type"].tolist())
    mode_order = ordered_modes(grouped["target_mode"].tolist())
    table = grouped.pivot(index="model_type", columns="target_mode", values=metric_name)
    table = table.reindex(index=model_type_order, columns=mode_order)
    table.index.name = "model_type"
    table.columns.name = "forecasting_mode"
    return table


def write_model_type_overview(output_dir: Path, runs_df: pd.DataFrame) -> None:
    """Write model type overview.

    Args:
        output_dir (Path): Output directory for artifacts.
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        None: None.
    """
    best_overview = build_best_experiments_by_model_type(runs_df)
    model_summary = build_model_type_summary(runs_df, best_overview)

    best_overview.to_csv(output_dir / "00_best_experiments_by_model_type.csv", index=False)
    model_summary.to_csv(output_dir / "00_model_type_summary.csv", index=False)
    write_model_type_overview_markdown(output_dir / "00_best_experiments_by_model_type.md", best_overview, model_summary)


def build_best_experiments_by_model_type(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build best experiments by model type.

    Args:
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if runs_df.empty:
        return pd.DataFrame(columns=best_overview_columns())

    sort_columns = ["target_metric", "model_type", "target_mode", "r2", "rmse", "run_started_at", "experiment_name"]
    ascending = [True, True, True, False, True, False, True]
    best = (
        runs_df.sort_values(sort_columns, ascending=ascending, na_position="last")
        .drop_duplicates(subset=["target_metric", "model_type", "target_mode"], keep="first")
        .sort_values(["target_metric", "model_type", "target_mode"])
        .reset_index(drop=True)
    )
    return best[best_overview_columns()]


def build_model_type_summary(runs_df: pd.DataFrame, best_overview: pd.DataFrame) -> pd.DataFrame:
    """Build model type summary.

    Args:
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.
        best_overview (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if runs_df.empty:
        return pd.DataFrame(
            columns=[
                "target_metric",
                "model_type",
                "target_mode",
                "n_experiments",
                "best_experiment",
                "best_r2",
                "best_rmse",
                "mean_r2",
                "mean_rmse",
            ]
        )

    grouped = (
        runs_df.groupby(["target_metric", "model_type", "target_mode"], dropna=False, as_index=False)
        .agg(
            n_experiments=("experiment_name", "nunique"),
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values(["target_metric", "model_type", "target_mode"])
        .reset_index(drop=True)
    )

    best_lookup = best_overview[
        ["target_metric", "model_type", "target_mode", "experiment_name", "r2", "rmse"]
    ].rename(columns={"experiment_name": "best_experiment", "r2": "best_r2", "rmse": "best_rmse"})
    return grouped.merge(best_lookup, on=["target_metric", "model_type", "target_mode"], how="left")[
        [
            "target_metric",
            "model_type",
            "target_mode",
            "n_experiments",
            "best_experiment",
            "best_r2",
            "best_rmse",
            "mean_r2",
            "mean_rmse",
        ]
    ]


def write_model_type_overview_markdown(output_path: Path, best_overview: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    """Write model type overview markdown.

    Args:
        output_path (Path): Path value used by the function.
        best_overview (pd.DataFrame): pd.DataFrame value used by the function.
        model_summary (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        None: None.
    """
    lines = [
        "# Best Forecasting Experiments by Model Type",
        "",
        "Best means highest validation R2, with lower RMSE used as the tie-breaker.",
        "",
    ]

    if best_overview.empty:
        lines.append("No completed runs were available.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Winners",
            "",
            format_table_for_markdown(
                best_overview[
                    [
                        "target_metric",
                        "model_type",
                        "target_mode",
                        "experiment_name",
                        "dataset_name",
                        "horizon",
                        "r2",
                        "rmse",
                    ]
                ]
            ),
            "",
            "## Model Type Summary",
            "",
            format_table_for_markdown(model_summary),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def best_overview_columns() -> list[str]:
    """Best overview columns.

    Returns:
        list[str]: Result of the function.
    """
    return [
        "target_metric",
        "model_type",
        "target_mode",
        "experiment_name",
        "dataset_name",
        "dataset_number",
        "horizon",
        "r2",
        "rmse",
        "run_started_at",
        "run_dir",
    ]


def build_per_epoch_metric_table(runs_df: pd.DataFrame, *, target_mode: str, metric_name: str) -> pd.DataFrame:
    """Build per epoch metric table.

    Args:
        runs_df (pd.DataFrame): pd.DataFrame value used by the function.
        target_mode (str): str value used by the function.
        metric_name (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    mode_runs = runs_df[runs_df["target_mode"] == target_mode].copy()
    if mode_runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for run in mode_runs.itertuples(index=False):
        predictions_path = getattr(run, "validation_predictions_path", None)
        if predictions_path is None:
            continue

        frame = pd.read_parquet(predictions_path)
        required_columns = {"epoch", "target", "prediction"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{predictions_path} is missing required columns: {missing}")

        for epoch, epoch_frame in frame.groupby("epoch", sort=True):
            if float(epoch) > MAX_PREDICTION_EPOCH:
                continue
            rows.append(
                {
                    "dataset_name": run.dataset_name,
                    "dataset_number": run.dataset_number,
                    "prediction_epoch": epoch,
                    metric_name: compute_metric(epoch_frame["target"], epoch_frame["prediction"], metric_name=metric_name),
                }
            )

    if not rows:
        return pd.DataFrame()

    grouped = pd.DataFrame(rows).groupby(
        ["dataset_name", "dataset_number", "prediction_epoch"], dropna=False, as_index=False
    )[metric_name].mean()
    dataset_order = ordered_dataset_names(mode_runs[["dataset_name", "dataset_number"]].drop_duplicates())
    epoch_order = sorted(pd.Index(grouped["prediction_epoch"]).unique().tolist())
    table = grouped.pivot(index="dataset_name", columns="prediction_epoch", values=metric_name)
    table = table.reindex(index=dataset_order, columns=epoch_order)
    table.index.name = "dataset"
    table.columns.name = "prediction_epoch"
    return _map_dataset_index(table)


def compute_metric(y_true: pd.Series, y_pred: pd.Series, *, metric_name: str) -> float:
    """Compute metric.

    Args:
        y_true (pd.Series): pd.Series value used by the function.
        y_pred (pd.Series): pd.Series value used by the function.
        metric_name (str): str value used by the function.

    Returns:
        float: Result of the function.
    """
    y_true_values = y_true.to_numpy(dtype="float64")
    y_pred_values = y_pred.to_numpy(dtype="float64")

    if metric_name == "r2":
        if len(y_true_values) < 2:
            return float("nan")
        score = r2_score(y_true_values, y_pred_values)
        return float(score) if np.isfinite(score) else float("nan")

    if metric_name == "rmse":
        score = float(np.sqrt(mean_squared_error(y_true_values, y_pred_values)))
        return score if np.isfinite(score) else float("nan")

    raise ValueError(f"Unsupported metric: {metric_name}")


def render_heatmap(
    table: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    metric_name: str | None = None,
    clip_negative: bool = False,
    reverse_scale: bool = False,
    annotate_values: bool = True,
    compact: bool = False,
    save_dpi: int = 200,
    save_pdf: bool = False,
) -> None:
    """Render heatmap.

    Args:
        table (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Path value used by the function.
        title (str): str value used by the function.
        x_label (str): str value used by the function.
        y_label (str): str value used by the function.
        colorbar_label (str): str value used by the function.
        metric_name (str | None): str | None value used by the function.
        clip_negative (bool): bool value used by the function.
        reverse_scale (bool): bool value used by the function.
        annotate_values (bool): bool value used by the function.
        compact (bool): bool value used by the function.
        save_dpi (int): int value used by the function.

    Returns:
        None: None.
    """
    if table.empty or not np.isfinite(table.to_numpy(dtype="float64")).any():
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    plot_table = clip_negative_r2_for_plot(table) if clip_negative else table
    values = plot_table.to_numpy(dtype="float64")
    vmin, vmax = heatmap_value_limits(metric_name, clip_negative=clip_negative)

    if compact:
        fig_width = max(10.0, 0.18 * table.shape[1] + 4.0)
        fig_height = max(4.2, 0.5 * table.shape[0] + 1.8)
    else:
        fig_width = max(7.0, 1.2 * table.shape[1] + 4.0)
        fig_height = max(4.0, 0.55 * table.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = heatmap_colormap(reverse_scale=reverse_scale)
    cmap.set_bad("#E6E6E6")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([str(column) for column in table.columns], fontsize=7 if compact else None)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(index) for index in table.index], fontsize=8 if compact else None)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if not compact:
        ax.set_xticks(np.arange(-0.5, table.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, table.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    if annotate_values:
        for row_index in range(table.shape[0]):
            for column_index in range(table.shape[1]):
                value = values[row_index, column_index]
                if not np.isfinite(value):
                    continue
                rgba = image.cmap(image.norm(value))
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=annotation_text_color(rgba),
                    fontsize=9,
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_experiment_performance_bar(
    runs_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    metric_name: str,
    sort_ascending: bool = True,
    save_dpi: int = 200,
) -> None:
    """Render a horizontal bar chart of per-experiment metric scores, one subplot per forecasting mode.

    Args:
        runs_df (pd.DataFrame): Rows from latest_runs, one row per experiment.
        output_path (Path): Destination PNG path.
        title (str): Figure-level title.
        metric_name (str): Column to plot ("r2" or "rmse").
        sort_ascending (bool): Sort direction — True puts best R² at top; False puts best (lowest) RMSE at top.
        save_dpi (int): Output resolution.

    Returns:
        None: None.
    """
    if runs_df.empty:
        return

    target_modes = ordered_modes(runs_df["target_mode"].dropna().unique().tolist())
    n_modes = len(target_modes)
    if n_modes == 0:
        return

    max_n_experiments = max(
        int((runs_df["target_mode"] == mode).sum()) for mode in target_modes
    )
    row_height = 0.32
    fig_height = max(4.0, row_height * max_n_experiments + 2.5)
    subplot_width = 6.5

    fig, axes = plt.subplots(1, n_modes, figsize=(subplot_width * n_modes, fig_height), squeeze=False)

    seen_model_types: set[str] = set()

    for col_idx, mode in enumerate(target_modes):
        ax = axes[0, col_idx]
        mode_df = runs_df[runs_df["target_mode"] == mode].copy()

        if mode_df.empty:
            ax.axis("off")
            ax.set_title(mode)
            continue

        mode_df = mode_df.sort_values(metric_name, ascending=sort_ascending, na_position="last").reset_index(drop=True)
        values = mode_df[metric_name].to_numpy(dtype="float64")
        y_pos = np.arange(len(mode_df))
        bar_colors = [MODEL_TYPE_COLORS.get(str(mt), MODEL_TYPE_COLORS["unknown"]) for mt in mode_df["model_type"]]

        ax.barh(y_pos, values, color=bar_colors, edgecolor="white", linewidth=0.4)

        # xlim must leave enough room for the value labels to the right of each bar
        if metric_name == "r2":
            xlim = (0.0, 1.18)
        else:
            finite_vals = values[np.isfinite(values)]
            upper = float(finite_vals.max()) * 1.4 if len(finite_vals) > 0 else 0.3
            xlim = (0.0, max(upper, 0.01))
        ax.set_xlim(*xlim)

        span = xlim[1] - xlim[0]
        for y, value in zip(y_pos, values):
            if np.isfinite(value):
                ax.text(
                    value + span * 0.012,
                    y,
                    f"{value:.4f}",
                    va="center",
                    fontsize=6,
                    clip_on=False,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(mode_df["experiment_name"].tolist(), fontsize=7)
        metric_label = R2_PLOT_LABEL if metric_name == "r2" else RMSE_PLOT_LABEL
        ax.set_xlabel(metric_label)
        # mode label goes as x-axis title to avoid fighting suptitle at the top
        ax.set_title(mode, fontsize=9, pad=4)
        ax.grid(axis="x", linewidth=0.5, alpha=0.4)

        seen_model_types.update(str(mt) for mt in mode_df["model_type"].dropna())

    legend_handles = [
        mpatches.Patch(color=MODEL_TYPE_COLORS.get(mt, MODEL_TYPE_COLORS["unknown"]), label=mt)
        for mt in MODEL_TYPE_ORDER
        if mt in seen_model_types
    ]

    # suptitle must be set before tight_layout so rect can reserve space for it
    fig.suptitle(title, fontsize=11, y=0.99)
    bottom_rect = 0.06 if legend_handles else 0.02
    fig.tight_layout(rect=[0, bottom_rect, 1, 0.96])

    if legend_handles:
        fig.legend(handles=legend_handles, title="Model type", loc="lower right", fontsize=8,
                   bbox_to_anchor=(1.0, 0.0), bbox_transform=fig.transFigure)

    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)


def heatmap_colormap(*, reverse_scale: bool = False) -> matplotlib.colors.Colormap:
    """Heatmap colormap.

    Args:
        reverse_scale (bool): bool value used by the function.

    Returns:
        matplotlib.colors.Colormap: Result of the function.
    """
    return (plt.cm.YlGnBu_r if reverse_scale else plt.cm.YlGnBu).copy()


def heatmap_value_limits(metric_name: str | None, *, clip_negative: bool = False) -> tuple[float | None, float | None]:
    """Heatmap value limits.

    Args:
        metric_name (str | None): str | None value used by the function.
        clip_negative (bool): bool value used by the function.

    Returns:
        tuple[float | None, float | None]: Result of the function.
    """
    if metric_name == "r2":
        return 0.0, 1.0
    if metric_name == "rmse":
        return 0.0, 0.3
    if clip_negative:
        return 0.0, None
    return None, None


def annotation_text_color(rgba: tuple[float, float, float, float]) -> str:
    """Annotation text color.

    Args:
        rgba (tuple[float, float, float, float]): tuple[float, float, float, float] value used by the function.

    Returns:
        str: Resolved string value.
    """
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance >= 0.6 else "white"


def clip_negative_r2_for_plot(values: Any) -> Any:
    """Clip negative r2 for plot.

    Args:
        values (Any): Values to process.

    Returns:
        Any: Result of the function.
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.clip(lower=0)
    return max(float(values), 0.0)


def parse_run_started_at(run_dir: Path, metrics_path: Path) -> datetime:
    """Parse run started at.

    Args:
        run_dir (Path): Path value used by the function.
        metrics_path (Path): Path value used by the function.

    Returns:
        datetime: Result of the function.
    """
    if run_dir.name == "artifacts":
        run_dir = run_dir.parent

    parent_date_dir = next((parent.name for parent in run_dir.parents if DATE_DIR_RE.match(parent.name)), None)
    if parent_date_dir is not None:
        try:
            return datetime.strptime(f"{parent_date_dir} {run_dir.name}", "%Y-%m-%d %H-%M-%S-%f")
        except ValueError:
            pass

    for timestamp_format in FULL_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(run_dir.name, timestamp_format)
        except ValueError:
            continue

    return datetime.fromtimestamp(metrics_path.stat().st_mtime)


def experiment_name_from_run_dir(run_dir: Path) -> str:
    """Experiment name from run dir.

    Args:
        run_dir (Path): Path value used by the function.

    Returns:
        str: Resolved string value.
    """
    for candidate in (run_dir.name, run_dir.parent.name):
        if is_experiment_name_candidate(candidate):
            return candidate
    return run_dir.name


def is_experiment_name_candidate(value: str) -> bool:
    """Is experiment name candidate.

    Args:
        value (str): Value to process.

    Returns:
        bool: Result of the function.
    """
    normalized = value.strip()
    if not normalized or normalized.lower() in {"artifacts", "evaluation", "results"}:
        return False
    if DATE_DIR_RE.match(normalized):
        return False
    if re.match(r"^\d{2}-\d{2}-\d{2}-\d{6}$", normalized):
        return False
    for timestamp_format in FULL_TIMESTAMP_FORMATS:
        try:
            datetime.strptime(normalized, timestamp_format)
            return False
        except ValueError:
            continue
    return True


def dataset_name_from_config(config: dict[str, Any], *, fallback: str) -> str:
    """Dataset name from config.

    Args:
        config (dict[str, Any]): dict[str, Any] value used by the function.
        fallback (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    dataset_path = _dataset_path_from_config(config)
    if not isinstance(dataset_path, str) or not dataset_path:
        return fallback

    dataset_name = _dataset_name_from_path(dataset_path)
    dataset_name = dataset_name.removesuffix("__prepared_dataset")
    dataset_name = dataset_name.removesuffix("_full_all_metrics_long")
    return dataset_name or fallback


def _dataset_path_from_config(config: dict[str, Any]) -> str | None:
    """Resolve the most informative dataset path from the config."""
    for field_name in ("parquet_path", "pt_train_path", "pt_path", "target_parquet_path"):
        candidate = nested_get(config, "data", field_name)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _dataset_name_from_path(dataset_path: str) -> str:
    """Derive a dataset label from a parquet or PT path."""
    path = Path(dataset_path)
    stem = path.stem
    if stem in {"weight_statistics_train", "weight_statistics_val", "weight_statistics_test"}:
        return path.parent.name
    return stem


def dataset_number_from_name(dataset_name: str) -> float:
    """Dataset number from name.

    Args:
        dataset_name (str): str value used by the function.

    Returns:
        float: Result of the function.
    """
    match = re.match(r"^(?P<number>\d+)_", dataset_name)
    if match is not None:
        return float(match.group("number"))

    match = re.match(r"^ds(?P<number>\d+)\b", dataset_name, re.IGNORECASE)
    if match is not None:
        return float(match.group("number"))

    return float("inf")


def dataset_name_from_experiment_name(experiment_name: str) -> str:
    """Dataset name from experiment name.

    Args:
        experiment_name (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    match = re.match(r"^(ds\d+)(?:\b|_)", experiment_name, re.IGNORECASE)
    if match is not None:
        return match.group(1)
    return experiment_name


def model_type_from_experiment_name(experiment_name: str) -> str:
    """Model type from experiment name.

    Args:
        experiment_name (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    normalized = experiment_name.lower()
    known_models = {
        "mlp_latent": "mlp_latent",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "ridge": "ridge",
        "mlp": "mlp",
    }
    for token, model_type in known_models.items():
        if re.search(rf"(^|_){re.escape(token)}($|_)", normalized):
            return model_type
    return "unknown"


def ordered_dataset_names(df: pd.DataFrame) -> list[str]:
    """Ordered dataset names.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    dataset_metadata = (
        df[["dataset_name", "dataset_number"]]
        .drop_duplicates()
        .sort_values(["dataset_number", "dataset_name"], na_position="last")
    )
    return dataset_metadata["dataset_name"].tolist()


def ordered_modes(modes: Sequence[str]) -> list[str]:
    """Ordered modes.

    Args:
        modes (Sequence[str]): Sequence[str] value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    unique_modes = {str(mode) for mode in modes}
    ordered = [mode for mode in FORECAST_MODE_ORDER if mode in unique_modes]
    ordered.extend(sorted(unique_modes.difference(ordered)))
    return ordered


def ordered_model_types(model_types: Sequence[str]) -> list[str]:
    """Ordered model types.

    Args:
        model_types (Sequence[str]): Sequence[str] value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    unique_model_types = {str(model_type) for model_type in model_types}
    ordered = [model_type for model_type in MODEL_TYPE_ORDER if model_type in unique_model_types]
    ordered.extend(sorted(unique_model_types.difference(ordered)))
    return ordered


def nested_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested get.

    Args:
        mapping (dict[str, Any]): dict[str, Any] value used by the function.
        *keys (str): str value used by the function.
        default (Any): Value used by the function.

    Returns:
        Any: Result of the function.
    """
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load json if exists.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any]: Result of the function.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_hydra_config_if_exists(path: Path) -> dict[str, Any]:
    """Load hydra config if exists.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any]: Result of the function.
    """
    if not path.exists():
        return {}
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else {}


def merge_config_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge config mappings.

    Args:
        base (dict[str, Any]): dict[str, Any] value used by the function.
        override (dict[str, Any]): dict[str, Any] value used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_config_mappings(current, value)
        else:
            merged[key] = value
    return merged


def first_non_empty(*values: Any) -> str:
    """First non empty.

    Args:
        *values (Any): Values to process.

    Returns:
        str: Resolved string value.
    """
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def parse_optional_int(value: Any) -> int | None:
    """Parse optional int.

    Args:
        value (Any): Value to process.

    Returns:
        int | None: Result of the function.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_table_for_markdown(df: pd.DataFrame) -> str:
    """Format table for markdown.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        str: Resolved string value.
    """
    if df.empty:
        return "(no rows)"

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(format_float_for_markdown)
        else:
            display_df[column] = display_df[column].map(format_value_for_markdown)

    header = "| " + " | ".join(str(column) for column in display_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display_df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def format_float_for_markdown(value: Any) -> str:
    """Format float for markdown.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def format_value_for_markdown(value: Any) -> str:
    """Format value for markdown.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def sanitize_file_name(value: str) -> str:
    """Sanitize file name.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "unknown"


def with_target_metric(title: str, target_metric: str) -> str:
    """With target metric.

    Args:
        title (str): str value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    return f"{title} ({target_metric})"


if __name__ == "__main__":
    raise SystemExit(main())
