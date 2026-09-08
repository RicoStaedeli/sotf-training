from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from .shared import (
    compute_binary_auc,
    compute_prediction_metric,
    dataset_number_from_sources,
    dataset_slug_from_parquet,
    float_or_nan,
    goal_adjusted_scores,
    nested_get,
    parse_run_started_at,
    pretty_dataset_name,
    rank_to_percentile,
    regret_against_oracle,
    resolve_baseline_and_comparison_predictors,
    resolve_model_zoo_metadata,
    safe_ratio,
    weighted_average,
)

def load_completed_runs(runs_root: Path) -> pd.DataFrame:
    """Load completed runs.

    Args:
        runs_root (Path): Path value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.rglob("simulation_summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "resolved_config.json"
        manifest_path = run_dir / "run_manifest.json"
        round_details_path = run_dir / "round_details.parquet"
        final_survivors_path = run_dir / "final_survivors.parquet"
        if not config_path.exists() or not round_details_path.exists() or not final_survivors_path.exists():
            continue
        rows.append(load_single_run(run_dir, manifest_path=manifest_path))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def resolve_results_root(path: Path) -> Path:
    """Resolve results root.

    Args:
        path (Path): Path to the input resource.

    Returns:
        Path: Resolved path.
    """
    path = path.resolve()
    results_dir = path / "results"
    if results_dir.is_dir():
        return results_dir
    return path


def normalize_predictor_name(value: Any) -> str:
    """Normalize predictor name.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    normalized = str(value).strip().lower()
    alias_map = {
        "forecast": "forecast_model",
        "forecast-model": "forecast_model",
        "forecasting_model": "forecast_model",
        "online-lstm": "online_lstm",
        "online_latent_window_lstm": "online_lstm",
        "online-latent-window-lstm": "online_lstm",
        "baseline": "naive",
    }
    return alias_map.get(normalized, normalized)


def parse_experiment_path_specs(values: Sequence[str] | None) -> list[tuple[str, Path]]:
    """Parse experiment path specs.

    Args:
        values (Sequence[str] | None): Values to process.

    Returns:
        list[tuple[str, Path]]: Computed experiment path specs.
    """
    if not values:
        return []
    parsed: list[tuple[str, Path]] = []
    seen_predictors: set[str] = set()
    for raw_value in values:
        predictor_part, separator, path_part = str(raw_value).partition("=")
        predictor = normalize_predictor_name(predictor_part)
        path_text = path_part.strip()
        if separator != "=" or not predictor or not path_text:
            raise SystemExit(
                f"Invalid --experiment-path value `{raw_value}`. Expected format: predictor=/path/to/experiment_or_results"
            )
        if predictor in seen_predictors:
            raise SystemExit(f"Duplicate predictor `{predictor}` in --experiment-path entries.")
        parsed.append((predictor, Path(path_text).resolve()))
        seen_predictors.add(predictor)
    return parsed


def load_runs_from_explicit_paths(experiment_paths: Sequence[tuple[str, Path]]) -> pd.DataFrame:
    """Load runs from explicit paths.

    Args:
        experiment_paths (Sequence[tuple[str, Path]]): Sequence[tuple[str, Path]] value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    frames: list[pd.DataFrame] = []
    for predictor_name, experiment_path in experiment_paths:
        runs = load_completed_runs(resolve_results_root(experiment_path))
        if runs.empty:
            raise SystemExit(f"No completed SOTF runs found under {experiment_path}")
        runs = runs.copy()
        runs["predictor"] = runs["predictor"].map(normalize_predictor_name)
        predictor_runs = runs[runs["predictor"] == predictor_name].copy()
        if predictor_runs.empty:
            available_predictors = sorted(set(runs["predictor"].dropna().astype(str)))
            available_text = ", ".join(available_predictors) if available_predictors else "none"
            raise SystemExit(
                f"No runs with predictor `{predictor_name}` found under {experiment_path}. Available predictors: {available_text}"
            )
        frames.append(predictor_runs)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_single_run(run_dir: Path, *, manifest_path: Path) -> dict[str, Any]:
    """Load single run.

    Args:
        run_dir (Path): Directory for the run.
        manifest_path (Path): Path to the manifest.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    summary = json.loads((run_dir / "simulation_summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    experiment_name = str(
        manifest.get(
            "experiment_name",
            nested_get(
                config,
                "requested",
                "output",
                "experiment_name",
                default=nested_get(config, "effective", "output", "experiment_name", default=run_dir.parent.name),
            ),
        )
    )
    parquet_path = Path(
        str(
            manifest.get(
                "dataset_path",
                nested_get(
                    config,
                    "requested",
                    "data",
                    "parquet_path",
                    default=nested_get(
                        config,
                        "forecast_model_bundle_config",
                        "data",
                        "parquet_path",
                        default="",
                    ),
                ),
            )
        )
    )
    dataset_number = manifest.get("dataset_id", dataset_number_from_sources(experiment_name=experiment_name, parquet_path=parquet_path))
    dataset_slug = str(manifest.get("dataset_name", dataset_slug_from_parquet(parquet_path, fallback=experiment_name)))
    _zoo_meta = resolve_model_zoo_metadata(
        dataset_number=dataset_number,
        dataset_slug=dataset_slug,
        dataset_label=pretty_dataset_name(dataset_slug),
    )
    dataset_label = (
        (_zoo_meta.get("plot_name", "").strip() or pretty_dataset_name(dataset_slug))
        if _zoo_meta
        else pretty_dataset_name(dataset_slug)
    )
    predictor = normalize_predictor_name(
        manifest.get("predictor", summary.get("predictor", nested_get(config, "effective", "predictor", default="unknown")))
    )
    target_column = str(
        manifest.get("target_column", summary.get("target_column", nested_get(config, "requested", "data", "target_column", default="unknown")))
    )
    target_mode = str(
        manifest.get("target_mode", nested_get(config, "effective", "sequence", "target_mode", default=nested_get(config, "requested", "sequence", "target_mode", default="unknown")))
    )
    horizon = int(
        manifest.get("horizon", nested_get(config, "effective", "sequence", "horizon", default=nested_get(config, "requested", "sequence", "horizon", default=-1)))
    )
    selection_start_epoch = int(
        nested_get(manifest, "selection", "start_epoch", default=nested_get(config, "requested", "selection", "start_epoch", default=-1))
    )
    selection_interval = int(
        nested_get(manifest, "selection", "interval", default=nested_get(config, "requested", "selection", "interval", default=-1))
    )
    selection_survival_rate = float(
        nested_get(
            manifest,
            "selection",
            "survival_rate",
            default=nested_get(config, "requested", "selection", "survival_rate", default=np.nan),
        )
    )
    metric_goal = str(
        nested_get(manifest, "selection", "metric_goal", default=nested_get(config, "requested", "selection", "metric_goal", default="maximize"))
    ).lower()
    epoch_from_checkpoint_path = bool(nested_get(config, "requested", "data", "epoch_from_checkpoint_path", default=False))
    raw_checkpoint_stride = int(summary.get("checkpoint_stride", 1))
    # When epoch_from_checkpoint_path=True, epochs are already real epoch numbers so no stride
    # scaling is needed, even if an older run stored checkpoint_stride > 1 in its summary.
    checkpoint_stride = 1 if epoch_from_checkpoint_path else raw_checkpoint_stride
    data_source = str(nested_get(config, "requested", "data", "source", default="prepared_dataset"))

    return {
        "experiment_name": experiment_name,
        "predictor": predictor,
        "dataset_number": -1 if dataset_number is None else int(dataset_number),
        "dataset_label": dataset_label,
        "dataset_slug": dataset_slug,
        "dataset_parquet_path": str(parquet_path),
        "data_source": data_source,
        "run_dir": str(run_dir),
        "run_started_at": parse_run_started_at(run_dir),
        "run_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "summary_path": str(run_dir / "simulation_summary.json"),
        "config_path": str(run_dir / "resolved_config.json"),
        "round_details_path": str(run_dir / "round_details.parquet"),
        "final_survivors_path": str(run_dir / "final_survivors.parquet"),
        "target_column": target_column,
        "target_mode": target_mode,
        "horizon": horizon,
        "selection_start_epoch": selection_start_epoch,
        "selection_interval": selection_interval,
        "selection_survival_rate": selection_survival_rate,
        "metric_goal": metric_goal,
        "checkpoint_stride": checkpoint_stride,
        "initial_population": int(summary.get("initial_population", 0)),
        "final_population": int(summary.get("final_population", 0)),
        "rounds_executed": int(summary.get("rounds_executed", 0)),
        "retained_oracle_best": bool(summary.get("retained_oracle_best", False)),
        "oracle_top_k_recall": float_or_nan(summary.get("oracle_top_k_recall")),
        "mean_survivor_rank_gap": float_or_nan(summary.get("mean_survivor_rank_gap")),
        "mean_absolute_prediction_error": float_or_nan(summary.get("mean_absolute_prediction_error")),
        "oracle_best_final_target": float_or_nan(summary.get("oracle_best_final_target")),
        "best_survivor_final_target": float_or_nan(summary.get("best_survivor_final_target")),
    }


def select_latest_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Select latest runs.

    Args:
        runs_df (pd.DataFrame): Data frame containing runs values.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if runs_df.empty:
        return runs_df.copy()

    key_columns = [
        "experiment_name",
        "predictor",
        "dataset_number",
        "dataset_slug",
        "target_column",
        "target_mode",
        "horizon",
        "selection_start_epoch",
        "selection_interval",
        "selection_survival_rate",
    ]
    latest = (
        runs_df.sort_values(key_columns + ["run_started_at", "run_dir"])
        .groupby(key_columns, as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def filter_runs(
    runs_df: pd.DataFrame,
    *,
    predictors: Sequence[str] | None,
    target_column: str | None,
    target_mode: str | None,
    horizon: int | None,
    start_epoch: int | None,
    interval: int | None,
    survival_rate: float | None,
) -> pd.DataFrame:
    """Filter runs.

    Args:
        runs_df (pd.DataFrame): Data frame containing runs values.
        predictors (Sequence[str] | None): Sequence[str] | None value used by the function.
        target_column (str | None): str | None value used by the function.
        target_mode (str | None): str | None value used by the function.
        horizon (int | None): int | None value used by the function.
        start_epoch (int | None): int | None value used by the function.
        interval (int | None): int | None value used by the function.
        survival_rate (float | None): float | None value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    filtered = runs_df.copy()
    if "predictor" in filtered.columns:
        filtered["predictor"] = filtered["predictor"].map(normalize_predictor_name)
    if predictors:
        allowed = {normalize_predictor_name(value) for value in predictors}
        filtered = filtered[filtered["predictor"].isin(allowed)]
    if target_column is not None:
        filtered = filtered[filtered["target_column"] == str(target_column)]
    if target_mode is not None:
        filtered = filtered[filtered["target_mode"] == str(target_mode)]
    if horizon is not None:
        filtered = filtered[filtered["horizon"] == int(horizon)]
    if start_epoch is not None:
        filtered = filtered[filtered["selection_start_epoch"] == int(start_epoch)]
    if interval is not None:
        filtered = filtered[filtered["selection_interval"] == int(interval)]
    if survival_rate is not None:
        filtered = filtered[np.isclose(filtered["selection_survival_rate"].to_numpy(dtype="float64"), float(survival_rate), equal_nan=False)]
    return filtered.reset_index(drop=True)


def evaluate_runs(latest_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate runs.

    Args:
        latest_runs (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Computed runs.
    """
    round_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for run in latest_runs.to_dict("records"):
        round_details = pd.read_parquet(Path(run["round_details_path"]))
        final_survivors = pd.read_parquet(Path(run["final_survivors_path"]))

        max_epoch_val = (
            float(pd.to_numeric(round_details["max_epoch"], errors="coerce").dropna().max())
            if "max_epoch" in round_details.columns
            else np.nan
        )

        checkpoint_stride = _resolve_checkpoint_stride(run, max_epoch_val)
        real_max_epoch = max_epoch_val * checkpoint_stride if not np.isnan(max_epoch_val) else np.nan

        dataset_round_metrics = compute_round_metrics(run=run, round_details=round_details)
        round_rows.extend(dataset_round_metrics.to_dict("records"))
        dataset_rows.append(compute_dataset_summary(run=run, round_metrics=dataset_round_metrics, final_survivors=final_survivors, max_epoch=real_max_epoch))

    round_metrics = pd.DataFrame(round_rows)
    dataset_summary = pd.DataFrame(dataset_rows)

    if not round_metrics.empty:
        round_metrics = round_metrics.sort_values(["predictor", "dataset_number", "round_index"]).reset_index(drop=True)
    if not dataset_summary.empty:
        dataset_summary = dataset_summary.sort_values(["predictor", "dataset_number", "experiment_name"]).reset_index(drop=True)

    return round_metrics, dataset_summary


def build_roc_curve_points(
    latest_runs: pd.DataFrame,
    *,
    oracle_label_source: str = "horizon",
) -> pd.DataFrame:
    """Build roc curve points.

    Args:
        latest_runs (pd.DataFrame): pd.DataFrame value used by the function.
        oracle_label_source (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    rows: list[pd.DataFrame] = []
    for run in latest_runs.to_dict("records"):
        round_details = pd.read_parquet(Path(run["round_details_path"]))
        curve = compute_run_roc_curve(
            run=run,
            round_details=round_details,
            oracle_label_source=oracle_label_source,
        )
        if not curve.empty:
            rows.append(curve)

    if not rows:
        return empty_roc_curve_points_frame()

    return pd.concat(rows, ignore_index=True, sort=False).sort_values(
        ["predictor", "dataset_number", "fpr", "tpr"]
    ).reset_index(drop=True)


def empty_roc_curve_points_frame() -> pd.DataFrame:
    """Create roc curve points frame.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    return pd.DataFrame(
        columns=[
            "experiment_name",
            "predictor",
            "dataset_number",
            "dataset_label",
            "dataset_slug",
            "target_column",
            "target_mode",
            "horizon",
            "selection_start_epoch",
            "selection_interval",
            "selection_survival_rate",
            "positive_count",
            "negative_count",
            "roc_auc",
            "threshold",
            "fpr",
            "tpr",
        ]
    )


def compute_run_roc_curve(
    *,
    run: dict[str, Any],
    round_details: pd.DataFrame,
    oracle_label_source: str = "horizon",
) -> pd.DataFrame:
    """Compute run roc curve.

    Args:
        run (dict[str, Any]): Run metadata used by the function.
        round_details (pd.DataFrame): pd.DataFrame value used by the function.
        oracle_label_source (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    required_columns = {"predicted_future_target"}
    if round_details.empty or not required_columns.issubset(round_details.columns):
        return pd.DataFrame()

    if oracle_label_source == "horizon":
        if "oracle_selected" not in round_details.columns:
            return pd.DataFrame()
        labels = round_details["oracle_selected"]
    elif oracle_label_source == "final_epoch":
        labels = compute_final_epoch_oracle_labels(
            round_details,
            metric_goal=run["metric_goal"],
        )
        if labels.empty:
            return pd.DataFrame()
    else:
        raise ValueError(f"Unsupported oracle_label_source: {oracle_label_source}")

    scores = goal_adjusted_scores(round_details["predicted_future_target"], metric_goal=run["metric_goal"])
    valid = ~(labels.isna() | scores.isna())
    if oracle_label_source == "horizon" and "actual_future_target" in round_details.columns:
        valid &= round_details["actual_future_target"].notna()
    if oracle_label_source == "final_epoch":
        valid &= round_details["final_target"].notna()
    labels_array = labels.loc[valid].astype(bool).astype(int).to_numpy(dtype="int64")
    scores_array = scores.loc[valid].to_numpy(dtype="float64")
    if len(labels_array) == 0 or len(np.unique(labels_array)) < 2:
        return pd.DataFrame()

    fpr, tpr, thresholds = roc_curve(labels_array, scores_array)
    roc_auc = float(auc(fpr, tpr))
    positive_count = int(labels_array.sum())
    negative_count = int(len(labels_array) - positive_count)

    return pd.DataFrame(
        {
            "experiment_name": run["experiment_name"],
            "predictor": run["predictor"],
            "dataset_number": run["dataset_number"],
            "dataset_label": run["dataset_label"],
            "dataset_slug": run["dataset_slug"],
            "target_column": run["target_column"],
            "target_mode": run["target_mode"],
            "horizon": run["horizon"],
            "selection_start_epoch": run["selection_start_epoch"],
            "selection_interval": run["selection_interval"],
            "selection_survival_rate": run["selection_survival_rate"],
            "positive_count": positive_count,
            "negative_count": negative_count,
            "roc_auc": roc_auc,
            "threshold": thresholds,
            "fpr": fpr,
            "tpr": tpr,
        }
    )


def compute_final_epoch_oracle_labels(round_details: pd.DataFrame, *, metric_goal: str) -> pd.Series:
    """Compute final epoch oracle labels.

    Args:
        round_details (pd.DataFrame): pd.DataFrame value used by the function.
        metric_goal (str): str value used by the function.

    Returns:
        pd.Series: Computed series.
    """
    if "final_target" not in round_details.columns:
        return pd.Series(dtype="boolean")

    labels = pd.Series(False, index=round_details.index, dtype="boolean")
    round_group_key = "round_index" if "round_index" in round_details.columns else None
    groups = (
        round_details.groupby(round_group_key, sort=True, dropna=False)
        if round_group_key is not None
        else [(0, round_details)]
    )

    for _, group in groups:
        final_target = pd.to_numeric(group["final_target"], errors="coerce")
        predicted_target = pd.to_numeric(group["predicted_future_target"], errors="coerce")
        valid_group = group.loc[~(final_target.isna() | predicted_target.isna())].copy()
        if valid_group.empty:
            continue

        keep_count = infer_round_keep_count(valid_group)
        if keep_count <= 0:
            continue
        keep_count = min(keep_count, len(valid_group))
        ranked = sort_by_metric_goal(valid_group, value_column="final_target", metric_goal=metric_goal)
        labels.loc[ranked.head(keep_count).index] = True

    return labels


def infer_round_keep_count(group: pd.DataFrame) -> int:
    """Infer round keep count.

    Args:
        group (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        int: Process exit code.
    """
    if "selected" in group.columns:
        selected = pd.Series(group["selected"], index=group.index).fillna(False).astype(bool)
        return int(selected.sum())
    if "oracle_selected" in group.columns:
        oracle_selected = pd.Series(group["oracle_selected"], index=group.index).fillna(False).astype(bool)
        return int(oracle_selected.sum())
    return max(1, int(math.ceil(len(group) / 2.0)))


def sort_by_metric_goal(frame: pd.DataFrame, *, value_column: str, metric_goal: str) -> pd.DataFrame:
    """Sort by metric goal.

    Args:
        frame (pd.DataFrame): Data frame to process.
        value_column (str): str value used by the function.
        metric_goal (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    ascending = str(metric_goal).lower() == "minimize"
    sort_columns = [value_column]
    ascending_values = [ascending]
    if "model_id" in frame.columns:
        sort_columns.append("model_id")
        ascending_values.append(True)
    return frame.sort_values(
        by=sort_columns,
        ascending=ascending_values,
        kind="stable",
    )


def compute_round_metrics(*, run: dict[str, Any], round_details: pd.DataFrame) -> pd.DataFrame:
    """Compute round metrics.

    Args:
        run (dict[str, Any]): Run metadata used by the function.
        round_details (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    scored_rows = round_details.dropna(subset=["actual_future_target"]).copy()
    if scored_rows.empty:
        return pd.DataFrame()

    scored_rows["selected"] = scored_rows["selected"].astype(bool)
    scored_rows["oracle_selected"] = scored_rows["oracle_selected"].astype(bool)
    scored_rows["oracle_best_model"] = scored_rows["oracle_best_model"].astype(bool)

    rows: list[dict[str, Any]] = []
    for round_index, group in scored_rows.groupby("round_index", sort=True):
        group = group.copy()
        policy_kill = ~group["selected"]
        oracle_kill = ~group["oracle_selected"]
        oracle_keep = group["oracle_selected"]
        decision_match = group["selected"] == group["oracle_selected"]

        true_positive_kills = int((policy_kill & oracle_kill).sum())
        false_positive_kills = int((policy_kill & oracle_keep).sum())
        false_negative_kills = int((group["selected"] & oracle_kill).sum())
        correctly_kept = int((group["selected"] & oracle_keep).sum())
        candidate_count = int(len(group))
        policy_kill_count = int(policy_kill.sum())
        oracle_kill_count = int(oracle_kill.sum())
        oracle_keep_count = int(oracle_keep.sum())

        predicted_rank = pd.to_numeric(group["predicted_rank"], errors="coerce")
        actual_rank = pd.to_numeric(group["actual_future_rank"], errors="coerce")
        rank_alignment = predicted_rank.corr(actual_rank, method="spearman")
        predicted_target = pd.to_numeric(group["predicted_future_target"], errors="coerce")
        actual_target = pd.to_numeric(group["actual_future_target"], errors="coerce")
        valid_prediction_mask = ~(predicted_target.isna() | actual_target.isna())
        prediction_count = int(valid_prediction_mask.sum())
        prediction_r2 = compute_prediction_metric(
            actual_target.loc[valid_prediction_mask],
            predicted_target.loc[valid_prediction_mask],
            metric_name="r2",
        )
        prediction_rmse = compute_prediction_metric(
            actual_target.loc[valid_prediction_mask],
            predicted_target.loc[valid_prediction_mask],
            metric_name="rmse",
        )
        roc_auc = compute_binary_auc(
            labels=group["oracle_selected"],
            scores=goal_adjusted_scores(predicted_target, metric_goal=run["metric_goal"]),
        )

        rows.append(
            {
                "experiment_name": run["experiment_name"],
                "predictor": run["predictor"],
                "dataset_number": run["dataset_number"],
                "dataset_label": run["dataset_label"],
                "dataset_slug": run["dataset_slug"],
                "target_column": run["target_column"],
                "target_mode": run["target_mode"],
                "horizon": run["horizon"],
                "selection_start_epoch": run["selection_start_epoch"],
                "selection_interval": run["selection_interval"],
                "selection_survival_rate": run["selection_survival_rate"],
                "round_index": int(round_index),
                "decision_epoch": float(group["decision_epoch"].iloc[0]),
                "candidate_count": candidate_count,
                "policy_keep_count": int(group["selected"].sum()),
                "policy_kill_count": policy_kill_count,
                "oracle_keep_count": oracle_keep_count,
                "oracle_kill_count": oracle_kill_count,
                "decision_accuracy": float(decision_match.mean()),
                "kill_precision": safe_ratio(true_positive_kills, policy_kill_count),
                "kill_recall": safe_ratio(true_positive_kills, oracle_kill_count),
                "keep_precision": safe_ratio(correctly_kept, int(group["selected"].sum())),
                "keep_recall": safe_ratio(correctly_kept, oracle_keep_count),
                "false_kill_rate": safe_ratio(false_positive_kills, oracle_keep_count),
                "false_keep_rate": safe_ratio(false_negative_kills, oracle_kill_count),
                "roc_auc": roc_auc,
                "prediction_count": prediction_count,
                "prediction_r2": prediction_r2,
                "prediction_rmse": prediction_rmse,
                "killed_round_oracle_best": bool((policy_kill & group["oracle_best_model"]).any()),
                "rank_alignment": float(rank_alignment) if pd.notna(rank_alignment) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_dataset_summary(
    *,
    run: dict[str, Any],
    round_metrics: pd.DataFrame,
    final_survivors: pd.DataFrame,
    max_epoch: float = np.nan,
) -> dict[str, Any]:
    """Compute dataset summary.

    Args:
        run (dict[str, Any]): Run metadata used by the function.
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        final_survivors (pd.DataFrame): pd.DataFrame value used by the function.
        max_epoch (float): Maximum training epoch for the dataset (used to compute compute savings).

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    best_survivor_rank = np.nan
    median_survivor_rank = np.nan
    best_survivor_rank_percentile = np.nan
    median_survivor_rank_percentile = np.nan

    if not final_survivors.empty and "full_population_final_rank" in final_survivors.columns:
        ranks = pd.to_numeric(final_survivors["full_population_final_rank"], errors="coerce").dropna()
        if not ranks.empty:
            best_survivor_rank = float(ranks.min())
            median_survivor_rank = float(ranks.median())
            best_survivor_rank_percentile = rank_to_percentile(best_survivor_rank, run["initial_population"])
            median_survivor_rank_percentile = rank_to_percentile(median_survivor_rank, run["initial_population"])

    if round_metrics.empty:
        decision_accuracy_weighted = np.nan
        kill_precision_weighted = np.nan
        kill_recall_weighted = np.nan
        keep_precision_weighted = np.nan
        keep_recall_weighted = np.nan
        false_kill_rate_weighted = np.nan
        false_keep_rate_weighted = np.nan
        roc_auc_weighted = np.nan
        prediction_r2_weighted = np.nan
        prediction_rmse_weighted = np.nan
        rank_alignment_mean = np.nan
        ever_killed_round_oracle_best = False
    else:
        weights = pd.to_numeric(round_metrics["candidate_count"], errors="coerce").fillna(0.0)
        prediction_weights = pd.to_numeric(round_metrics["prediction_count"], errors="coerce").fillna(0.0)
        decision_accuracy_weighted = weighted_average(round_metrics["decision_accuracy"], weights)
        kill_precision_weighted = weighted_average(round_metrics["kill_precision"], round_metrics["policy_kill_count"])
        kill_recall_weighted = weighted_average(round_metrics["kill_recall"], round_metrics["oracle_kill_count"])
        keep_precision_weighted = weighted_average(round_metrics["keep_precision"], round_metrics["policy_keep_count"])
        keep_recall_weighted = weighted_average(round_metrics["keep_recall"], round_metrics["oracle_keep_count"])
        false_kill_rate_weighted = weighted_average(round_metrics["false_kill_rate"], round_metrics["oracle_keep_count"])
        false_keep_rate_weighted = weighted_average(round_metrics["false_keep_rate"], round_metrics["oracle_kill_count"])
        roc_auc_weighted = weighted_average(round_metrics["roc_auc"], weights)
        prediction_r2_weighted = weighted_average(round_metrics["prediction_r2"], prediction_weights)
        prediction_rmse_weighted = weighted_average(round_metrics["prediction_rmse"], prediction_weights)
        rank_alignment_mean = weighted_average(round_metrics["rank_alignment"], weights)
        ever_killed_round_oracle_best = bool(round_metrics["killed_round_oracle_best"].fillna(False).any())

    best_survivor_regret = regret_against_oracle(
        metric_goal=run["metric_goal"],
        oracle_value=run["oracle_best_final_target"],
        survivor_value=run["best_survivor_final_target"],
    )

    # Use the actual first decision epoch, not the configured start_epoch. Predictors that
    # require a warm-up window (e.g. min_history > 1) skip early scheduled rounds, so the
    # configured start_epoch overstates compute savings.
    if not round_metrics.empty and "decision_epoch" in round_metrics.columns:
        effective_start_epoch = float(pd.to_numeric(round_metrics["decision_epoch"], errors="coerce").dropna().min())
    else:
        effective_start_epoch = float(run["selection_start_epoch"]) if run["selection_start_epoch"] != -1 else np.nan
    interval = float(run["selection_interval"]) if run["selection_interval"] != -1 else np.nan
    compute_savings_fraction = _compute_savings_fraction(
        initial_population=run["initial_population"],
        max_epoch=max_epoch,
        start_epoch=effective_start_epoch,
        interval=interval,
        rounds_executed=run["rounds_executed"],
        survival_rate=run["selection_survival_rate"],
    )

    return {
        **run,
        "max_epoch": max_epoch,
        "compute_savings_fraction": compute_savings_fraction,
        "decision_accuracy_weighted": decision_accuracy_weighted,
        "kill_precision_weighted": kill_precision_weighted,
        "kill_recall_weighted": kill_recall_weighted,
        "keep_precision_weighted": keep_precision_weighted,
        "keep_recall_weighted": keep_recall_weighted,
        "false_kill_rate_weighted": false_kill_rate_weighted,
        "false_keep_rate_weighted": false_keep_rate_weighted,
        "roc_auc_weighted": roc_auc_weighted,
        "prediction_r2_weighted": prediction_r2_weighted,
        "prediction_rmse_weighted": prediction_rmse_weighted,
        "rank_alignment_mean": rank_alignment_mean,
        "ever_killed_round_oracle_best": ever_killed_round_oracle_best,
        "best_survivor_regret": best_survivor_regret,
        "best_survivor_full_rank": best_survivor_rank,
        "median_survivor_full_rank": median_survivor_rank,
        "best_survivor_rank_percentile": best_survivor_rank_percentile,
        "median_survivor_rank_percentile": median_survivor_rank_percentile,
    }


def _resolve_checkpoint_stride(run: dict[str, Any], checkpoint_max_epoch: float) -> float:
    """Return the number of real training epochs per checkpoint index for this run.

    For latent-forecast experiments the PT file stores sequential checkpoint indices (0, 1, …)
    where each index represents multiple real training epochs.  The prepared-dataset parquet is
    the authoritative source of real max epoch: we always infer the stride from it when available,
    ignoring the stored ``checkpoint_stride`` which can be wrong when the PT file's epoch column
    already contains real epoch numbers rather than sequential indices.
    """
    if np.isnan(checkpoint_max_epoch) or checkpoint_max_epoch <= 0:
        return float(run.get("checkpoint_stride", 1))
    parquet_path = Path(str(run.get("dataset_parquet_path", "")))
    if parquet_path.exists():
        try:
            real_max = float(pd.read_parquet(parquet_path, columns=["epoch"])["epoch"].max())
            inferred = real_max / checkpoint_max_epoch
            if inferred > 1.0:
                log.debug(
                    "Inferred checkpoint_stride=%.0f for '%s' (parquet max_epoch=%.0f, round_details max_epoch=%.0f)",
                    inferred,
                    run.get("experiment_name", "?"),
                    real_max,
                    checkpoint_max_epoch,
                )
            return max(1.0, inferred)
        except Exception:
            pass
    return max(1.0, float(run.get("checkpoint_stride", 1)))


def _compute_savings_fraction(
    *,
    initial_population: int,
    max_epoch: float,
    start_epoch: float,
    interval: float,
    rounds_executed: int,
    survival_rate: float,
) -> float:
    """Estimate the fraction of training compute saved by SOTF selection vs. training all models fully.

    Uses an analytic formula: full compute = n * T; SOTF compute accounts for the shrinking
    population across selection rounds.
    """
    if np.isnan(max_epoch) or np.isnan(survival_rate) or np.isnan(start_epoch) or np.isnan(interval):
        return np.nan
    n = int(initial_population)
    T = float(max_epoch)
    start = float(start_epoch)
    K = int(rounds_executed)
    iv = float(interval)
    s = float(survival_rate)
    if n <= 0 or T <= 0 or iv <= 0 or not (0.0 < s <= 1.0) or start < 0:
        return np.nan
    if K == 0:
        return 0.0
    full_compute = n * T
    # Phase 0: all n models train from epoch 1 to start_epoch
    sotf_compute = n * start
    # Phases 1..K-1: reduced population trains between consecutive decision epochs
    for k in range(1, K):
        sotf_compute += n * (s ** k) * iv
    # Phase K: survivors after last round train to max_epoch
    last_phase = T - start - (K - 1) * iv
    sotf_compute += n * (s ** K) * max(0.0, last_phase)
    if full_compute <= 0:
        return np.nan
    return float(1.0 - sotf_compute / full_compute)


def build_predictor_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    """Build predictor summary.

    Args:
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if dataset_summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for predictor, group in dataset_summary.groupby("predictor", sort=True):
        rows.append(
            {
                "predictor": predictor,
                "dataset_count": int(len(group)),
                "mean_oracle_top_k_recall": float(group["oracle_top_k_recall"].mean()),
                "mean_decision_accuracy_weighted": float(group["decision_accuracy_weighted"].mean()),
                "mean_keep_precision_weighted": float(group["keep_precision_weighted"].mean()),
                "mean_keep_recall_weighted": float(group["keep_recall_weighted"].mean()),
                "mean_kill_precision_weighted": float(group["kill_precision_weighted"].mean()),
                "mean_kill_recall_weighted": float(group["kill_recall_weighted"].mean()),
                "mean_false_kill_rate_weighted": float(group["false_kill_rate_weighted"].mean()),
                "mean_roc_auc_weighted": float(group["roc_auc_weighted"].mean()),
                "mean_prediction_r2_weighted": float(group["prediction_r2_weighted"].mean()),
                "mean_prediction_rmse_weighted": float(group["prediction_rmse_weighted"].mean()),
                "oracle_best_retained_rate": float(group["retained_oracle_best"].mean()),
                "mean_best_survivor_regret": float(group["best_survivor_regret"].mean()),
                "mean_best_survivor_rank_percentile": float(group["best_survivor_rank_percentile"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("predictor").reset_index(drop=True)


def build_predictor_comparison(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    """Build predictor comparison.

    Args:
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if dataset_summary.empty:
        return pd.DataFrame()

    resolved = resolve_baseline_and_comparison_predictors(dataset_summary["predictor"])
    if resolved is None:
        return pd.DataFrame()
    comparison_predictors, baseline_predictor = resolved

    merge_keys = [
        "dataset_number",
        "dataset_label",
        "dataset_slug",
        "target_column",
        "target_mode",
        "horizon",
        "selection_start_epoch",
        "selection_interval",
        "selection_survival_rate",
    ]
    baseline_df = dataset_summary[dataset_summary["predictor"] == baseline_predictor].copy()
    if baseline_df.empty:
        return pd.DataFrame()

    comparisons: list[pd.DataFrame] = []
    for comparison_predictor in comparison_predictors:
        comparison_df = dataset_summary[dataset_summary["predictor"] == comparison_predictor].copy()
        if comparison_df.empty:
            continue
        merged = comparison_df.merge(
            baseline_df,
            on=merge_keys,
            suffixes=("_comparison", "_baseline"),
            how="inner",
        )
        if merged.empty:
            continue
        comparisons.append(
            pd.DataFrame(
                {
                    "dataset_number": merged["dataset_number"],
                    "dataset_label": merged["dataset_label"],
                    "target_column": merged["target_column"],
                    "comparison_predictor": comparison_predictor,
                    "baseline_predictor": baseline_predictor,
                    "comparison_oracle_top_k_recall": merged["oracle_top_k_recall_comparison"],
                    "baseline_oracle_top_k_recall": merged["oracle_top_k_recall_baseline"],
                    "delta_oracle_top_k_recall": (
                        merged["oracle_top_k_recall_comparison"] - merged["oracle_top_k_recall_baseline"]
                    ),
                    "comparison_decision_accuracy_weighted": merged["decision_accuracy_weighted_comparison"],
                    "baseline_decision_accuracy_weighted": merged["decision_accuracy_weighted_baseline"],
                    "delta_decision_accuracy_weighted": (
                        merged["decision_accuracy_weighted_comparison"] - merged["decision_accuracy_weighted_baseline"]
                    ),
                    "comparison_false_kill_rate_weighted": merged["false_kill_rate_weighted_comparison"],
                    "baseline_false_kill_rate_weighted": merged["false_kill_rate_weighted_baseline"],
                    "delta_false_kill_rate_weighted": (
                        merged["false_kill_rate_weighted_comparison"] - merged["false_kill_rate_weighted_baseline"]
                    ),
                    "comparison_best_survivor_regret": merged["best_survivor_regret_comparison"],
                    "baseline_best_survivor_regret": merged["best_survivor_regret_baseline"],
                    "delta_best_survivor_regret": (
                        merged["best_survivor_regret_comparison"] - merged["best_survivor_regret_baseline"]
                    ),
                    "comparison_best_survivor_rank_percentile": merged["best_survivor_rank_percentile_comparison"],
                    "baseline_best_survivor_rank_percentile": merged["best_survivor_rank_percentile_baseline"],
                    "delta_best_survivor_rank_percentile": (
                        merged["best_survivor_rank_percentile_comparison"]
                        - merged["best_survivor_rank_percentile_baseline"]
                    ),
                }
            )
        )

    if not comparisons:
        return pd.DataFrame()
    combined = pd.concat(comparisons, ignore_index=True)
    return combined.sort_values(["dataset_number", "comparison_predictor"]).reset_index(drop=True)
