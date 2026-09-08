from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_squared_error, r2_score, roc_curve


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATASET_ID_RE = re.compile(r"ds(\d+)", re.IGNORECASE)
RUN_TIME_FORMATS = ("%Y-%m-%d %H-%M-%S-%f", "%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d-%H-%M-%S-%f")
DATASET_NAME_SUFFIXES = (
    "_full_all_metrics_long",
    "_all_metrics_long",
    "_full_all_metrics",
    "_all_metrics",
)
WEIGHT_STATISTIC_SPLIT_RE = re.compile(r"^weight_statistics_(train|val|test)\.pt$")
MODEL_ZOO_INFORMATION_PATH = Path(__file__).resolve().parents[3] / "learning-dynamics" / "configs" / "model_zoo_informations.json"


def _dataset_name_token(path: Path) -> str:
    """Extract the path token that carries the dataset identity."""
    if WEIGHT_STATISTIC_SPLIT_RE.match(path.name):
        return path.parent.name
    return path.name

def regret_against_oracle(*, metric_goal: str, oracle_value: float, survivor_value: float) -> float:
    """Regret against oracle.

    Args:
        metric_goal (str): str value used by the function.
        oracle_value (float): float value used by the function.
        survivor_value (float): float value used by the function.

    Returns:
        float: Result of the function.
    """
    if np.isnan(oracle_value) or np.isnan(survivor_value):
        return np.nan
    if metric_goal == "minimize":
        return float(survivor_value - oracle_value)
    return float(oracle_value - survivor_value)


def dataset_number_from_sources(*, experiment_name: str, parquet_path: Path) -> int | None:
    """Dataset number from sources.

    Args:
        experiment_name (str): str value used by the function.
        parquet_path (Path): Path to the parquet.

    Returns:
        int | None: Result of the function.
    """
    match = DATASET_ID_RE.search(experiment_name)
    if match:
        return int(match.group(1))

    parquet_stem = _dataset_name_token(parquet_path)
    if "_" in parquet_stem:
        prefix = parquet_stem.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
    return None


def dataset_slug_from_parquet(parquet_path: Path, *, fallback: str) -> str:
    """Dataset slug from parquet.

    Args:
        parquet_path (Path): Path to the parquet.
        fallback (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    token = _dataset_name_token(parquet_path)
    if token:
        stem = token.removesuffix(".parquet").removesuffix(".pt")
        stem = stem.removesuffix("__prepared_dataset")
        return stem
    return fallback


def pretty_dataset_name(value: str) -> str:
    """Format dataset name.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    text = str(value).strip()
    for suffix in DATASET_NAME_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def dataset_number_from_value(value: Any) -> int | None:
    """Extract a dataset number from a numeric or string identifier."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = DATASET_ID_RE.search(text)
    if match:
        return int(match.group(1))
    prefix = text.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    if text.isdigit():
        return int(text)
    return None


def parse_run_started_at(run_dir: Path) -> datetime:
    """Parse run started at.

    Args:
        run_dir (Path): Directory for the run.

    Returns:
        datetime: Computed run started at.
    """
    date_component = next((part.name for part in run_dir.parents if DATE_DIR_RE.match(part.name)), None)
    if date_component is not None:
        for time_format in RUN_TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_component} {run_dir.name}", time_format)
            except ValueError:
                continue
    return datetime.fromtimestamp(run_dir.stat().st_mtime)


def nested_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Nested get.

    Args:
        mapping (Any): Mapping to convert.
        default (Any): Fallback value used when conversion fails.
        *keys (str): str value used by the function.

    Returns:
        Any: Result of the function.
    """
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Safe ratio.

    Args:
        numerator (int | float): int | float value used by the function.
        denominator (int | float): int | float value used by the function.

    Returns:
        float: Result of the function.
    """
    if float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def compute_prediction_metric(
    y_true: pd.Series | Sequence[float],
    y_pred: pd.Series | Sequence[float],
    *,
    metric_name: str,
) -> float:
    """Compute prediction metric.

    Args:
        y_true (pd.Series | Sequence[float]): pd.Series | Sequence[float] value used by the function.
        y_pred (pd.Series | Sequence[float]): pd.Series | Sequence[float] value used by the function.
        metric_name (str): str value used by the function.

    Returns:
        float: Computed prediction metric.
    """
    y_true_values = np.asarray(y_true, dtype="float64")
    y_pred_values = np.asarray(y_pred, dtype="float64")
    if len(y_true_values) == 0 or len(y_pred_values) == 0:
        return np.nan

    if metric_name == "r2":
        if len(y_true_values) < 2:
            return np.nan
        score = r2_score(y_true_values, y_pred_values)
        return float(score) if np.isfinite(score) else np.nan

    if metric_name == "rmse":
        score = np.sqrt(mean_squared_error(y_true_values, y_pred_values))
        return float(score) if np.isfinite(score) else np.nan

    raise ValueError(f"Unsupported metric: {metric_name}")


def goal_adjusted_scores(scores: pd.Series | Sequence[float], *, metric_goal: str) -> pd.Series:
    """Goal adjusted scores.

    Args:
        scores (pd.Series | Sequence[float]): pd.Series | Sequence[float] value used by the function.
        metric_goal (str): str value used by the function.

    Returns:
        pd.Series: Computed series.
    """
    adjusted = pd.to_numeric(pd.Series(scores), errors="coerce")
    if str(metric_goal).lower() == "minimize":
        return -adjusted
    return adjusted


def compute_binary_auc(labels: pd.Series | Sequence[bool], scores: pd.Series | Sequence[float]) -> float:
    """Compute binary auc.

    Args:
        labels (pd.Series | Sequence[bool]): pd.Series | Sequence[bool] value used by the function.
        scores (pd.Series | Sequence[float]): pd.Series | Sequence[float] value used by the function.

    Returns:
        float: Computed binary auc.
    """
    label_series = pd.Series(labels)
    score_series = pd.to_numeric(pd.Series(scores), errors="coerce")
    valid = ~(label_series.isna() | score_series.isna())
    if not valid.any():
        return np.nan

    y_true = label_series.loc[valid].astype(bool).astype(int).to_numpy(dtype="int64")
    y_score = score_series.loc[valid].to_numpy(dtype="float64")
    if len(np.unique(y_true)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)
    return float(score) if np.isfinite(score) else np.nan


def float_or_nan(value: Any) -> float:
    """Convert or nan.

    Args:
        value (Any): Value to process.

    Returns:
        float: Result of the function.
    """
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def weighted_average(values: pd.Series | Sequence[float], weights: pd.Series | Sequence[float]) -> float:
    """Weighted average.

    Args:
        values (pd.Series | Sequence[float]): Values to process.
        weights (pd.Series | Sequence[float]): pd.Series | Sequence[float] value used by the function.

    Returns:
        float: Result of the function.
    """
    values_array = np.asarray(values, dtype="float64")
    weights_array = np.asarray(weights, dtype="float64")
    valid = ~(np.isnan(values_array) | np.isnan(weights_array))
    if not valid.any():
        return np.nan
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    if np.allclose(weights_array.sum(), 0.0):
        return np.nan
    return float(np.average(values_array, weights=weights_array))


def rank_to_percentile(rank: float, population: int) -> float:
    """Rank to percentile.

    Args:
        rank (float): float value used by the function.
        population (int): int value used by the function.

    Returns:
        float: Result of the function.
    """
    if np.isnan(rank) or population <= 1:
        return np.nan
    return float(1.0 - ((rank - 1.0) / float(population - 1)))


def build_round_metric_table(
    round_metrics: pd.DataFrame,
    *,
    metric_column: str,
    x_axis_column: str = "decision_epoch",
) -> pd.DataFrame:
    """Build round metric table.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        metric_column (str): str value used by the function.
        x_axis_column (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if round_metrics.empty or metric_column not in round_metrics.columns:
        return pd.DataFrame()

    metric_axis_column = x_axis_column if x_axis_column in round_metrics.columns else "round_index"
    if metric_axis_column not in round_metrics.columns:
        return pd.DataFrame()

    grouped = (
        round_metrics.loc[:, ["dataset_label", "dataset_number", metric_axis_column, metric_column]]
        .groupby(["dataset_label", "dataset_number", metric_axis_column], as_index=False, dropna=False)[metric_column]
        .mean()
    )
    dataset_order = (
        grouped.loc[:, ["dataset_label", "dataset_number"]]
        .drop_duplicates()
        .sort_values(["dataset_number", "dataset_label"], na_position="last")["dataset_label"]
        .tolist()
    )
    axis_order = sorted(pd.Index(grouped[metric_axis_column]).dropna().unique().tolist())
    table = grouped.pivot(index="dataset_label", columns=metric_axis_column, values=metric_column)
    table = table.reindex(index=dataset_order, columns=axis_order)
    table.index.name = "dataset"
    table.columns.name = metric_axis_column
    return table


def clip_negative_r2_for_plot(values: pd.DataFrame | pd.Series | float) -> pd.DataFrame | pd.Series | float:
    """Clip negative r2 for plot.

    Args:
        values (pd.DataFrame | pd.Series | float): Values to process.

    Returns:
        pd.DataFrame | pd.Series | float: Result of the function.
    """
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.clip(lower=0.0)
    return max(float(values), 0.0)


def contains_finite_values(table: pd.DataFrame) -> bool:
    """Contains finite values.

    Args:
        table (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        bool: Boolean result.
    """
    return not table.empty and np.isfinite(table.to_numpy(dtype="float64")).any()


def round_metric_plot_max(table: pd.DataFrame, *, default: float) -> float:
    """Round metric plot max.

    Args:
        table (pd.DataFrame): pd.DataFrame value used by the function.
        default (float): Fallback value used when conversion fails.

    Returns:
        float: Result of the function.
    """
    if table.empty:
        return default
    finite_values = table.to_numpy(dtype="float64")
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return default
    return max(default, float(finite_values.max()))


def dataset_sort_key(value: str) -> tuple[int, str]:
    """Dataset sort key.

    Args:
        value (str): Value to process.

    Returns:
        tuple[int, str]: Result of the function.
    """
    match = DATASET_ID_RE.search(str(value))
    if match:
        return int(match.group(1)), str(value)
    return 10**9, str(value)


def build_filter_summary(
    *,
    experiment_root: str,
    baseline_root: str | None,
    experiment_paths: Sequence[str] | None,
    predictors: Sequence[str] | None,
    target_column: str | None,
    target_mode: str | None,
    horizon: int | None,
    start_epoch: int | None,
    interval: int | None,
    survival_rate: float | None,
) -> dict[str, Any]:
    """Build filter summary.

    Args:
        experiment_root (str): str value used by the function.
        baseline_root (str | None): str | None value used by the function.
        experiment_paths (Sequence[str] | None): Sequence[str] | None value used by the function.
        predictors (Sequence[str] | None): Sequence[str] | None value used by the function.
        target_column (str | None): str | None value used by the function.
        target_mode (str | None): str | None value used by the function.
        horizon (int | None): int | None value used by the function.
        start_epoch (int | None): int | None value used by the function.
        interval (int | None): int | None value used by the function.
        survival_rate (float | None): float | None value used by the function.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    return {
        "experiment_root": experiment_root,
        "baseline_root": baseline_root or "none",
        "experiment_paths": list(experiment_paths) if experiment_paths else "none",
        "predictors": list(predictors) if predictors else "all",
        "target_column": target_column or "all",
        "target_mode": target_mode or "all",
        "horizon": horizon if horizon is not None else "all",
        "start_epoch": start_epoch if start_epoch is not None else "all",
        "interval": interval if interval is not None else "all",
        "survival_rate": survival_rate if survival_rate is not None else "all",
    }


def _dataset_zoo_name(frame: pd.DataFrame) -> str:
    """Return the zoo name for the single dataset in frame, or '' if multiple/unknown."""
    dataset_number = unique_plot_value(frame, "dataset_number")
    dataset_slug = unique_plot_value(frame, "dataset_slug")
    dataset_label = unique_plot_value(frame, "dataset_label")

    if "mixed" in (dataset_number, dataset_slug, dataset_label):
        return ""

    def _none_if_na(v: Any) -> Any:
        return None if v in (None, "n/a") else v

    metadata = resolve_model_zoo_metadata(
        dataset_number=_none_if_na(dataset_number),
        dataset_slug=_none_if_na(dataset_slug),
        dataset_label=_none_if_na(dataset_label),
    )
    if metadata is not None:
        zoo_name = metadata.get("zoo_name", "")
        if zoo_name:
            return zoo_name

    fallback = _none_if_na(dataset_label) or _none_if_na(dataset_slug)
    if fallback and str(fallback).strip().lower() not in ("none", "nan"):
        return str(fallback).strip()
    return ""


def build_experiment_subtitle(frame: pd.DataFrame) -> str:
    """Build experiment subtitle.

    Args:
        frame (pd.DataFrame): Data frame to process.

    Returns:
        str: Resolved string value.
    """
    if frame.empty:
        return ""

    target_mode = unique_plot_value(frame, "target_mode")
    horizon = unique_plot_value(frame, "horizon")
    start_epoch = unique_plot_value(frame, "selection_start_epoch")
    interval = unique_plot_value(frame, "selection_interval")
    survival_rate = unique_plot_value(frame, "selection_survival_rate")

    # Cross-batch context: some hyper-parameters vary across batches.
    is_cross_batch = any(v == "mixed" for v in (start_epoch, survival_rate, horizon))

    parts: list[str] = []
    if target_mode == "final_value":
        parts.append("Final value prediction")
    elif target_mode not in ("mixed", "n/a", None):
        parts.append(f"Forecast horizon: {format_plot_value(horizon)}")

    if start_epoch not in ("mixed", "n/a"):
        parts.append(f"Start epoch: {format_plot_value(start_epoch)}")
    if interval not in ("mixed", "n/a"):
        parts.append(f"Decision interval: {format_plot_value(interval)}")
    if survival_rate not in ("mixed", "n/a"):
        parts.append(f"Survival rate: {format_plot_value(survival_rate)}")

    if is_cross_batch:
        zoo_name = _dataset_zoo_name(frame)
        if zoo_name:
            parts.append(zoo_name)

    return " | ".join(parts)


def build_model_zoo_subtitle(frame: pd.DataFrame, *, info_path: Path | None = None) -> str:
    """Build a subtitle line with model-zoo metadata for the datasets in the frame."""
    if frame.empty:
        return ""

    columns = [column for column in ("dataset_number", "dataset_slug", "dataset_label") if column in frame.columns]
    if not columns:
        return ""

    dataset_rows = frame.loc[:, columns].drop_duplicates().copy()
    sort_columns = [column for column in ("dataset_number", "dataset_label", "dataset_slug") if column in dataset_rows.columns]
    if sort_columns:
        dataset_rows = dataset_rows.sort_values(sort_columns, na_position="last", kind="stable")

    entries: list[tuple[str, str]] = []  # (label, details)
    for dataset_row in dataset_rows.to_dict("records"):
        metadata = resolve_model_zoo_metadata(
            dataset_number=dataset_row.get("dataset_number"),
            dataset_slug=dataset_row.get("dataset_slug"),
            dataset_label=dataset_row.get("dataset_label"),
            info_path=info_path,
        )
        if metadata is None:
            continue

        label = model_zoo_reference_label(metadata=metadata, dataset_row=dataset_row)
        details = []
        if metadata.get("model_type"):
            details.append(f"model={metadata['model_type']}")
        if metadata.get("dataset"):
            details.append(f"data={metadata['dataset']}")
        if metadata.get("initialization"):
            details.append(f"init={metadata['initialization']}")
        if label or details:
            entries.append((label, ", ".join(details)))

    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0][0]  # Just the zoo name — used as per-batch secondary subtitle
    return f"Model zoo info: {'; '.join(f'{lbl}: {det}' for lbl, det in entries if det)}"


def model_zoo_reference_label(*, metadata: dict[str, str], dataset_row: dict[str, Any]) -> str:
    """Build the left-hand label for model-zoo subtitle entries."""
    plot_name = str(metadata.get("plot_name", "")).strip()
    if plot_name:
        return plot_name
    zoo_name = str(metadata.get("zoo_name", "")).strip()
    if zoo_name:
        return zoo_name

    dataset_number = dataset_number_from_value(dataset_row.get("dataset_number"))
    if dataset_number is not None and dataset_number >= 0:
        return f"DS{dataset_number}"

    for key in ("dataset_label", "dataset_slug"):
        value = dataset_row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return "dataset"


def resolve_model_zoo_metadata(
    *,
    dataset_number: Any,
    dataset_slug: Any,
    dataset_label: Any,
    info_path: Path | None = None,
) -> dict[str, str] | None:
    """Resolve model-zoo metadata for a dataset from the configured information file."""
    lookup = load_model_zoo_information_lookup(info_path)
    if not lookup:
        return None

    parsed_dataset_number = dataset_number_from_value(dataset_number)
    if parsed_dataset_number is not None:
        metadata = lookup.get(("number", parsed_dataset_number))
        if metadata is not None:
            return metadata

    for candidate in (dataset_slug, dataset_label):
        normalized_candidate = normalize_model_zoo_key(candidate)
        if not normalized_candidate:
            continue
        metadata = lookup.get(("slug", normalized_candidate))
        if metadata is not None:
            return metadata
    return None


@lru_cache(maxsize=None)
def load_model_zoo_information_lookup(info_path: Path | None = None) -> dict[tuple[str, Any], dict[str, str]]:
    """Load the model-zoo information file into a lookup keyed by dataset number and slug."""
    path = (info_path if info_path is not None else MODEL_ZOO_INFORMATION_PATH).resolve()
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}

    lookup: dict[tuple[str, Any], dict[str, str]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue

        metadata = {
            "zoo_name": str(entry.get("zoo_name", "")).strip(),
            "plot_name": str(entry.get("plot_name", "")).strip(),
            "model_type": str(entry.get("model_type", "")).strip(),
            "dataset": str(entry.get("dataset", "")).strip(),
            "initialization": str(entry.get("Initialization", "")).strip(),
        }
        if not any(metadata.values()):
            continue

        zoo_name = entry.get("zoo_name")
        dataset_number = dataset_number_from_value(zoo_name)
        if dataset_number is not None:
            lookup[("number", dataset_number)] = metadata

        normalized_zoo_name = normalize_model_zoo_key(zoo_name)
        if normalized_zoo_name:
            lookup[("slug", normalized_zoo_name)] = metadata

    return lookup


def normalize_model_zoo_key(value: Any) -> str:
    """Normalize a dataset slug or zoo-name key for metadata lookup."""
    if value is None:
        return ""
    return pretty_dataset_name(str(value).strip()).casefold()


def shared_remaining_population_trace(dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Shared remaining population trace.

    Args:
        dataset_df (pd.DataFrame): Data frame containing dataset values.

    Returns:
        tuple[pd.DataFrame, str | None]: Result of the function.
    """
    if dataset_df.empty:
        return pd.DataFrame(columns=["decision_epoch", "policy_keep_count"]), None

    remaining = (
        dataset_df.loc[:, ["predictor", "decision_epoch", "policy_keep_count"]]
        .drop_duplicates()
        .sort_values(["decision_epoch", "predictor"])
    )
    if remaining.empty:
        return pd.DataFrame(columns=["decision_epoch", "policy_keep_count"]), None

    pivot = remaining.pivot_table(
        index="decision_epoch",
        columns="predictor",
        values="policy_keep_count",
        aggfunc="first",
        sort=True,
    ).sort_index()
    predictor_order = resolve_comparison_predictor_order(pd.Index(pivot.columns))
    if len(predictor_order) < 2:
        ordered_columns = list(pivot.columns)
    else:
        ordered_columns = [name for name in predictor_order if name in pivot.columns]
    reference = pivot[ordered_columns[0]]
    mismatch = False
    for column in ordered_columns[1:]:
        if not reference.equals(pivot[column]):
            mismatch = True
            break

    trace = reference.rename("policy_keep_count").reset_index()
    warning = None
    if mismatch:
        warning = "Warning: remaining model counts differ between predictors"
    return trace, warning


def unique_plot_value(frame: pd.DataFrame, column: str) -> Any:
    """Unique plot value.

    Args:
        frame (pd.DataFrame): Data frame to process.
        column (str): str value used by the function.

    Returns:
        Any: Result of the function.
    """
    if column not in frame.columns:
        return "n/a"
    series = pd.Series(frame[column]).dropna()
    if series.empty:
        return "n/a"
    unique_values = list(pd.unique(series))
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def format_plot_value(value: Any) -> str:
    """Format plot value.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "n/a"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    if text in {"nan", "None"}:
        return "n/a"
    return text


def apply_figure_titles(fig, *, title: str, subtitle: str, secondary_subtitle: str = "") -> float:
    """Apply figure titles with stable spacing.

    Args:
        fig: Matplotlib figure to update.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.
        secondary_subtitle (str): Additional subtitle text for the output.

    Returns:
        float: Top boundary to reserve for subplot layout.
    """
    title_line_count = max(1, str(title).count("\n") + 1)
    subtitle_line_count = 0 if not subtitle else max(1, str(subtitle).count("\n") + 1)
    secondary_line_count = 0 if not secondary_subtitle else max(1, str(secondary_subtitle).count("\n") + 1)
    figure_height = max(1.0, float(fig.get_figheight()))
    title_y = 0.99
    title_line_height = (20.0 / 72.0 / figure_height) * 1.15
    subtitle_gap = (10.0 / 72.0 / figure_height) * 0.3
    subtitle_line_height = (15.0 / 72.0 / figure_height) * 1.3
    secondary_gap = (10.0 / 72.0 / figure_height) * 0.3
    plot_gap = 0.0
    title_bottom = title_y - title_line_height * title_line_count

    fig.suptitle(title, fontsize=20, y=title_y)
    current_bottom = title_bottom
    if subtitle:
        subtitle_y = current_bottom - subtitle_gap
        fig.text(
            0.5,
            subtitle_y,
            subtitle,
            ha="center",
            va="top",
            fontsize=15,
            color="#555555",
            wrap=True,
        )
        current_bottom = subtitle_y - subtitle_line_height * subtitle_line_count
    if secondary_subtitle:
        secondary_y = current_bottom - secondary_gap
        fig.text(
            0.5,
            secondary_y,
            secondary_subtitle,
            ha="center",
            va="top",
            fontsize=13,
            color="#555555",
            wrap=True,
        )
        current_bottom = secondary_y - subtitle_line_height * secondary_line_count

    layout_top = current_bottom - plot_gap
    return max(0.72, layout_top)


def predictor_display_name(value: str) -> str:
    """Predictor display name.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    normalized = str(value).strip().lower()
    if normalized == "forecast_model":
        return "Forecasting model"
    if normalized == "xgboost":
        return "Online XGBoost"
    if normalized == "online_lstm":
        return "Online latent-window LSTM"
    if normalized == "naive":
        return "Decision based on functional metric"
    return str(value).replace("_", " ")


def resolve_comparison_predictor_order(values: Sequence[Any] | pd.Series | pd.Index) -> list[str]:
    """Resolve comparison predictor order.

    Args:
        values (Sequence[Any] | pd.Series | pd.Index): Values to process.

    Returns:
        list[str]: List of string values.
    """
    predictors = [str(value) for value in values if pd.notna(value)]
    available = sorted(set(predictors))
    if len(available) < 2:
        return []

    preferred_non_baseline = ["forecast_model", "xgboost", "online_lstm"]
    order: list[str] = []
    for predictor_name in preferred_non_baseline:
        if predictor_name in available:
            order.append(predictor_name)

    for predictor_name in available:
        if predictor_name == "naive" or predictor_name in order:
            continue
        order.append(predictor_name)

    if "naive" in available:
        order.append("naive")
    return order


def resolve_baseline_and_comparison_predictors(
    values: Sequence[Any] | pd.Series | pd.Index,
) -> tuple[list[str], str] | None:
    """Resolve baseline and comparison predictors.

    Args:
        values (Sequence[Any] | pd.Series | pd.Index): Values to process.

    Returns:
        tuple[list[str], str] | None: Computed baseline and comparison predictors.
    """
    predictor_order = resolve_comparison_predictor_order(values)
    if len(predictor_order) < 2 or "naive" not in predictor_order:
        return None
    comparison_predictors = [name for name in predictor_order if name != "naive"]
    if not comparison_predictors:
        return None
    return comparison_predictors, "naive"


def resolve_comparison_predictors(values: Sequence[Any] | pd.Series | pd.Index) -> tuple[str, str] | None:
    """Resolve comparison predictors.

    Args:
        values (Sequence[Any] | pd.Series | pd.Index): Values to process.

    Returns:
        tuple[str, str] | None: Computed comparison predictors.
    """
    resolved = resolve_baseline_and_comparison_predictors(values)
    if resolved is None:
        return None
    comparison_predictors, baseline_predictor = resolved
    if len(comparison_predictors) != 1:
        return None
    return comparison_predictors[0], baseline_predictor


def predictor_style(predictor_name: str, *, role_index: int) -> dict[str, str]:
    """Predictor style.

    Args:
        predictor_name (str): str value used by the function.
        role_index (int): int value used by the function.

    Returns:
        dict[str, str]: Result of the function.
    """
    if predictor_name == "forecast_model":
        return {"color": "#1d3557", "marker": "o", "label": predictor_display_name(predictor_name)}
    if predictor_name == "xgboost":
        return {"color": "#457b9d", "marker": "^", "label": predictor_display_name(predictor_name)}
    if predictor_name == "online_lstm":
        return {"color": "#2a9d8f", "marker": "D", "label": predictor_display_name(predictor_name)}
    if predictor_name == "naive":
        return {"color": "#a8dadc", "marker": "s", "label": predictor_display_name(predictor_name)}
    color_cycle = ["#ff7f11", "#2a9d8f", "#6f4e7c", "#d62828", "#264653"]
    marker_cycle = ["D", "P", "X", "v", ">"]
    cycle_index = role_index % len(color_cycle)
    return {
        "color": color_cycle[cycle_index],
        "marker": marker_cycle[cycle_index],
        "label": predictor_display_name(predictor_name),
    }


def unique_predictor_name(series: pd.Series | None, *, default: str) -> str:
    """Unique predictor name.

    Args:
        series (pd.Series | None): pd.Series | None value used by the function.
        default (str): Fallback value used when conversion fails.

    Returns:
        str: Resolved string value.
    """
    if series is None:
        return default
    values = pd.Series(series).dropna().astype(str).unique().tolist()
    if len(values) == 1:
        return values[0]
    return default


def sanitize_file_stem(value: str) -> str:
    """Sanitize file stem.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower() or "output"


def format_rank(value: float) -> str:
    """Format rank.

    Args:
        value (float): Value to process.

    Returns:
        str: Resolved string value.
    """
    if np.isnan(value):
        return "n/a"
    return str(int(round(value)))


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    """Dataframe to markdown.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.
        columns (list[str]): Column names to process.

    Returns:
        str: Resolved string value.
    """
    if df.empty:
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        return "\n".join([header, divider])
    subset = df.loc[:, columns].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "nan")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])
