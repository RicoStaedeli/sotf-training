from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, cast

import matplotlib
import matplotlib.axes
import matplotlib.colors
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


PROJECT_ROOT = Path(__file__).resolve().parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ZOO_INFO_PATH = _REPO_ROOT / "learning-dynamics" / "configs" / "model_zoo_informations.json"
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "metric-prediction" / "outputs"
DEFAULT_BATCH_FILES = [
    PROJECT_ROOT / "metric-prediction" / "experiment_batches" / "ds1-ds4.yaml",
    PROJECT_ROOT / "metric-prediction" / "experiment_batches" / "ds5-ds7.yaml",
]
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FULL_TIMESTAMP_FORMATS = ("%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d-%H-%M-%S-%f")

EXPERIMENT_RE = re.compile(
    r"^ds(?P<dataset_number>\d+)_(?P<model>ridge|lightgbm|xgboost|mlp)"
    r"(?:_(?P<scope>model_only|layer_only|model_and_layer))?"
    r"_(?P<family>[A-Za-z0-9_]+?)"
    r"(?:_(?P<target_variant>test_acc|test_loss|train_acc|train_loss))?"
    r"(?:_epoch\d+)?$"
)
EXPERIMENT_NAME_RE = re.compile(r"experiment_name:\s*([A-Za-z0-9_.-]+)")

MODEL_LABELS = {
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}
SCOPE_LABELS = {
    "model_only": "Model only",
    "layer_only": "Layer only",
    "model_and_layer": "Model + layer",
}
FAMILY_LABELS = {
    "all_families": "All families",
    "distribution": "Distribution",
    "distribution_norm": "Distribution + norm",
    "matrix_distance_sharpness": "Matrix + distance + sharpness",
    "latent": "Latent (autoencoder)",
}
MODEL_COLORS = {
    "ridge": "#a8dadc",   # light teal — simplest baseline
    "lightgbm": "#f4a261",  # warm orange
    "xgboost": "#1d3557",   # dark navy — strongest performer
    "mlp": "#2a9d8f",       # teal green
}
R2_PLOT_LABEL = r"$R^2$"
RMSE_PLOT_LABEL = "RMSE"


@dataclass(frozen=True)
class ExperimentParts:
    experiment_name: str
    dataset_id: str
    dataset_number: int
    model: str
    scope: str
    family_combo: str
    target_metric: str | None


def default_report_dir(outputs_root: Path) -> Path:
    """Default report dir.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    return outputs_root / "evaluation" / "ds1-ds7-overview"


def batch_files_dir(outputs_root: Path) -> Path:
    """Batch files dir.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    return outputs_root / "batch_files"


def results_dir(outputs_root: Path) -> Path:
    """Results dir.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    return outputs_root / "results"


def _contains_top_level_batch_files(outputs_root: Path) -> bool:
    """Contains top level batch files.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        bool: Result of the function.
    """
    return any(path.is_file() for path in outputs_root.glob("*.yaml")) or any(path.is_file() for path in outputs_root.glob("*.yml"))


def _has_completed_runs(outputs_root: Path) -> bool:
    """Has completed runs.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        bool: Result of the function.
    """
    return any(outputs_root.rglob("artifacts/metrics.json"))


def _is_usable_outputs_root(outputs_root: Path) -> bool:
    """Is usable outputs root.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        bool: Result of the function.
    """
    return (
        batch_files_dir(outputs_root).is_dir()
        or results_dir(outputs_root).is_dir()
        or _contains_top_level_batch_files(outputs_root)
        or _has_completed_runs(outputs_root)
    )


def resolve_outputs_root(outputs_root: Path) -> Path:
    """Resolve outputs root.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    resolved = outputs_root.resolve()
    if _is_usable_outputs_root(resolved):
        return resolved

    batch_experiments_candidate = resolved.parent / "batch_experiments" / resolved.name
    if _is_usable_outputs_root(batch_experiments_candidate):
        return batch_experiments_candidate.resolve()

    return resolved


def discover_batch_files(outputs_root: Path) -> list[Path]:
    """Discover batch files.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        list[Path]: Result of the function.
    """
    search_root = batch_files_dir(outputs_root) if batch_files_dir(outputs_root).is_dir() else outputs_root
    batch_files = sorted(search_root.glob("*.yaml")) + sorted(search_root.glob("*.yml"))
    return [path.resolve() for path in batch_files if path.is_file()]


def resolve_batch_files(outputs_root: Path, batch_files: list[Path] | None) -> list[Path]:
    """Resolve batch files.

    Args:
        outputs_root (Path): Path value used by the function.
        batch_files (list[Path] | None): list[Path] | None value used by the function.

    Returns:
        list[Path]: Result of the function.
    """
    if batch_files is not None:
        return [path.resolve() for path in batch_files]

    discovered_batch_files = discover_batch_files(outputs_root)
    if discovered_batch_files:
        return discovered_batch_files

    return DEFAULT_BATCH_FILES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv (list[str] | None): list[str] | None value used by the function.

    Returns:
        argparse.Namespace: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize metric-prediction batch experiment results.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root directory containing Hydra run outputs.",
    )
    parser.add_argument(
        "--batch-file",
        dest="batch_files",
        type=Path,
        action="append",
        default=None,
        help="Batch config to use for expected experiment discovery. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where plots, tables, and the summary report will be written. "
            "Defaults to <outputs-root>/evaluation/ds1-ds7-overview."
        ),
    )
    parser.add_argument(
        "--extra-outputs-root",
        dest="extra_outputs_roots",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional results root to merge into the evaluation. "
            "Can be passed multiple times. Runs from all roots are combined before ranking."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top completed runs to show in the ranking plot and summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main.

    Args:
        argv (list[str] | None): list[str] | None value used by the function.

    Returns:
        int: Result of the function.
    """
    args = parse_args(argv)
    args.outputs_root = resolve_outputs_root(args.outputs_root)
    batch_files = resolve_batch_files(args.outputs_root, args.batch_files)
    output_dir = (args.output_dir if args.output_dir is not None else default_report_dir(args.outputs_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_df = load_expected_experiments(batch_files)
    completed_all_df = load_completed_runs(args.outputs_root)
    for extra_root in (args.extra_outputs_roots or []):
        extra_root = resolve_outputs_root(extra_root)
        extra_df = load_completed_runs(extra_root)
        if not extra_df.empty:
            completed_all_df = pd.concat([completed_all_df, extra_df], ignore_index=True)
            print(f"Merged {len(extra_df)} additional runs from {extra_root}")
    if completed_all_df.empty:
        raise SystemExit(f"No completed runs with metrics.json found under {args.outputs_root}")

    latest_completed_df, duplicates_df = select_latest_runs(completed_all_df)

    metric_report_rows: list[dict[str, Any]] = []
    for target_metric in target_metric_values(latest_completed_df, expected_df):
        metric_output_dir = output_dir / target_metric
        metric_output_dir.mkdir(parents=True, exist_ok=True)

        metric_expected_df = filter_by_target_metric(expected_df, target_metric)
        metric_completed_all_df = filter_by_target_metric(completed_all_df, target_metric)
        metric_duplicates_df = filter_by_target_metric(duplicates_df, target_metric)
        metric_completed_df = filter_by_target_metric(latest_completed_df, target_metric)
        metric_completed_df = select_metric_completed_runs(metric_completed_df, metric_expected_df)
        if metric_completed_df.empty and metric_expected_df.empty:
            continue
        if metric_completed_df.empty:
            print(f"  No completed runs matched expected experiments for target '{target_metric}' — skipping report.")
            print("  Hint: experiment names in --batch-file must match run directory names.")
            continue

        metric_completed_df = add_rank_columns(metric_completed_df)
        completion_df = build_completion_table(metric_expected_df, metric_completed_df)
        tables = build_summary_tables(metric_completed_df, completion_df)

        write_tables(
            metric_output_dir,
            metric_completed_all_df,
            metric_completed_df,
            metric_duplicates_df,
            completion_df,
            tables,
        )
        write_summary_report(
            metric_output_dir,
            args.outputs_root,
            batch_files,
            metric_completed_df,
            completion_df,
            tables,
            args.top_n,
            target_metric=target_metric,
        )
        render_plots(metric_output_dir, metric_completed_df, completion_df, tables, args.top_n, target_metric)
        write_per_dataset_reports(metric_output_dir, metric_completed_df, completion_df, args.top_n, target_metric)

        completion_counts = completion_df["status"].value_counts() if not completion_df.empty else pd.Series(dtype=int)
        completed_count = int(completion_counts.get("completed", len(metric_completed_df)))
        missing_count = int(completion_counts.get("missing", 0))
        total_expected = completed_count + missing_count
        metric_report_rows.append(
            {
                "target_metric": target_metric,
                "completed_runs": completed_count,
                "missing_runs": missing_count,
                "completion_rate": completed_count / total_expected if total_expected else 1.0,
                "report_dir": str(metric_output_dir),
            }
        )

    write_metric_index_report(output_dir, args.outputs_root, batch_files, metric_report_rows)
    if not metric_report_rows:
        raise SystemExit("Completed runs were found, but none could be grouped into metric-specific reports.")
    print(f"Wrote evaluation reports to {output_dir}")
    return 0


def load_expected_experiments(batch_files: Iterable[Path]) -> pd.DataFrame:
    """Load expected experiments.

    Args:
        batch_files (Iterable[Path]): Iterable[Path] value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for batch_file in batch_files:
        if not batch_file.exists():
            continue
        content = batch_file.read_text(encoding="utf-8")
        # Regex scanning keeps commented experiment entries, which matters here because
        # the batch files were partially commented out after runs had already completed.
        for match in EXPERIMENT_NAME_RE.findall(content):
            if (str(batch_file), match) in seen:
                continue
            seen.add((str(batch_file), match))
            parts = parse_experiment_name(match)
            if parts is None:
                print(f"  Warning: skipping unrecognized experiment name in batch file (will not appear in completion table): {match}")
                continue
            rows.append(
                {
                    "source_batch": str(batch_file),
                    "experiment_name": match,
                    **parts_to_row(parts),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_batch",
                "experiment_name",
                "dataset_id",
                "dataset_number",
                "model",
                "scope",
                "family_combo",
                "target_metric",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["experiment_name"]).sort_values("experiment_name").reset_index(drop=True)


def load_completed_runs(outputs_root: Path) -> pd.DataFrame:
    """Load completed runs.

    Args:
        outputs_root (Path): Path value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    rows: list[dict[str, Any]] = []
    source_outputs_root = outputs_root.resolve()
    search_root = results_dir(outputs_root) if results_dir(outputs_root).is_dir() else outputs_root
    for metrics_path in sorted(search_root.rglob("artifacts/metrics.json")):
        try:
            row = load_single_run(metrics_path)
        except Exception as exc:  # pragma: no cover - defensive reporting for local runs
            print(f"Skipping {metrics_path}: {exc}")
            continue
        row["source_outputs_root"] = str(source_outputs_root)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["run_started_at", "experiment_name"]).reset_index(drop=True)


def load_single_run(metrics_path: Path) -> dict[str, Any]:
    """Load single run.

    Args:
        metrics_path (Path): Path value used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    artifacts_dir = metrics_path.parent
    run_dir = artifacts_dir.parent
    experiment_dir = run_dir.parent
    group_dir = experiment_dir.parent

    # Two supported layouts:
    #  4-level: <group>/<experiment_name>/<run_timestamp>/artifacts/metrics.json
    #  3-level: <batch_root>/results/<experiment_name>/artifacts/metrics.json
    # Detect the 3-level layout by checking whether the parent of run_dir is "results"
    # (meaning run_dir itself is the experiment directory, not a timestamp subdir).
    if experiment_dir.name == "results":
        experiment_name = run_dir.name
        run_group = group_dir.name
    else:
        experiment_name = experiment_dir.name
        if group_dir.name == "results":
            run_group = group_dir.parent.name
        else:
            run_group = group_dir.name
    run_started_at, run_date = parse_run_timestamp(run_group, run_dir.name, metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    config_path = run_dir / ".hydra" / "config.yaml"
    _raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False) if config_path.exists() else {}
    config: dict[str, Any] = cast(dict[str, Any], _raw) if isinstance(_raw, dict) else {}

    config_parts = parse_config_parts(experiment_name, config)
    config_target_column = nested_get(config, "data", "target_column")
    target_metric = config_parts.target_metric or (str(config_target_column) if isinstance(config_target_column, str) else "unknown")
    used_features = load_json_if_exists(artifacts_dir / "used_features.json")
    tune_summary = load_json_if_exists(artifacts_dir / "ray_tune_summary.json")

    row = {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "artifacts_dir": str(artifacts_dir),
        "run_group": run_group,
        "run_date": run_date,
        "run_time": run_dir.name,
        "run_started_at": run_started_at,
        "target_metric": target_metric,
        "target_column": target_metric,
        "config_target_column": config_target_column,
        "dataset_path": dataset_path_from_config(config),
        "dataset_label": dataset_label_from_config(config, config_parts.dataset_id),
        "n_features_from_used_features": used_features.get("count") if used_features else None,
        "tune_enabled": bool(nested_get(config, "tune", "enabled", default=False)),
        "tune_metric": nested_get(config, "tune", "metric"),
        "best_tune_metric": tune_summary.get("best_metrics", {}).get("rmse") if isinstance(tune_summary, dict) else None,
        **parts_to_row(config_parts),
        **metrics,
    }
    return row


def parse_run_timestamp(date_part: str, time_part: str, metrics_path: Path) -> tuple[datetime, str]:
    """Parse run timestamp.

    Args:
        date_part (str): str value used by the function.
        time_part (str): str value used by the function.
        metrics_path (Path): Path value used by the function.

    Returns:
        tuple[datetime, str]: Result of the function.
    """
    if DATE_DIR_RE.match(date_part):
        started_at = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S-%f")
        return started_at, date_part

    for timestamp_format in FULL_TIMESTAMP_FORMATS:
        try:
            started_at = datetime.strptime(time_part, timestamp_format)
            return started_at, started_at.date().isoformat()
        except ValueError:
            continue

    started_at = datetime.fromtimestamp(metrics_path.stat().st_mtime)
    return started_at, started_at.date().isoformat()


def parse_config_parts(experiment_name: str, config: dict[str, Any]) -> ExperimentParts:
    """Parse config parts.

    Args:
        experiment_name (str): str value used by the function.
        config (dict[str, Any]): dict[str, Any] value used by the function.

    Returns:
        ExperimentParts: Result of the function.
    """
    parsed = parse_experiment_name(experiment_name)
    data_path = dataset_path_from_config(config)
    config_target_column = nested_get(config, "data", "target_column")

    dataset_id = parsed.dataset_id if parsed is not None else "unknown"
    dataset_number = parsed.dataset_number if parsed is not None else 0
    if isinstance(data_path, str):
        dataset_match = re.match(r"(?P<number>\d+)_", dataset_name_from_path(data_path))
        if dataset_match:
            dataset_number = int(dataset_match.group("number"))
            dataset_id = f"ds{dataset_number}"

    name_model = parsed.model if parsed is not None else None
    model = str(nested_get(config, "model", "name", default=name_model))

    name_scope = parsed.scope if parsed is not None else "model_and_layer"
    include_model_metrics = bool(nested_get(config, "features", "include_model_metrics", default=name_scope != "layer_only"))
    include_layer_metrics = bool(nested_get(config, "features", "include_layer_metrics", default=name_scope != "model_only"))
    if include_model_metrics and include_layer_metrics:
        scope = "model_and_layer"
    elif include_model_metrics:
        scope = "model_only"
    elif include_layer_metrics:
        scope = "layer_only"
    else:
        scope = name_scope

    metric_families = nested_get(config, "features", "metric_families", default=None)
    name_family = parsed.family_combo if parsed is not None else "unknown"
    family_combo = normalize_family_combo(metric_families, fallback=name_family)

    name_target = parsed.target_metric if parsed is not None else None
    target_metric = name_target or (str(config_target_column) if isinstance(config_target_column, str) else None)

    return ExperimentParts(
        experiment_name=experiment_name,
        dataset_id=dataset_id,
        dataset_number=dataset_number,
        model=model,
        scope=scope,
        family_combo=family_combo,
        target_metric=target_metric,
    )


def parse_experiment_name(experiment_name: str) -> ExperimentParts | None:
    """Parse experiment name.

    Args:
        experiment_name (str): str value used by the function.

    Returns:
        ExperimentParts | None: Result of the function.
    """
    match = EXPERIMENT_RE.match(experiment_name)
    if not match:
        return None
    dataset_number = int(match.group("dataset_number"))
    return ExperimentParts(
        experiment_name=experiment_name,
        dataset_id=f"ds{dataset_number}",
        dataset_number=dataset_number,
        model=match.group("model"),
        scope=match.group("scope") or "model_and_layer",
        family_combo=match.group("family"),
        target_metric=match.group("target_variant"),
    )


def normalize_family_combo(metric_families: Any, fallback: str) -> str:
    """Normalize family combo.

    Args:
        metric_families (Any): Value used by the function.
        fallback (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    if metric_families is None:
        return fallback
    families = [str(value) for value in metric_families]
    if not families:
        return "all_families"
    joined = "_".join(families)
    return joined if joined in FAMILY_LABELS else fallback


def dataset_label_from_config(config: dict[str, Any], dataset_id: str) -> str:
    """Dataset label from config.

    Args:
        config (dict[str, Any]): dict[str, Any] value used by the function.
        dataset_id (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    dataset_path = dataset_path_from_config(config)
    if not isinstance(dataset_path, str):
        return dataset_id
    stem = dataset_name_from_path(dataset_path)
    stem = stem.removesuffix("__prepared_dataset")
    return stem


def dataset_path_from_config(config: dict[str, Any]) -> str | None:
    """Resolve the most informative dataset path from a run config."""
    for field_name in ("parquet_path", "pt_train_path", "pt_path", "target_parquet_path"):
        candidate = nested_get(config, "data", field_name)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def dataset_name_from_path(dataset_path: str) -> str:
    """Derive a dataset label from a parquet or PT path."""
    path = Path(dataset_path)
    stem = path.stem
    if stem in {"weight_statistics_train", "weight_statistics_val", "weight_statistics_test"}:
        return path.parent.name
    return stem


def dataset_display_label_from_values(dataset_label: Any, dataset_id: Any) -> str:
    """Dataset display label from values.

    Args:
        dataset_label (Any): Value used by the function.
        dataset_id (Any): Value used by the function.

    Returns:
        str: Resolved string value.
    """
    if isinstance(dataset_label, str) and dataset_label:
        return format_dataset_display_label(dataset_label)
    return str(dataset_id)


def dataset_directory_name(dataset_display_label: str) -> str:
    """Dataset directory name.

    Args:
        dataset_display_label (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    normalized = re.sub(r"[\s\\/]+", "_", str(dataset_display_label).strip())
    return normalized or "unknown_dataset"


def add_dataset_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add dataset display columns.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    display_df = df.copy()
    if "dataset_id" not in display_df.columns:
        return display_df

    dataset_labels = (
        display_df["dataset_label"]
        if "dataset_label" in display_df.columns
        else pd.Series(pd.NA, index=display_df.index, dtype="object")
    )
    display_df["dataset_display_label"] = [
        dataset_display_label_from_values(dataset_label, dataset_id)
        for dataset_label, dataset_id in zip(dataset_labels, display_df["dataset_id"], strict=False)
    ]
    display_df["dataset_dir_name"] = display_df["dataset_display_label"].map(dataset_directory_name)
    return display_df


def build_dataset_metadata(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Build dataset metadata.

    Args:
        *dfs (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    metadata_frames: list[pd.DataFrame] = []
    for df in dfs:
        if df.empty or "dataset_id" not in df.columns:
            continue
        display_df = add_dataset_display_columns(df)
        metadata_frame = display_df.copy()
        if "dataset_number" not in metadata_frame.columns:
            metadata_frame["dataset_number"] = np.nan
        metadata_frames.append(
            metadata_frame[["dataset_id", "dataset_number", "dataset_display_label", "dataset_dir_name"]]
        )

    if not metadata_frames:
        return pd.DataFrame(columns=["dataset_id", "dataset_number", "dataset_display_label", "dataset_dir_name"])

    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata["has_extracted_label"] = metadata["dataset_display_label"] != metadata["dataset_id"]
    metadata = metadata.sort_values(
        ["dataset_number", "has_extracted_label", "dataset_display_label", "dataset_id"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    metadata = metadata.drop(columns=["has_extracted_label"])
    return metadata.drop_duplicates(subset=["dataset_id"], keep="first").reset_index(drop=True)


def dataset_display_labels(df: pd.DataFrame) -> list[str]:
    """Dataset display labels.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    if df.empty:
        return []
    return build_dataset_metadata(df)["dataset_display_label"].tolist()


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


def format_dataset_display_label(dataset_label: str) -> str:
    """Format dataset display label.

    Args:
        dataset_label (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    zoo_name = dataset_label.removesuffix("_full_all_metrics_long")
    lookup = _load_zoo_plot_name_lookup()
    return lookup.get(zoo_name, zoo_name)


def with_target_metric(title: str, target_metric: str) -> str:
    """With target metric.

    Args:
        title (str): str value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    return f"{title} ({target_metric})"


def with_plot_r2(text: str) -> str:
    """With plot r2.

    Args:
        text (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    return text.replace("R2", R2_PLOT_LABEL)


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


def mask_high_rmse_for_plot(values: Any, ceiling: float = 1.0) -> Any:
    """Mask high rmse for plot.

    Args:
        values (Any): Values to process.
        ceiling (float): float value used by the function.

    Returns:
        Any: Result of the function.
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.where(values <= ceiling)
    v = float(values)
    return float("nan") if v > ceiling else v


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load json if exists.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any] | None: Result of the function.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def parts_to_row(parts: ExperimentParts) -> dict[str, Any]:
    """Parts to row.

    Args:
        parts (ExperimentParts): ExperimentParts value used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    return {
        "dataset_id": parts.dataset_id,
        "dataset_number": parts.dataset_number,
        "model": parts.model,
        "model_label": MODEL_LABELS.get(parts.model, parts.model),
        "scope": parts.scope,
        "scope_label": SCOPE_LABELS.get(parts.scope, parts.scope),
        "family_combo": parts.family_combo,
        "family_label": FAMILY_LABELS.get(parts.family_combo, parts.family_combo),
        "target_metric": parts.target_metric,
    }


def target_metric_values(*dfs: pd.DataFrame) -> list[str]:
    """Target metric values.

    Args:
        *dfs (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    target_metrics: set[str] = set()
    for df in dfs:
        if df.empty or "target_metric" not in df.columns:
            continue
        target_metrics.update(str(value) for value in df["target_metric"].dropna().unique())
    return sorted(target_metrics)


def filter_by_target_metric(df: pd.DataFrame, target_metric: str) -> pd.DataFrame:
    """Filter by target metric.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if df.empty or "target_metric" not in df.columns:
        return df.copy()
    return df[df["target_metric"] == target_metric].copy()


def select_metric_completed_runs(completed_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    """Select metric completed runs.

    Args:
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        expected_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if completed_df.empty:
        return completed_df.copy()
    if expected_df.empty:
        return completed_df.copy()
    return completed_df[completed_df["experiment_name"].isin(expected_df["experiment_name"])].copy()


def select_latest_runs(completed_all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select latest runs.

    Args:
        completed_all_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Result of the function.
    """
    sorted_df = completed_all_df.sort_values(["experiment_name", "run_started_at"])
    subset = ["experiment_name"]
    if "source_outputs_root" in sorted_df.columns:
        subset = ["source_outputs_root", "experiment_name"]
    duplicate_mask = sorted_df.duplicated(subset=subset, keep="last")
    duplicates_df = sorted_df[duplicate_mask].copy()
    latest_df = sorted_df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    return latest_df, duplicates_df.reset_index(drop=True)


def add_rank_columns(completed_df: pd.DataFrame) -> pd.DataFrame:
    """Add rank columns.

    Args:
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    df = completed_df.copy()
    metric_specs = {"r2": False, "rmse": True, "mae": True}
    global_group_columns = [column for column in ["target_metric"] if column in df.columns]
    dataset_group_columns = [column for column in ["target_metric", "dataset_id"] if column in df.columns]

    for metric_name, ascending in metric_specs.items():
        if global_group_columns:
            df[f"{metric_name}_rank_global"] = (
                df.groupby(global_group_columns)[metric_name].rank(ascending=ascending, method="average")
            )
        else:
            df[f"{metric_name}_rank_global"] = df[metric_name].rank(ascending=ascending, method="average")

        if dataset_group_columns:
            df[f"{metric_name}_rank_dataset"] = (
                df.groupby(dataset_group_columns)[metric_name].rank(ascending=ascending, method="average")
            )
        else:
            df[f"{metric_name}_rank_dataset"] = df[metric_name].rank(ascending=ascending, method="average")

    df["rank_score_global"] = df[[f"{metric}_rank_global" for metric in metric_specs]].mean(axis=1)
    df["rank_score_dataset"] = df[[f"{metric}_rank_dataset" for metric in metric_specs]].mean(axis=1)
    return df.sort_values(["rank_score_global", "r2", "rmse"], ascending=[True, False, True]).reset_index(drop=True)


def build_completion_table(expected_df: pd.DataFrame, completed_df: pd.DataFrame) -> pd.DataFrame:
    """Build completion table.

    Args:
        expected_df (pd.DataFrame): pd.DataFrame value used by the function.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if expected_df.empty:
        completion_columns = [
            "experiment_name",
            "dataset_id",
            "dataset_number",
            "model",
            "model_label",
            "scope",
            "scope_label",
            "family_combo",
            "family_label",
            "target_metric",
        ]
        available_columns = [column for column in completion_columns if column in completed_df.columns]
        completion_df = completed_df[available_columns].copy()
        if "model_label" not in completion_df.columns and "model" in completion_df.columns:
            completion_df["model_label"] = completion_df["model"].map(MODEL_LABELS).fillna(completion_df["model"])
        if "scope_label" not in completion_df.columns and "scope" in completion_df.columns:
            completion_df["scope_label"] = completion_df["scope"].map(SCOPE_LABELS).fillna(completion_df["scope"])
        if "family_label" not in completion_df.columns and "family_combo" in completion_df.columns:
            completion_df["family_label"] = (
                completion_df["family_combo"].map(FAMILY_LABELS).fillna(completion_df["family_combo"])
            )
        completion_df["status"] = "completed"
        return completion_df

    completion_df = expected_df.merge(
        completed_df[
            [
                "experiment_name",
                "run_started_at",
                "run_dir",
                "r2",
                "rmse",
                "mae",
                "n_features",
                "n_train",
                "n_validation",
            ]
        ],
        on="experiment_name",
        how="left",
    )
    completion_df["status"] = np.where(completion_df["run_started_at"].notna(), "completed", "missing")
    return completion_df.sort_values(["target_metric", "dataset_number", "model", "scope", "family_combo"]).reset_index(
        drop=True
    )


def build_summary_tables(completed_df: pd.DataFrame, completion_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build summary tables.

    Args:
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        dict[str, pd.DataFrame]: Result of the function.
    """
    best_overall = (
        completed_df.sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).head(1).reset_index(drop=True)
    )
    best_per_dataset = best_rows(completed_df, ["dataset_id"])
    top_runs = completed_df.sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).reset_index(drop=True)

    model_best_per_dataset = best_rows(completed_df, ["dataset_id", "model"])
    family_best_per_dataset = best_rows(completed_df, ["dataset_id", "family_combo"])
    scope_best_per_dataset = best_rows(completed_df, ["dataset_id", "scope"])
    combo_best_per_dataset = best_rows(completed_df, ["dataset_id", "model", "scope", "family_combo"])

    model_summary = aggregate_group_summary(model_best_per_dataset, ["model", "model_label"])
    family_summary = aggregate_group_summary(family_best_per_dataset, ["family_combo", "family_label"])
    scope_summary = aggregate_group_summary(scope_best_per_dataset, ["scope", "scope_label"])
    combo_summary = aggregate_group_summary(combo_best_per_dataset, ["model", "scope", "family_combo"])

    completion_by_dataset = (
        completion_df.groupby(["dataset_id", "status"]).size().unstack(fill_value=0).reset_index()
        if not completion_df.empty
        else pd.DataFrame(columns=["dataset_id", "completed", "missing"])
    )
    if not completion_by_dataset.empty:
        dataset_metadata = build_dataset_metadata(completed_df, completion_df)
        for column in ["completed", "missing"]:
            if column not in completion_by_dataset:
                completion_by_dataset[column] = 0
        completion_by_dataset = completion_by_dataset.merge(dataset_metadata, on="dataset_id", how="left")
        completion_by_dataset["total"] = completion_by_dataset["completed"] + completion_by_dataset["missing"]
        completion_by_dataset["completion_rate"] = np.where(
            completion_by_dataset["total"] > 0,
            completion_by_dataset["completed"] / completion_by_dataset["total"],
            np.nan,
        )
        completion_by_dataset = completion_by_dataset.sort_values(
            ["dataset_number", "dataset_display_label", "dataset_id"], na_position="last"
        ).reset_index(drop=True)

    completion_by_model = (
        completion_df.groupby(["model", "status"]).size().unstack(fill_value=0).reset_index()
        if not completion_df.empty
        else pd.DataFrame(columns=["model", "completed", "missing"])
    )
    if not completion_by_model.empty:
        for column in ["completed", "missing"]:
            if column not in completion_by_model:
                completion_by_model[column] = 0
        completion_by_model["total"] = completion_by_model["completed"] + completion_by_model["missing"]
        completion_by_model["completion_rate"] = np.where(
            completion_by_model["total"] > 0,
            completion_by_model["completed"] / completion_by_model["total"],
            np.nan,
        )
        completion_by_model["model_label"] = completion_by_model["model"].map(MODEL_LABELS).fillna(completion_by_model["model"])
        completion_by_model = completion_by_model.sort_values("model").reset_index(drop=True)

    return {
        "best_overall": best_overall,
        "best_per_dataset": best_per_dataset,
        "top_runs": top_runs,
        "model_best_per_dataset": model_best_per_dataset,
        "family_best_per_dataset": family_best_per_dataset,
        "scope_best_per_dataset": scope_best_per_dataset,
        "combo_best_per_dataset": combo_best_per_dataset,
        "model_summary": model_summary,
        "family_summary": family_summary,
        "scope_summary": scope_summary,
        "combo_summary": combo_summary,
        "completion_by_dataset": completion_by_dataset,
        "completion_by_model": completion_by_model,
    }


def best_rows(df: pd.DataFrame, grouping_columns: list[str], *, metric_name: str = "r2") -> pd.DataFrame:
    """Best rows.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.
        grouping_columns (list[str]): list[str] value used by the function.
        metric_name (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if metric_name == "r2":
        sort_columns = grouping_columns + ["r2", "rmse", "mae"]
        ascending = [True] * len(grouping_columns) + [False, True, True]
    elif metric_name == "rmse":
        sort_columns = grouping_columns + ["rmse", "mae", "r2"]
        ascending = [True] * len(grouping_columns) + [True, True, False]
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    ordered = df.sort_values(sort_columns, ascending=ascending)
    return ordered.drop_duplicates(subset=grouping_columns, keep="first").reset_index(drop=True)


def aggregate_group_summary(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate group summary.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.
        group_columns (list[str]): list[str] value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(group_columns, dropna=False)
        .agg(
            datasets_covered=("dataset_id", "nunique"),
            wins=("dataset_id", "size"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_mae=("mae", "mean"),
            mean_rank=("rank_score_dataset", "mean"),
            mean_features=("n_features", "mean"),
        )
        .reset_index()
        .sort_values(["mean_rank", "mean_r2", "mean_rmse"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return summary


def write_tables(
    output_dir: Path,
    completed_all_df: pd.DataFrame,
    completed_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Write tables.

    Args:
        output_dir (Path): Output directory for artifacts.
        completed_all_df (pd.DataFrame): pd.DataFrame value used by the function.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        duplicates_df (pd.DataFrame): pd.DataFrame value used by the function.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.
        tables (dict[str, pd.DataFrame]): dict[str, pd.DataFrame] value used by the function.

    Returns:
        None: None.
    """
    completed_all_df.to_csv(output_dir / "completed_runs_all.csv", index=False)
    completed_df.to_csv(output_dir / "completed_runs_latest.csv", index=False)
    duplicates_df.to_csv(output_dir / "duplicate_runs_ignored.csv", index=False)
    completion_df.to_csv(output_dir / "completion_status.csv", index=False)

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def write_metric_index_report(
    output_dir: Path,
    outputs_root: Path,
    batch_files: list[Path],
    metric_report_rows: list[dict[str, Any]],
) -> None:
    """Write metric index report.

    Args:
        output_dir (Path): Output directory for artifacts.
        outputs_root (Path): Path value used by the function.
        batch_files (list[Path]): list[Path] value used by the function.
        metric_report_rows (list[dict[str, Any]]): list[dict[str, Any]] value used by the function.

    Returns:
        None: None.
    """
    summary_df = pd.DataFrame(metric_report_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("target_metric").reset_index(drop=True)
    summary_df.to_csv(output_dir / "metric_reports.csv", index=False)

    lines = [
        "# Metric Prediction Batch Evaluation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Outputs root: `{outputs_root}`",
        f"Batch files: {', '.join(f'`{path}`' for path in batch_files if path.exists()) or 'not provided'}",
        "",
        "## Metrics",
        "",
        "```text",
        format_table_for_text(summary_df) if not summary_df.empty else "(no rows)",
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_report(
    output_dir: Path,
    outputs_root: Path,
    batch_files: list[Path],
    completed_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    top_n: int,
    *,
    target_metric: str,
) -> None:
    """Write summary report.

    Args:
        output_dir (Path): Output directory for artifacts.
        outputs_root (Path): Path value used by the function.
        batch_files (list[Path]): list[Path] value used by the function.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.
        tables (dict[str, pd.DataFrame]): dict[str, pd.DataFrame] value used by the function.
        top_n (int): int value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    best_overall = tables["best_overall"].iloc[0]
    completion_counts = completion_df["status"].value_counts() if not completion_df.empty else pd.Series(dtype=int)
    completed_count = int(completion_counts.get("completed", len(completed_df)))
    missing_count = int(completion_counts.get("missing", 0))
    total_expected = completed_count + missing_count
    completion_rate = completed_count / total_expected if total_expected else 1.0

    lines = [
        f"# Metric Prediction Batch Evaluation: {target_metric}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Outputs root: `{outputs_root}`",
        f"Batch files: {', '.join(f'`{path}`' for path in batch_files if path.exists()) or 'not provided'}",
        "",
        "## Coverage",
        "",
        f"- Completed runs matched to the selected batches: {completed_count}",
        f"- Missing expected runs: {missing_count}",
        f"- Completion rate: {completion_rate:.1%}",
        "",
        "## Best Overall Completed Run",
        "",
        f"- Experiment: `{best_overall['experiment_name']}`",
        f"- Dataset: `{dataset_display_label_from_values(best_overall['dataset_label'], best_overall['dataset_id'])}` (`{best_overall['dataset_id']}`)",
        f"- Model: {best_overall['model_label']}",
        f"- Feature scope: {best_overall['scope_label']}",
        f"- Metric families: {best_overall['family_label']}",
        f"- Validation metrics: R2={best_overall['r2']:.4f}, RMSE={best_overall['rmse']:.4f}, MAE={best_overall['mae']:.4f}",
        "",
        "## Best Run Per Dataset",
        "",
        "```text",
        format_table_for_text(
            add_dataset_display_columns(tables["best_per_dataset"])[
                [
                    "dataset_display_label",
                    "dataset_id",
                    "experiment_name",
                    "model_label",
                    "scope_label",
                    "family_label",
                    "r2",
                    "rmse",
                    "mae",
                    "n_features",
                ]
            ].rename(columns={"dataset_display_label": "dataset"})
        ),
        "```",
        "",
        "## Model Summary",
        "",
        "Best available configuration per dataset, then averaged across datasets.",
        "",
        "```text",
        format_table_for_text(tables["model_summary"]),
        "```",
        "",
        "## Metric Family Combination Summary",
        "",
        "Best available model and scope per dataset, then averaged across datasets.",
        "",
        "```text",
        format_table_for_text(tables["family_summary"]),
        "```",
        "",
        "## Feature Scope Summary",
        "",
        "```text",
        format_table_for_text(tables["scope_summary"]),
        "```",
        "",
        f"## Top {top_n} Completed Runs",
        "",
        "```text",
        format_table_for_text(
            add_dataset_display_columns(tables["top_runs"].head(top_n))[
                [
                    "dataset_display_label",
                    "experiment_name",
                    "dataset_id",
                    "model_label",
                    "scope_label",
                    "family_label",
                    "r2",
                    "rmse",
                    "mae",
                    "n_features",
                ]
            ].rename(columns={"dataset_display_label": "dataset"})
        ),
        "```",
    ]

    if missing_count:
        lines.extend(
            [
                "",
                "## Completion by Dataset",
                "",
                "```text",
                format_table_for_text(tables["completion_by_dataset"]),
                "```",
            ]
        )

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def format_table_for_text(df: pd.DataFrame) -> str:
    """Format table for text.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        str: Resolved string value.
    """
    if df.empty:
        return "(no rows)"

    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return formatted.to_string(index=False)


def write_per_dataset_reports(
    output_dir: Path,
    completed_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    top_n: int,
    target_metric: str,
) -> None:
    """Write per dataset reports.

    Args:
        output_dir (Path): Output directory for artifacts.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.
        top_n (int): int value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    per_dataset_root = output_dir / "per_dataset"
    per_dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_metadata = build_dataset_metadata(completed_df, completion_df)
    for dataset in dataset_metadata.itertuples():
        dataset_output_dir = per_dataset_root / str(dataset.dataset_dir_name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        write_single_dataset_report(
            dataset_output_dir,
            str(dataset.dataset_id),
            str(dataset.dataset_display_label),
            completed_df,
            completion_df,
            top_n,
            target_metric,
        )


def replace_output_path(path: Path) -> None:
    """Replace output path.

    Args:
        path (Path): Path to process.

    Returns:
        None: None.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def copy_best_run_artifacts(dataset_output_dir: Path, artifacts_dir: Path | None) -> None:
    """Copy best run artifacts.

    Args:
        dataset_output_dir (Path): Path value used by the function.
        artifacts_dir (Path | None): Directory for artifacts.

    Returns:
        None: None.
    """
    target_artifacts_dir = dataset_output_dir / "artifacts"
    replace_output_path(target_artifacts_dir)
    if artifacts_dir is None or not artifacts_dir.is_dir():
        return
    shutil.copytree(artifacts_dir, target_artifacts_dir)


def write_single_dataset_report(
    dataset_output_dir: Path,
    dataset_id: str,
    dataset_display_label: str,
    completed_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    top_n: int,
    target_metric: str,
) -> None:
    """Write single dataset report.

    Args:
        dataset_output_dir (Path): Path value used by the function.
        dataset_id (str): str value used by the function.
        dataset_display_label (str): str value used by the function.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.
        top_n (int): int value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    dataset_completed = completed_df[completed_df["dataset_id"] == dataset_id].copy()
    dataset_expected = completion_df[completion_df["dataset_id"] == dataset_id].copy()

    dataset_completed.to_csv(dataset_output_dir / "completed_runs.csv", index=False)
    dataset_expected.to_csv(dataset_output_dir / "completion_status.csv", index=False)

    if dataset_completed.empty:
        copy_best_run_artifacts(dataset_output_dir, None)
        lines = [
            f"# {dataset_display_label} Evaluation: {target_metric}",
            "",
            f"Dataset id: `{dataset_id}`",
            "",
            "No completed runs found for this dataset.",
        ]
        (dataset_output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return

    best_run = dataset_completed.sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).iloc[0]
    model_best = best_rows(dataset_completed, ["model"])
    family_best = best_rows(dataset_completed, ["family_combo"])
    scope_best = best_rows(dataset_completed, ["scope"])
    model_scope_best = best_rows(dataset_completed, ["model", "scope"])
    model_family_best = best_rows(dataset_completed, ["model", "family_combo"])
    family_scope_best = best_rows(dataset_completed, ["family_combo", "scope"])
    dataset_top = dataset_completed.sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).head(top_n)

    model_summary = summarize_single_dataset(model_best, ["model", "model_label"])
    family_summary = summarize_single_dataset(family_best, ["family_combo", "family_label"])
    scope_summary = summarize_single_dataset(scope_best, ["scope", "scope_label"])
    model_scope_summary = summarize_single_dataset(model_scope_best, ["model", "scope"])
    model_family_summary = summarize_single_dataset(model_family_best, ["model", "family_combo"])
    family_scope_summary = summarize_single_dataset(family_scope_best, ["family_combo", "scope"])

    model_best.to_csv(dataset_output_dir / "best_by_model.csv", index=False)
    family_best.to_csv(dataset_output_dir / "best_by_family.csv", index=False)
    scope_best.to_csv(dataset_output_dir / "best_by_scope.csv", index=False)
    model_scope_summary.to_csv(dataset_output_dir / "best_by_model_scope.csv", index=False)
    model_family_summary.to_csv(dataset_output_dir / "best_by_model_family.csv", index=False)
    family_scope_summary.to_csv(dataset_output_dir / "best_by_family_scope.csv", index=False)
    dataset_top.to_csv(dataset_output_dir / "top_runs.csv", index=False)
    model_summary.to_csv(dataset_output_dir / "model_summary.csv", index=False)
    family_summary.to_csv(dataset_output_dir / "family_summary.csv", index=False)
    scope_summary.to_csv(dataset_output_dir / "scope_summary.csv", index=False)
    copy_best_run_artifacts(dataset_output_dir, Path(str(best_run["artifacts_dir"])))

    completed_count = len(dataset_expected[dataset_expected["status"] == "completed"]) if not dataset_expected.empty else len(dataset_completed)
    missing_count = len(dataset_expected[dataset_expected["status"] == "missing"]) if not dataset_expected.empty else 0
    total_count = completed_count + missing_count
    completion_rate = completed_count / total_count if total_count else 1.0

    lines = [
        f"# {dataset_display_label} Evaluation: {target_metric}",
        "",
        f"Dataset id: `{dataset_id}`",
        f"Dataset label: `{dataset_display_label}`",
        "",
        "## Coverage",
        "",
        f"- Completed runs: {completed_count}",
        f"- Missing expected runs: {missing_count}",
        f"- Completion rate: {completion_rate:.1%}",
        "",
        "## Best Completed Run",
        "",
        f"- Experiment: `{best_run['experiment_name']}`",
        f"- Model: {best_run['model_label']}",
        f"- Feature scope: {best_run['scope_label']}",
        f"- Metric families: {best_run['family_label']}",
        f"- Validation metrics: R2={best_run['r2']:.4f}, RMSE={best_run['rmse']:.4f}, MAE={best_run['mae']:.4f}",
        "",
        "## Best By Model",
        "",
        "```text",
        format_table_for_text(
            model_best[
                ["model_label", "experiment_name", "scope_label", "family_label", "r2", "rmse", "mae", "n_features"]
            ]
        ),
        "```",
        "",
        "## Best By Metric Family Combination",
        "",
        "```text",
        format_table_for_text(
            family_best[
                ["family_label", "experiment_name", "model_label", "scope_label", "r2", "rmse", "mae", "n_features"]
            ]
        ),
        "```",
        "",
        "## Best By Feature Scope",
        "",
        "```text",
        format_table_for_text(
            scope_best[
                ["scope_label", "experiment_name", "model_label", "family_label", "r2", "rmse", "mae", "n_features"]
            ]
        ),
        "```",
        "",
        f"## Top {min(top_n, len(dataset_top))} Completed Runs",
        "",
        "```text",
        format_table_for_text(
            dataset_top[
                ["experiment_name", "model_label", "scope_label", "family_label", "r2", "rmse", "mae", "n_features"]
            ]
        ),
        "```",
    ]

    (dataset_output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    render_single_dataset_plots(
        dataset_output_dir,
        dataset_display_label,
        dataset_completed,
        dataset_expected,
        model_best,
        family_best,
        scope_best,
        model_scope_best,
        model_family_best,
        family_scope_best,
        dataset_top,
        target_metric,
    )


def summarize_single_dataset(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize single dataset.

    Args:
        df (pd.DataFrame): pd.DataFrame value used by the function.
        group_columns (list[str]): list[str] value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(group_columns, dropna=False)
        .agg(
            best_r2=("r2", "max"),
            best_rmse=("rmse", "min"),
            best_mae=("mae", "min"),
            mean_features=("n_features", "mean"),
        )
        .reset_index()
        .sort_values(["best_r2", "best_rmse", "best_mae"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def render_plots(
    output_dir: Path,
    completed_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    top_n: int,
    target_metric: str,
) -> None:
    """Render plots.

    Args:
        output_dir (Path): Output directory for artifacts.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.
        tables (dict[str, pd.DataFrame]): dict[str, pd.DataFrame] value used by the function.
        top_n (int): int value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    plot_completion_overview(output_dir, tables["completion_by_dataset"], tables["completion_by_model"], target_metric)
    plot_top_runs(output_dir, tables["top_runs"], top_n, target_metric)
    plot_best_model_heatmap(output_dir, tables["model_best_per_dataset"], target_metric)
    plot_best_model_rmse_heatmap(output_dir, best_rows(completed_df, ["dataset_id", "model"], metric_name="rmse"), target_metric)
    plot_best_family_heatmap(output_dir, tables["family_best_per_dataset"], target_metric)
    plot_best_family_rmse_heatmap(
        output_dir,
        best_rows(completed_df, ["dataset_id", "family_combo"], metric_name="rmse"),
        target_metric,
    )
    plot_best_scope_heatmap(output_dir, tables["scope_best_per_dataset"], target_metric)
    plot_best_scope_rmse_heatmap(output_dir, best_rows(completed_df, ["dataset_id", "scope"], metric_name="rmse"), target_metric)
    plot_feature_count_scatter(output_dir, completed_df, target_metric)
    if not completion_df.empty and "status" in completion_df:
        plot_completion_heatmap(output_dir, completion_df, target_metric)


def render_single_dataset_plots(
    dataset_output_dir: Path,
    dataset_display_label: str,
    dataset_completed: pd.DataFrame,
    dataset_expected: pd.DataFrame,
    model_best: pd.DataFrame,
    family_best: pd.DataFrame,
    scope_best: pd.DataFrame,
    model_scope_best: pd.DataFrame,
    model_family_best: pd.DataFrame,
    family_scope_best: pd.DataFrame,
    dataset_top: pd.DataFrame,
    target_metric: str,
) -> None:
    """Render single dataset plots.

    Args:
        dataset_output_dir (Path): Path value used by the function.
        dataset_display_label (str): str value used by the function.
        dataset_completed (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_expected (pd.DataFrame): pd.DataFrame value used by the function.
        model_best (pd.DataFrame): pd.DataFrame value used by the function.
        family_best (pd.DataFrame): pd.DataFrame value used by the function.
        scope_best (pd.DataFrame): pd.DataFrame value used by the function.
        model_scope_best (pd.DataFrame): pd.DataFrame value used by the function.
        model_family_best (pd.DataFrame): pd.DataFrame value used by the function.
        family_scope_best (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_top (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    plot_dataset_best_bars(dataset_output_dir, dataset_display_label, model_best, family_best, scope_best, target_metric)
    plot_dataset_top_runs(dataset_output_dir, dataset_display_label, dataset_top, target_metric)
    plot_dataset_heatmap(
        dataset_output_dir / "03_model_scope_heatmap_r2.png",
        model_scope_best,
        index="model_label",
        columns="scope_label",
        title=with_target_metric(with_plot_r2(f"{dataset_display_label}: best R2 by model and feature scope"), target_metric),
        metric_name="r2",
        value_label=R2_PLOT_LABEL,
        clip_negative=True,
    )
    plot_dataset_heatmap(
        dataset_output_dir / "04_model_family_heatmap_r2.png",
        model_family_best,
        index="model_label",
        columns="family_label",
        title=with_target_metric(
            with_plot_r2(f"{dataset_display_label}: best R2 by model and metric-family combination"),
            target_metric,
        ),
        metric_name="r2",
        value_label=R2_PLOT_LABEL,
        clip_negative=True,
    )
    plot_dataset_heatmap(
        dataset_output_dir / "05_family_scope_heatmap_r2.png",
        family_scope_best,
        index="family_label",
        columns="scope_label",
        title=with_target_metric(
            with_plot_r2(f"{dataset_display_label}: best R2 by metric-family combination and scope"),
            target_metric,
        ),
        metric_name="r2",
        value_label=R2_PLOT_LABEL,
        clip_negative=True,
    )
    plot_dataset_heatmap(
        dataset_output_dir / "03_model_scope_heatmap_rmse.png",
        best_rows(dataset_completed, ["model", "scope"], metric_name="rmse"),
        index="model_label",
        columns="scope_label",
        title=with_target_metric(f"{dataset_display_label}: best RMSE by model and feature scope", target_metric),
        metric_name="rmse",
        value_label=RMSE_PLOT_LABEL,
        clip_rmse_ceiling=1.0,
        reverse_cmap=True,
    )
    plot_dataset_heatmap(
        dataset_output_dir / "04_model_family_heatmap_rmse.png",
        best_rows(dataset_completed, ["model", "family_combo"], metric_name="rmse"),
        index="model_label",
        columns="family_label",
        title=with_target_metric(
            f"{dataset_display_label}: best RMSE by model and metric-family combination",
            target_metric,
        ),
        metric_name="rmse",
        value_label=RMSE_PLOT_LABEL,
        clip_rmse_ceiling=1.0,
        reverse_cmap=True,
    )
    plot_dataset_heatmap(
        dataset_output_dir / "05_family_scope_heatmap_rmse.png",
        best_rows(dataset_completed, ["family_combo", "scope"], metric_name="rmse"),
        index="family_label",
        columns="scope_label",
        title=with_target_metric(
            f"{dataset_display_label}: best RMSE by metric-family combination and scope",
            target_metric,
        ),
        metric_name="rmse",
        value_label=RMSE_PLOT_LABEL,
        clip_rmse_ceiling=1.0,
        reverse_cmap=True,
    )
    if not dataset_expected.empty:
        plot_dataset_completion(dataset_output_dir, dataset_display_label, dataset_expected, target_metric)


def annotate_vertical_bars(
    ax: matplotlib.axes.Axes,
    bars: Any,
    *,
    fmt: str,
    padding_fraction: float = 0.01,
    fontsize: int = 8,
) -> None:
    """Annotate vertical bars.

    Args:
        ax (matplotlib.axes.Axes): matplotlib.axes.Axes value used by the function.
        bars (Any): Value used by the function.
        fmt (str): str value used by the function.
        padding_fraction (float): float value used by the function.
        fontsize (int): int value used by the function.

    Returns:
        None: None.
    """
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min if y_max > y_min else 1.0
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + span * padding_fraction,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def annotate_horizontal_bars(
    ax: matplotlib.axes.Axes,
    bars: Any,
    *,
    fmt: str,
    padding_fraction: float = 0.01,
    fontsize: int = 8,
) -> None:
    """Annotate horizontal bars.

    Args:
        ax (matplotlib.axes.Axes): matplotlib.axes.Axes value used by the function.
        bars (Any): Value used by the function.
        fmt (str): str value used by the function.
        padding_fraction (float): float value used by the function.
        fontsize (int): int value used by the function.

    Returns:
        None: None.
    """
    x_min, x_max = ax.get_xlim()
    span = x_max - x_min if x_max > x_min else 1.0
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + span * padding_fraction,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            ha="left",
            va="center",
            fontsize=fontsize,
        )


def plot_dataset_best_bars(
    dataset_output_dir: Path,
    dataset_display_label: str,
    model_best: pd.DataFrame,
    family_best: pd.DataFrame,
    scope_best: pd.DataFrame,
    target_metric: str,
) -> None:
    """Plot dataset best bars.

    Args:
        dataset_output_dir (Path): Path value used by the function.
        dataset_display_label (str): str value used by the function.
        model_best (pd.DataFrame): pd.DataFrame value used by the function.
        family_best (pd.DataFrame): pd.DataFrame value used by the function.
        scope_best (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Target metric: {target_metric}")
    model_r2 = clip_negative_r2_for_plot(model_best["r2"])
    family_r2 = clip_negative_r2_for_plot(family_best["r2"])
    scope_r2 = clip_negative_r2_for_plot(scope_best["r2"])

    model_bars = axes[0].bar(
        model_best["model_label"],
        model_r2,
        color=[MODEL_COLORS.get(model, "#8d99ae") for model in model_best["model"]],
    )
    axes[0].set_title(f"{dataset_display_label}: best model")
    axes[0].set_ylabel(with_plot_r2("Validation R2"))
    axes[0].set_ylim(bottom=0)
    annotate_vertical_bars(axes[0], model_bars, fmt="{:.4f}")

    family_bars = axes[1].bar(family_best["family_label"], family_r2, color="#457b9d")
    axes[1].set_title(f"{dataset_display_label}: best metric-family combination")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(bottom=0)
    annotate_vertical_bars(axes[1], family_bars, fmt="{:.4f}")

    scope_bars = axes[2].bar(scope_best["scope_label"], scope_r2, color="#e9c46a")
    axes[2].set_title(f"{dataset_display_label}: best feature scope")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_ylim(bottom=0)
    annotate_vertical_bars(axes[2], scope_bars, fmt="{:.4f}")

    fig.savefig(dataset_output_dir / "01_best_category_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_top_runs(
    dataset_output_dir: Path,
    dataset_display_label: str,
    dataset_top: pd.DataFrame,
    target_metric: str,
) -> None:
    """Plot dataset top runs.

    Args:
        dataset_output_dir (Path): Path value used by the function.
        dataset_display_label (str): str value used by the function.
        dataset_top (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if dataset_top.empty:
        return

    plot_df = dataset_top.copy()
    plot_df["plot_r2"] = clip_negative_r2_for_plot(plot_df["r2"])
    plot_df = plot_df.sort_values(["plot_r2", "r2"], ascending=[True, True])
    labels = [
        f"{row.model_label} | {row.scope_label} | {row.family_label}"
        for row in plot_df.itertuples()
    ]
    colors = [MODEL_COLORS.get(model, "#8d99ae") for model in plot_df["model"]]

    fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * len(plot_df))))
    bars = ax.barh(labels, plot_df["plot_r2"], color=colors)
    ax.set_xlabel(with_plot_r2("Validation R2"))
    ax.set_xlim(left=0)
    ax.set_title(with_target_metric(f"{dataset_display_label}: top completed runs", target_metric))
    annotate_horizontal_bars(ax, bars, fmt="{:.4f}")
    fig.savefig(dataset_output_dir / "02_top_runs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_heatmap(
    output_path: Path,
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    title: str,
    metric_name: str,
    value_label: str,
    clip_negative: bool = False,
    clip_rmse_ceiling: float | None = None,
    reverse_cmap: bool = False,
) -> None:
    """Plot dataset heatmap.

    Args:
        output_path (Path): Path value used by the function.
        frame (pd.DataFrame): Data frame to process.
        index (str): Index value used by the function.
        columns (str): Columns to process.
        title (str): str value used by the function.
        metric_name (str): str value used by the function.
        value_label (str): str value used by the function.
        clip_negative (bool): bool value used by the function.
        clip_rmse_ceiling (float | None): float | None value used by the function.
        reverse_cmap (bool): bool value used by the function.

    Returns:
        None: None.
    """
    if frame.empty:
        return
    pivot = frame.pivot(index=index, columns=columns, values=metric_name).sort_index()
    if clip_negative:
        pivot = clip_negative_r2_for_plot(pivot)
    if clip_rmse_ceiling is not None:
        pivot = mask_high_rmse_for_plot(pivot, ceiling=clip_rmse_ceiling)
    draw_heatmap(
        pivot,
        output_path,
        title=title,
        value_label=value_label,
        reverse_cmap=reverse_cmap,
    )


def plot_dataset_completion(
    dataset_output_dir: Path,
    dataset_display_label: str,
    dataset_expected: pd.DataFrame,
    target_metric: str,
) -> None:
    """Plot dataset completion.

    Args:
        dataset_output_dir (Path): Path value used by the function.
        dataset_display_label (str): str value used by the function.
        dataset_expected (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    plot_df = dataset_expected.copy()
    plot_df["completed"] = (plot_df["status"] == "completed").astype(int)
    plot_df["row_label"] = plot_df["model_label"] + " | " + plot_df["scope_label"]
    plot_df["column_label"] = plot_df["family_label"]

    completion_matrix = (
        plot_df.pivot_table(
            index="row_label",
            columns="column_label",
            values="completed",
            aggfunc="max",
        )
        .sort_index()
    )
    draw_heatmap(
        completion_matrix,
        dataset_output_dir / "06_completion_matrix.png",
        title=with_target_metric(f"{dataset_display_label}: completion matrix", target_metric),
        value_label="Completed",
        value_format="{:.0f}",
        vmin=0,
        vmax=1,
    )


def plot_completion_overview(
    output_dir: Path,
    completion_by_dataset: pd.DataFrame,
    completion_by_model: pd.DataFrame,
    target_metric: str,
) -> None:
    """Plot completion overview.

    Args:
        output_dir (Path): Output directory for artifacts.
        completion_by_dataset (pd.DataFrame): pd.DataFrame value used by the function.
        completion_by_model (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if completion_by_dataset.empty or completion_by_model.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Target metric: {target_metric}")

    x = np.arange(len(completion_by_dataset))
    completed = completion_by_dataset["completed"].to_numpy()
    missing = completion_by_dataset["missing"].to_numpy()
    completed_bars = axes[0].bar(x, completed, color="#2a9d8f", label="Completed")
    missing_bars = axes[0].bar(x, missing, bottom=completed, color="#8d99ae", label="Missing")
    dataset_tick_labels = (
        completion_by_dataset["dataset_display_label"]
        if "dataset_display_label" in completion_by_dataset.columns
        else completion_by_dataset["dataset_id"]
    )
    axes[0].set_xticks(x, dataset_tick_labels)
    axes[0].set_ylabel("Experiments")
    axes[0].set_title("Completion by dataset")
    axes[0].legend()
    annotate_vertical_bars(axes[0], completed_bars, fmt="{:.0f}")
    annotate_vertical_bars(axes[0], missing_bars, fmt="{:.0f}")

    model_rates = completion_by_model["completion_rate"].to_numpy()
    model_rate_bars = axes[1].bar(
        completion_by_model["model_label"],
        model_rates,
        color=[MODEL_COLORS.get(model, "#8d99ae") for model in completion_by_model["model"]],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Completion rate")
    axes[1].set_title("Completion by model")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    annotate_vertical_bars(axes[1], model_rate_bars, fmt="{:.1%}")

    fig.savefig(output_dir / "01_completion_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_runs(output_dir: Path, top_runs: pd.DataFrame, top_n: int, target_metric: str) -> None:
    """Plot top runs.

    Args:
        output_dir (Path): Output directory for artifacts.
        top_runs (pd.DataFrame): pd.DataFrame value used by the function.
        top_n (int): int value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if top_runs.empty:
        return

    plot_df = add_dataset_display_columns(top_runs.head(top_n))
    plot_df["plot_r2"] = clip_negative_r2_for_plot(plot_df["r2"])
    plot_df = plot_df.sort_values(["plot_r2", "r2"], ascending=[True, True])
    labels = [
        f"{row.dataset_display_label} | {row.model_label} | {row.scope_label} | {row.family_label}"
        for row in plot_df.itertuples()
    ]
    colors = [MODEL_COLORS.get(model, "#8d99ae") for model in plot_df["model"]]

    fig, ax = plt.subplots(figsize=(14, max(6, 0.4 * len(plot_df))))
    bars = ax.barh(labels, plot_df["plot_r2"], color=colors)
    ax.set_xlabel(with_plot_r2("Validation R2"))
    ax.set_xlim(left=0)
    ax.set_title(with_target_metric(with_plot_r2(f"Top {len(plot_df)} completed runs by R2"), target_metric))

    x_min, x_max = ax.get_xlim()
    span = x_max - x_min if x_max > x_min else 1.0
    for bar, r2, rmse, mae in zip(bars, plot_df["plot_r2"], plot_df["rmse"], plot_df["mae"], strict=True):
        ax.text(
            r2 + span * 0.01,
            bar.get_y() + bar.get_height() / 2,
            rf"{R2_PLOT_LABEL}={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}",
            va="center",
            fontsize=8,
        )

    fig.savefig(output_dir / "02_top_runs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_model_heatmap(output_dir: Path, df: pd.DataFrame, target_metric: str) -> None:
    """Plot best model heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if df.empty:
        return
    display_df = add_dataset_display_columns(df)
    ordered_columns = [MODEL_LABELS[key] for key in MODEL_LABELS]
    pivot = display_df.pivot(index="dataset_display_label", columns="model_label", values="r2").reindex(dataset_display_labels(display_df))
    pivot = clip_negative_r2_for_plot(pivot[[col for col in ordered_columns if col in pivot.columns]])
    draw_heatmap(
        pivot,
        output_dir / "03_best_model_heatmap_r2.png",
        title=with_target_metric(with_plot_r2("Best available R2 per dataset and model"), target_metric),
        value_label=R2_PLOT_LABEL,
    )


def plot_best_model_rmse_heatmap(output_dir: Path, df: pd.DataFrame, target_metric: str) -> None:
    """Plot best model rmse heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if df.empty:
        return
    display_df = add_dataset_display_columns(df)
    ordered_columns = [MODEL_LABELS[key] for key in MODEL_LABELS]
    pivot = display_df.pivot(index="dataset_display_label", columns="model_label", values="rmse").reindex(dataset_display_labels(display_df))
    pivot = mask_high_rmse_for_plot(pivot[[col for col in ordered_columns if col in pivot.columns]])
    draw_heatmap(
        pivot,
        output_dir / "03_best_model_heatmap_rmse.png",
        title=with_target_metric("Best available RMSE per dataset and model", target_metric),
        value_label=RMSE_PLOT_LABEL,
        reverse_cmap=True,
    )


def plot_best_family_heatmap(output_dir: Path, df: pd.DataFrame, target_metric: str) -> None:
    """Plot best family heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if df.empty:
        return
    display_df = add_dataset_display_columns(df)
    ordered_columns = [FAMILY_LABELS[key] for key in FAMILY_LABELS]
    pivot = display_df.pivot(index="dataset_display_label", columns="family_label", values="r2").reindex(dataset_display_labels(display_df))
    pivot = clip_negative_r2_for_plot(pivot[[col for col in ordered_columns if col in pivot.columns]])
    draw_heatmap(
        pivot,
        output_dir / "04_best_family_heatmap_r2.png",
        title=with_target_metric(
            with_plot_r2("Best available R2 per dataset and metric-family combination"),
            target_metric,
        ),
        value_label=R2_PLOT_LABEL,
    )


def plot_best_family_rmse_heatmap(output_dir: Path, df: pd.DataFrame, target_metric: str) -> None:
    """Plot best family rmse heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if df.empty:
        return
    display_df = add_dataset_display_columns(df)
    ordered_columns = [FAMILY_LABELS[key] for key in FAMILY_LABELS]
    pivot = display_df.pivot(index="dataset_display_label", columns="family_label", values="rmse").reindex(dataset_display_labels(display_df))
    pivot = mask_high_rmse_for_plot(pivot[[col for col in ordered_columns if col in pivot.columns]])
    draw_heatmap(
        pivot,
        output_dir / "04_best_family_heatmap_rmse.png",
        title=with_target_metric(
            "Best available RMSE per dataset and metric-family combination",
            target_metric,
        ),
        value_label=RMSE_PLOT_LABEL,
        reverse_cmap=True,
    )


def plot_best_scope_heatmap(output_dir: Path, df: pd.DataFrame, target_metric: str) -> None:
    """Plot best scope heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if df.empty:
        return
    display_df = add_dataset_display_columns(df)
    ordered_columns = [SCOPE_LABELS[key] for key in SCOPE_LABELS]
    pivot = display_df.pivot(index="dataset_display_label", columns="scope_label", values="r2").reindex(dataset_display_labels(display_df))
    pivot = clip_negative_r2_for_plot(pivot[[col for col in ordered_columns if col in pivot.columns]])
    draw_heatmap(
        pivot,
        output_dir / "05_best_scope_heatmap_r2.png",
        title=with_target_metric(with_plot_r2("Best available R2 per dataset and feature scope"), target_metric),
        value_label=R2_PLOT_LABEL,
    )


def plot_best_scope_rmse_heatmap(output_dir: Path, df: pd.DataFrame, target_metric: str) -> None:
    """Plot best scope rmse heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if df.empty:
        return
    display_df = add_dataset_display_columns(df)
    ordered_columns = [SCOPE_LABELS[key] for key in SCOPE_LABELS]
    pivot = display_df.pivot(index="dataset_display_label", columns="scope_label", values="rmse").reindex(dataset_display_labels(display_df))
    pivot = mask_high_rmse_for_plot(pivot[[col for col in ordered_columns if col in pivot.columns]])
    draw_heatmap(
        pivot,
        output_dir / "05_best_scope_heatmap_rmse.png",
        title=with_target_metric("Best available RMSE per dataset and feature scope", target_metric),
        value_label=RMSE_PLOT_LABEL,
        reverse_cmap=True,
    )


def plot_feature_count_scatter(output_dir: Path, completed_df: pd.DataFrame, target_metric: str) -> None:
    """Plot feature count scatter.

    Args:
        output_dir (Path): Output directory for artifacts.
        completed_df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if completed_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in completed_df.groupby("model"):
        model = str(model)
        ax.scatter(
            group["n_features"],
            clip_negative_r2_for_plot(group["r2"]),
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#8d99ae"),
            alpha=0.8,
        )
    ax.set_xlabel("Number of fitted features")
    ax.set_ylabel(with_plot_r2("Validation R2"))
    ax.set_ylim(bottom=0)
    ax.set_title(with_target_metric(with_plot_r2("Feature count versus validation R2"), target_metric))
    ax.legend()

    fig.savefig(output_dir / "06_feature_count_vs_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_completion_heatmap(output_dir: Path, completion_df: pd.DataFrame, target_metric: str) -> None:
    """Plot completion heatmap.

    Args:
        output_dir (Path): Output directory for artifacts.
        completion_df (pd.DataFrame): pd.DataFrame value used by the function.
        target_metric (str): str value used by the function.

    Returns:
        None: None.
    """
    if completion_df.empty or "status" not in completion_df:
        return

    display_df = add_dataset_display_columns(completion_df)
    matrix = (
        display_df.assign(completed=lambda frame: (frame["status"] == "completed").astype(int))
        .pivot_table(
            index="dataset_display_label",
            columns="experiment_name",
            values="completed",
            aggfunc="max",
        )
        .reindex(dataset_display_labels(display_df))
    )

    if matrix.shape[1] > 60:
        return

    draw_heatmap(
        matrix,
        output_dir / "07_completion_heatmap.png",
        title=with_target_metric("Experiment completion matrix", target_metric),
        value_label="Completed",
        value_format="{:.0f}",
        vmin=0,
        vmax=1,
    )


def draw_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    value_label: str,
    value_format: str = "{:.3f}",
    vmin: float | None = None,
    vmax: float | None = None,
    reverse_cmap: bool = False,
) -> None:
    """Draw heatmap.

    Args:
        frame (pd.DataFrame): Data frame to process.
        output_path (Path): Path value used by the function.
        title (str): str value used by the function.
        value_label (str): str value used by the function.
        value_format (str): str value used by the function.
        vmin (float | None): float | None value used by the function.
        vmax (float | None): float | None value used by the function.
        reverse_cmap (bool): bool value used by the function.

    Returns:
        None: None.
    """
    if frame.empty:
        return

    data = frame.to_numpy(dtype=float)
    cmap = matplotlib.colormaps["YlGnBu_r" if reverse_cmap else "YlGnBu"].copy()
    cmap.set_bad("#E6E6E6")

    fig_width = max(8, 1.2 * len(frame.columns))
    fig_height = max(4, 0.7 * len(frame.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(frame.columns)), frame.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(frame.index)), frame.index)
    ax.set_title(title)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if math.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                value_format.format(value),
                ha="center",
                va="center",
                fontsize=8,
                color=heatmap_text_color(value, cmap=cmap, norm=image.norm),
            )

    color_bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    color_bar.set_label(value_label)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def heatmap_text_color(
    value: float,
    *,
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
) -> str:
    """Heatmap text color.

    Args:
        value (float): Value to process.
        cmap (matplotlib.colors.Colormap): matplotlib.colors.Colormap value used by the function.
        norm (matplotlib.colors.Normalize): matplotlib.colors.Normalize value used by the function.

    Returns:
        str: Resolved string value.
    """
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance >= 0.6 else "white"


if __name__ == "__main__":
    raise SystemExit(main())
