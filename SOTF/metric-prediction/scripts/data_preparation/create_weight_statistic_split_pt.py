from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype

# Allow running this script directly via `python path/to/script.py`.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.logging_utils import setup_logging

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "metric-prediction/cache"
DEFAULT_OUTPUT_DIRNAME = "weight_statistic_datasets"
DEFAULT_TARGET_COLUMN = "test_acc"
EXPECTED_SPLITS = ("train", "val", "test")
TARGET_METRIC_COLUMNS = ("test_acc", "train_acc", "test_loss", "train_loss")
DEFAULT_METADATA_COLUMNS = ("sample_idx", "model_id", "path", "epoch", "hyperparameter_key")


def _build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        argparse.ArgumentParser: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val/test .pt files for a prepared weight-statistic dataset "
            "using a latent-representation split CSV."
        )
    )
    parser.add_argument(
        "split_csv_path",
        help=(
            "Path to the latent split CSV with columns model_id and split_name, for "
            "example metric-prediction/cache/latent_representation_datasets/"
            "ds_12_resnet18_tinyimagenet_ckpts_encoded_checkpoints/averaged/"
            "latent_representation_splits.csv."
        ),
    )
    parser.add_argument(
        "--prepared-dataset-path",
        default=None,
        help=(
            "Prepared weight-statistic parquet to split. When omitted, the script "
            "tries to infer it from a ds_<id> token in split_csv_path and resolves "
            "the matching cache/prepared_datasets/*__prepared_dataset.parquet file."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=(
            "Cache root containing prepared_datasets/ and the output "
            f"{DEFAULT_OUTPUT_DIRNAME}/ directory. Defaults to {DEFAULT_CACHE_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional explicit output directory. Defaults to "
            "metric-prediction/cache/weight_statistic_datasets/<dataset_name>/."
        ),
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help=f"Target column to store separately in the .pt payload. Defaults to {DEFAULT_TARGET_COLUMN}.",
    )
    parser.add_argument(
        "--feature-dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Floating-point dtype used for the saved feature tensor. Defaults to float32.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if the prepared dataset contains models missing from the split CSV "
            "or if the split CSV contains models missing from the prepared dataset."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )
    return parser


def _resolve_existing_file(raw_path: str | Path, label: str) -> Path:
    """Resolve existing file.

    Args:
        raw_path (str | Path): Raw path value to resolve.
        label (str): Field label used for validation.

    Returns:
        Path: Resolved path.
    """
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _resolve_cache_dir(raw_path: str | Path) -> Path:
    """Resolve cache dir.

    Args:
        raw_path (str | Path): Raw path value to resolve.

    Returns:
        Path: Resolved path.
    """
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Cache path is not a directory: {path}")
    return path


def _find_repo_root(start_path: Path) -> Path:
    """Find repository root for stable relative-path serialization."""
    search_root = start_path.resolve()
    for candidate in (search_root, *search_root.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start_path}")


def _path_relative_to_repo_root(path: Path, *, repo_root: Path) -> str:
    """Serialize a path relative to the repository root."""
    return Path(os.path.relpath(path, repo_root)).as_posix()


def _feature_metadata_cache_path(prepared_dataset_path: Path) -> Path:
    """Return the sidecar feature-metadata path for a prepared dataset."""
    return prepared_dataset_path.with_name(f"{prepared_dataset_path.stem}__feature_metadata.parquet")


def _load_feature_metadata(prepared_dataset_path: Path) -> pd.DataFrame:
    """Load feature metadata sidecar for a prepared dataset if available."""
    feature_metadata_path = _feature_metadata_cache_path(prepared_dataset_path)
    if not feature_metadata_path.exists():
        return pd.DataFrame()

    frame = pd.read_parquet(feature_metadata_path)
    if "feature" not in frame.columns:
        raise ValueError(f"Prepared feature metadata is missing the 'feature' column: {feature_metadata_path}")
    return frame.set_index("feature")


def _infer_prepared_dataset_path(split_csv_path: Path, cache_dir: Path) -> Path:
    """Infer prepared dataset path from split CSV.

    Args:
        split_csv_path (Path): Path to the split CSV.
        cache_dir (Path): Cache directory root.

    Returns:
        Path: Resolved path.
    """
    ds_match = re.search(r"(?:^|[\\/])ds_(\d+)(?:_|[\\/])", str(split_csv_path))
    if ds_match is None:
        raise ValueError(
            "Could not infer dataset id from split_csv_path. Pass --prepared-dataset-path explicitly."
        )

    dataset_id = ds_match.group(1)
    prepared_dir = cache_dir / "prepared_datasets"
    matches = sorted(prepared_dir.glob(f"{dataset_id}_*__prepared_dataset.parquet"))
    matches = [path for path in matches if "__feature_metadata" not in path.name]
    if not matches:
        raise FileNotFoundError(
            "Could not find a prepared dataset parquet matching "
            f"dataset id {dataset_id} under {prepared_dir}."
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple prepared datasets matched the inferred dataset id. "
            f"Pass --prepared-dataset-path explicitly. Matches: {matches}"
        )
    return matches[0]


def _dataset_name_from_prepared_path(prepared_dataset_path: Path) -> str:
    """Dataset name from prepared path.

    Args:
        prepared_dataset_path (Path): Path to the prepared dataset.

    Returns:
        str: Result of the function.
    """
    suffix = "__prepared_dataset.parquet"
    if prepared_dataset_path.name.endswith(suffix):
        return prepared_dataset_path.name[: -len(suffix)]
    return prepared_dataset_path.stem


def _resolve_output_dir(
    *,
    raw_output_dir: str | None,
    cache_dir: Path,
    dataset_name: str,
) -> Path:
    """Resolve output dir.

    Args:
        raw_output_dir (str | None): Optional explicit output directory.
        cache_dir (Path): Cache directory root.
        dataset_name (str): Dataset name used for the default output path.

    Returns:
        Path: Resolved path.
    """
    if raw_output_dir:
        return Path(raw_output_dir).expanduser().resolve()
    return cache_dir / DEFAULT_OUTPUT_DIRNAME / dataset_name


def _load_split_csv(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load split CSV with exact and basename lookup tables.

    Args:
        path (Path): Path to the CSV.

    Returns:
        tuple[dict[str, str], dict[str, str]]: Result of the function.
    """
    exact_split_by_model_id: dict[str, str] = {}
    basename_split_by_model_id: dict[str, str] = {}
    basename_conflicts: set[str] = set()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required_columns = {"model_id", "split_name"}
        missing_columns = sorted(required_columns.difference(fieldnames))
        if missing_columns:
            raise ValueError(f"Split CSV is missing required columns {missing_columns}: {path}")

        for row in reader:
            model_id = str(row["model_id"]).strip()
            split_name = _normalize_split_name(row["split_name"])
            if not model_id:
                continue

            previous_split = exact_split_by_model_id.get(model_id)
            if previous_split is not None and previous_split != split_name:
                raise ValueError(
                    f"Model ID appears with multiple split names in {path}: "
                    f"{model_id!r} ({previous_split}, {split_name})"
                )
            exact_split_by_model_id[model_id] = split_name

            basename = Path(model_id).name
            previous_basename_split = basename_split_by_model_id.get(basename)
            if previous_basename_split is not None and previous_basename_split != split_name:
                basename_conflicts.add(basename)
            else:
                basename_split_by_model_id[basename] = split_name

    for basename in basename_conflicts:
        basename_split_by_model_id.pop(basename, None)

    if not exact_split_by_model_id:
        raise ValueError(f"Split CSV contains no model IDs: {path}")

    return exact_split_by_model_id, basename_split_by_model_id


def _normalize_split_name(raw_split_name: str) -> str:
    """Normalize split name.

    Args:
        raw_split_name (str): Raw split name.

    Returns:
        str: Result of the function.
    """
    split_name = str(raw_split_name).strip().lower()
    if split_name in {"train", "training"}:
        return "train"
    if split_name in {"val", "valid", "validation", "valt"}:
        return "val"
    if split_name == "test":
        return "test"
    raise ValueError(
        "Unsupported split_name in split CSV. Expected train, val, validation, or test; "
        f"received {raw_split_name!r}."
    )


def _assign_splits(
    frame: pd.DataFrame,
    *,
    exact_split_by_model_id: dict[str, str],
    basename_split_by_model_id: dict[str, str],
) -> pd.Series:
    """Assign splits to a prepared dataset frame.

    Args:
        frame (pd.DataFrame): Prepared dataset frame.
        exact_split_by_model_id (dict[str, str]): Exact model_id split map.
        basename_split_by_model_id (dict[str, str]): Basename fallback split map.

    Returns:
        pd.Series: Result of the function.
    """
    model_ids = frame["model_id"].astype(str)
    assigned = model_ids.map(exact_split_by_model_id)
    missing_mask = assigned.isna()
    if missing_mask.any():
        assigned.loc[missing_mask] = model_ids.loc[missing_mask].map(
            lambda model_id: basename_split_by_model_id.get(Path(model_id).name)
        )
    return assigned


def _feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    """Determine feature columns to persist.

    Args:
        frame (pd.DataFrame): Prepared dataset frame.
        target_column (str): Target column to exclude from features.

    Returns:
        list[str]: Result of the function.
    """
    excluded_columns = set(DEFAULT_METADATA_COLUMNS) | set(TARGET_METRIC_COLUMNS) | {target_column, "split_name"}
    return [
        column
        for column in frame.columns
        if column not in excluded_columns and is_numeric_dtype(frame[column])
    ]


def _serialize_metadata_value(series: pd.Series) -> list[Any]:
    """Serialize metadata series into a torch-save-friendly list.

    Args:
        series (pd.Series): Series to serialize.

    Returns:
        list[Any]: Result of the function.
    """
    if is_numeric_dtype(series):
        values = series.to_numpy(copy=True)
        return values.tolist()
    return series.astype("string").fillna("").tolist()


def _build_payload(
    split_frame: pd.DataFrame,
    *,
    split_name: str,
    dataset_name: str,
    repo_root: Path,
    prepared_dataset_path: Path,
    split_csv_path: Path,
    target_column: str,
    feature_columns: list[str],
    feature_dtype: str,
    feature_metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Build split payload.

    Args:
        split_frame (pd.DataFrame): Split frame to serialize.
        split_name (str): Name of the split.
        dataset_name (str): Dataset name.
        prepared_dataset_path (Path): Source prepared parquet.
        split_csv_path (Path): Source split CSV.
        target_column (str): Target column to store separately.
        feature_columns (list[str]): Feature columns to save.
        feature_dtype (str): Floating-point dtype for the feature tensor.

    Returns:
        dict[str, Any]: Result of the function.
    """
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "source_prepared_dataset_path": _path_relative_to_repo_root(prepared_dataset_path, repo_root=repo_root),
        "source_split_csv_path": _path_relative_to_repo_root(split_csv_path, repo_root=repo_root),
        "sample_idx": torch.as_tensor(
            split_frame["sample_idx"].to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
        ),
        "model_ids": split_frame["model_id"].astype(str).tolist(),
        "feature_names": list(feature_columns),
        "features": torch.as_tensor(
            split_frame.loc[:, feature_columns].to_numpy(dtype=np.float32 if feature_dtype == "float32" else np.float64, copy=True)
        ),
    }

    if "path" in split_frame.columns:
        payload["checkpoint_paths"] = split_frame["path"].astype(str).tolist()
    if "epoch" in split_frame.columns:
        epoch_series = pd.to_numeric(split_frame["epoch"], errors="coerce")
        if epoch_series.isna().any():
            payload["epochs"] = epoch_series.tolist()
        else:
            payload["epochs"] = epoch_series.astype(np.int64).tolist()
    if target_column in split_frame.columns:
        payload["target_name"] = target_column
        payload["targets"] = torch.as_tensor(
            split_frame[target_column].to_numpy(dtype=np.float32, copy=True)
        )

    metadata_columns = [
        column for column in split_frame.columns
        if column not in feature_columns and column != "split_name"
    ]
    payload["metadata_columns"] = metadata_columns
    payload["metadata"] = {
        column: _serialize_metadata_value(split_frame[column])
        for column in metadata_columns
    }
    if not feature_metadata.empty:
        aligned_feature_metadata = feature_metadata.loc[feature_columns].reset_index()
        payload["feature_metadata"] = {
            column: aligned_feature_metadata[column].astype("string").fillna("").tolist()
            for column in aligned_feature_metadata.columns
        }
    return payload


def create_weight_statistic_split_pt_files(
    *,
    split_csv_path: Path,
    prepared_dataset_path: Path,
    output_dir: Path,
    target_column: str,
    feature_dtype: str,
    strict: bool,
) -> list[Path]:
    """Create split PT files.

    Args:
        split_csv_path (Path): Split CSV path.
        prepared_dataset_path (Path): Prepared parquet path.
        output_dir (Path): Output directory.
        target_column (str): Target column to store separately.
        feature_dtype (str): Floating-point dtype for saved features.
        strict (bool): Whether to fail on unmatched models.

    Returns:
        list[Path]: Result of the function.
    """
    exact_split_by_model_id, basename_split_by_model_id = _load_split_csv(split_csv_path)
    frame = pd.read_parquet(prepared_dataset_path)

    required_columns = {"sample_idx", "model_id"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Prepared dataset is missing required columns {missing_columns}: {prepared_dataset_path}"
        )

    dataset_name = _dataset_name_from_prepared_path(prepared_dataset_path)
    repo_root = _find_repo_root(Path(__file__).resolve())
    feature_metadata = _load_feature_metadata(prepared_dataset_path)
    assigned_splits = _assign_splits(
        frame,
        exact_split_by_model_id=exact_split_by_model_id,
        basename_split_by_model_id=basename_split_by_model_id,
    )
    frame = frame.copy()
    frame["split_name"] = assigned_splits

    unmatched_rows = frame["split_name"].isna()
    unmatched_model_ids = sorted(frame.loc[unmatched_rows, "model_id"].astype(str).unique().tolist())
    frame_model_ids = set(frame["model_id"].astype(str))
    frame_model_id_basenames = {Path(model_id).name for model_id in frame_model_ids}
    missing_from_weight_dataset = sorted(
        model_id
        for model_id in exact_split_by_model_id
        if model_id not in frame_model_ids and Path(model_id).name not in frame_model_id_basenames
    )

    if strict and unmatched_model_ids:
        raise ValueError(
            "Prepared dataset contains models that are missing from the split CSV. "
            f"First unmatched model IDs: {unmatched_model_ids[:10]}"
        )
    if strict and missing_from_weight_dataset:
        raise ValueError(
            "Split CSV contains models that are missing from the prepared dataset. "
            f"First missing model IDs: {missing_from_weight_dataset[:10]}"
        )

    if unmatched_model_ids:
        log.warning(
            "Dropping %d rows from %d prepared-dataset models that were not found in the split CSV",
            int(unmatched_rows.sum()),
            len(unmatched_model_ids),
        )
    if missing_from_weight_dataset:
        log.warning(
            "Split CSV contains %d model IDs that are not present in the prepared dataset",
            len(missing_from_weight_dataset),
        )

    frame = frame.loc[frame["split_name"].isin(EXPECTED_SPLITS)].copy()
    if frame.empty:
        raise ValueError("No prepared-dataset rows matched train/val/test entries in the split CSV.")

    feature_columns = _feature_columns(frame, target_column)
    if not feature_columns:
        raise ValueError("No numeric feature columns remain after excluding metadata and target columns.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for split_name in EXPECTED_SPLITS:
        split_frame = frame.loc[frame["split_name"] == split_name].copy()
        if split_frame.empty:
            raise ValueError(f"Split CSV did not produce any rows for split '{split_name}'.")

        payload = _build_payload(
            split_frame,
            split_name=split_name,
            dataset_name=dataset_name,
            repo_root=repo_root,
            prepared_dataset_path=prepared_dataset_path,
            split_csv_path=split_csv_path,
            target_column=target_column,
            feature_columns=feature_columns,
            feature_dtype=feature_dtype,
            feature_metadata=feature_metadata,
        )
        output_path = output_dir / f"weight_statistics_{split_name}.pt"
        torch.save(payload, output_path)
        written_paths.append(output_path)

        log.info(
            "Wrote %s with %d rows, %d unique models, and %d features",
            output_path,
            len(split_frame),
            int(split_frame["model_id"].astype(str).nunique()),
            len(feature_columns),
        )

    return written_paths


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    args = _build_parser().parse_args()
    setup_logging(level=args.log_level)

    cache_dir = _resolve_cache_dir(args.cache_dir)
    split_csv_path = _resolve_existing_file(args.split_csv_path, "split_csv_path")
    prepared_dataset_path = (
        _resolve_existing_file(args.prepared_dataset_path, "prepared_dataset_path")
        if args.prepared_dataset_path
        else _infer_prepared_dataset_path(split_csv_path, cache_dir)
    )
    dataset_name = _dataset_name_from_prepared_path(prepared_dataset_path)
    output_dir = _resolve_output_dir(
        raw_output_dir=args.output_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
    )

    log.info("Using split CSV: %s", split_csv_path)
    log.info("Using prepared dataset: %s", prepared_dataset_path)
    log.info("Writing output to: %s", output_dir)

    written_paths = create_weight_statistic_split_pt_files(
        split_csv_path=split_csv_path,
        prepared_dataset_path=prepared_dataset_path,
        output_dir=output_dir,
        target_column=str(args.target_column),
        feature_dtype=str(args.feature_dtype),
        strict=bool(args.strict),
    )
    log.info("Finished writing %d split files", len(written_paths))


if __name__ == "__main__":
    main()
