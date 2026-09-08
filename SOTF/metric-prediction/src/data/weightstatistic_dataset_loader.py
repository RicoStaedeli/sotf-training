from __future__ import annotations

from fnmatch import fnmatch
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from data.weightstatistic_dataset import WeightStatisticDataset, _drop_sparse_and_constant_features

log = logging.getLogger(__name__)

LONG_FORM_REQUIRED_COLUMNS = {
    "sample_idx",
    "model_id",
    "path",
    "epoch",
    "layer_name",
    "layer_safe_name",
    "layer_type",
    "metric_family",
    "metric",
    "scope",
    "value",
}
TARGET_METRIC_COLUMNS = (
    "test_acc",
    "train_acc",
    "test_loss",
    "train_loss",
)


def load_weightstatistic_dataset(
    data_cfg: DictConfig,
    feature_cfg: DictConfig,
) -> WeightStatisticDataset:
    """Load prepared weight statistic dataset.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.

    Returns:
        WeightStatisticDataset: Computed weight-statistic dataset.
    """
    parquet_path = _resolve_weightstatistic_dataset_path(data_cfg)
    log.info("Loading prepared dataset from %s", parquet_path)
    feature_metadata = _load_feature_metadata(parquet_path)
    read_columns = _resolve_weightstatistic_dataset_read_columns(
        parquet_path=parquet_path,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )
    if read_columns is None:
        frame = pd.read_parquet(parquet_path)
    else:
        frame = pd.read_parquet(parquet_path, columns=read_columns)
    return _weightstatistic_dataset_from_frame(
        frame,
        parquet_path=parquet_path,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )


def load_weightstatistic_split_datasets(
    data_cfg: DictConfig,
    feature_cfg: DictConfig,
) -> tuple[WeightStatisticDataset, WeightStatisticDataset]:
    """Load pre-split weight-statistic datasets from .pt files.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.

    Returns:
        tuple[WeightStatisticDataset, WeightStatisticDataset]: Result of the function.
    """
    pt_train_path = _require_path(data_cfg.pt_train_path, "data.pt_train_path")
    pt_val_path = _require_path(data_cfg.pt_val_path, "data.pt_val_path")

    log.info("Loading pre-split weight-statistic datasets from PT files")
    train_dataset = _load_weightstatistic_pt_dataset(
        pt_train_path,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
        split_name="train",
    )
    val_dataset = _load_weightstatistic_pt_dataset(
        pt_val_path,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
        split_name="val",
    )
    log.info(
        "Pre-split weight-statistic datasets ready: train=%d val=%d",
        len(train_dataset.features),
        len(val_dataset.features),
    )
    return train_dataset, val_dataset


def load_weightstatistic_pt_dataset(
    data_cfg: DictConfig,
    feature_cfg: DictConfig,
) -> WeightStatisticDataset:
    """Load one pre-split weight-statistic dataset from a .pt file.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.

    Returns:
        WeightStatisticDataset: Result of the function.
    """
    pt_path = _require_path(data_cfg.pt_path, "data.pt_path")
    return _load_weightstatistic_pt_dataset(
        pt_path,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
    )

def _resolve_weightstatistic_dataset_path(data_cfg: DictConfig) -> Path:
    """Resolve weight-statistic dataset path.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    parquet_path = _require_path(data_cfg.parquet_path, "data.parquet_path")
    if parquet_path.name.endswith("__prepared_dataset.parquet"):
        return parquet_path

    cache_path = _weightstatistic_dataset_cache_path(parquet_path=parquet_path, data_cfg=data_cfg)
    if cache_path.exists():
        return cache_path

    raise FileNotFoundError(
        "Prepared dataset cache is missing for "
        f"{parquet_path}. Expected prepared dataset at {cache_path}."
    )


def _load_weightstatistic_pt_dataset(
    pt_path: Path,
    *,
    data_cfg: DictConfig,
    feature_cfg: DictConfig,
    split_name: str = "",
) -> WeightStatisticDataset:
    """Load weight-statistic dataset from a pre-split PT payload.

    Args:
        pt_path (Path): Path to the PT payload.
        data_cfg (DictConfig): Configuration object used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.
        split_name (str): Optional split name for logging.

    Returns:
        WeightStatisticDataset: Result of the function.
    """
    prefix = f"[{split_name}] " if split_name else ""
    log.info("%sLoading weight-statistic PT payload from %s", prefix, pt_path)
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{prefix}Expected {pt_path} to contain a dict payload, found {type(payload).__name__}.")

    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"{prefix}PT payload is missing a non-empty 'feature_names' list: {pt_path}")
    feature_names = [str(name) for name in feature_names]

    raw_features = payload.get("features")
    if raw_features is None:
        raise ValueError(f"{prefix}PT payload is missing 'features': {pt_path}")
    if isinstance(raw_features, torch.Tensor):
        feature_array = raw_features.detach().cpu().numpy()
    else:
        feature_array = np.asarray(raw_features)
    if feature_array.ndim != 2:
        raise ValueError(
            f"{prefix}Expected 'features' to be a rank-2 array/tensor, got shape {feature_array.shape!r}: {pt_path}"
        )
    if feature_array.shape[1] != len(feature_names):
        raise ValueError(
            f"{prefix}Feature name count {len(feature_names)} does not match feature matrix width {feature_array.shape[1]}: {pt_path}"
        )

    row_count = int(feature_array.shape[0])
    metadata_frame = _weightstatistic_pt_metadata_frame(payload, row_count=row_count, pt_path=pt_path, prefix=prefix)

    target_column = str(data_cfg.target_column)
    payload_target_name = str(payload.get("target_name", "")).strip() or None
    if "targets" in payload and payload_target_name and payload_target_name not in metadata_frame.columns:
        metadata_frame[payload_target_name] = _pt_payload_vector(
            payload["targets"],
            row_count=row_count,
            pt_path=pt_path,
            field_name="targets",
            prefix=prefix,
        )
    if target_column not in metadata_frame.columns:
        raise ValueError(
            f"{prefix}Configured target column '{target_column}' is not present in the PT payload metadata: {pt_path}"
        )

    metadata_frame = _enrich_metadata_from_source_parquet(
        metadata_frame,
        payload=payload,
        pt_path=pt_path,
        prefix=prefix,
    )

    feature_frame = pd.DataFrame(feature_array, columns=feature_names)
    frame = pd.concat([metadata_frame.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    feature_metadata = _load_feature_metadata_from_pt_payload(
        payload,
        data_cfg=data_cfg,
        pt_path=pt_path,
        prefix=prefix,
    )
    return _weightstatistic_dataset_from_frame(
        frame,
        parquet_path=pt_path,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )


def _weightstatistic_pt_metadata_frame(
    payload: dict[str, Any],
    *,
    row_count: int,
    pt_path: Path,
    prefix: str,
) -> pd.DataFrame:
    """Build metadata frame from PT payload.

    Args:
        payload (dict[str, Any]): PT payload.
        row_count (int): Expected row count.
        pt_path (Path): Path to the PT file.
        prefix (str): Logging prefix.

    Returns:
        pd.DataFrame: Result of the function.
    """
    metadata_payload = payload.get("metadata", {})
    if metadata_payload is None:
        metadata_payload = {}
    if not isinstance(metadata_payload, dict):
        raise ValueError(f"{prefix}PT payload field 'metadata' must be a dict: {pt_path}")

    metadata_columns = payload.get("metadata_columns", [])
    if metadata_columns and not isinstance(metadata_columns, list):
        raise ValueError(f"{prefix}PT payload field 'metadata_columns' must be a list when present: {pt_path}")

    normalized_metadata: dict[str, list[Any]] = {}
    for column, values in metadata_payload.items():
        normalized_metadata[str(column)] = _pt_payload_vector(
            values,
            row_count=row_count,
            pt_path=pt_path,
            field_name=f"metadata[{column}]",
            prefix=prefix,
        )

    metadata_frame = pd.DataFrame(normalized_metadata)
    if len(metadata_frame.index) not in {0, row_count}:
        raise ValueError(
            f"{prefix}Metadata frame length {len(metadata_frame.index)} does not match feature rows {row_count}: {pt_path}"
        )
    if metadata_frame.empty:
        metadata_frame = pd.DataFrame(index=pd.RangeIndex(row_count))

    if "sample_idx" not in metadata_frame.columns:
        if "sample_idx" in payload:
            metadata_frame["sample_idx"] = _pt_payload_vector(
                payload["sample_idx"],
                row_count=row_count,
                pt_path=pt_path,
                field_name="sample_idx",
                prefix=prefix,
            )
        else:
            metadata_frame["sample_idx"] = list(range(row_count))

    fallback_fields = {
        "model_id": "model_ids",
        "path": "checkpoint_paths",
        "epoch": "epochs",
    }
    for output_column, payload_field in fallback_fields.items():
        if output_column not in metadata_frame.columns and payload_field in payload:
            metadata_frame[output_column] = _pt_payload_vector(
                payload[payload_field],
                row_count=row_count,
                pt_path=pt_path,
                field_name=payload_field,
                prefix=prefix,
            )

    if metadata_columns:
        ordered_columns = [str(column) for column in metadata_columns if str(column) in metadata_frame.columns]
        remaining_columns = [column for column in metadata_frame.columns if column not in ordered_columns]
        metadata_frame = metadata_frame.loc[:, [*ordered_columns, *remaining_columns]]

    return metadata_frame


def _pt_payload_vector(
    values: Any,
    *,
    row_count: int,
    pt_path: Path,
    field_name: str,
    prefix: str,
) -> list[Any]:
    """Normalize one PT payload vector to a Python list.

    Args:
        values (Any): Values to normalize.
        row_count (int): Expected row count.
        pt_path (Path): Path to the PT file.
        field_name (str): Field name used for validation.
        prefix (str): Logging prefix.

    Returns:
        list[Any]: Result of the function.
    """
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
        if array.ndim == 0:
            raise ValueError(f"{prefix}PT payload field '{field_name}' is scalar, expected length {row_count}: {pt_path}")
        normalized = array.tolist()
    elif isinstance(values, np.ndarray):
        if values.ndim == 0:
            raise ValueError(f"{prefix}PT payload field '{field_name}' is scalar, expected length {row_count}: {pt_path}")
        normalized = values.tolist()
    elif isinstance(values, list):
        normalized = values
    elif isinstance(values, tuple):
        normalized = list(values)
    else:
        raise ValueError(
            f"{prefix}PT payload field '{field_name}' must be a tensor, ndarray, list, or tuple: {pt_path}"
        )

    if len(normalized) != row_count:
        raise ValueError(
            f"{prefix}PT payload field '{field_name}' has {len(normalized)} rows, expected {row_count}: {pt_path}"
        )
    return normalized


def _load_feature_metadata_from_pt_payload(
    payload: dict[str, Any],
    *,
    data_cfg: DictConfig,
    pt_path: Path,
    prefix: str,
) -> pd.DataFrame | None:
    """Load feature metadata for a PT payload.

    Args:
        payload (dict[str, Any]): PT payload.
        pt_path (Path): PT file path.
        prefix (str): Logging prefix.

    Returns:
        pd.DataFrame | None: Result of the function.
    """
    embedded_metadata = _deserialize_feature_metadata_payload(payload, pt_path=pt_path, prefix=prefix)
    if embedded_metadata is not None and not embedded_metadata.empty:
        log.info("%sLoaded feature metadata embedded in PT payload", prefix)
        return embedded_metadata

    parquet_override = data_cfg.get("parquet_path", None)
    if parquet_override:
        prepared_path = _resolve_weightstatistic_dataset_path(data_cfg)
        loaded = _load_feature_metadata(prepared_path)
        if loaded is not None and not loaded.empty:
            log.info("%sLoaded feature metadata from configured data.parquet_path=%s", prefix, prepared_path)
            return loaded
        log.warning(
            "%sConfigured data.parquet_path did not provide feature metadata sidecar: %s",
            prefix,
            prepared_path,
        )

    source_prepared_dataset_path = payload.get("source_prepared_dataset_path", None)
    if source_prepared_dataset_path is None:
        log.info("%sPT payload does not declare source_prepared_dataset_path; feature metadata unavailable", prefix)
        return pd.DataFrame()

    prepared_path = _resolve_payload_source_path(
        source_prepared_dataset_path,
        pt_path=pt_path,
    )
    if not prepared_path.exists():
        log.warning(
            "%sSource prepared dataset path from PT payload does not exist, skipping feature metadata: %s",
            prefix,
            prepared_path,
        )
        return pd.DataFrame()

    try:
        return _load_feature_metadata(prepared_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning(
            "%sFailed to load feature metadata from %s for PT payload %s: %s",
            prefix,
            prepared_path,
            pt_path,
            exc,
        )
        return pd.DataFrame()


def _deserialize_feature_metadata_payload(
    payload: dict[str, Any],
    *,
    pt_path: Path,
    prefix: str,
) -> pd.DataFrame | None:
    """Load embedded feature metadata from a PT payload."""
    raw_feature_metadata = payload.get("feature_metadata", None)
    if raw_feature_metadata is None:
        return None
    if not isinstance(raw_feature_metadata, dict):
        raise ValueError(f"{prefix}PT payload field 'feature_metadata' must be a dict: {pt_path}")

    frame = pd.DataFrame(
        {
            str(column): list(values)
            for column, values in raw_feature_metadata.items()
        }
    )
    if frame.empty:
        return pd.DataFrame()
    if "feature" not in frame.columns:
        raise ValueError(f"{prefix}PT payload feature_metadata is missing required 'feature' column: {pt_path}")
    return frame.set_index("feature")


def _enrich_metadata_from_source_parquet(
    metadata_frame: pd.DataFrame,
    *,
    payload: dict[str, Any],
    pt_path: Path,
    prefix: str,
) -> pd.DataFrame:
    """Add TARGET_METRIC_COLUMNS missing from the PT metadata by reading the source prepared parquet.

    Handles old PT files that stored only model_ids/checkpoint_paths/epochs/targets at the
    top level and never serialised train_acc / test_loss / train_loss into the payload.
    """
    missing_cols = [col for col in TARGET_METRIC_COLUMNS if col not in metadata_frame.columns]
    if not missing_cols:
        return metadata_frame

    source_path_raw = payload.get("source_prepared_dataset_path")
    if not source_path_raw:
        return metadata_frame

    prepared_path = _resolve_payload_source_path(str(source_path_raw), pt_path=pt_path)
    if not prepared_path.exists():
        log.warning(
            "%sCannot enrich metadata with %s — source prepared dataset not found: %s",
            prefix,
            missing_cols,
            prepared_path,
        )
        return metadata_frame

    try:
        schema_cols = set(_read_parquet_schema_columns(prepared_path))
        cols_to_load = [col for col in missing_cols if col in schema_cols]
        if not cols_to_load:
            return metadata_frame

        source_frame = pd.read_parquet(prepared_path, columns=["sample_idx", *cols_to_load])
        source_frame = source_frame.set_index("sample_idx")

        for col in cols_to_load:
            metadata_frame[col] = pd.to_numeric(
                metadata_frame["sample_idx"].map(source_frame[col]),
                errors="coerce",
            )
        log.info(
            "%sEnriched PT metadata with %d columns from source parquet: %s",
            prefix,
            len(cols_to_load),
            cols_to_load,
        )
    except Exception as exc:
        log.warning(
            "%sFailed to enrich PT metadata from source parquet %s: %s",
            prefix,
            prepared_path,
            exc,
        )

    return metadata_frame


def _resolve_payload_source_path(raw_path: str | Path, *, pt_path: Path) -> Path:
    """Resolve a payload source path, supporting repo-root-relative paths."""
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()

    for candidate in (pt_path.resolve(), *pt_path.resolve().parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return (candidate / path).resolve()
    return path.resolve()


def _weightstatistic_dataset_from_frame(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    data_cfg: DictConfig,
    feature_cfg: DictConfig,
    feature_metadata: pd.DataFrame | None = None,
) -> WeightStatisticDataset:
    """Build weight-statistic dataset from frame.

    Args:
        frame (pd.DataFrame): Data frame to process.
        parquet_path (Path): Path to the parquet.
        data_cfg (DictConfig): Configuration object used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.

    Returns:
        WeightStatisticDataset: Computed weight-statistic dataset.
    """
    _validate_weightstatistic_dataset_frame(frame, parquet_path=parquet_path, data_cfg=data_cfg)
    frame = _filter_weightstatistic_dataset_by_target_model_threshold(
        frame,
        parquet_path=parquet_path,
        data_cfg=data_cfg,
    )
    frame = _filter_weightstatistic_dataset_by_epoch(
        frame,
        parquet_path=parquet_path,
        data_cfg=data_cfg,
    )

    target_column = str(data_cfg.target_column)
    metadata_columns = [str(column) for column in data_cfg.metadata_columns if column in frame.columns]
    prepared = frame.rename(columns={"target": target_column} if target_column not in frame.columns else {})
    prepared = prepared.set_index("sample_idx")

    if prepared.index.has_duplicates:
        raise ValueError(f"Prepared dataset contains duplicate sample_idx values: {parquet_path}")

    prepared = prepared.dropna(subset=[target_column])
    if prepared.empty:
        raise ValueError(f"Prepared dataset has no rows with target '{target_column}': {parquet_path}")

    excluded_columns = {target_column, *metadata_columns, *TARGET_METRIC_COLUMNS}
    feature_columns = _select_feature_columns(
        available_columns=[column for column in prepared.columns if column not in excluded_columns],
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )
    wants_weight_stat_features = bool(feature_cfg.include_model_metrics) or bool(feature_cfg.include_layer_metrics)
    if not feature_columns and wants_weight_stat_features:
        raise ValueError("No feature columns remain after applying feature selection.")

    features = prepared.loc[:, feature_columns].astype("float64")
    if feature_columns:
        features = _drop_sparse_and_constant_features(features, feature_cfg)
        if features.empty:
            raise ValueError("No usable feature columns remain after sparse/constant filtering.")
    if bool(feature_cfg.sort_columns):
        features = features.sort_index(axis=1)

    target = prepared.loc[features.index, target_column].astype("float64")
    obs_metric_cols = [col for col in TARGET_METRIC_COLUMNS if col in prepared.columns and col not in metadata_columns]
    all_metadata_cols = metadata_columns + obs_metric_cols
    metadata = prepared.loc[features.index, all_metadata_cols].copy() if all_metadata_cols else pd.DataFrame(index=features.index)
    metadata.index.name = "sample_idx"
    aligned_feature_metadata = _align_feature_metadata(
        feature_metadata=feature_metadata,
        feature_columns=features.columns,
    )

    log.info(
        "Prepared dataset contains %d samples, %d features, metadata columns=%s, feature metadata rows=%d",
        len(features),
        features.shape[1],
        metadata_columns,
        len(aligned_feature_metadata),
    )
    return WeightStatisticDataset(
        features=features,
        target=target,
        metadata=metadata,
        feature_metadata=aligned_feature_metadata,
    )


def _filter_weightstatistic_dataset_by_epoch(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    data_cfg: DictConfig,
) -> pd.DataFrame:
    """Filter prepared dataset by epoch.

    Args:
        frame (pd.DataFrame): Data frame to process.
        parquet_path (Path): Path to the parquet.
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    epoch_filter = _normalize_epoch_filter(data_cfg.get("epoch_filter", []))
    if not epoch_filter:
        return frame

    if "epoch" not in frame.columns:
        raise ValueError(
            "Configured data.epoch_filter requires the prepared dataset to contain an 'epoch' column: "
            f"{parquet_path}"
        )

    epoch_values = pd.to_numeric(frame["epoch"], errors="coerce")
    filtered = frame.loc[epoch_values.isin(epoch_filter)].copy()
    if filtered.empty:
        raise ValueError(
            f"Prepared dataset has no rows for data.epoch_filter={epoch_filter}: {parquet_path}"
        )

    log.info(
        "Applied data.epoch_filter=%s and kept %d of %d prepared rows",
        epoch_filter,
        len(filtered),
        len(frame),
    )
    return filtered


def _filter_weightstatistic_dataset_by_target_model_threshold(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    data_cfg: DictConfig,
) -> pd.DataFrame:
    """Filter prepared dataset by target model threshold.

    Args:
        frame (pd.DataFrame): Data frame to process.
        parquet_path (Path): Path to the parquet.
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    threshold = _normalize_optional_numeric(
        data_cfg.get("target_model_min_threshold", None),
        field_name="data.target_model_min_threshold",
    )
    if threshold is None:
        return frame

    required_columns = {"model_id", "epoch"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Configured data.target_model_min_threshold requires the prepared dataset to contain "
            + ", ".join(f"'{column}'" for column in missing_columns)
            + f": {parquet_path}"
        )

    target_column = _resolved_target_column_name(frame, data_cfg=data_cfg)
    working = frame.loc[:, ["sample_idx", "model_id", "epoch", target_column]].copy()
    working["model_id"] = working["model_id"].astype("string")
    working["epoch_numeric"] = pd.to_numeric(working["epoch"], errors="coerce")
    working["target_numeric"] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=["model_id", "epoch_numeric", "target_numeric"]).copy()
    if working.empty:
        raise ValueError(
            "Prepared dataset has no rows with valid model_id, epoch, and target values for "
            f"data.target_model_min_threshold={threshold}: {parquet_path}"
        )

    working = working.sort_values(["model_id", "epoch_numeric", "sample_idx"], kind="stable")
    final_targets = working.groupby("model_id", sort=False)["target_numeric"].last()
    keep_model_ids = final_targets.index[final_targets > threshold]
    filtered = frame.loc[frame["model_id"].astype("string").isin(keep_model_ids)].copy()
    if filtered.empty:
        raise ValueError(
            "Prepared dataset has no models whose final target exceeds "
            f"data.target_model_min_threshold={threshold}: {parquet_path}"
        )

    log.info(
        "Applied data.target_model_min_threshold=%s and kept %d of %d models, %d of %d rows",
        threshold,
        int(len(keep_model_ids)),
        int(final_targets.index.nunique()),
        int(len(filtered)),
        int(len(frame)),
    )
    return filtered


def _normalize_epoch_filter(raw_epoch_filter: Any) -> list[float]:
    """Normalize epoch filter.

    Args:
        raw_epoch_filter (Any): Value used by the function.

    Returns:
        list[float]: Result of the function.
    """
    if raw_epoch_filter is None:
        return []

    if isinstance(raw_epoch_filter, (str, int, float)):
        values = [raw_epoch_filter]
    else:
        values = list(raw_epoch_filter)

    normalized: list[float] = []
    for value in values:
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            raise ValueError(f"data.epoch_filter must contain only numeric values. Received: {value!r}")
        normalized.append(float(numeric_value))

    return list(dict.fromkeys(normalized))


def _normalize_optional_numeric(raw_value: Any, *, field_name: str) -> float | None:
    """Normalize optional numeric.

    Args:
        raw_value (Any): Raw value to process.
        field_name (str): Field name used for validation.

    Returns:
        float | None: Result of the function.
    """
    if raw_value is None:
        return None

    numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        raise ValueError(f"{field_name} must be numeric when set. Received: {raw_value!r}")
    return float(numeric_value)


def _validate_weightstatistic_dataset_frame(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    data_cfg: DictConfig,
) -> None:
    """Validate prepared dataset frame.

    Args:
        frame (pd.DataFrame): Data frame to process.
        parquet_path (Path): Path to the parquet.
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    if "sample_idx" not in frame.columns:
        raise ValueError(f"Prepared dataset is missing sample_idx: {parquet_path}")

    target_column = str(data_cfg.target_column)
    if target_column not in frame.columns and "target" not in frame.columns:
        raise ValueError(
            f"Prepared dataset is missing target column '{target_column}': {parquet_path}"
        )

    if LONG_FORM_REQUIRED_COLUMNS.issubset(frame.columns):
        cache_path = _weightstatistic_dataset_cache_path(parquet_path=parquet_path, data_cfg=data_cfg)
        cache_hint = f" or point data.parquet_path to {cache_path}" if cache_path is not None else ""
        raise ValueError(
            f"{parquet_path} looks like the raw long-form dataset, not the prepared wide dataset. "
            "Generate the prepared parquet first"
            f"{cache_hint}."
        )


def _resolved_target_column_name(frame: pd.DataFrame, *, data_cfg: DictConfig) -> str:
    """Resolved target column name.

    Args:
        frame (pd.DataFrame): Data frame to process.
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        str: Resolved string value.
    """
    target_column = str(data_cfg.target_column)
    if target_column in frame.columns:
        return target_column
    return "target"


def _load_feature_metadata(parquet_path: Path) -> pd.DataFrame | None:
    """Load feature metadata.

    Args:
        parquet_path (Path): Path to the parquet.

    Returns:
        pd.DataFrame | None: Result of the function.
    """
    feature_metadata_path = parquet_path.with_name(f"{parquet_path.stem}__feature_metadata.parquet")
    if not feature_metadata_path.exists():
        return None

    frame = pd.read_parquet(feature_metadata_path)
    if "feature" not in frame.columns:
        raise ValueError(f"Prepared feature metadata is missing the 'feature' column: {feature_metadata_path}")

    return frame.set_index("feature")


def _resolve_weightstatistic_dataset_read_columns(
    *,
    parquet_path: Path,
    data_cfg: DictConfig,
    feature_cfg: DictConfig,
    feature_metadata: pd.DataFrame | None,
) -> list[str] | None:
    """Resolve prepared dataset read columns.

    Args:
        parquet_path (Path): Path to the parquet.
        data_cfg (DictConfig): Configuration object used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.

    Returns:
        list[str] | None: Result of the function.
    """
    if feature_metadata is None or feature_metadata.empty:
        return None

    schema_columns = _read_parquet_schema_columns(parquet_path)
    if not schema_columns:
        return None

    available = set(schema_columns)
    target_column = str(data_cfg.target_column)
    target_candidates = [column for column in (target_column, "target") if column in available]
    if not target_candidates:
        log.warning(
            "Prepared dataset schema did not include target candidates '%s'/'target'. Falling back to full read.",
            target_column,
        )
        return None

    metadata_columns = [str(column) for column in data_cfg.metadata_columns if str(column) in available]
    excluded = {"sample_idx", *metadata_columns, *target_candidates, *TARGET_METRIC_COLUMNS}
    available_feature_columns = [column for column in schema_columns if column not in excluded]
    selected_feature_columns = _select_feature_columns(
        available_feature_columns,
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )

    read_columns: list[str] = ["sample_idx", *target_candidates, *metadata_columns]
    for column in TARGET_METRIC_COLUMNS:
        if column in available:
            read_columns.append(column)

    # Keep columns required by optional row-level filters.
    if "model_id" in available:
        read_columns.append("model_id")
    if "epoch" in available:
        read_columns.append("epoch")

    read_columns.extend(selected_feature_columns)
    deduplicated = list(dict.fromkeys(read_columns))
    log.info(
        "Column-pruned parquet read enabled: selected %d of %d columns (features=%d)",
        len(deduplicated),
        len(schema_columns),
        len(selected_feature_columns),
    )
    return deduplicated


def _read_parquet_schema_columns(parquet_path: Path) -> list[str]:
    """Read parquet schema columns.

    Args:
        parquet_path (Path): Path to the parquet.

    Returns:
        list[str]: Result of the function.
    """
    try:
        import pyarrow.dataset as ds
    except ImportError:
        return []

    dataset = ds.dataset(str(parquet_path), format="parquet")
    return [str(name) for name in dataset.schema.names]


def _align_feature_metadata(
    *,
    feature_metadata: pd.DataFrame | None,
    feature_columns: pd.Index,
) -> pd.DataFrame:
    """Align feature metadata.

    Args:
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.
        feature_columns (pd.Index): pd.Index value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if feature_metadata is None or feature_metadata.empty:
        empty = pd.DataFrame(index=feature_columns.copy())
        empty.index.name = "feature"
        return empty

    duplicated = feature_metadata.index.duplicated(keep=False)
    if duplicated.any():
        duplicates = feature_metadata.index[duplicated].unique().tolist()
        preview = ", ".join(str(name) for name in duplicates[:10])
        raise ValueError(f"Prepared feature metadata contains duplicate feature entries: {preview}")

    missing = feature_columns.difference(feature_metadata.index)
    if not missing.empty:
        preview = ", ".join(str(name) for name in missing[:10])
        raise ValueError(f"Prepared feature metadata is missing rows for selected features: {preview}")

    aligned = feature_metadata.loc[feature_columns].copy()
    aligned.index.name = "feature"
    return aligned


def _select_feature_columns(
    available_columns: list[str],
    *,
    feature_cfg: DictConfig,
    feature_metadata: pd.DataFrame | None = None,
) -> list[str]:
    """Select feature columns.

    Args:
        available_columns (list[str]): list[str] value used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    selected = _filter_feature_columns_by_scope(
        available_columns,
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )
    selected = _filter_feature_columns_by_metadata(
        selected_columns=selected,
        feature_cfg=feature_cfg,
        feature_metadata=feature_metadata,
    )
    include_names = [str(name) for name in feature_cfg.include_feature_names]
    exclude_names = [str(name) for name in feature_cfg.exclude_feature_names]
    include_patterns = [str(pattern) for pattern in feature_cfg.include_feature_patterns]
    exclude_patterns = [str(pattern) for pattern in feature_cfg.exclude_feature_patterns]

    if include_names:
        missing = sorted(set(include_names).difference(selected))
        if missing:
            glob_hint = (
                " Use features.include_feature_patterns for wildcard matching."
                if any(any(char in name for char in "*?[]") for name in include_names)
                else ""
            )
            raise ValueError(
                "Requested features.include_feature_names are not available in the prepared dataset: "
                + ", ".join(missing)
                + glob_hint
            )
        selected = [column for column in selected if column in include_names]

    if include_patterns:
        selected = [
            column
            for column in selected
            if any(fnmatch(column, pattern) for pattern in include_patterns)
        ]
        if not selected:
            raise ValueError(
                "No prepared-dataset features matched features.include_feature_patterns="
                + ", ".join(include_patterns)
            )

    if exclude_names:
        selected = [column for column in selected if column not in exclude_names]
    if exclude_patterns:
        selected = [
            column
            for column in selected
            if not any(fnmatch(column, pattern) for pattern in exclude_patterns)
        ]

    return selected


def _filter_feature_columns_by_metadata(
    selected_columns: list[str],
    *,
    feature_cfg: DictConfig,
    feature_metadata: pd.DataFrame | None,
) -> list[str]:
    """Filter feature columns by metadata.

    Args:
        selected_columns (list[str]): list[str] value used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    configured_filters = {
        "metric_family": [str(value) for value in feature_cfg.get("metric_families", [])],
        "metric": [str(value) for value in feature_cfg.get("metrics", [])],
        "scope": [str(value) for value in feature_cfg.get("scopes", [])],
        "layer_type": [str(value) for value in feature_cfg.get("layer_types", [])],
    }
    active_filters = {
        column: values
        for column, values in configured_filters.items()
        if values
    }
    if not active_filters:
        return list(selected_columns)

    if not selected_columns:
        return []

    if feature_metadata is None or feature_metadata.empty:
        raise ValueError(
            "Prepared feature metadata is required when filtering by "
            "features.metric_families / features.metrics / features.scopes / features.layer_types."
        )

    missing_features = pd.Index(selected_columns).difference(feature_metadata.index)
    if not missing_features.empty:
        preview = ", ".join(str(name) for name in missing_features[:10])
        raise ValueError(
            "Prepared feature metadata is missing rows for feature filtering: "
            + preview
        )

    metadata = feature_metadata.loc[selected_columns]
    keep_mask = pd.Series(True, index=metadata.index)
    for column, values in active_filters.items():
        if column not in metadata.columns:
            raise ValueError(
                f"Prepared feature metadata is missing required column '{column}' for configured feature filtering."
            )
        keep_mask &= metadata[column].fillna("").astype(str).isin(values)

    filtered = metadata.index[keep_mask].tolist()
    if filtered:
        log.info(
            "Prepared dataset metadata filtering kept %d feature columns using filters=%s",
            len(filtered),
            active_filters,
        )
        return filtered

    selectors = ", ".join(f"{column}={values}" for column, values in active_filters.items())
    raise ValueError(
        "No prepared-dataset feature columns match the configured metadata filters: "
        + selectors
    )


def _filter_feature_columns_by_scope(
    available_columns: list[str],
    *,
    feature_cfg: DictConfig,
    feature_metadata: pd.DataFrame | None,
) -> list[str]:
    """Filter feature columns by scope.

    Args:
        available_columns (list[str]): list[str] value used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    include_model_metrics = bool(feature_cfg.include_model_metrics)
    include_layer_metrics = bool(feature_cfg.include_layer_metrics)
    if not include_model_metrics and not include_layer_metrics:
        return []
    if include_model_metrics and include_layer_metrics:
        return list(available_columns)

    if feature_metadata is None or feature_metadata.empty:
        raise ValueError(
            "Prepared feature metadata is required when filtering by "
            "features.include_model_metrics / features.include_layer_metrics."
        )

    missing_features = pd.Index(available_columns).difference(feature_metadata.index)
    if not missing_features.empty:
        preview = ", ".join(str(name) for name in missing_features[:10])
        raise ValueError(
            "Prepared feature metadata is missing rows for feature scope filtering: "
            + preview
        )

    if "row_kind" not in feature_metadata.columns:
        raise ValueError(
            "Prepared feature metadata is missing required column 'row_kind' for feature scope filtering."
        )

    metadata = feature_metadata.loc[available_columns]
    requested_row_kind = "model" if include_model_metrics else "layers"
    selected = metadata.index[metadata["row_kind"].fillna("").astype(str) == requested_row_kind].tolist()
    if selected:
        log.info(
            "Prepared dataset feature scope selection kept %d %s feature columns",
            len(selected),
            "model-level" if include_model_metrics else "layer-level",
        )
        return selected

    requested_scope = "model-level" if include_model_metrics else "layer-level"
    raise ValueError(
        "No prepared-dataset feature columns match the requested "
        f"{requested_scope} scope. "
        "The prepared dataset may not contain that scope in its feature metadata."
    )


def _weightstatistic_dataset_cache_path(
    *,
    parquet_path: Path,
    data_cfg: DictConfig,
) -> Path:
    """Prepared dataset cache path.

    Args:
        parquet_path (Path): Path to the parquet.
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    cache_dir = Path(str(data_cfg.cache_dir)).expanduser().resolve()
    return cache_dir / f"{parquet_path.stem}__prepared_dataset.parquet"


def _require_path(raw_path: str | None, field_name: str) -> Path:
    """Require path.

    Args:
        raw_path (str | None): Raw path value to resolve.
        field_name (str): Field name used for validation.

    Returns:
        Path: Resolved path.
    """
    if not raw_path:
        raise ValueError(f"{field_name} must be set.")

    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path
