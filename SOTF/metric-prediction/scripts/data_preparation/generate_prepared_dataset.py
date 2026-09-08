from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly without setting PYTHONPATH.
# Adds metric-prediction/src (for data/utils packages) and learning-dynamics/src (for nnrun).
_SCRIPT_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _SCRIPT_DIR.parent
for _src in (_SCRIPT_DIR / "src", _REPO_ROOT / "learning-dynamics" / "src"):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import argparse
from dataclasses import dataclass
import logging
import os
import re
import pandas as pd
from nnrun.analyses.weight_statistics import load_or_build_partitioned_weight_metric_manifest
from data.weightstatistic_dataset import WeightStatisticDataset
from utils.logging_utils import setup_logging

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
TARGET_COLUMN = "test_acc"
DEFAULT_METADATA_COLUMNS = ["model_id", "path", "epoch"]
OPTIONAL_METADATA_COLUMNS = ["hyperparameter_key"]
FEATURE_METADATA_COLUMNS = [
    "row_kind",
    "layer_name",
    "layer_safe_name",
    "layer_type",
    "metric_family",
    "metric",
    "scope",
]
DEFAULT_FEATURE_NAME_PARTS = ["metric_family", "metric", "scope", "layer_safe_name"]
DEFAULT_LAYER_FEATURE_NAME_PARTS = ["metric_family", "metric", "scope", "row_kind", "layer_safe_name"]
DEFAULT_BATCH_SIZE = 250000
DEFAULT_CACHE_DIR_NAME = "cache"


@dataclass(slots=True)
class DataConfig:
    parquet_path: str
    cache_dir: str
    batch_size: int
    metadata_parquet_path: str | None = None


@dataclass(slots=True)
class FeatureConfig:
    drop_constant_features: bool
    sort_columns: bool


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


def _prepared_dataset_cache_path(
    *,
    parquet_path: Path,
    data_cfg: DataConfig,
) -> Path:
    """Prepared dataset cache path.

    Args:
        parquet_path (Path): Path to the parquet.
        data_cfg (DataConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    cache_dir = Path(data_cfg.cache_dir).expanduser().resolve() / "prepared_datasets"
    return cache_dir / f"{parquet_path.stem}__prepared_dataset.parquet"


def _partition_cache_dir(data_cfg: DataConfig) -> Path:
    """Partition cache dir.

    Args:
        data_cfg (DataConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    return Path(data_cfg.cache_dir).expanduser().resolve() / "partitioned_weight_statistic_datasets"


def _prepared_dataset_feature_metadata_cache_path(prepared_dataset_path: Path) -> Path:
    """Prepared dataset feature metadata cache path.

    Args:
        prepared_dataset_path (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    return prepared_dataset_path.with_name(f"{prepared_dataset_path.stem}__feature_metadata.parquet")


def _infer_partition_cache_inputs(
    parquet_path: Path,
) -> tuple[str, str]:
    """Infer partition cache inputs.

    Args:
        parquet_path (Path): Path to the parquet.

    Returns:
        tuple[str, str]: Result of the function.
    """
    match = re.match(r"(?P<ds_name>.+)_(?P<dataset_split>[^_]+)_all_metrics_long\.parquet$", parquet_path.name)
    if match is None:
        raise ValueError(
            "Could not infer partition-cache naming from parquet path. "
            "Expected 'complete_ws_tables/<dataset_name>_<dataset_split>_all_metrics_long.parquet'."
        )
    return match.group("ds_name"), match.group("dataset_split")


def _collect_partition_paths_by_kind(manifest: pd.DataFrame) -> tuple[list[Path], list[Path]]:
    """Collect partition paths by kind.

    Args:
        manifest (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        tuple[list[Path], list[Path]]: Result of the function.
    """
    model_paths = [
        Path(path)
        for path in manifest.loc[manifest["has_model_rows"], "model_path"].dropna().astype(str).tolist()
    ]
    layer_paths = [
        Path(path)
        for path in manifest.loc[manifest["has_layer_rows"], "layers_path"].dropna().astype(str).tolist()
    ]
    return list(dict.fromkeys(model_paths)), list(dict.fromkeys(layer_paths))


def _partition_file_columns(partition_path: Path, *, pq) -> set[str]:
    """Partition file columns.

    Args:
        partition_path (Path): Path value used by the function.
        pq (Any): Value used by the function.

    Returns:
        set[str]: Result of the function.
    """
    schema = pq.read_schema(partition_path)
    return set(schema.names)


def build_prepared_dataset_cache(
    data_cfg: DataConfig,
    feature_cfg: FeatureConfig,
) -> tuple[WeightStatisticDataset, Path]:
    """Build prepared dataset cache.

    Args:
        data_cfg (DataConfig): Configuration object used by the function.
        feature_cfg (FeatureConfig): Configuration for feature selection.

    Returns:
        tuple[WeightStatisticDataset, Path]: Result of the function.
    """
    parquet_path = _require_path(data_cfg.parquet_path, "base_dataset_path")
    cache_path = _prepared_dataset_cache_path(parquet_path=parquet_path, data_cfg=data_cfg)
    partition_cache_dir = _partition_cache_dir(data_cfg)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the partitioned parquet pipeline.") from exc

    ds_name, dataset_split = _infer_partition_cache_inputs(parquet_path)
    log.info(
        "Generating prepared dataset for dataset='%s' split='%s' cache_dir=%s",
        ds_name,
        dataset_split,
        partition_cache_dir,
    )

    manifest = load_or_build_partitioned_weight_metric_manifest(
        ds_name=ds_name,
        dataset_split=dataset_split,
        cache_dir=partition_cache_dir,
        source_path=parquet_path,
        rebuild=False,
        batch_size=int(data_cfg.batch_size),
    )
    if manifest is None or manifest.empty:
        raise ValueError(f"Partitioned cache manifest is unavailable for {parquet_path}.")

    model_paths, layer_paths = _collect_partition_paths_by_kind(manifest)
    selected_paths = list(dict.fromkeys([*model_paths, *layer_paths]))
    if not model_paths or not layer_paths:
        raise ValueError("Partition cache must contain both model and layer rows.")

    source_columns = set(_partition_file_columns(parquet_path, pq=pq))
    source_metric_columns = [column for column in TARGET_METRIC_COLUMNS if column in source_columns]
    if TARGET_COLUMN not in source_metric_columns:
        source_metric_columns = [TARGET_COLUMN, *source_metric_columns]

    available_columns = _partition_file_columns(selected_paths[0], pq=pq)
    metadata_columns = [
        *DEFAULT_METADATA_COLUMNS,
        *[column for column in OPTIONAL_METADATA_COLUMNS if column in available_columns],
    ]
    required_columns = sorted(
        LONG_FORM_REQUIRED_COLUMNS
        | set(source_metric_columns)
        | set(DEFAULT_METADATA_COLUMNS)
    )
    missing_columns = sorted(set(required_columns).difference(available_columns))

    if missing_columns:
        raise ValueError(
            "Partition files are missing columns required for prepared-dataset generation: "
            + ", ".join(missing_columns)
        )

    summary_metric_columns = source_metric_columns

    dataset, target_metrics = _build_prepared_dataset_from_partition_paths(
        model_paths=model_paths,
        layer_paths=layer_paths,
        summary_source_path=selected_paths[0],
        summary_metric_columns=summary_metric_columns,
        metadata_columns=metadata_columns,
        duckdb_temp_root=_default_duckdb_temp_root(cache_path),
        feature_cfg=feature_cfg,
    )
    _write_prepared_dataset_cache(
        cache_path,
        dataset,
        target_metrics=target_metrics,
    )

    if data_cfg.metadata_parquet_path:
        _enrich_with_metadata(cache_path, Path(data_cfg.metadata_parquet_path))

    return dataset, cache_path


def _build_prepared_dataset_from_partition_paths(
    *,
    model_paths: list[Path],
    layer_paths: list[Path],
    summary_source_path: Path,
    summary_metric_columns: list[str],
    metadata_columns: list[str],
    duckdb_temp_root: Path,
    feature_cfg: FeatureConfig,
) -> tuple[WeightStatisticDataset, pd.DataFrame]:
    """Build prepared dataset from partition paths.

    Args:
        model_paths (list[Path]): list[Path] value used by the function.
        layer_paths (list[Path]): list[Path] value used by the function.
        summary_source_path (Path): Path value used by the function.
        summary_metric_columns (list[str]): list[str] value used by the function.
        metadata_columns (list[str]): list[str] value used by the function.
        duckdb_temp_root (Path): Path value used by the function.
        feature_cfg (FeatureConfig): Configuration for feature selection.

    Returns:
        tuple[WeightStatisticDataset, pd.DataFrame]: Result of the function.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for prepared dataset generation.") from exc

    target_column = TARGET_COLUMN
    summary_columns = ["sample_idx", *summary_metric_columns, *metadata_columns]
    feature_blocks: list[pd.DataFrame] = []
    feature_metadata_blocks: list[pd.DataFrame] = []

    log.info(
        "Building prepared dataset with DuckDB from %d model files and %d layer files",
        len(model_paths),
        len(layer_paths),
    )

    con = duckdb.connect(database=":memory:")
    try:
        duckdb_settings = _configure_duckdb_runtime(con, temp_root=duckdb_temp_root)
        log.info(
            "DuckDB runtime settings: threads=%d memory_limit=%s temp_directory=%s preserve_insertion_order=false",
            duckdb_settings["threads"],
            duckdb_settings["memory_limit"],
            duckdb_settings["temp_directory"],
        )
        summary_frame = con.execute(
            _summary_query(
                path=summary_source_path,
                summary_columns=summary_columns,
            )
        ).fetchdf()
        if summary_frame.empty:
            raise ValueError("No summary rows remain after loading partitioned feature data.")
        summary_frame = summary_frame.set_index("sample_idx")

        if model_paths:
            for batch_index, path_batch in enumerate(_path_batches(model_paths), start=1):
                metadata_block = con.execute(
                    _feature_metadata_query(
                        paths=path_batch,
                        feature_name_parts=DEFAULT_FEATURE_NAME_PARTS,
                        row_kind="model",
                        layer_name="__model__",
                        layer_safe_name="__model__",
                        layer_type="model",
                    )
                ).fetchdf()
                if not metadata_block.empty:
                    feature_metadata_blocks.append(metadata_block)
                wide_block = con.execute(
                    _pivot_query(
                        partition_query=_partition_query(
                            paths=path_batch,
                            summary_columns=["sample_idx"],
                            feature_name_parts=DEFAULT_FEATURE_NAME_PARTS,
                            row_kind="model",
                            layer_name="__model__",
                            layer_safe_name="__model__",
                            layer_type="model",
                        )
                    )
                ).fetchdf()
                if wide_block.empty:
                    continue
                wide_block = wide_block.set_index("sample_idx")
                feature_blocks.append(wide_block)
                log.info(
                    "Built model feature block %d with %d samples and %d features",
                    batch_index,
                    wide_block.shape[0],
                    wide_block.shape[1],
                )

        if layer_paths:
            for batch_index, path_batch in enumerate(_path_batches(layer_paths), start=1):
                metadata_block = con.execute(
                    _feature_metadata_query(
                        paths=path_batch,
                        feature_name_parts=DEFAULT_LAYER_FEATURE_NAME_PARTS,
                        row_kind="layers",
                        layer_name=None,
                        layer_safe_name=None,
                        layer_type=None,
                    )
                ).fetchdf()
                if not metadata_block.empty:
                    feature_metadata_blocks.append(metadata_block)
                wide_block = con.execute(
                    _pivot_query(
                        partition_query=_partition_query(
                            paths=path_batch,
                            summary_columns=["sample_idx"],
                            feature_name_parts=DEFAULT_LAYER_FEATURE_NAME_PARTS,
                            row_kind="layers",
                            layer_name=None,
                            layer_safe_name=None,
                            layer_type=None,
                        )
                    )
                ).fetchdf()
                if wide_block.empty:
                    continue
                wide_block = wide_block.set_index("sample_idx")
                feature_blocks.append(wide_block)
                log.info(
                    "Built layer feature block %d with %d samples and %d features",
                    batch_index,
                    wide_block.shape[0],
                    wide_block.shape[1],
                )
    finally:
        con.close()

    if not feature_blocks:
        raise ValueError("No feature columns were produced from the selected partition files.")

    feature_table = pd.concat(feature_blocks, axis=1, sort=False)
    if feature_table.columns.has_duplicates:
        duplicate_names = pd.Index(feature_table.columns[feature_table.columns.duplicated()]).unique().tolist()
        preview = ", ".join(duplicate_names[:10])
        raise ValueError(f"Duplicate feature columns were produced: {preview}")

    log.info(
        "Concatenated %d DuckDB feature blocks into a %d x %d table",
        len(feature_blocks),
        feature_table.shape[0],
        feature_table.shape[1],
    )

    log.info(
        "DuckDB block build produced %d samples and %d raw features before filtering",
        feature_table.shape[0],
        feature_table.shape[1],
    )

    feature_table = _filter_feature_table(feature_table, feature_cfg)
    if feature_table.empty:
        raise ValueError("No usable features remain after filtering.")
    if feature_cfg.sort_columns:
        feature_table = feature_table.sort_index(axis=1)
    feature_metadata = _finalize_feature_metadata(
        feature_metadata_blocks=feature_metadata_blocks,
        feature_columns=feature_table.columns,
    )

    summary_table = summary_frame.loc[:, list(dict.fromkeys([*summary_metric_columns, *metadata_columns]))]
    summary_table = summary_table.dropna(subset=[target_column])
    combined = summary_table.join(feature_table, how="inner")
    if combined.empty:
        raise ValueError("No rows remain after joining targets with the feature table.")

    features = combined[feature_table.columns].astype("float64")
    target = combined[target_column].astype("float64")
    target_metrics = combined[summary_metric_columns].copy()
    metadata = combined[metadata_columns].copy() if metadata_columns else pd.DataFrame(index=combined.index)
    metadata.index.name = "sample_idx"
    target_metrics.index.name = "sample_idx"
    log.info(
        "Prepared dataset contains %d samples, %d features, and %d feature-metadata rows",
        len(features),
        features.shape[1],
        len(feature_metadata),
    )
    return (
        WeightStatisticDataset(
            features=features,
            target=target,
            metadata=metadata,
            feature_metadata=feature_metadata,
        ),
        target_metrics,
    )


def _write_prepared_dataset_cache(
    path: Path,
    dataset: WeightStatisticDataset,
    *,
    target_metrics: pd.DataFrame,
) -> None:
    """Write prepared dataset cache.

    Args:
        path (Path): Path to process.
        dataset (WeightStatisticDataset): Dataset to process.
        target_metrics (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        None: None.
    """
    frame = dataset.features.copy()
    for column in dataset.metadata.columns:
        frame.insert(len(frame.columns), column, dataset.metadata[column].to_numpy())
    for column in target_metrics.columns:
        frame.insert(len(frame.columns), column, target_metrics[column].to_numpy())
    frame = frame.reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    feature_metadata_path = _prepared_dataset_feature_metadata_cache_path(path)
    dataset.feature_metadata.reset_index().to_parquet(feature_metadata_path, index=False)
    log.info("Wrote prepared dataset cache: %s (%d rows, %d columns)", path, len(frame), len(frame.columns))
    log.info(
        "Wrote prepared feature metadata cache: %s (%d rows, %d columns)",
        feature_metadata_path,
        len(dataset.feature_metadata),
        len(dataset.feature_metadata.columns) + 1,
    )


def _enrich_with_metadata(prepared_path: Path, metadata_path: Path) -> None:
    """Join a metadata parquet onto the prepared dataset, adding any missing TARGET_METRIC_COLUMNS."""
    frame = pd.read_parquet(prepared_path)
    missing_cols = [col for col in TARGET_METRIC_COLUMNS if col not in frame.columns]
    if not missing_cols:
        log.info("All target metric columns already present in %s, skipping enrichment.", prepared_path)
        return

    metadata = pd.read_parquet(metadata_path)
    available_cols = [col for col in missing_cols if col in metadata.columns and not metadata[col].isna().all()]
    if not available_cols:
        log.warning(
            "Metadata parquet %s has no usable values for missing columns %s.",
            metadata_path,
            missing_cols,
        )
        return

    frame_key = pd.DataFrame({
        "model_id": frame["model_id"].astype(str).apply(lambda p: Path(p).name),
        "epoch": pd.to_numeric(frame["epoch"], errors="coerce").astype(int),
    })
    meta_join = metadata[["model_id", "epoch", *available_cols]].copy()
    meta_join["model_id"] = meta_join["model_id"].astype(str).apply(lambda p: Path(p).name)
    meta_join["epoch"] = pd.to_numeric(meta_join["epoch"], errors="coerce").astype(int)

    merged = frame_key.merge(meta_join, on=["model_id", "epoch"], how="left")
    for col in available_cols:
        frame[col] = pd.to_numeric(merged[col].values, errors="coerce")

    frame.to_parquet(prepared_path, index=False)
    log.info("Enriched prepared dataset with columns %s → %s", available_cols, prepared_path)


def _filter_feature_table(feature_table: pd.DataFrame, feature_cfg: FeatureConfig) -> pd.DataFrame:
    """Filter feature table.

    Args:
        feature_table (pd.DataFrame): pd.DataFrame value used by the function.
        feature_cfg (FeatureConfig): Configuration for feature selection.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    filtered = feature_table
    if feature_cfg.drop_constant_features:
        keep_non_constant = filtered.nunique(dropna=True) > 1
        filtered = filtered.loc[:, keep_non_constant]
    return filtered


def _summary_query(
    *,
    path: Path,
    summary_columns: list[str],
) -> str:
    """Summary query.

    Args:
        path (Path): Path to process.
        summary_columns (list[str]): list[str] value used by the function.

    Returns:
        str: Resolved string value.
    """
    aggregated_columns = []
    for column in summary_columns:
        if column == "sample_idx":
            aggregated_columns.append(_quote_identifier(column))
        else:
            aggregated_columns.append(
                f"first({_quote_identifier(column)}) AS {_quote_identifier(column)}"
            )
    return f"""
        SELECT
            {', '.join(aggregated_columns)}
        FROM read_parquet({_sql_string_literal(str(path))})
        GROUP BY {_quote_identifier('sample_idx')}
        ORDER BY {_quote_identifier('sample_idx')}
    """.strip()


def _pivot_query(*, partition_query: str) -> str:
    """Pivot query.

    Args:
        partition_query (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    return f"""
        WITH feature_rows AS (
            {partition_query}
        )
        PIVOT feature_rows
        ON feature_name
        USING first(value)
        GROUP BY {_quote_identifier('sample_idx')}
    """.strip()


def _path_batches(paths: list[Path]) -> list[list[Path]]:
    """Path batches.

    Args:
        paths (list[Path]): Paths to process.

    Returns:
        list[list[Path]]: Result of the function.
    """
    batch_size = _duckdb_feature_block_size()
    return [paths[index : index + batch_size] for index in range(0, len(paths), batch_size)]


def _partition_query(
    *,
    paths: list[Path],
    summary_columns: list[str],
    feature_name_parts: list[str],
    row_kind: str,
    layer_name: str | None,
    layer_safe_name: str | None,
    layer_type: str | None,
) -> str:
    """Partition query.

    Args:
        paths (list[Path]): Paths to process.
        summary_columns (list[str]): list[str] value used by the function.
        feature_name_parts (list[str]): list[str] value used by the function.
        row_kind (str): str value used by the function.
        layer_name (str | None): str | None value used by the function.
        layer_safe_name (str | None): str | None value used by the function.
        layer_type (str | None): str | None value used by the function.

    Returns:
        str: Resolved string value.
    """
    feature_name_expression = _feature_name_sql(
        parts=feature_name_parts,
        row_kind=row_kind,
        layer_name=layer_name,
        layer_safe_name=layer_safe_name,
        layer_type=layer_type,
    )
    select_columns = ", ".join(
        _source_column_expression(
            column=column,
            row_kind=row_kind,
            layer_name=layer_name,
            layer_safe_name=layer_safe_name,
            layer_type=layer_type,
        )
        for column in summary_columns
    )
    return f"""
        SELECT
            {select_columns},
            {feature_name_expression} AS feature_name,
            value
        FROM read_parquet([{', '.join(_sql_string_literal(str(path)) for path in paths)}])
    """.strip()


def _feature_metadata_query(
    *,
    paths: list[Path],
    feature_name_parts: list[str],
    row_kind: str,
    layer_name: str | None,
    layer_safe_name: str | None,
    layer_type: str | None,
) -> str:
    """Feature metadata query.

    Args:
        paths (list[Path]): Paths to process.
        feature_name_parts (list[str]): list[str] value used by the function.
        row_kind (str): str value used by the function.
        layer_name (str | None): str | None value used by the function.
        layer_safe_name (str | None): str | None value used by the function.
        layer_type (str | None): str | None value used by the function.

    Returns:
        str: Resolved string value.
    """
    feature_name_expression = _feature_name_sql(
        parts=feature_name_parts,
        row_kind=row_kind,
        layer_name=layer_name,
        layer_safe_name=layer_safe_name,
        layer_type=layer_type,
    )
    select_columns = ", ".join(
        _source_column_expression(
            column=column,
            row_kind=row_kind,
            layer_name=layer_name,
            layer_safe_name=layer_safe_name,
            layer_type=layer_type,
        )
        for column in FEATURE_METADATA_COLUMNS
    )
    return f"""
        SELECT DISTINCT
            {feature_name_expression} AS feature,
            {select_columns}
        FROM read_parquet([{', '.join(_sql_string_literal(str(path)) for path in paths)}])
    """.strip()


def _finalize_feature_metadata(
    *,
    feature_metadata_blocks: list[pd.DataFrame],
    feature_columns: pd.Index,
) -> pd.DataFrame:
    """Finalize feature metadata.

    Args:
        feature_metadata_blocks (list[pd.DataFrame]): list[pd.DataFrame] value used by the function.
        feature_columns (pd.Index): pd.Index value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if not feature_metadata_blocks:
        raise ValueError("No feature metadata rows were produced from the selected partition files.")

    combined = pd.concat(feature_metadata_blocks, ignore_index=True)
    combined = combined.drop_duplicates(ignore_index=True)
    duplicated = combined["feature"].duplicated(keep=False)
    if duplicated.any():
        conflicting = combined.loc[duplicated, "feature"].drop_duplicates().tolist()
        preview = ", ".join(conflicting[:10])
        raise ValueError(f"Conflicting feature metadata were produced: {preview}")

    feature_metadata = combined.set_index("feature")
    missing_features = feature_columns.difference(feature_metadata.index)
    if not missing_features.empty:
        preview = ", ".join(str(column) for column in missing_features[:10])
        raise ValueError(f"Feature metadata is missing entries for prepared features: {preview}")

    feature_metadata = feature_metadata.loc[feature_columns].copy()
    feature_metadata.index.name = "feature"
    return feature_metadata


def _feature_name_sql(
    *,
    parts: list[str],
    row_kind: str,
    layer_name: str | None,
    layer_safe_name: str | None,
    layer_type: str | None,
) -> str:
    """Feature name sql.

    Args:
        parts (list[str]): list[str] value used by the function.
        row_kind (str): str value used by the function.
        layer_name (str | None): str | None value used by the function.
        layer_safe_name (str | None): str | None value used by the function.
        layer_type (str | None): str | None value used by the function.

    Returns:
        str: Resolved string value.
    """
    expressions = [
        _feature_name_part_expression(
            part=part,
            row_kind=row_kind,
            layer_name=layer_name,
            layer_safe_name=layer_safe_name,
            layer_type=layer_type,
        )
        for part in parts
    ]
    return f"concat_ws('__', {', '.join(expressions)})"


def _feature_name_part_expression(
    *,
    part: str,
    row_kind: str,
    layer_name: str | None,
    layer_safe_name: str | None,
    layer_type: str | None,
) -> str:
    """Feature name part expression.

    Args:
        part (str): str value used by the function.
        row_kind (str): str value used by the function.
        layer_name (str | None): str | None value used by the function.
        layer_safe_name (str | None): str | None value used by the function.
        layer_type (str | None): str | None value used by the function.

    Returns:
        str: Resolved string value.
    """
    if part == "row_kind":
        return _sql_string_literal(row_kind)
    if part == "layer_name" and layer_name is not None:
        return _sql_string_literal(layer_name)
    if part == "layer_safe_name" and layer_safe_name is not None:
        return _sql_string_literal(layer_safe_name)
    if part == "layer_type" and layer_type is not None:
        return _sql_string_literal(layer_type)
    return f"CAST({_quote_identifier(part)} AS VARCHAR)"


def _source_column_expression(
    *,
    column: str,
    row_kind: str,
    layer_name: str | None,
    layer_safe_name: str | None,
    layer_type: str | None,
) -> str:
    """Source column expression.

    Args:
        column (str): Column to process.
        row_kind (str): str value used by the function.
        layer_name (str | None): str | None value used by the function.
        layer_safe_name (str | None): str | None value used by the function.
        layer_type (str | None): str | None value used by the function.

    Returns:
        str: Resolved string value.
    """
    if column == "row_kind":
        return f"{_sql_string_literal(row_kind)} AS {_quote_identifier(column)}"
    if column == "layer_name" and layer_name is not None:
        return f"{_sql_string_literal(layer_name)} AS {_quote_identifier(column)}"
    if column == "layer_safe_name" and layer_safe_name is not None:
        return f"{_sql_string_literal(layer_safe_name)} AS {_quote_identifier(column)}"
    if column == "layer_type" and layer_type is not None:
        return f"{_sql_string_literal(layer_type)} AS {_quote_identifier(column)}"
    return _quote_identifier(column)


def _quote_identifier(value: str) -> str:
    """Quote identifier.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    return '"' + value.replace('"', '""') + '"'


def _sql_string_literal(value: str) -> str:
    """Sql string literal.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    return "'" + value.replace("'", "''") + "'"


def _default_duckdb_temp_root(cache_path: Path) -> Path:
    """Default duckdb temp root.

    Args:
        cache_path (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    configured = os.environ.get("METRIC_PREDICTION_DUCKDB_TEMP_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return cache_path.parent / ".duckdb_tmp"


def _configure_duckdb_runtime(con, *, temp_root: Path) -> dict[str, object]:
    """Configure duckdb runtime.

    Args:
        con (Any): Value used by the function.
        temp_root (Path): Path value used by the function.

    Returns:
        dict[str, object]: Result of the function.
    """
    temp_root.mkdir(parents=True, exist_ok=True)

    threads = _duckdb_threads()
    memory_limit = _duckdb_memory_limit()
    temp_directory = str(temp_root)

    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit={_sql_string_literal(memory_limit)}")
    con.execute(f"SET temp_directory={_sql_string_literal(temp_directory)}")
    con.execute("SET preserve_insertion_order=false")

    return {
        "threads": threads,
        "memory_limit": memory_limit,
        "temp_directory": temp_directory,
    }


def _duckdb_threads() -> int:
    """Duckdb threads.

    Returns:
        int: Result of the function.
    """
    configured = os.environ.get("METRIC_PREDICTION_DUCKDB_THREADS")
    if configured:
        return max(1, int(configured))
    return min(8, max(1, (os.cpu_count() or 1) // 2))


def _duckdb_feature_block_size() -> int:
    """Duckdb feature block size.

    Returns:
        int: Result of the function.
    """
    configured = os.environ.get("METRIC_PREDICTION_DUCKDB_FEATURE_BLOCK_SIZE")
    if configured:
        return max(1, int(configured))
    return 8


def _duckdb_memory_limit() -> str:
    """Duckdb memory limit.

    Returns:
        str: Resolved string value.
    """
    configured = os.environ.get("METRIC_PREDICTION_DUCKDB_MEMORY_LIMIT")
    if configured:
        return configured

    total_bytes = _total_physical_memory_bytes()
    if total_bytes is None:
        return "16GB"

    target_bytes = min(int(total_bytes * 0.5), 32 * 1024**3)
    target_bytes = max(target_bytes, 8 * 1024**3)
    return _format_duckdb_memory_limit(target_bytes)


def _total_physical_memory_bytes() -> int | None:
    """Total physical memory bytes.

    Returns:
        int | None: Result of the function.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(phys_pages)


def _format_duckdb_memory_limit(num_bytes: int) -> str:
    """Format duckdb memory limit.

    Args:
        num_bytes (int): int value used by the function.

    Returns:
        str: Resolved string value.
    """
    gib = max(1, num_bytes // 1024**3)
    return f"{gib}GB"


def _build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        argparse.ArgumentParser: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description="Generate a prepared wide metric-prediction dataset from a long-form parquet file.",
    )
    parser.add_argument(
        "base_dataset_path",
        help="Path to the long-form parquet dataset.",
    )
    parser.add_argument(
        "--metadata-parquet",
        default=None,
        metavar="PARQUET",
        help=(
            "Path to a model-zoo metadata parquet produced by extract_model_zoo_metadata.py. "
            "Columns missing from the prepared dataset (train_acc, test_loss, train_loss) "
            "are joined in from this table after the dataset is written."
        ),
    )
    return parser


def _build_configs(
    base_dataset_path: str,
    metadata_parquet_path: str | None = None,
) -> tuple[DataConfig, FeatureConfig]:
    """Build configs.

    Args:
        base_dataset_path (str): str value used by the function.
        metadata_parquet_path (str | None): Optional path to a metadata parquet for enrichment.

    Returns:
        tuple[DataConfig, FeatureConfig]: Result of the function.
    """
    data_cfg = DataConfig(
        parquet_path=base_dataset_path,
        cache_dir=str(Path.cwd() / "metric-prediction" / DEFAULT_CACHE_DIR_NAME),
        batch_size=DEFAULT_BATCH_SIZE,
        metadata_parquet_path=metadata_parquet_path,
    )
    feature_cfg = FeatureConfig(
        drop_constant_features=True,
        sort_columns=True,
    )
    return data_cfg, feature_cfg


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    args = _build_parser().parse_args()
    setup_logging("INFO")

    data_cfg, feature_cfg = _build_configs(
        args.base_dataset_path,
        metadata_parquet_path=args.metadata_parquet,
    )
    dataset, cache_path = build_prepared_dataset_cache(data_cfg, feature_cfg)
    log.info(
        "Wrote prepared dataset cache at %s with %d samples and %d features",
        cache_path,
        len(dataset.features),
        dataset.features.shape[1],
    )
    log.info(f"Wrote prepared dataset cache: {cache_path}")
    log.info(f"Samples: {len(dataset.features)}")
    log.info(f"Features: {dataset.features.shape[1]}")


if __name__ == "__main__":
    main()
