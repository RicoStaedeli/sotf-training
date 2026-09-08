from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import logging
from pathlib import Path
from pprint import pformat
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from nnrun.analyses.base import AnalysisResult
from nnrun.analyses.result_store import make_result_stores
from nnrun.compute.weight_statistics_compute import (
    write_metrics_parquet_for_dataset,
)
from nnrun.zoo_info import zoo_plot_name

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------
def _complete_cache_root(cache_dir: Path) -> Path:
    root = cache_dir / "complete_ws_tables"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _per_metric_cache_root(cache_dir: Path) -> Path:
    root = cache_dir / "per_metric_ws_tables"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_cache_path(
    cache_dir: Path,
    ds_name: str,
    dataset_split: str,
    epoch_lst: Optional[Sequence[int]] = None,
) -> Path:
    complete_root = _complete_cache_root(cache_dir)
    # Keep filename stable for backwards compatibility, but note epoch_lst is not encoded here.
    return complete_root / f"{ds_name}_{dataset_split}_all_metrics_long.parquet"


def _safe_cache_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "empty"


def _partition_cache_root(
    cache_dir: Path,
    ds_name: str,
    dataset_split: str,
) -> Path:
    root = _per_metric_cache_root(cache_dir) / f"{ds_name}_{dataset_split}_combo_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _partition_manifest_path(
    cache_dir: Path,
    ds_name: str,
    dataset_split: str,
) -> Path:
    return _partition_cache_root(cache_dir, ds_name, dataset_split) / "manifest.parquet"


def _partition_combo_path(
    cache_dir: Path,
    ds_name: str,
    dataset_split: str,
    metric_family: str,
    metric: str,
    scope: str,
    row_kind: str,
) -> Path:
    root = _partition_cache_root(cache_dir, ds_name, dataset_split)
    file_name = "__".join(
        [
            _safe_cache_component(metric_family),
            _safe_cache_component(metric),
            _safe_cache_component(scope),
            _safe_cache_component(row_kind),
        ]
    )
    return root / f"{file_name}.parquet"


def _normalize_cache_signature_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _normalize_cache_signature_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_signature_value(v) for v in value]
    return value


def _plot_table_cache_root(
    cache_dir: Path,
    ds_name: str,
    dataset_split: str,
    *,
    signature: Mapping[str, Any],
) -> Path:
    normalized_signature = _normalize_cache_signature_value(dict(signature))
    root = (
        cache_dir
        / "plot_tables"
        / f"{_safe_cache_component(ds_name)}_{_safe_cache_component(dataset_split)}"
    )
    root.mkdir(parents=True, exist_ok=True)

    # Remove legacy hashed subdirectories from the previous cache layout.
    for child in sorted(root.iterdir(), key=lambda path: len(path.parts), reverse=True):
        if not child.is_dir():
            continue
        for nested in sorted(child.rglob("*"), key=lambda path: len(path.parts), reverse=True):
            if nested.is_file():
                nested.unlink()
            elif nested.is_dir():
                nested.rmdir()
        child.rmdir()

    signature_path = root / "signature.json"
    existing_signature: Optional[Any] = None
    if signature_path.exists():
        try:
            existing_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_signature = None

    if existing_signature != normalized_signature:
        for cached_file in root.glob("*.parquet"):
            cached_file.unlink()
        signature_path.write_text(json.dumps(normalized_signature, indent=2, sort_keys=True), encoding="utf-8")

    return root


def _plot_table_cache_path(cache_root: Path, table_key: str) -> Path:
    return cache_root / f"{_safe_cache_component(table_key)}.parquet"


def _load_cached_plot_table(
    cache_root: Path,
    table_key: str,
    *,
    force_recompute: bool = False,
) -> Optional[pd.DataFrame]:
    cache_path = _plot_table_cache_path(cache_root, table_key)
    if not cache_path.exists() or force_recompute:
        return None

    log.info("Loading cached plot table '%s' from %s", table_key, cache_path)
    return pd.read_parquet(cache_path)


def _write_cached_plot_table(
    cache_root: Path,
    table_key: str,
    df: pd.DataFrame,
) -> Path:
    cache_path = _plot_table_cache_path(cache_root, table_key)
    df.to_parquet(cache_path, index=False)
    log.info("Saved plot table cache '%s' to %s", table_key, cache_path)
    return cache_path


def build_partitioned_weight_metric_cache(
    *,
    ds_name: str,
    dataset_split: str,
    cache_dir: Path,
    source_path: Optional[Path] = None,
    batch_size: int = 250_000,
) -> Optional[pd.DataFrame]:
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        log.warning("pyarrow is required for partitioned weight-metric cache building: %s", exc)
        return None

    source_path = (
        source_path.expanduser().resolve()
        if source_path is not None
        else _default_cache_path(cache_dir, ds_name, dataset_split)
    )
    if not source_path.exists():
        log.warning("Cannot build partitioned cache; source cache is missing: %s", source_path)
        return None

    partition_root = _partition_cache_root(cache_dir, ds_name, dataset_split)
    manifest_path = _partition_manifest_path(cache_dir, ds_name, dataset_split)
    if manifest_path.exists():
        manifest_path.unlink()

    for old_part in partition_root.glob("*.parquet"):
        if old_part.name != manifest_path.name:
            old_part.unlink()

    dataset = ds.dataset(source_path, format="parquet")
    read_columns = list(dataset.schema.names)
    scanner = dataset.scanner(columns=read_columns, batch_size=batch_size)

    writers: Dict[Tuple[str, str, str, str], Any] = {}
    writer_schemas: Dict[Tuple[str, str, str, str], Any] = {}
    stats: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    log.info(
        "Building partitioned weight cache for dataset '%s' from %s into %s",
        ds_name,
        source_path,
        partition_root,
    )

    try:
        for record_batch in scanner.to_batches():
            batch_df = record_batch.to_pandas()
            if batch_df.empty:
                continue

            batch_df = batch_df.dropna(
                subset=["metric_family", "metric", "scope", "layer_name", "model_id", "epoch", "value"]
            )
            if batch_df.empty:
                continue

            batch_df["row_kind"] = np.where(batch_df["layer_name"] == "__model__", "model", "layers")

            for key, subdf in batch_df.groupby(["metric_family", "metric", "scope", "row_kind"], sort=False):
                metric_family, metric, scope, row_kind = key
                combo_key = (str(metric_family), str(metric), str(scope))
                out_path = _partition_combo_path(
                    cache_dir=cache_dir,
                    ds_name=ds_name,
                    dataset_split=dataset_split,
                    metric_family=str(metric_family),
                    metric=str(metric),
                    scope=str(scope),
                    row_kind=str(row_kind),
                )

                payload_df = subdf.drop(columns=["row_kind"])
                writer_key = combo_key + (str(row_kind),)
                writer = writers.get(writer_key)
                if writer is None:
                    table = pa.Table.from_pandas(payload_df, preserve_index=False)
                    writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                    writers[writer_key] = writer
                    writer_schemas[writer_key] = table.schema
                else:
                    table = pa.Table.from_pandas(
                        payload_df,
                        schema=writer_schemas[writer_key],
                        preserve_index=False,
                    )
                writer.write_table(table)

                stat = stats.setdefault(
                    combo_key,
                    {
                        "metric_family": str(metric_family),
                        "metric": str(metric),
                        "scope": str(scope),
                        "model_path": None,
                        "layers_path": None,
                        "model_rows": 0,
                        "layers_rows": 0,
                    },
                )
                path_key = f"{row_kind}_path"
                rows_key = f"{row_kind}_rows"
                stat[path_key] = str(out_path)
                stat[rows_key] += int(len(subdf))
    finally:
        for writer in writers.values():
            writer.close()

    manifest = pd.DataFrame(stats.values())
    if manifest.empty:
        log.warning("Partitioned cache build produced no combo files for dataset '%s'.", ds_name)
        return manifest

    manifest["has_model_rows"] = manifest["model_rows"] > 0
    manifest["has_layer_rows"] = manifest["layers_rows"] > 0
    manifest = manifest.sort_values(["metric_family", "metric", "scope"], kind="stable").reset_index(drop=True)
    manifest.to_parquet(manifest_path, index=False)
    log.info(
        "Built partitioned weight cache for dataset '%s': combos=%d manifest=%s",
        ds_name,
        len(manifest),
        manifest_path,
    )
    return manifest


def load_or_build_partitioned_weight_metric_manifest(
    *,
    ds_name: str,
    dataset_split: str,
    cache_dir: Path,
    source_path: Optional[Path] = None,
    rebuild: bool = False,
    batch_size: int = 250_000,
) -> Optional[pd.DataFrame]:
    manifest_path = _partition_manifest_path(cache_dir, ds_name, dataset_split)
    if manifest_path.exists() and not rebuild:
        return pd.read_parquet(manifest_path)

    return build_partitioned_weight_metric_cache(
        ds_name=ds_name,
        dataset_split=dataset_split,
        cache_dir=cache_dir,
        source_path=source_path,
        batch_size=batch_size,
    )


def ensure_weight_metric_cache(
    ds_name: str,
    root_dir: Path,
    dstype: str,
    dataset_split: str,
    epoch_lst: Sequence[int],
    cache_dir: Path,
    force_recompute: bool = False,
    skip_missing_cache: bool = True,
    preload_properties: bool = True,
    validate_checkpoints: bool = False,
    show_progress: bool = True,
    compute_percentiles: bool = True,
    compute_full_model_theta: bool = True,
    num_workers: int = 1,
    cache_write_chunk_rows: int = 1_000_000,
) -> Optional[Path]:
    cache_path = _default_cache_path(
        cache_dir=cache_dir,
        ds_name=ds_name,
        dataset_split=dataset_split,
        epoch_lst=epoch_lst,
    )

    if cache_path.exists() and not force_recompute:
        return cache_path

    if skip_missing_cache and not force_recompute and not cache_path.exists():
        log.info(
            "Skipping dataset '%s' because cached weight-statistics table is missing at %s.",
            ds_name,
            cache_path,
        )
        return None

    log.info(
        "Cache miss. Computing df_long for %s with chunked parquet writing "
        "(num_workers=%d, compute_full_model_theta=%s)",
        ds_name,
        int(num_workers),
        bool(compute_full_model_theta),
    )
    write_metrics_parquet_for_dataset(
        root_dir=root_dir,
        dstype=dstype,
        output_path=cache_path,
        epoch_lst=epoch_lst,
        dataset_split=dataset_split,
        preload_properties=preload_properties,
        validate_checkpoints=validate_checkpoints,
        show_progress=show_progress,
        compute_percentiles=compute_percentiles,
        compute_full_model_theta=compute_full_model_theta,
        num_workers=num_workers,
        chunk_rows=cache_write_chunk_rows,
    )
    log.info("Saved df_long cache to %s", cache_path)
    return cache_path


def load_or_compute_df_long(
    ds_name: str,
    root_dir: Path,
    dstype: str,
    dataset_split: str,
    epoch_lst: Sequence[int],
    cache_dir: Path,
    force_recompute: bool = False,
    skip_missing_cache: bool = True,
    preload_properties: bool = True,
    validate_checkpoints: bool = False,
    show_progress: bool = True,
    parquet_columns: Optional[Sequence[str]] = None,
    parquet_filters: Optional[Sequence[Any]] = None,
    compute_percentiles: bool = True,
    compute_full_model_theta: bool = True,
    num_workers: int = 1,
    cache_write_chunk_rows: int = 1_000_000,
) -> Optional[pd.DataFrame]:
    cache_path = ensure_weight_metric_cache(
        ds_name=ds_name,
        root_dir=root_dir,
        dstype=dstype,
        dataset_split=dataset_split,
        epoch_lst=epoch_lst,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        skip_missing_cache=skip_missing_cache,
        preload_properties=preload_properties,
        validate_checkpoints=validate_checkpoints,
        show_progress=show_progress,
        compute_percentiles=compute_percentiles,
        compute_full_model_theta=compute_full_model_theta,
        num_workers=num_workers,
        cache_write_chunk_rows=cache_write_chunk_rows,
    )
    if cache_path is None:
        return None

    log.info("Loading cached df_long from %s", cache_path)
    if parquet_columns:
        return pd.read_parquet(cache_path, columns=list(parquet_columns), filters=parquet_filters)
    return pd.read_parquet(cache_path, filters=parquet_filters)


# -----------------------------------------------------------------------------
# Filtering / aggregation
# -----------------------------------------------------------------------------
def _resolve_group_cols_from_available_columns(
    available_columns: Sequence[str],
    requested_group_cols: Optional[Sequence[str]],
) -> List[str]:
    if not requested_group_cols:
        return []

    available = {str(col) for col in available_columns}
    found = [c for c in requested_group_cols if c in available]
    missing = [c for c in requested_group_cols if c not in available]
    if missing:
        log.warning("Ignoring missing group columns: %s", missing)
    return found


def _resolve_group_cols(
    df: pd.DataFrame,
    requested_group_cols: Optional[Sequence[str]],
) -> List[str]:
    return _resolve_group_cols_from_available_columns(df.columns.tolist(), requested_group_cols)


def _to_int_epoch(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _infer_available_epochs(
    df: pd.DataFrame,
    *,
    epoch_col: str,
    epoch_range: Optional[Tuple[int, int]] = None,
) -> List[int]:
    # Respect caller-provided epoch_col; fall back if absent.
    if epoch_col not in df.columns:
        fallback = "epoch" if epoch_col != "epoch" else "training_iteration"
        if fallback not in df.columns:
            return []
        epoch_col = fallback

    epochs = _to_int_epoch(df[epoch_col]).dropna().astype(int)
    if epoch_range is not None:
        lo, hi = epoch_range
        epochs = epochs[(epochs >= int(lo)) & (epochs <= int(hi))]

    return sorted(epochs.unique().tolist())


def _read_parquet_schema_columns(parquet_path: Path) -> Optional[List[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        log.warning("pyarrow is required to inspect parquet schema for %s: %s", parquet_path, exc)
        return None

    return list(pq.read_schema(parquet_path).names)


def _build_plot_parquet_filters(
    *,
    metric_family: str,
    metric: Optional[str],
    scope: str,
    epoch_col: str,
    epoch_range: Optional[Tuple[int, int]],
) -> Optional[List[Tuple[str, str, Any]]]:
    filters: List[Tuple[str, str, Any]] = []

    family_key = str(metric_family).lower()
    metric_key = None if metric is None else str(metric).lower()
    scope_key = str(scope).lower()

    if family_key == "unterthiner":
        filters.append((epoch_col, "!=", 0))
        if epoch_range is not None:
            lo, hi = epoch_range
            filters.append((epoch_col, ">=", int(lo)))
            filters.append((epoch_col, "<=", int(hi)))
        return filters

    if family_key != "all":
        filters.append(("metric_family", "==", metric_family))
    if metric is not None and metric_key != "all":
        filters.append(("metric", "==", metric))
    if scope_key != "all":
        filters.append(("scope", "==", scope))

    filters.append((epoch_col, "!=", 0))
    if epoch_range is not None:
        lo, hi = epoch_range
        filters.append((epoch_col, ">=", int(lo)))
        filters.append((epoch_col, "<=", int(hi)))

    return filters or None


def _optimize_df_dtypes(df: pd.DataFrame, extra_group_cols: Sequence[str]) -> pd.DataFrame:
    # Copy once here so downstream code can rely on stable ownership.
    out = df.copy()

    cat_candidates = [
        "dataset_name",
        "metric_family",
        "metric",
        "scope",
        "layer_name",
        "layer_safe_name",
        "layer_type",
        "__combo_key__",
        *list(extra_group_cols),
    ]
    for col in cat_candidates:
        if col in out.columns and out[col].dtype == "object":
            nunique = out[col].nunique(dropna=False)
            # Strings with repetition benefit most from category.
            if nunique <= max(10_000, int(len(out) * 0.5)):
                out[col] = out[col].astype("category")

    if "value" in out.columns and out["value"].dtype != np.float64:
        out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")

    # Normalize epoch-like columns to numeric once.
    for col in ("epoch", "training_iteration"):
        if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _discover_metric_combinations_for_analysis(
    df: pd.DataFrame,
    *,
    metric_family: str,
    metric: Optional[str],
    scope: str,
) -> List[tuple[str, str, str]]:
    family_key = str(metric_family).lower()
    scope_key = str(scope).lower()
    metric_key = None if metric is None else str(metric).lower()

    d = df
    if d.empty:
        return []

    if family_key == "unterthiner":
        d = d[
            (d["metric_family"] == "distribution")
            | (
                (d["metric_family"] == "norm")
                & (d["metric"].isin(["l1", "l2"]))
                & (d["scope"].isin(["weight_only", "bias_only"]))
            )
        ]
    else:
        if family_key != "all":
            d = d[d["metric_family"] == metric_family]
        if metric is not None and metric_key != "all":
            d = d[d["metric"] == metric]
        if scope_key != "all":
            d = d[d["scope"] == scope]

    if d.empty:
        return []

    if metric is None and family_key not in {"all", "unterthiner"} and scope_key != "all":
        available_metrics = sorted(d["metric"].dropna().astype(str).unique().tolist())
        if not available_metrics:
            return []
        first_metric = available_metrics[0]
        d = d[d["metric"] == first_metric]
        log.info("No metric specified; using first available metric '%s'.", first_metric)

    combos = (
        d[["metric_family", "metric", "scope"]]
        .drop_duplicates()
        .sort_values(["metric_family", "metric", "scope"])
    )
    return [tuple(x) for x in combos.to_numpy()]


def _build_partition_combo_source_map(manifest: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    if manifest is None or manifest.empty:
        return {}

    combo_sources: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in manifest.itertuples(index=False):
        combo_key = (str(row.metric_family), str(row.metric), str(row.scope))
        payload: Dict[str, str] = {}
        if bool(getattr(row, "has_model_rows", False)) and getattr(row, "model_path", None):
            payload["model_path"] = str(row.model_path)
        if bool(getattr(row, "has_layer_rows", False)) and getattr(row, "layers_path", None):
            payload["layers_path"] = str(row.layers_path)
        combo_sources[combo_key] = payload

    return combo_sources


def _prepare_combo_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__combo_key__"] = (
        out["metric_family"].astype(str)
        + "\x1f"
        + out["metric"].astype(str)
        + "\x1f"
        + out["scope"].astype(str)
    )
    if out["__combo_key__"].dtype == "object":
        out["__combo_key__"] = out["__combo_key__"].astype("category")
    return out


def _filter_metric_table_by_combo_key(
    df: pd.DataFrame,
    combo_key: str,
    include_model_row: bool = False,
) -> pd.DataFrame:
    out = df[df["__combo_key__"] == combo_key]
    if include_model_row:
        return out[out["layer_name"] == "__model__"]
    return out[out["layer_name"] != "__model__"]


def _load_partition_combo_frame(
    parquet_path: str | Path,
    *,
    columns: Sequence[str],
    epoch_col: str,
    epoch_range: Optional[Tuple[int, int]],
    extra_group_cols: Sequence[str],
) -> pd.DataFrame:
    filters: List[Tuple[str, str, Any]] = [(epoch_col, "!=", 0)]
    if epoch_range is not None:
        lo, hi = epoch_range
        filters.append((epoch_col, ">=", int(lo)))
        filters.append((epoch_col, "<=", int(hi)))

    frame = pd.read_parquet(
        Path(parquet_path),
        columns=list(dict.fromkeys(columns)),
        filters=filters,
    )
    if frame.empty:
        return frame
    return _optimize_df_dtypes(frame, extra_group_cols)


def _aggregate_metric_over_epoch(
    df_metric: pd.DataFrame,
    epoch_col: str = "epoch",
    group_cols: Optional[Sequence[str]] = None,
    include_quantiles: bool = True,
) -> pd.DataFrame:
    group_cols = list(group_cols or [])
    by = [epoch_col] + group_cols

    grouped = df_metric.groupby(
        by,
        observed=True,
        sort=False,
        dropna=False,
    )["value"]

    # Fast path for standard reducers.
    base = grouped.agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        count="count",
    )

    if not include_quantiles:
        return base.reset_index().sort_values(by, kind="stable")

    # Compute all quantiles in one grouped pass instead of 4 separate lambda aggregations.
    q = (
        grouped.quantile([0.10, 0.25, 0.75, 0.90])
        .unstack(level=-1)
        .rename(columns={0.10: "q10", 0.25: "q25", 0.75: "q75", 0.90: "q90"})
    )

    out = base.join(q, how="left").reset_index()
    return out.sort_values(by, kind="stable")


def _aggregate_layers_metric_over_epoch(
    df_metric: pd.DataFrame,
    epoch_col: str = "epoch",
    group_cols: Optional[Sequence[str]] = None,
    stat: str = "mean",
) -> pd.DataFrame:
    group_cols = list(group_cols or [])
    by = [epoch_col] + group_cols + ["layer_order", "layer_name", "layer_safe_name", "layer_type"]

    grouped = df_metric.groupby(
        by,
        observed=True,
        sort=False,
        dropna=False,
        as_index=False,
    )["value"]

    if stat == "mean":
        out = grouped.mean()
    elif stat == "median":
        out = grouped.median()
    else:
        raise ValueError(f"Unsupported stat '{stat}'. Use 'mean' or 'median'.")

    return out.sort_values(by, kind="stable")


def _log_df_size(label: str, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        log.info("%s: <none>", label)
        return

    rows, cols = df.shape
    mem_mib = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    log.info("%s: rows=%d cols=%d memory=%.2f MiB", label, rows, cols, mem_mib)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _plot_single_metric_dataset_aggregate_long(
    df_stats: pd.DataFrame,
    metric_name: str,
    ds_name: str,
    epoch_col: str = "epoch",
    group_cols: Optional[Sequence[str]] = None,
    png_dpi: int = 200,
) -> plt.Figure:
    group_cols = list(group_cols or [])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=png_dpi)

    if not group_cols:
        sub = df_stats.sort_values(epoch_col, kind="stable")
        x = sub[epoch_col].to_numpy()
        ax.plot(x, sub["mean"].to_numpy(), marker="o", label="mean")
        ax.plot(x, sub["median"].to_numpy(), linestyle="--", label="median")

        if {"q25", "q75"}.issubset(sub.columns):
            ax.fill_between(x, sub["q25"].to_numpy(), sub["q75"].to_numpy(), alpha=0.25, label="25-75%")
        if {"q10", "q90"}.issubset(sub.columns):
            ax.fill_between(x, sub["q10"].to_numpy(), sub["q90"].to_numpy(), alpha=0.15, label="10-90%")
    else:
        for group_key, subdf in df_stats.groupby(group_cols, observed=True, dropna=False, sort=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label = ", ".join(f"{k}={v}" for k, v in zip(group_cols, group_key))
            subdf = subdf.sort_values(epoch_col, kind="stable")
            ax.plot(subdf[epoch_col].to_numpy(), subdf["mean"].to_numpy(), marker="o", label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{ds_name} | {metric_name} over dataset")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def _plot_all_layers_lines_long(
    df_agg: pd.DataFrame,
    ds_name: str,
    metric_label: str,
    epoch_col: str = "epoch",
    group_cols: Optional[Sequence[str]] = None,
    png_dpi: int = 200,
    max_groups_in_legend: int = 20,
    max_lines_to_plot: int = 300,
    show_legend_threshold: int = 60,
) -> Optional[plt.Figure]:
    group_cols = list(group_cols or [])

    layer_key_cols = ["layer_order", "layer_name", "layer_type"]
    if group_cols:
        n_lines = df_agg[group_cols + layer_key_cols].drop_duplicates().shape[0]
    else:
        n_lines = df_agg[layer_key_cols].drop_duplicates().shape[0]

    if n_lines > max_lines_to_plot:
        log.warning(
            "Skipping all-layer line plot for '%s' because it would draw %d lines (> %d).",
            ds_name,
            n_lines,
            max_lines_to_plot,
        )
        return None

    fig, ax = plt.subplots(figsize=(12, 7), dpi=png_dpi)
    use_markers = n_lines <= 40

    if not group_cols:
        grouped = df_agg.groupby(layer_key_cols, observed=True, dropna=False, sort=False)
        for layer_key, subdf in grouped:
            layer_order, layer_name, layer_type = layer_key
            subdf = subdf.sort_values(epoch_col, kind="stable")
            label = f"{int(layer_order):03d} | {layer_name} | {layer_type}"
            ax.plot(
                subdf[epoch_col].to_numpy(),
                subdf["value"].to_numpy(),
                marker="o" if use_markers else None,
                label=label,
            )
    else:
        grouped_outer = list(df_agg.groupby(group_cols, observed=True, dropna=False, sort=False))
        if len(grouped_outer) > max_groups_in_legend:
            log.warning("Too many groups (%d) for a readable all-layer line plot.", len(grouped_outer))

        for group_key, group_df in grouped_outer:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_label = ", ".join(f"{k}={v}" for k, v in zip(group_cols, group_key))

            for layer_key, subdf in group_df.groupby(layer_key_cols, observed=True, dropna=False, sort=False):
                layer_order, layer_name, layer_type = layer_key
                subdf = subdf.sort_values(epoch_col, kind="stable")
                label = f"{group_label} | {int(layer_order):03d} | {layer_name} | {layer_type}"
                ax.plot(
                    subdf[epoch_col].to_numpy(),
                    subdf["value"].to_numpy(),
                    marker="o" if use_markers else None,
                    label=label,
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{ds_name} | All-layer {metric_label} over dataset")
    ax.grid(True, alpha=0.3)

    if n_lines <= show_legend_threshold:
        ax.legend(fontsize=7, ncol=2)
    else:
        log.info("Suppressing legend for '%s' all-layer plot with %d lines.", ds_name, n_lines)

    return fig


def _plot_layer_epoch_heatmap_long(
    df_agg: pd.DataFrame,
    ds_name: str,
    metric_label: str,
    epoch_col: str = "epoch",
    group_filter_label: Optional[str] = None,
    png_dpi: int = 200,
) -> Optional[plt.Figure]:
    if df_agg.empty:
        return None

    heat = (
        df_agg.pivot_table(
            index=["layer_order", "layer_name"],
            columns=epoch_col,
            values="value",
            aggfunc="mean",
            observed=True,
        )
        .sort_index(level=0)
        .sort_index(axis=1)
    )

    if heat.empty:
        return None

    matrix = heat.to_numpy()
    fig, ax = plt.subplots(figsize=(14, 6), dpi=png_dpi)
    im = ax.imshow(matrix, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(list(heat.columns))

    y_labels = [f"{int(idx[0]):03d} | {idx[1]}" for idx in heat.index]
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    title = f"{ds_name} | Heatmap of {metric_label} over dataset"
    if group_filter_label:
        title += f" | {group_filter_label}"

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label=metric_label)
    return fig


def _prepare_distribution_histogram(
    values: pd.Series,
    *,
    bins: int,
) -> Dict[str, Any]:
    arr = values.to_numpy(dtype=np.float64, copy=False)
    if arr.size == 0:
        return {
            "plot_values": arr,
            "hist_bins": 1,
            "mean_value": np.nan,
            "median_value": np.nan,
            "xlabel": "Value",
            "mode_label": "linear",
            "clipped_count": 0,
        }

    unique_values = int(values.nunique(dropna=True))
    hist_bins = min(int(bins), max(10, int(np.sqrt(len(values)))))
    if unique_values <= 1:
        hist_bins = 1

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    median_val = float(np.median(arr))
    q95 = float(np.quantile(arr, 0.95))
    q99 = float(np.quantile(arr, 0.99))
    clipped_count = 0

    # Heavy-tailed nonnegative metrics become unreadable on a linear axis when a
    # few checkpoints explode. In that case, plot log10(1 + x) instead.
    positive_heavy_tail = (
        min_val >= 0.0
        and max_val > 0.0
        and (
            (median_val > 0.0 and q99 / max(median_val, 1e-12) >= 1_000.0)
            or (q95 > 0.0 and max_val / max(q95, 1e-12) >= 100.0)
        )
    )
    if positive_heavy_tail:
        plot_values = np.log10(arr + 1.0)
        return {
            "plot_values": plot_values,
            "hist_bins": hist_bins,
            "mean_value": float(np.log10(float(np.mean(arr)) + 1.0)),
            "median_value": float(np.log10(median_val + 1.0)),
            "xlabel": "log10(1 + value)",
            "mode_label": "log1p",
            "clipped_count": 0,
        }

    clipped_high = (
        arr.size >= 25
        and np.isfinite(q99)
        and np.isfinite(max_val)
        and max_val > q99
        and (q95 > 0.0 and max_val / max(q95, 1e-12) >= 20.0)
    )
    if clipped_high:
        plot_values = np.clip(arr, None, q99)
        clipped_count = int(np.sum(arr > q99))
        return {
            "plot_values": plot_values,
            "hist_bins": hist_bins,
            "mean_value": float(np.mean(plot_values)),
            "median_value": float(np.median(plot_values)),
            "xlabel": "Value (clipped at p99)",
            "mode_label": "clip_p99",
            "clipped_count": clipped_count,
        }

    return {
        "plot_values": arr,
        "hist_bins": hist_bins,
        "mean_value": float(np.mean(arr)),
        "median_value": median_val,
        "xlabel": "Value",
        "mode_label": "linear",
        "clipped_count": 0,
    }


def _plot_metric_distribution_grid_long(
    distribution_tables: Mapping[tuple[str, str, str], Any],
    combos: Sequence[tuple[str, str, str]],
    ds_name: str,
    png_dpi: int = 200,
    nrows: int = 3,
    ncols: int = 4,
    bins: int = 40,
) -> List[Tuple[int, plt.Figure]]:
    if not distribution_tables or not combos:
        return []

    plots_per_page = max(1, int(nrows) * int(ncols))
    page_figures: List[Tuple[int, plt.Figure]] = []
    legend_handles = [
        Line2D([0], [0], color="#E45756", linestyle="-", linewidth=1.5, label="mean"),
        Line2D([0], [0], color="#F58518", linestyle="--", linewidth=1.5, label="median"),
    ]

    for page_idx, start in enumerate(range(0, len(combos), plots_per_page), start=1):
        page_combos = combos[start : start + plots_per_page]
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols, 3.4 * nrows),
            dpi=png_dpi,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        for ax in axes_flat[len(page_combos) :]:
            ax.axis("off")

        for ax, (metric_family, metric, scope) in zip(axes_flat, page_combos):
            source = distribution_tables.get((metric_family, metric, scope))
            if isinstance(source, (str, Path)):
                df_metric = pd.read_parquet(source, columns=["layer_name", "value"])
            else:
                df_metric = source
            if df_metric is None or df_metric.empty:
                ax.axis("off")
                continue

            row_kind_label = "model" if df_metric["layer_name"].eq("__model__").all() else "layers"

            raw_values = pd.to_numeric(df_metric["value"], errors="coerce")
            finite_mask = np.isfinite(raw_values.to_numpy(dtype=np.float64, na_value=np.nan))
            values = raw_values.loc[finite_mask]
            dropped_non_finite = int(len(raw_values) - len(values))

            if values.empty:
                ax.axis("off")
                continue

            hist_meta = _prepare_distribution_histogram(values, bins=bins)
            plot_values = hist_meta["plot_values"]

            ax.hist(
                plot_values,
                bins=hist_meta["hist_bins"],
                color="#4C78A8",
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.axvline(hist_meta["mean_value"], color="#E45756", linewidth=1.5)
            ax.axvline(hist_meta["median_value"], color="#F58518", linestyle="--", linewidth=1.5)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_title(f"{metric_family}\n{metric} | {scope}", fontsize=10)
            ax.set_xlabel(hist_meta["xlabel"], fontsize=9)
            ax.set_ylabel("Count")
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
            ax.text(
                0.98,
                0.95,
                (
                    f"n={len(values)} | dropped={dropped_non_finite} | "
                    f"clipped={hist_meta['clipped_count']} | {hist_meta['mode_label']} | {row_kind_label}"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )

        total_pages = int(np.ceil(len(combos) / plots_per_page))
        fig.suptitle(
            f"{ds_name} | Weight-statistic distributions ({page_idx}/{total_pages})",
            fontsize=14,
            y=0.995,
        )
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        page_figures.append((page_idx, fig))

    return page_figures


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def run_weight_analysis(
    df: pd.DataFrame,
    config: Dict[str, Any],
    registry: Dict[str, Any],
) -> AnalysisResult:
    if "dataset_name" not in df.columns:
        raise ValueError("weight_analysis: df must contain dataset_name column.")

    requested_metric_family = str(config.get("metric_family", "norm"))
    requested_metric = config.get("metric", None)
    requested_scope = str(config.get("scope", "default"))

    selected_layer_name = config.get("layer_name")
    epoch_col = str(config.get("epoch_col", "epoch"))
    source_epoch_col = str(config.get("source_epoch_col", "training_iteration"))
    epoch_range = tuple(config.get("epoch_range", [])) or None
    png_dpi = int(config.get("png_dpi", 200))
    dataset_split = str(config.get("dataset_split", "train"))
    cache_dir = Path(config.get("cache_dir", "./cache/weight_metrics"))
    force_recompute = bool(config.get("force_recompute", False))
    cache_only = bool(config.get("cache_only", False))
    skip_missing_cache = bool(config.get("skip_missing_cache", True))

    groupby_cols = config.get("groupby_cols", [])
    include_single_metric_plot = bool(config.get("include_single_metric_plot", True))
    include_all_layer_plot = bool(config.get("include_all_layer_plot", True))
    include_heatmap = bool(config.get("include_heatmap", True))
    include_distribution_grid = bool(config.get("include_distribution_grid", True))
    heatmap_group_selection = config.get("heatmap_group_selection", None)

    # New tuning knobs
    include_quantiles = bool(config.get("include_quantiles", True))
    max_all_layer_lines = int(config.get("max_all_layer_lines", 300))
    all_layer_legend_threshold = int(config.get("all_layer_legend_threshold", 60))
    distribution_grid_rows = int(config.get("distribution_grid_rows", 3))
    distribution_grid_cols = int(config.get("distribution_grid_cols", 4))
    distribution_grid_bins = int(config.get("distribution_grid_bins", 40))
    distribution_grid_prefer_layer_rows = bool(config.get("distribution_grid_prefer_layer_rows", True))
    compute_percentiles = bool(config.get("compute_percentiles", True))
    compute_full_model_theta = bool(config.get("compute_full_model_theta", True))
    weight_compute_num_workers = int(config.get("weight_compute_num_workers", 1))
    cache_write_chunk_rows = int(config.get("cache_write_chunk_rows", 1_000_000))
    force_recompute_plot_tables = bool(config.get("force_recompute_plot_tables", False))
    use_partitioned_cache = bool(config.get("use_partitioned_cache", True))
    partition_batch_size = int(config.get("partition_batch_size", 250_000))

    effective_config = {
        "metric_family": requested_metric_family,
        "metric": requested_metric,
        "scope": requested_scope,
        "layer_name": selected_layer_name,
        "epoch_col": epoch_col,
        "source_epoch_col": source_epoch_col,
        "epoch_range": list(epoch_range) if epoch_range is not None else [],
        "png_dpi": png_dpi,
        "dataset_split": dataset_split,
        "cache_dir": str(cache_dir),
        "force_recompute": force_recompute,
        "cache_only": cache_only,
        "skip_missing_cache": skip_missing_cache,
        "groupby_cols": list(groupby_cols),
        "include_single_metric_plot": include_single_metric_plot,
        "include_all_layer_plot": include_all_layer_plot,
        "include_heatmap": include_heatmap,
        "include_distribution_grid": include_distribution_grid,
        "heatmap_group_selection": heatmap_group_selection,
        "include_quantiles": include_quantiles,
        "max_all_layer_lines": max_all_layer_lines,
        "all_layer_legend_threshold": all_layer_legend_threshold,
        "distribution_grid_rows": distribution_grid_rows,
        "distribution_grid_cols": distribution_grid_cols,
        "distribution_grid_bins": distribution_grid_bins,
        "distribution_grid_prefer_layer_rows": distribution_grid_prefer_layer_rows,
        "compute_percentiles": compute_percentiles,
        "compute_full_model_theta": compute_full_model_theta,
        "weight_compute_num_workers": weight_compute_num_workers,
        "cache_write_chunk_rows": cache_write_chunk_rows,
        "force_recompute_plot_tables": force_recompute_plot_tables,
        "use_partitioned_cache": use_partitioned_cache,
        "partition_batch_size": partition_batch_size,
    }
    log.info("run_weight_analysis config:\n%s", pformat(effective_config, sort_dicts=True))

    tables, figures = make_result_stores(config, default_png_dpi=png_dpi)
    datasets = sorted(df["dataset_name"].dropna().unique().tolist())

    # Read only what we need from parquet if possible.
    parquet_cols = {
        epoch_col,
        source_epoch_col,
        "metric_family",
        "metric",
        "scope",
        "layer_name",
        "layer_order",
        "layer_safe_name",
        "layer_type",
        "value",
        *list(groupby_cols),
    }

    for ds_name in datasets:
        ds_cfg = registry.get(ds_name)
        if ds_cfg is None:
            log.warning("weight_analysis: dataset '%s' is missing in registry. Skipping.", ds_name)
            continue

        root_dir = Path(ds_cfg.root_dir)
        dstype = ds_cfg.dstype

        log.info("Root dir: %s", root_dir)
        log.info("Dataset Type: %s", dstype)

        dfd = df[df["dataset_name"] == ds_name] if "dataset_name" in df.columns else pd.DataFrame()
        _log_df_size(f"{ds_name} source dataset", dfd)
        epoch_lst = _infer_available_epochs(
            dfd,
            epoch_col=source_epoch_col,
            epoch_range=None,
        )
        if not epoch_lst:
            epoch_lst = list(range(0, 51))
            log.warning(
                "Could not infer epochs for dataset '%s' from dataframe column '%s'. Falling back to 0..50.",
                ds_name,
                source_epoch_col,
            )

        log.info("Epoch lst: %s", epoch_lst)

        plot_table_force_recompute = force_recompute or force_recompute_plot_tables
        plot_parquet_filters = _build_plot_parquet_filters(
            metric_family=requested_metric_family,
            metric=requested_metric,
            scope=requested_scope,
            epoch_col=epoch_col,
            epoch_range=epoch_range,
        )
        weight_cache_path = _default_cache_path(
            cache_dir=cache_dir,
            ds_name=ds_name,
            dataset_split=dataset_split,
            epoch_lst=epoch_lst,
        )
        df_long: Optional[pd.DataFrame] = None
        partition_manifest: Optional[pd.DataFrame] = None
        combo_source_map: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        valid_groupby_cols: List[str] = list(groupby_cols)
        plot_source_mode = "full_cache"

        if cache_only:
            df_long = load_or_compute_df_long(
                ds_name=ds_name,
                root_dir=root_dir,
                dstype=dstype,
                dataset_split=dataset_split,
                epoch_lst=epoch_lst,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
                skip_missing_cache=skip_missing_cache,
                preload_properties=True,
                validate_checkpoints=False,
                show_progress=True,
                parquet_columns=sorted(parquet_cols),
                compute_percentiles=compute_percentiles,
                compute_full_model_theta=compute_full_model_theta,
                num_workers=weight_compute_num_workers,
                cache_write_chunk_rows=cache_write_chunk_rows,
            )
            if df_long is None:
                continue
            _log_df_size(f"{ds_name} cached weight table", df_long)
            tables[f"{ds_name}_{dataset_split}_all_metrics_long"] = df_long
            log.info(
                "cache_only=True for dataset '%s'. Returning cached weight-statistics table without plots.",
                ds_name,
            )
            continue

        ensured_cache_path = ensure_weight_metric_cache(
            ds_name=ds_name,
            root_dir=root_dir,
            dstype=dstype,
            dataset_split=dataset_split,
            epoch_lst=epoch_lst,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            skip_missing_cache=skip_missing_cache,
            preload_properties=True,
            validate_checkpoints=False,
            show_progress=True,
            compute_percentiles=compute_percentiles,
            compute_full_model_theta=compute_full_model_theta,
            num_workers=weight_compute_num_workers,
            cache_write_chunk_rows=cache_write_chunk_rows,
        )
        if ensured_cache_path is None:
            continue
        weight_cache_path = ensured_cache_path

        cache_columns = _read_parquet_schema_columns(weight_cache_path) if use_partitioned_cache else None
        if cache_columns is not None:
            if epoch_col not in cache_columns:
                raise KeyError(f"Epoch column '{epoch_col}' not found.")
            valid_groupby_cols = _resolve_group_cols_from_available_columns(cache_columns, groupby_cols)
            partition_manifest = load_or_build_partitioned_weight_metric_manifest(
                ds_name=ds_name,
                dataset_split=dataset_split,
                cache_dir=cache_dir,
                source_path=weight_cache_path,
                rebuild=force_recompute,
                batch_size=partition_batch_size,
            )
            if partition_manifest is not None:
                combo_source_map = _build_partition_combo_source_map(partition_manifest)
                plot_source_mode = "partitioned"
                _log_df_size(f"{ds_name} partition manifest", partition_manifest)

        if plot_source_mode != "partitioned":
            df_long = load_or_compute_df_long(
                ds_name=ds_name,
                root_dir=root_dir,
                dstype=dstype,
                dataset_split=dataset_split,
                epoch_lst=epoch_lst,
                cache_dir=cache_dir,
                force_recompute=False,
                skip_missing_cache=skip_missing_cache,
                preload_properties=True,
                validate_checkpoints=False,
                show_progress=True,
                parquet_columns=sorted(parquet_cols),
                parquet_filters=plot_parquet_filters,
                compute_percentiles=compute_percentiles,
                compute_full_model_theta=compute_full_model_theta,
                num_workers=weight_compute_num_workers,
                cache_write_chunk_rows=cache_write_chunk_rows,
            )
            if df_long is None:
                continue
            _log_df_size(f"{ds_name} cached weight table", df_long)
            if epoch_col not in df_long.columns:
                raise KeyError(f"Epoch column '{epoch_col}' not found.")
            valid_groupby_cols = _resolve_group_cols(df_long, groupby_cols)

        weight_cache_stat = weight_cache_path.stat() if weight_cache_path.exists() else None
        plot_cache_root = _plot_table_cache_root(
            cache_dir=cache_dir,
            ds_name=ds_name,
            dataset_split=dataset_split,
            signature={
                "metric_family": requested_metric_family,
                "metric": requested_metric,
                "scope": requested_scope,
                "layer_name": selected_layer_name,
                "epoch_col": epoch_col,
                "epoch_range": list(epoch_range) if epoch_range is not None else [],
                "groupby_cols": list(valid_groupby_cols),
                "heatmap_group_selection": heatmap_group_selection,
                "include_quantiles": include_quantiles,
                "distribution_grid_prefer_layer_rows": distribution_grid_prefer_layer_rows,
                "weight_cache_path": str(weight_cache_path),
                "weight_cache_mtime_ns": getattr(weight_cache_stat, "st_mtime_ns", None),
                "weight_cache_size": getattr(weight_cache_stat, "st_size", None),
            },
        )

        df_plot_cache: Optional[pd.DataFrame] = None
        combo_table_cache: Dict[str, pd.DataFrame] = {}
        combo_partition_cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
        combo_read_columns = [
            epoch_col,
            "layer_name",
            "layer_order",
            "layer_safe_name",
            "layer_type",
            "value",
            *list(valid_groupby_cols),
        ]

        def ensure_df_plot() -> pd.DataFrame:
            nonlocal df_plot_cache
            if df_long is None:
                return pd.DataFrame()
            if df_plot_cache is None:
                epoch_mask = pd.Series(True, index=df_long.index)
                epoch_mask &= df_long[epoch_col] != 0
                if epoch_range is not None:
                    lo, hi = epoch_range
                    epoch_mask &= (df_long[epoch_col] >= lo) & (df_long[epoch_col] <= hi)

                df_plot_cache = df_long.loc[epoch_mask]
                _log_df_size(f"{ds_name} filtered plotting table", df_plot_cache)

                df_plot_cache = _optimize_df_dtypes(df_plot_cache, valid_groupby_cols)
                _log_df_size(f"{ds_name} optimized plotting table", df_plot_cache)

                df_plot_cache = _prepare_combo_index(df_plot_cache)
            return df_plot_cache

        def get_combo_df(combo_key: str, combo_label: str) -> pd.DataFrame:
            cached_combo = combo_table_cache.get(combo_key)
            if cached_combo is None:
                df_plot = ensure_df_plot()
                cached_combo = df_plot[df_plot["__combo_key__"] == combo_key]
                combo_table_cache[combo_key] = cached_combo
                _log_df_size(f"{ds_name} combo table [{combo_label}]", cached_combo)
            return cached_combo

        def get_combo_partition_df(
            combo: Tuple[str, str, str],
            combo_label: str,
            row_kind: str,
        ) -> pd.DataFrame:
            cache_key = combo + (row_kind,)
            cached_partition = combo_partition_cache.get(cache_key)
            if cached_partition is not None:
                return cached_partition

            path = combo_source_map.get(combo, {}).get(f"{row_kind}_path")
            if path is None:
                cached_partition = pd.DataFrame(columns=combo_read_columns)
            else:
                cached_partition = _load_partition_combo_frame(
                    path,
                    columns=combo_read_columns,
                    epoch_col=epoch_col,
                    epoch_range=epoch_range,
                    extra_group_cols=valid_groupby_cols,
                )
            combo_partition_cache[cache_key] = cached_partition
            _log_df_size(f"{ds_name} combo {row_kind} [{combo_label}]", cached_partition)
            return cached_partition

        def get_model_combo_df(combo: Tuple[str, str, str], combo_label: str) -> pd.DataFrame:
            if plot_source_mode == "partitioned":
                return get_combo_partition_df(combo, combo_label, "model")

            combo_key = f"{combo[0]}\x1f{combo[1]}\x1f{combo[2]}"
            df_combo = get_combo_df(combo_key, combo_label)
            if df_combo.empty:
                return df_combo
            return df_combo[df_combo["layer_name"] == "__model__"]

        def get_layers_combo_df(combo: Tuple[str, str, str], combo_label: str) -> pd.DataFrame:
            if plot_source_mode == "partitioned":
                return get_combo_partition_df(combo, combo_label, "layers")

            combo_key = f"{combo[0]}\x1f{combo[1]}\x1f{combo[2]}"
            df_combo = get_combo_df(combo_key, combo_label)
            if df_combo.empty:
                return df_combo
            return df_combo[df_combo["layer_name"] != "__model__"]

        resolved_combos_key = f"{ds_name}_{dataset_split}_resolved_metric_combinations"
        resolved_combos_df = _load_cached_plot_table(
            plot_cache_root,
            resolved_combos_key,
            force_recompute=plot_table_force_recompute,
        )
        if resolved_combos_df is None:
            combo_source_df = (
                partition_manifest[["metric_family", "metric", "scope"]]
                if partition_manifest is not None
                else ensure_df_plot()
            )
            combos = _discover_metric_combinations_for_analysis(
                combo_source_df,
                metric_family=requested_metric_family,
                metric=requested_metric,
                scope=requested_scope,
            )
            if not combos:
                available_families = sorted(
                    combo_source_df["metric_family"].dropna().astype(str).unique().tolist()
                )
                raise ValueError(
                    f"No metrics found for metric_family='{requested_metric_family}', "
                    f"metric='{requested_metric}', scope='{requested_scope}'. "
                    f"Available families: {available_families}"
                )

            resolved_combos_df = pd.DataFrame(
                combos,
                columns=["metric_family", "metric", "scope"],
            )
            _write_cached_plot_table(plot_cache_root, resolved_combos_key, resolved_combos_df)
        else:
            combos = list(
                resolved_combos_df[["metric_family", "metric", "scope"]].itertuples(index=False, name=None)
            )

        if not combos:
            combo_source_df = (
                partition_manifest[["metric_family", "metric", "scope"]]
                if partition_manifest is not None
                else ensure_df_plot()
            )
            available_families = sorted(
                combo_source_df["metric_family"].dropna().astype(str).unique().tolist()
            )
            raise ValueError(
                f"No metrics found for metric_family='{requested_metric_family}', "
                f"metric='{requested_metric}', scope='{requested_scope}'. "
                f"Available families: {available_families}"
            )

        tables[resolved_combos_key] = resolved_combos_df
        _log_df_size(f"{ds_name} resolved metric combinations", resolved_combos_df)

        if include_distribution_grid:
            distribution_tables: Dict[tuple[str, str, str], Any] = {}
            for combo in combos:
                metric_family, metric, scope = combo
                combo_label = f"{metric_family} | {metric} | {scope}"
                distribution_table_key = f"{ds_name}_{dataset_split}_{metric_family}_{metric}_{scope}_distribution_source"
                distribution_cache_path = _plot_table_cache_path(plot_cache_root, distribution_table_key)
                if distribution_cache_path.exists() and not plot_table_force_recompute:
                    log.info(
                        "Using cached plot table '%s' from %s",
                        distribution_table_key,
                        distribution_cache_path,
                    )
                    distribution_tables[combo] = distribution_cache_path
                    continue

                df_layers = get_layers_combo_df(combo, combo_label)
                df_model = get_model_combo_df(combo, combo_label)
                if df_layers.empty and df_model.empty:
                    continue

                # Only histogram inputs are needed for the distribution cache.
                df_layers = df_layers[["layer_name", "value"]].copy() if not df_layers.empty else df_layers
                df_model = df_model[["layer_name", "value"]].copy() if not df_model.empty else df_model

                if distribution_grid_prefer_layer_rows and not df_layers.empty:
                    df_distribution = df_layers.copy()
                elif not df_model.empty:
                    df_distribution = df_model.copy()
                else:
                    df_distribution = df_layers.copy()

                if df_distribution.empty:
                    continue
                _log_df_size(f"{ds_name} distribution source [{combo_label}]", df_distribution)
                _write_cached_plot_table(plot_cache_root, distribution_table_key, df_distribution)
                distribution_tables[combo] = distribution_cache_path

            distribution_figures = _plot_metric_distribution_grid_long(
                distribution_tables=distribution_tables,
                combos=combos,
                ds_name=f"{zoo_plot_name(ds_name)} [{dataset_split}]",
                png_dpi=png_dpi,
                nrows=distribution_grid_rows,
                ncols=distribution_grid_cols,
                bins=distribution_grid_bins,
            )
            for page_idx, fig in distribution_figures:
                figures.append(
                    {f"{ds_name}_{dataset_split}_weight_metric_distribution_page_{page_idx:02d}": fig}
                )

        for combo in combos:
            metric_family, metric, scope = combo
            combo_label = f"{metric_family} | {metric} | {scope}"

            model_table_key = f"{ds_name}_{dataset_split}_{metric_family}_model_{metric}_{scope}_epoch_stats"
            df_model_stats = _load_cached_plot_table(
                plot_cache_root,
                model_table_key,
                force_recompute=plot_table_force_recompute,
            )
            if df_model_stats is None:
                df_model_plot = get_model_combo_df(combo, combo_label)
                if not df_model_plot.empty:
                    df_model_stats = _aggregate_metric_over_epoch(
                        df_model_plot,
                        epoch_col=epoch_col,
                        group_cols=valid_groupby_cols,
                        include_quantiles=include_quantiles,
                    )
                    _write_cached_plot_table(plot_cache_root, model_table_key, df_model_stats)

            if df_model_stats is not None and not df_model_stats.empty:
                tables[model_table_key] = df_model_stats
                _log_df_size(
                    f"{ds_name} model aggregate [{metric_family}/{metric}/{scope}]",
                    df_model_stats,
                )

                fig = _plot_single_metric_dataset_aggregate_long(
                    df_stats=df_model_stats,
                    metric_name=f"{metric_family} | model | {metric} | {scope}",
                    ds_name=f"{zoo_plot_name(ds_name)} [{dataset_split}]",
                    epoch_col=epoch_col,
                    group_cols=valid_groupby_cols,
                    png_dpi=png_dpi,
                )
                figures.append({f"{ds_name}_{dataset_split}_{metric_family}_model_{metric}_{scope}_aggregate": fig})
            else:
                log.warning(
                    "No model-level rows for dataset='%s', metric_family='%s', metric='%s', scope='%s'.",
                    ds_name, metric_family, metric, scope
                )

            df_layers_plot = get_layers_combo_df(combo, combo_label)
            if df_layers_plot.empty:
                log.info(
                    "No layer-level rows for dataset='%s', metric_family='%s', metric='%s', scope='%s'. "
                    "Skipping single-layer/all-layer/heatmap plots.",
                    ds_name, metric_family, metric, scope
                )
                continue

            dataset_layer_name = selected_layer_name
            if dataset_layer_name is None:
                first_layer = (
                    df_layers_plot[["layer_order", "layer_name"]]
                    .drop_duplicates()
                    .sort_values("layer_order", kind="stable")
                    .iloc[0]
                )
                dataset_layer_name = first_layer["layer_name"]

            single_layer_table_key = (
                f"{ds_name}_{dataset_split}_{metric_family}_{dataset_layer_name}_{metric}_{scope}_epoch_stats"
            )
            df_single_layer_stats = _load_cached_plot_table(
                plot_cache_root,
                single_layer_table_key,
                force_recompute=plot_table_force_recompute,
            )
            if df_single_layer_stats is None:
                df_single_layer = df_layers_plot[df_layers_plot["layer_name"] == dataset_layer_name]
                if df_single_layer.empty:
                    raise KeyError(f"Configured layer_name '{dataset_layer_name}' not found for {combo_label}.")

                df_single_layer_stats = _aggregate_metric_over_epoch(
                    df_single_layer,
                    epoch_col=epoch_col,
                    group_cols=valid_groupby_cols,
                    include_quantiles=include_quantiles,
                )
                _write_cached_plot_table(plot_cache_root, single_layer_table_key, df_single_layer_stats)

            tables[single_layer_table_key] = df_single_layer_stats
            _log_df_size(
                f"{ds_name} single-layer aggregate [{metric_family}/{metric}/{scope}]",
                df_single_layer_stats,
            )

            if include_single_metric_plot:
                fig = _plot_single_metric_dataset_aggregate_long(
                    df_stats=df_single_layer_stats,
                    metric_name=f"{metric_family} | {dataset_layer_name} | {metric} | {scope}",
                    ds_name=f"{zoo_plot_name(ds_name)} [{dataset_split}]",
                    epoch_col=epoch_col,
                    group_cols=valid_groupby_cols,
                    png_dpi=png_dpi,
                )
                figures.append(
                    {f"{ds_name}_{dataset_split}_{metric_family}_{dataset_layer_name}_{metric}_{scope}_aggregate": fig}
                )

            all_layers_table_key = f"{ds_name}_{dataset_split}_{metric_family}_all_layers_{metric}_{scope}_mean_long"
            df_layers_agg = _load_cached_plot_table(
                plot_cache_root,
                all_layers_table_key,
                force_recompute=plot_table_force_recompute,
            )
            if df_layers_agg is None:
                df_layers_agg = _aggregate_layers_metric_over_epoch(
                    df_layers_plot,
                    epoch_col=epoch_col,
                    group_cols=valid_groupby_cols,
                    stat="mean",
                )
                _write_cached_plot_table(plot_cache_root, all_layers_table_key, df_layers_agg)

            tables[all_layers_table_key] = df_layers_agg
            _log_df_size(
                f"{ds_name} all-layer aggregate [{metric_family}/{metric}/{scope}]",
                df_layers_agg,
            )

            if include_all_layer_plot:
                fig = _plot_all_layers_lines_long(
                    df_agg=df_layers_agg,
                    ds_name=f"{zoo_plot_name(ds_name)} [{dataset_split}]",
                    metric_label=combo_label,
                    epoch_col=epoch_col,
                    group_cols=valid_groupby_cols,
                    png_dpi=png_dpi,
                    max_lines_to_plot=max_all_layer_lines,
                    show_legend_threshold=all_layer_legend_threshold,
                )
                if fig is not None:
                    figures.append({f"{ds_name}_{dataset_split}_{metric_family}_all_layers_{metric}_{scope}": fig})

            if include_heatmap:
                if not valid_groupby_cols:
                    heatmap_df = df_layers_agg
                    fig = _plot_layer_epoch_heatmap_long(
                        df_agg=heatmap_df,
                        ds_name=f"{zoo_plot_name(ds_name)} [{dataset_split}]",
                        metric_label=combo_label,
                        epoch_col=epoch_col,
                        group_filter_label=None,
                        png_dpi=png_dpi,
                    )
                    if fig is not None:
                        figures.append({f"{ds_name}_{dataset_split}_{metric_family}_heatmap_{metric}_{scope}": fig})
                else:
                    heatmap_table_key = f"{ds_name}_{dataset_split}_{metric_family}_heatmap_{metric}_{scope}_source"
                    heatmap_df = _load_cached_plot_table(
                        plot_cache_root,
                        heatmap_table_key,
                        force_recompute=plot_table_force_recompute,
                    )
                    label_parts: List[str] = []
                    if heatmap_df is None:
                        this_heatmap_group_selection = heatmap_group_selection
                        if this_heatmap_group_selection is None and not df_layers_agg.empty:
                            first_row = df_layers_agg.iloc[0]
                            this_heatmap_group_selection = {col: first_row[col] for col in valid_groupby_cols}

                        if this_heatmap_group_selection:
                            mask = pd.Series(True, index=df_layers_agg.index)
                            for col in valid_groupby_cols:
                                if col in this_heatmap_group_selection:
                                    val = this_heatmap_group_selection[col]
                                    mask &= df_layers_agg[col] == val
                                    label_parts.append(f"{col}={val}")

                            heatmap_df = df_layers_agg.loc[mask]
                            if not heatmap_df.empty:
                                _write_cached_plot_table(plot_cache_root, heatmap_table_key, heatmap_df)

                    if heatmap_df is not None and not heatmap_df.empty:
                        if not label_parts:
                            if heatmap_group_selection:
                                label_parts = [
                                    f"{col}={heatmap_group_selection[col]}"
                                    for col in valid_groupby_cols
                                    if col in heatmap_group_selection
                                ]
                            else:
                                first_row = heatmap_df.iloc[0]
                                label_parts = [f"{col}={first_row[col]}" for col in valid_groupby_cols]

                        tables[heatmap_table_key] = heatmap_df
                        _log_df_size(
                            f"{ds_name} heatmap source [{metric_family}/{metric}/{scope}]",
                            heatmap_df,
                        )
                        fig = _plot_layer_epoch_heatmap_long(
                            df_agg=heatmap_df,
                            ds_name=f"{zoo_plot_name(ds_name)} [{dataset_split}]",
                            metric_label=combo_label,
                            epoch_col=epoch_col,
                            group_filter_label=", ".join(label_parts) if label_parts else None,
                            png_dpi=png_dpi,
                        )
                        if fig is not None:
                            figures.append({f"{ds_name}_{dataset_split}_{metric_family}_heatmap_{metric}_{scope}": fig})
                    else:
                        log.warning(
                            "Heatmap selection produced an empty table for dataset '%s', combo '%s'.",
                            ds_name,
                            combo_label,
                        )

    return AnalysisResult(tables=tables, figures=figures)
