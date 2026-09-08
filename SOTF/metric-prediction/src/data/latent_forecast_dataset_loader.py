from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from data.weightstatistic_dataset import WeightStatisticDataset
from data.weightstatistic_dataset_loader import TARGET_METRIC_COLUMNS

log = logging.getLogger(__name__)


def load_parquet_only_forecast_dataset(data_cfg: DictConfig) -> WeightStatisticDataset:
    """Load a forecast dataset from parquet only (no latent .pt files).

    Each (model_id, epoch) row's window_sequence is the history of available
    performance metric columns up to and including that epoch, giving the
    hierarchical transformer a temporal sequence to process.
    """
    target_parquet_path = _require_path(data_cfg.target_parquet_path, "data.target_parquet_path")
    target_column = str(data_cfg.target_column)
    raw_threshold = data_cfg.get("target_model_min_threshold", None)
    target_model_min_threshold = float(raw_threshold) if raw_threshold is not None else None

    return _load_parquet_as_window_forecast_dataset(
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        target_model_min_threshold=target_model_min_threshold,
    )


def load_latent_forecast_dataset(data_cfg: DictConfig) -> WeightStatisticDataset:
    """Load latent forecast dataset.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        WeightStatisticDataset: Computed weight-statistic dataset.
    """
    pt_path = _require_path(data_cfg.pt_path, "data.pt_path")
    target_parquet_path = _require_path(data_cfg.target_parquet_path, "data.target_parquet_path")
    id_normalize_basename = bool(data_cfg.get("id_normalize_basename", True))
    epoch_from_checkpoint_path = bool(data_cfg.get("epoch_from_checkpoint_path", False))
    target_column = str(data_cfg.target_column)
    raw_threshold = data_cfg.get("target_model_min_threshold", None)
    target_model_min_threshold = float(raw_threshold) if raw_threshold is not None else None

    return _load_pt_as_latent_forecast_dataset(
        pt_path,
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        id_normalize_basename=id_normalize_basename,
        target_model_min_threshold=target_model_min_threshold,
        epoch_from_checkpoint_path=epoch_from_checkpoint_path,
    )


def load_latent_forecast_split_datasets(
    data_cfg: DictConfig,
) -> "tuple[WeightStatisticDataset, WeightStatisticDataset]":
    """Load latent forecast split datasets.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        "tuple[WeightStatisticDataset, WeightStatisticDataset]": Result of the function.
    """
    pt_train_path = _require_path(data_cfg.pt_train_path, "data.pt_train_path")
    pt_val_path = _require_path(data_cfg.pt_val_path, "data.pt_val_path")
    target_parquet_path = _require_path(data_cfg.target_parquet_path, "data.target_parquet_path")
    id_normalize_basename = bool(data_cfg.get("id_normalize_basename", True))
    epoch_from_checkpoint_path = bool(data_cfg.get("epoch_from_checkpoint_path", False))
    target_column = str(data_cfg.target_column)
    raw_threshold = data_cfg.get("target_model_min_threshold", None)
    target_model_min_threshold = float(raw_threshold) if raw_threshold is not None else None

    log.info("Pre-split mode: loading train and val per-epoch PT files separately")
    train_prepared = _load_pt_as_latent_forecast_dataset(
        pt_train_path,
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        id_normalize_basename=id_normalize_basename,
        target_model_min_threshold=target_model_min_threshold,
        epoch_from_checkpoint_path=epoch_from_checkpoint_path,
        split_name="train",
    )
    val_prepared = _load_pt_as_latent_forecast_dataset(
        pt_val_path,
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        id_normalize_basename=id_normalize_basename,
        target_model_min_threshold=target_model_min_threshold,
        epoch_from_checkpoint_path=epoch_from_checkpoint_path,
        split_name="val",
    )
    log.info(
        "Pre-split datasets ready: train=%d rows (%d models), val=%d rows (%d models)",
        len(train_prepared.features),
        train_prepared.metadata["model_id"].nunique(),
        len(val_prepared.features),
        val_prepared.metadata["model_id"].nunique(),
    )
    return train_prepared, val_prepared


def _load_manifest_as_raw(manifest_path: Path) -> dict:
    """Load a chunked-token manifest and merge all chunk files into one raw dict.

    The encoding script saves large token datasets as a manifest file plus a
    directory of chunk files instead of a single ``.pt`` when the total size
    would exceed memory.  The manifest contains:
        ``model_ids``, ``checkpoint_paths``, ``epoch``, ``chunk_files`` (list of
        filenames), ``chunk_dir`` (absolute path recorded at encode time), and
        ``total_checkpoints``.

    Falls back to a sibling ``*_chunks/`` directory next to the manifest when
    the stored ``chunk_dir`` has moved (e.g. on a different machine or after
    reorganising the cache).
    """
    log.info("Detected chunked manifest, loading from: %s", manifest_path)
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)

    chunk_files: list[str] = manifest["chunk_files"]
    stored_chunk_dir = Path(manifest["chunk_dir"])

    if stored_chunk_dir.exists():
        chunk_dir = stored_chunk_dir
    else:
        # Derive sibling directory name: "..._train_tokens_manifest.pt" →
        # "..._train_tokens_chunks/"
        stem = manifest_path.stem  # e.g. "foo_train_tokens_manifest"
        sibling_name = stem.replace("_manifest", "_chunks")
        chunk_dir = manifest_path.parent / sibling_name
        if not chunk_dir.exists():
            raise FileNotFoundError(
                f"Chunk directory not found.\n"
                f"  Stored path  : {stored_chunk_dir}\n"
                f"  Sibling path : {chunk_dir}\n"
                f"Move the chunk directory to one of these locations or re-run "
                f"the latent encoding script."
            )
        log.warning(
            "Stored chunk_dir not found (%s); falling back to sibling: %s",
            stored_chunk_dir,
            chunk_dir,
        )

    model_ids: list[str] = manifest["model_ids"]
    checkpoint_paths: list[str] = manifest["checkpoint_paths"]
    epoch_indices: list[int] = manifest["epoch"]
    embeddings: list[torch.Tensor] = []

    log.info("Loading %d chunk files from %s", len(chunk_files), chunk_dir)
    for i, chunk_name in enumerate(chunk_files):
        chunk_path = chunk_dir / chunk_name
        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        embeddings.extend(chunk_data["embeddings"])
        del chunk_data
        if (i + 1) % 10 == 0 or i + 1 == len(chunk_files):
            log.info(
                "  Loaded %d / %d chunks (%d embeddings so far)",
                i + 1, len(chunk_files), len(embeddings),
            )

    log.info(
        "Merged %d chunks → %d checkpoint token embeddings",
        len(chunk_files), len(model_ids),
    )
    return {
        "mode": manifest.get("mode", "token"),
        "model_ids": model_ids,
        "checkpoint_paths": checkpoint_paths,
        "epoch": epoch_indices,
        "embeddings": embeddings,
    }


def _load_pt_as_latent_forecast_dataset(
    pt_path: Path,
    *,
    target_parquet_path: Path,
    target_column: str,
    id_normalize_basename: bool,
    target_model_min_threshold: float | None,
    epoch_from_checkpoint_path: bool = False,
    split_name: str = "",
) -> WeightStatisticDataset:
    """Load pt as latent forecast dataset.

    Args:
        pt_path (Path): Path value used by the function.
        target_parquet_path (Path): Path value used by the function.
        target_column (str): str value used by the function.
        id_normalize_basename (bool): bool value used by the function.
        target_model_min_threshold (float | None): float | None value used by the function.
        split_name (str): str value used by the function.

    Returns:
        WeightStatisticDataset: Computed weight-statistic dataset.
    """
    prefix = f"[{split_name}] " if split_name else ""
    log.info("%sLoading per-epoch latent representations from %s", prefix, pt_path)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Chunked manifest: the encoding script splits huge token datasets into
    # chunk files + a manifest instead of one monolithic .pt file.
    if isinstance(raw, dict) and "chunk_files" in raw:
        raw = _load_manifest_as_raw(pt_path)
    model_ids, epochs, embeddings = _extract_ids_epochs_and_embeddings(raw)

    if epoch_from_checkpoint_path:
        checkpoint_paths = raw.get("checkpoint_paths", [])
        if checkpoint_paths:
            epochs = _epochs_from_checkpoint_paths(checkpoint_paths)
            log.info("%sEpoch indices overridden from checkpoint paths (%d entries)", prefix, len(epochs))
        else:
            log.warning("%sepoch_from_checkpoint_path=True but no 'checkpoint_paths' found in PT file — keeping original epoch column", prefix)

    if id_normalize_basename:
        model_ids = [Path(mid).name for mid in model_ids]
        log.info("%sNormalized model IDs to basename", prefix)

    n_rows = len(embeddings) if isinstance(embeddings, list) else int(embeddings.shape[0])
    n_dims = _embedding_dim(embeddings)
    is_window_mode = isinstance(embeddings, list)
    if is_window_mode:
        lengths = [int(item.shape[0]) for item in embeddings]
        log.info(
            "%s%d rows × variable windows × %d embedding dimensions (min=%d median=%.1f max=%d windows)",
            prefix,
            n_rows,
            n_dims,
            min(lengths),
            float(np.median(lengths)),
            max(lengths),
        )
    else:
        log.info("%s%d rows × %d embedding dimensions", prefix, n_rows, n_dims)

    log.info("%sLoading target values from %s (column=%s)", prefix, target_parquet_path, target_column)
    extra_metric_cols = _readable_metric_columns(target_parquet_path, exclude={target_column})
    read_columns = ["model_id", "epoch", target_column] + extra_metric_cols
    target_df = pd.read_parquet(target_parquet_path, columns=read_columns)
    target_df["model_id"] = target_df["model_id"].astype(str)
    target_df["epoch"] = pd.to_numeric(target_df["epoch"], errors="coerce")

    target_lookup: dict[tuple[str, int], float] = {
        (str(row["model_id"]), int(row["epoch"])): float(row[target_column])
        for _, row in target_df.dropna(subset=["epoch", target_column]).iterrows()
    }
    perf_lookup: dict[tuple[str, int], dict[str, float]] = {}
    if extra_metric_cols:
        for _, row in target_df.dropna(subset=["epoch"]).iterrows():
            key = (str(row["model_id"]), int(row["epoch"]))
            perf_lookup[key] = {col: float(row[col]) if pd.notna(row[col]) else float("nan") for col in extra_metric_cols}

    targets = np.array(
        [target_lookup.get((mid, int(ep)), np.nan) for mid, ep in zip(model_ids, epochs)],
        dtype=np.float64,
    )

    n_matched = int(np.sum(~np.isnan(targets)))
    log.info("%sMatched %d / %d rows to (model_id, epoch) target values", prefix, n_matched, n_rows)
    if n_matched == 0:
        sample_parquet = list(target_df["model_id"].iloc[:3].astype(str))
        sample_pt = model_ids[:3]
        log.error("%sSample parquet model_ids: %s", prefix, sample_parquet)
        log.error("%sSample latent model_ids:  %s", prefix, sample_pt)
        raise ValueError(
            f"{prefix}No rows matched any (model_id, epoch) in the target parquet for target '{target_column}'. "
            "Check that pt_path and target_parquet_path use the same model zoo and epoch numbering."
        )

    valid_mask = ~np.isnan(targets)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        log.warning("%s%d / %d rows had no matching target and were dropped", prefix, n_dropped, n_rows)

    embeddings = _filter_embeddings(embeddings, valid_mask)
    model_ids = [mid for mid, keep in zip(model_ids, valid_mask) if keep]
    epochs = [ep for ep, keep in zip(epochs, valid_mask) if keep]
    targets = targets[valid_mask]

    if target_model_min_threshold is not None:
        model_max: dict[str, float] = {}
        for mid, t in zip(model_ids, targets):
            if mid not in model_max or t > model_max[mid]:
                model_max[mid] = t
        keep_models = {mid for mid, max_t in model_max.items() if max_t >= target_model_min_threshold}
        keep_mask = np.array([mid in keep_models for mid in model_ids], dtype=bool)
        n_before = len(model_ids)
        embeddings = _filter_embeddings(embeddings, keep_mask)
        model_ids = [m for m, k in zip(model_ids, keep_mask) if k]
        epochs = [e for e, k in zip(epochs, keep_mask) if k]
        targets = targets[keep_mask]
        log.info(
            "%sModel threshold filter (>= %.3f): kept %d / %d rows (%d / %d unique models)",
            prefix,
            target_model_min_threshold,
            len(model_ids),
            n_before,
            len(keep_models),
            len(model_max),
        )

    sample_idx = np.arange(len(model_ids))

    if is_window_mode:
        feature_columns = ["window_sequence"]
        features = pd.DataFrame(
            {
                "window_sequence": [
                    np.asarray(item, dtype=np.float32)
                    for item in embeddings
                ]
            }
        )
    else:
        feature_columns = [f"z_{i}" for i in range(n_dims)]
        features = pd.DataFrame(embeddings.astype(np.float64), columns=feature_columns)
    features.index = pd.Index(sample_idx, name="sample_idx")

    target_series = pd.Series(targets.astype(np.float64), index=features.index, name=target_column)

    metadata = pd.DataFrame(
        {"model_id": model_ids, "epoch": epochs},
        index=features.index,
    )
    for col in extra_metric_cols:
        metadata[col] = [
            perf_lookup.get((mid, int(ep)), {}).get(col, float("nan"))
            for mid, ep in zip(model_ids, epochs)
        ]
    metadata.index.name = "sample_idx"

    feature_metadata = pd.DataFrame(index=pd.Index(feature_columns, name="feature"))

    log.info(
        "%s%d samples ready, %d features, %d unique models, target=%s",
        prefix,
        len(features),
        n_dims,
        len(set(model_ids)),
        target_column,
    )
    return WeightStatisticDataset(
        features=features,
        target=target_series,
        metadata=metadata,
        feature_metadata=feature_metadata,
    )


def _epochs_from_checkpoint_paths(paths: list[str]) -> list[int]:
    """Extract real training epoch numbers from checkpoint path strings.

    Expects paths containing a segment like ``checkpoint_000075`` and returns
    the numeric part (e.g. 75).  Raises ``ValueError`` if any path does not
    match the expected pattern.
    """
    epochs: list[int] = []
    for p in paths:
        m = re.search(r"checkpoint_0*(\d+)", str(p))
        if not m:
            raise ValueError(
                f"Cannot extract epoch number from checkpoint path: {p!r}. "
                "Expected a segment like 'checkpoint_000075'."
            )
        epochs.append(int(m.group(1)))
    return epochs


def _extract_ids_epochs_and_embeddings(raw: Any) -> tuple[list[str], list[int], np.ndarray | list[np.ndarray]]:
    """Extract ids epochs and embeddings.

    Args:
        raw (Any): Value used by the function.

    Returns:
        tuple[list[str], list[int], np.ndarray]: Result of the function.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected .pt file to contain a dict, got {type(raw)}.")

    id_keys = [k for k in raw if "id" in k.lower() or "model" in k.lower()]
    epoch_keys = [k for k in raw if "epoch" in k.lower() or "step" in k.lower()]
    vec_keys = [k for k in raw if any(x in k.lower() for x in ("embed", "z", "latent", "vec", "repr"))]

    model_ids: list[str] | None = None
    for k in id_keys:
        val = raw[k]
        model_ids = [str(v) for v in val]
        log.info("Using '%s' as model IDs (%d entries)", k, len(model_ids))
        break

    epochs: list[int] | None = None
    for k in epoch_keys:
        val = raw[k]
        if isinstance(val, torch.Tensor):
            epochs = val.cpu().numpy().astype(int).tolist()
        else:
            epochs = [int(v) for v in val]
        log.info("Using '%s' as epoch indices (%d entries)", k, len(epochs))
        break

    embeddings: np.ndarray | list[np.ndarray] | None = None
    for k in vec_keys:
        val = raw[k]
        if isinstance(val, torch.Tensor):
            embeddings = val.numpy()
            log.info("Using '%s' as embeddings, shape=%s", k, embeddings.shape)
            break
        if isinstance(val, list):
            embeddings = [_as_window_embedding(item, key=k, index=index) for index, item in enumerate(val)]
            log.info("Using '%s' as window embeddings (%d variable-length sequences)", k, len(embeddings))
            break

    if model_ids is None or epochs is None or embeddings is None:
        available = {k: (tuple(raw[k].shape) if isinstance(raw[k], torch.Tensor) else type(raw[k]).__name__) for k in raw}
        missing = []
        if model_ids is None:
            missing.append("model IDs (key with 'id' or 'model')")
        if epochs is None:
            missing.append("epoch indices (key with 'epoch' or 'step')")
        if embeddings is None:
            missing.append("embeddings (key with 'embed', 'z', 'latent', 'vec', or 'repr')")
        raise ValueError(
            f"Could not detect required keys in .pt file. Missing: {', '.join(missing)}. "
            f"Available keys: {available}"
        )

    embedding_rows = len(embeddings) if isinstance(embeddings, list) else int(embeddings.shape[0])
    if not (len(model_ids) == len(epochs) == embedding_rows):
        raise ValueError(
            f"Length mismatch: model_ids={len(model_ids)}, epochs={len(epochs)}, "
            f"embeddings rows={embedding_rows}. All must be equal."
        )

    return model_ids, epochs, embeddings


def _as_window_embedding(value: Any, *, key: str, index: int) -> np.ndarray:
    """Convert one saved window embedding sequence to an array."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(
            f"Expected raw['{key}'][{index}] to have shape [n_windows, embedding_dim], "
            f"received {array.shape}."
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"Window embedding sequence raw['{key}'][{index}] must be non-empty.")
    return array.astype(np.float32, copy=False)


def _embedding_dim(embeddings: np.ndarray | list[np.ndarray]) -> int:
    """Return embedding dimension for average or window-mode latent features."""
    if isinstance(embeddings, list):
        if not embeddings:
            raise ValueError("Window embedding list is empty.")
        dims = {int(item.shape[1]) for item in embeddings}
        if len(dims) != 1:
            raise ValueError(f"All window embedding sequences must share one embedding dimension, got {sorted(dims)}.")
        return dims.pop()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected tensor embeddings to have shape [n_rows, embedding_dim], got {embeddings.shape}.")
    return int(embeddings.shape[1])


def _filter_embeddings(
    embeddings: np.ndarray | list[np.ndarray],
    mask: np.ndarray,
) -> np.ndarray | list[np.ndarray]:
    """Apply a row mask to either dense or window-list embeddings."""
    if isinstance(embeddings, list):
        return [item for item, keep in zip(embeddings, mask) if bool(keep)]
    return embeddings[mask]


def _readable_metric_columns(parquet_path: Path, *, exclude: set[str]) -> list[str]:
    """Return TARGET_METRIC_COLUMNS that exist in the parquet schema and are not excluded."""
    try:
        import pyarrow.dataset as ds
        available = {str(name) for name in ds.dataset(str(parquet_path), format="parquet").schema.names}
    except Exception:
        available = set()
    return [col for col in TARGET_METRIC_COLUMNS if col in available and col not in exclude]


def _load_parquet_as_window_forecast_dataset(
    *,
    target_parquet_path: Path,
    target_column: str,
    target_model_min_threshold: float | None,
) -> WeightStatisticDataset:
    """Build a WeightStatisticDataset from parquet metrics.

    For each (model_id, epoch) row, the window_sequence is the history of
    performance metric values from the first observed epoch up to and including
    the current epoch.  This lets the hierarchical transformer treat the
    performance trajectory as its input sequence.
    """
    extra_metric_cols = _readable_metric_columns(target_parquet_path, exclude={target_column})
    metric_cols = [target_column] + extra_metric_cols
    read_columns = ["model_id", "epoch"] + metric_cols

    log.info("Loading parquet-only forecast data from %s (column=%s)", target_parquet_path, target_column)
    df = pd.read_parquet(target_parquet_path, columns=read_columns)
    df["model_id"] = df["model_id"].astype(str)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch", target_column]).sort_values(["model_id", "epoch"]).reset_index(drop=True)

    if target_model_min_threshold is not None:
        model_max = df.groupby("model_id")[target_column].max()
        keep_models = model_max[model_max >= target_model_min_threshold].index
        df = df[df["model_id"].isin(keep_models)].reset_index(drop=True)

    model_ids_out: list[str] = []
    epochs_out: list[int] = []
    window_sequences_out: list[np.ndarray] = []
    targets_out: list[float] = []
    extra_metrics_out: dict[str, list[float]] = {col: [] for col in extra_metric_cols}

    for model_id, group in df.groupby("model_id", sort=False):
        group = group.sort_values("epoch")
        metric_matrix = group[metric_cols].to_numpy(dtype=np.float32)
        # Forward-fill then zero-fill so the sequences contain no NaN
        metric_df = pd.DataFrame(metric_matrix, columns=metric_cols).ffill().fillna(0.0)
        metric_matrix = metric_df.to_numpy(dtype=np.float32)
        target_values = group[target_column].to_numpy(dtype=np.float64)
        epoch_values = group["epoch"].to_numpy(dtype=int)

        for i in range(len(group)):
            model_ids_out.append(str(model_id))
            epochs_out.append(int(epoch_values[i]))
            window_sequences_out.append(metric_matrix[: i + 1])
            targets_out.append(float(target_values[i]))
            for col in extra_metric_cols:
                val = group[col].iloc[i]
                extra_metrics_out[col].append(float(val) if pd.notna(val) else float("nan"))

    n = len(model_ids_out)
    sample_idx = np.arange(n)
    features = pd.DataFrame(
        {"window_sequence": window_sequences_out},
        index=pd.Index(sample_idx, name="sample_idx"),
    )
    target_series = pd.Series(
        np.array(targets_out, dtype=np.float64),
        index=pd.Index(sample_idx, name="sample_idx"),
        name=target_column,
    )
    metadata = pd.DataFrame(
        {"model_id": model_ids_out, "epoch": epochs_out, **extra_metrics_out},
        index=pd.Index(sample_idx, name="sample_idx"),
    )
    feature_metadata = pd.DataFrame(index=pd.Index(["window_sequence"], name="feature"))

    log.info(
        "Parquet-only dataset: %d samples, %d unique models, target=%s, seq_dims=%d",
        n, len(set(model_ids_out)), target_column, len(metric_cols),
    )
    return WeightStatisticDataset(
        features=features,
        target=target_series,
        metadata=metadata,
        feature_metadata=feature_metadata,
    )


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
