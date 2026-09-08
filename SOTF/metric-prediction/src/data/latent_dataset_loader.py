from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from data.dataset_splitter import DatasetSplit
from data.weightstatistic_dataset import WeightStatisticDataset

log = logging.getLogger(__name__)

TARGET_METRIC_COLUMNS = ("test_acc", "train_acc", "test_loss", "train_loss")


def load_latent_dataset(data_cfg: DictConfig) -> WeightStatisticDataset:
    """Load latent dataset.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        WeightStatisticDataset: Computed weight-statistic dataset.
    """
    pt_path = _require_path(data_cfg.pt_path, "data.pt_path")
    target_parquet_path = _require_path(data_cfg.target_parquet_path, "data.target_parquet_path")
    id_normalize_basename = bool(data_cfg.get("id_normalize_basename", True))
    target_column = str(data_cfg.target_column)
    target_epoch = data_cfg.get("target_epoch", "last")
    return _load_pt_as_weightstatistic_dataset(
        pt_path,
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        target_epoch=target_epoch,
        id_normalize_basename=id_normalize_basename,
    )


def load_latent_split_dataset(data_cfg: DictConfig) -> DatasetSplit:
    """Load latent split dataset.

    Args:
        data_cfg (DictConfig): Configuration object used by the function.

    Returns:
        DatasetSplit: Computed dataset split.
    """

    pt_train_path = _require_path(data_cfg.pt_train_path, "data.pt_train_path")
    pt_val_path = _require_path(data_cfg.pt_val_path, "data.pt_val_path")
    target_parquet_path = _require_path(data_cfg.target_parquet_path, "data.target_parquet_path")
    target_column = str(data_cfg.target_column)
    target_epoch = data_cfg.get("target_epoch", "last")
    id_normalize_basename = bool(data_cfg.get("id_normalize_basename", True))

    log.info("Loading pre-split latent datasets (train + val)")
    train_ds = _load_pt_as_weightstatistic_dataset(
        pt_train_path,
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        target_epoch=target_epoch,
        id_normalize_basename=id_normalize_basename,
        split_name="train",
    )
    val_ds = _load_pt_as_weightstatistic_dataset(
        pt_val_path,
        target_parquet_path=target_parquet_path,
        target_column=target_column,
        target_epoch=target_epoch,
        id_normalize_basename=id_normalize_basename,
        split_name="val",
    )
    log.info(
        "Pre-split latent datasets ready: train=%d val=%d",
        len(train_ds.features),
        len(val_ds.features),
    )
    return DatasetSplit(
        X_train=train_ds.features,
        X_val=val_ds.features,
        y_train=train_ds.target,
        y_val=val_ds.target,
        metadata_train=train_ds.metadata,
        metadata_val=val_ds.metadata,
    )


def _load_pt_as_weightstatistic_dataset(
    pt_path: Path,
    *,
    target_parquet_path: Path,
    target_column: str,
    target_epoch: Any,
    id_normalize_basename: bool,
    split_name: str = "",
) -> WeightStatisticDataset:
    """Load pt as weight-statistic dataset.

    Args:
        pt_path (Path): Path value used by the function.
        target_parquet_path (Path): Path value used by the function.
        target_column (str): str value used by the function.
        target_epoch (Any): Value used by the function.
        id_normalize_basename (bool): bool value used by the function.
        split_name (str): str value used by the function.

    Returns:
        WeightStatisticDataset: Computed weight-statistic dataset.
    """
    prefix = f"[{split_name}] " if split_name else ""
    log.info("%sLoading latent representations from %s", prefix, pt_path)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    model_ids, embeddings = _extract_ids_and_embeddings(raw)

    if id_normalize_basename:
        model_ids = [Path(mid).name for mid in model_ids]
        log.info("%sNormalized model IDs to basename", prefix)

    n_models, n_dims = embeddings.shape
    log.info("%s%d models × %d dimensions", prefix, n_models, n_dims)

    targets = _load_targets(
        target_parquet_path=target_parquet_path,
        model_ids=model_ids,
        target_column=target_column,
        target_epoch=target_epoch,
    )

    valid_mask = ~np.isnan(targets)
    if not valid_mask.any():
        raise ValueError(
            f"{prefix}No latent models matched any entry in the target parquet for target '{target_column}'. "
            f"Check that pt_path and target_parquet_path refer to the same model zoo."
        )
    n_missing = int((~valid_mask).sum())
    if n_missing > 0:
        log.warning("%s%d / %d models had no target value and were dropped", prefix, n_missing, n_models)

    embeddings = embeddings[valid_mask]
    model_ids = [mid for mid, keep in zip(model_ids, valid_mask) if keep]

    feature_columns = [f"z_{i}" for i in range(n_dims)]
    sample_idx = np.arange(len(model_ids))

    features = pd.DataFrame(embeddings.astype(np.float64), columns=feature_columns)
    features.index = pd.Index(sample_idx, name="sample_idx")

    target = pd.Series(targets[valid_mask].astype(np.float64), index=features.index, name=target_column)

    metadata = pd.DataFrame({"model_id": model_ids}, index=features.index)
    metadata.index.name = "sample_idx"

    feature_metadata = pd.DataFrame(index=pd.Index(feature_columns, name="feature"))

    log.info(
        "%s%d samples ready, %d features, target=%s",
        prefix, len(features), n_dims, target_column,
    )
    return WeightStatisticDataset(
        features=features,
        target=target,
        metadata=metadata,
        feature_metadata=feature_metadata,
    )


def _extract_ids_and_embeddings(raw: Any) -> tuple[list[str], np.ndarray]:
    """Extract ids and embeddings.

    Args:
        raw (Any): Value used by the function.

    Returns:
        tuple[list[str], np.ndarray]: Result of the function.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected .pt file to contain a dict, got {type(raw)}.")

    id_keys = [k for k in raw if "id" in k.lower() or "model" in k.lower()]
    vec_keys = [k for k in raw if any(x in k.lower() for x in ("embed", "z", "latent", "vec", "repr"))]

    model_ids: list[str] | None = None
    for k in id_keys:
        val = raw[k]
        model_ids = [str(v) for v in val]
        log.info("Using '%s' as model IDs (%d entries)", k, len(model_ids))
        break

    embeddings: np.ndarray | None = None
    for k in vec_keys:
        val = raw[k]
        if isinstance(val, torch.Tensor):
            embeddings = val.numpy()
            log.info("Using '%s' as embeddings, shape=%s", k, embeddings.shape)
            break

    if model_ids is None or embeddings is None:
        available = {k: (raw[k].shape if isinstance(raw[k], torch.Tensor) else type(raw[k])) for k in raw}
        raise ValueError(
            f"Could not detect model ID and embedding keys in .pt file. Available keys: {available}"
        )

    if len(model_ids) != embeddings.shape[0]:
        raise ValueError(
            f"model_ids length {len(model_ids)} does not match embeddings rows {embeddings.shape[0]}."
        )

    return model_ids, embeddings


def _load_targets(
    *,
    target_parquet_path: Path,
    model_ids: list[str],
    target_column: str,
    target_epoch: Any,
) -> np.ndarray:
    """Load targets.

    Args:
        target_parquet_path (Path): Path value used by the function.
        model_ids (list[str]): list[str] value used by the function.
        target_column (str): str value used by the function.
        target_epoch (Any): Value used by the function.

    Returns:
        np.ndarray: Result of the function.
    """
    read_columns = ["model_id", target_column]
    if target_epoch != "last":
        read_columns.append("epoch")

    df = pd.read_parquet(target_parquet_path, columns=read_columns)

    if target_epoch == "last":
        if "epoch" in df.columns:
            df = df.sort_values("epoch").groupby("model_id", sort=False).last().reset_index()
        else:
            df = df.groupby("model_id", sort=False).last().reset_index()
        log.info("Using last available epoch per model as target")
    else:
        epoch_num = int(target_epoch)
        df = df.loc[df["epoch"] == epoch_num].copy()
        log.info("Filtered to epoch=%d, got %d rows", epoch_num, len(df))

    id_to_target: dict[str, float] = dict(
        zip(df["model_id"].astype(str), df[target_column].astype(float))
    )

    targets = np.array([id_to_target.get(mid, np.nan) for mid in model_ids], dtype=np.float64)
    n_matched = int(np.sum(~np.isnan(targets)))
    log.info("Matched %d / %d latent models to target values", n_matched, len(model_ids))

    if n_matched == 0:
        sample_parquet = df["model_id"].iloc[:3].tolist() if len(df) > 0 else []
        sample_latent = model_ids[:3]
        log.error("Sample parquet model_ids: %s", sample_parquet)
        log.error("Sample latent model_ids:  %s", sample_latent)

    return targets


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
