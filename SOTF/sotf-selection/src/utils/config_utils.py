from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf

_INVALID_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_DUPLICATE_UNDERSCORES = re.compile(r"_+")
_DATASET_ID_PREFIX = re.compile(r"^(?:ds_)?(\d+)_")
_WEIGHT_STATISTIC_SPLIT = re.compile(r"^weight_statistics_(train|val|test)\.pt$")
_LATENT_SUBDIR = re.compile(r"^(windows|averaged|token)$")


def sanitize_path_component(value: object) -> str:
    """Return a filesystem-safe path component."""
    text = str(value).strip()
    sanitized = _INVALID_PATH_CHARS.sub("_", text)
    sanitized = _DUPLICATE_UNDERSCORES.sub("_", sanitized).strip("._-")
    return sanitized or "run"


def resolve_dataset_identity_path(
    source: object | None,
    pt_path: object | None,
    parquet_path: object | None,
    target_parquet_path: object | None = None,
) -> str:
    """Resolve the path used to identify and label a dataset."""
    normalized_source = str(source or "prepared_dataset").strip().lower()
    if normalized_source == "latent_forecast":
        candidates = (target_parquet_path, parquet_path, pt_path)
    else:
        candidates = (pt_path, parquet_path, target_parquet_path)

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text.lower() != "none":
            return text
    return ""


def _dataset_name_token(value: object) -> str:
    """Extract the path token that carries the dataset identity."""
    path = Path(str(value).strip())
    if _WEIGHT_STATISTIC_SPLIT.match(path.name):
        return path.parent.name
    if _LATENT_SUBDIR.match(path.parent.name):
        return path.parent.parent.name
    return path.name


def dataset_id_from_identity_path(value: object) -> int | None:
    """Extract a numeric dataset id from a dataset identity path."""
    token = _dataset_name_token(value)
    match = _DATASET_ID_PREFIX.match(token)
    if match:
        return int(match.group(1))
    return None


def dataset_id_from_path(value: object) -> str:
    """Extract a numeric dataset id from a dataset identity path."""
    dataset_id = dataset_id_from_identity_path(value)
    if dataset_id is None:
        return "unknown"
    return str(dataset_id)


def dataset_slug_from_path(value: object) -> str:
    """Build a stable dataset slug from a dataset identity path."""
    token = _dataset_name_token(value)
    stem = token.removesuffix(".parquet").removesuffix("__prepared_dataset").removesuffix(".pt")
    return sanitize_path_component(stem)


def dataset_source_path(
    source: object | None,
    pt_path: object | None,
    parquet_path: object | None,
    target_parquet_path: object | None = None,
) -> str:
    """OmegaConf resolver wrapper returning the dataset identity path."""
    return resolve_dataset_identity_path(
        source=source,
        pt_path=pt_path,
        parquet_path=parquet_path,
        target_parquet_path=target_parquet_path,
    )


def register_omegaconf_resolvers() -> None:
    """Register the OmegaConf resolvers used by SOTF configs."""
    if not OmegaConf.has_resolver("path_safe"):
        OmegaConf.register_new_resolver("path_safe", sanitize_path_component)
    if not OmegaConf.has_resolver("dataset_source_path"):
        OmegaConf.register_new_resolver("dataset_source_path", dataset_source_path)
    if not OmegaConf.has_resolver("dataset_id_from_path"):
        OmegaConf.register_new_resolver("dataset_id_from_path", dataset_id_from_path)
    if not OmegaConf.has_resolver("dataset_slug_from_path"):
        OmegaConf.register_new_resolver("dataset_slug_from_path", dataset_slug_from_path)
