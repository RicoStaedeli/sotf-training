from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf

_INVALID_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_DUPLICATE_UNDERSCORES = re.compile(r"_+")
_DATASET_ID_PREFIX = re.compile(r"^(\d+)_")


def sanitize_path_component(value: object) -> str:
    """Sanitize path component.

    Args:
        value (object): Value to process.

    Returns:
        str: Resolved string value.
    """
    text = str(value).strip()
    sanitized = _INVALID_PATH_CHARS.sub("_", text)
    sanitized = _DUPLICATE_UNDERSCORES.sub("_", sanitized).strip("._-")
    return sanitized or "run"


def dataset_id_from_path(value: object) -> str:
    """Dataset id from path.

    Args:
        value (object): Value to process.

    Returns:
        str: Resolved string value.
    """
    name = Path(str(value).strip()).name
    match = _DATASET_ID_PREFIX.match(name)
    if match:
        return match.group(1)
    return "unknown"


def dataset_slug_from_path(value: object) -> str:
    """Dataset slug from path.

    Args:
        value (object): Value to process.

    Returns:
        str: Resolved string value.
    """
    name = Path(str(value).strip()).name
    stem = name.removesuffix(".parquet").removesuffix("__prepared_dataset")
    return sanitize_path_component(stem)


def register_omegaconf_resolvers() -> None:
    """Register omegaconf resolvers.

    Returns:
        None: None.
    """
    if not OmegaConf.has_resolver("path_safe"):
        OmegaConf.register_new_resolver("path_safe", sanitize_path_component)
    if not OmegaConf.has_resolver("dataset_id_from_path"):
        OmegaConf.register_new_resolver("dataset_id_from_path", dataset_id_from_path)
    if not OmegaConf.has_resolver("dataset_slug_from_path"):
        OmegaConf.register_new_resolver("dataset_slug_from_path", dataset_slug_from_path)
