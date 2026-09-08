from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import logging
import pickle
import pkgutil
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import pandas as pd
import torch

log = logging.getLogger(__name__)

_LEGACY_METRIC_PREDICTION_PACKAGES = (
    "config",
    "data",
    "forecasting",
    "models",
    "trainer",
    "utils",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write json.

    Args:
        path (Path): Path to process.
        payload (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        None: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote JSON artifact: %s", path)


def write_pickle(path: Path, payload: Any) -> None:
    """Write pickle.

    Args:
        path (Path): Path to process.
        payload (Any): Value used by the function.

    Returns:
        None: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    log.info("Wrote pickle artifact: %s", path)


def write_torch(path: Path, payload: Any) -> None:
    """Write torch.

    Args:
        path (Path): Path to process.
        payload (Any): Value used by the function.

    Returns:
        None: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    log.info("Wrote PyTorch artifact: %s", path)


def write_model_bundle(artifacts_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write model bundle.

    Args:
        artifacts_dir (Path): Directory for artifacts.
        payload (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        Path: Resolved path.
    """
    path = artifacts_dir / model_bundle_filename(payload.get("model"))
    if path.suffix == ".pt":
        write_torch(path, dict(payload))
    else:
        write_pickle(path, dict(payload))
    return path


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load model bundle.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any]: Result of the function.
    """
    suffix = path.suffix.lower()
    with _legacy_metric_prediction_aliases():
        if suffix == ".pt":
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError as exc:
                if "weights_only" not in str(exc):
                    raise
                payload = torch.load(path, map_location="cpu")
        else:
            with path.open("rb") as handle:
                payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected model bundle payload to be a dict, got {type(payload).__name__}.")
    return payload


def model_bundle_filename(model: Any) -> str:
    """Model bundle filename.

    Args:
        model (Any): Model to process.

    Returns:
        str: Resolved string value.
    """
    return "model_bundle.pt" if is_torch_model(model) else "model_bundle.pkl"


def is_torch_model(model: Any) -> bool:
    """Is torch model.

    Args:
        model (Any): Model to process.

    Returns:
        bool: Result of the function.
    """
    if isinstance(model, torch.nn.Module):
        return True
    network = getattr(model, "_network", None)
    return isinstance(network, torch.nn.Module)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write frame.

    Args:
        path (Path): Path to process.
        frame (pd.DataFrame): Data frame to process.

    Returns:
        None: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log.info("Wrote parquet artifact: %s (%d rows, %d columns)", path, len(frame), len(frame.columns))


@contextmanager
def _legacy_metric_prediction_aliases():
    created_modules: dict[str, types.ModuleType] = {}
    try:
        legacy_package = _ensure_legacy_package(created_modules)
        for package_name in _LEGACY_METRIC_PREDICTION_PACKAGES:
            package = importlib.import_module(package_name)
            alias_name = f"metric_prediction.{package_name}"
            sys.modules.setdefault(alias_name, package)
            setattr(legacy_package, package_name, package)
            created_modules.setdefault(alias_name, package)

            package_paths = getattr(package, "__path__", None)
            if package_paths is None:
                continue
            for child in pkgutil.iter_modules(package_paths):
                child_alias = f"{alias_name}.{child.name}"
                child_module = importlib.import_module(f"{package_name}.{child.name}")
                sys.modules.setdefault(child_alias, child_module)
                created_modules.setdefault(child_alias, child_module)

        preprocessing_module = importlib.import_module("data.feature_preprocessing")
        sys.modules.setdefault("data.preprocessing", preprocessing_module)
        created_modules.setdefault("data.preprocessing", preprocessing_module)
        sys.modules.setdefault("metric_prediction.data.preprocessing", preprocessing_module)
        created_modules.setdefault("metric_prediction.data.preprocessing", preprocessing_module)
        scaling_module = importlib.import_module("data.scalers")
        sys.modules.setdefault("data.scaling", scaling_module)
        created_modules.setdefault("data.scaling", scaling_module)
        sys.modules.setdefault("metric_prediction.data.scaling", scaling_module)
        created_modules.setdefault("metric_prediction.data.scaling", scaling_module)
        yield
    finally:
        for name, module in list(created_modules.items()):
            if sys.modules.get(name) is module:
                sys.modules.pop(name, None)


def _ensure_legacy_package(created_modules: dict[str, types.ModuleType]) -> types.ModuleType:
    package = sys.modules.get("metric_prediction")
    if isinstance(package, types.ModuleType):
        return package

    package = types.ModuleType("metric_prediction")
    package.__path__ = []
    sys.modules["metric_prediction"] = package
    created_modules["metric_prediction"] = package
    return package
