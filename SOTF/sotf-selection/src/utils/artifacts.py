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

_METRIC_PREDICTION_TOP_LEVEL_PACKAGES = (
    "config",
    "data",
    "forecasting",
    "models",
    "trainer",
    "utils",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote JSON artifact: %s", path)


def write_pickle(path: Path, payload: Any) -> None:
    """Write a pickle artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    log.info("Wrote pickle artifact: %s", path)


def write_torch(path: Path, payload: Any) -> None:
    """Write a PyTorch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    log.info("Wrote PyTorch artifact: %s", path)


def write_model_bundle(artifacts_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write a model bundle with the appropriate serialization format."""
    path = artifacts_dir / model_bundle_filename(payload.get("model"))
    if path.suffix == ".pt":
        write_torch(path, dict(payload))
    else:
        write_pickle(path, dict(payload))
    return path


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load a serialized model bundle."""
    suffix = path.suffix.lower()
    with _metric_prediction_import_context():
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
    """Return the expected filename for a serialized model bundle."""
    return "model_bundle.pt" if is_torch_model(model) else "model_bundle.pkl"


def is_torch_model(model: Any) -> bool:
    """Detect whether a model should be serialized with torch."""
    if isinstance(model, torch.nn.Module):
        return True
    network = getattr(model, "_network", None)
    return isinstance(network, torch.nn.Module)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe artifact as parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log.info("Wrote parquet artifact: %s (%d rows, %d columns)", path, len(frame), len(frame.columns))


@contextmanager
def _metric_prediction_import_context():
    metric_prediction_src = _find_metric_prediction_src_dir()
    if metric_prediction_src is None:
        yield
        return

    managed_prefixes = (*_METRIC_PREDICTION_TOP_LEVEL_PACKAGES, "metric_prediction")
    saved_modules: dict[str, types.ModuleType] = {}
    for name in list(sys.modules):
        if _matches_any_prefix(name, managed_prefixes):
            saved_modules[name] = sys.modules.pop(name)

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(metric_prediction_src))
    try:
        _register_metric_prediction_compat_aliases()
        yield
    finally:
        for name in list(sys.modules):
            if _matches_any_prefix(name, managed_prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = original_sys_path


def _find_metric_prediction_src_dir() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "metric-prediction" / "src"
        if candidate.is_dir():
            return candidate
    return None


def _matches_any_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _register_metric_prediction_compat_aliases() -> None:
    legacy_package = types.ModuleType("metric_prediction")
    legacy_package.__path__ = []
    sys.modules["metric_prediction"] = legacy_package

    for package_name in _METRIC_PREDICTION_TOP_LEVEL_PACKAGES:
        package = importlib.import_module(package_name)
        alias_name = f"metric_prediction.{package_name}"
        sys.modules[alias_name] = package
        setattr(legacy_package, package_name, package)

        package_paths = getattr(package, "__path__", None)
        if package_paths is None:
            continue
        for child in pkgutil.iter_modules(package_paths):
            sys.modules[f"{alias_name}.{child.name}"] = importlib.import_module(f"{package_name}.{child.name}")

    preprocessing_module = importlib.import_module("data.feature_preprocessing")
    scaling_module = importlib.import_module("data.scalers")

    sys.modules["data.preprocessing"] = preprocessing_module
    sys.modules["data.scaling"] = scaling_module
    sys.modules["metric_prediction.data.preprocessing"] = preprocessing_module
    sys.modules["metric_prediction.data.scaling"] = scaling_module
