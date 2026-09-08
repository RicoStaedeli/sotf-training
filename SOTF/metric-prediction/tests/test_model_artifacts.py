from __future__ import annotations

import importlib
import pickle
import sys
import types

import numpy as np
import pandas as pd
import torch

from data.feature_preprocessing import FeaturePreprocessor
from utils.artifacts import load_model_bundle, write_model_bundle


class DummyTorchRegressor:
    def __init__(self) -> None:
        self._network = torch.nn.Linear(2, 1)


class DummyTabularRegressor:
    def __init__(self) -> None:
        self.coefficients_ = np.array([1.0, -1.0])


def test_write_model_bundle_uses_pt_for_torch_backed_models(tmp_path) -> None:
    path = write_model_bundle(
        tmp_path,
        {
            "model": DummyTorchRegressor(),
            "preprocessor": None,
            "feature_columns": ["a", "b"],
        },
    )

    assert path.name == "model_bundle.pt"
    assert path.exists()

    restored = load_model_bundle(path)
    assert isinstance(restored["model"], DummyTorchRegressor)
    assert restored["feature_columns"] == ["a", "b"]


def test_write_model_bundle_keeps_pickle_for_non_torch_models(tmp_path) -> None:
    path = write_model_bundle(
        tmp_path,
        {
            "model": DummyTabularRegressor(),
            "preprocessor": None,
            "feature_columns": ["a", "b"],
        },
    )

    assert path.name == "model_bundle.pkl"
    assert path.exists()

    restored = load_model_bundle(path)
    assert isinstance(restored["model"], DummyTabularRegressor)
    np.testing.assert_allclose(restored["model"].coefficients_, np.array([1.0, -1.0]))


def test_load_model_bundle_supports_legacy_metric_prediction_namespace(tmp_path) -> None:
    path = tmp_path / "legacy_model_bundle.pkl"
    preprocessor = FeaturePreprocessor(strategy="median")
    preprocessor.fit_transform(pd.DataFrame({"a": [1.0, 2.0]}))

    legacy_root = types.ModuleType("metric_prediction")
    legacy_root.__path__ = []
    legacy_data = types.ModuleType("metric_prediction.data")
    preprocessing_module = importlib.import_module("data.feature_preprocessing")
    scaling_module = importlib.import_module("data.scalers")

    previous_aliases = {
        name: sys.modules.get(name)
        for name in (
            "metric_prediction",
            "metric_prediction.data",
            "metric_prediction.data.preprocessing",
            "metric_prediction.data.scaling",
        )
    }
    sys.modules["metric_prediction"] = legacy_root
    sys.modules["metric_prediction.data"] = legacy_data
    sys.modules["metric_prediction.data.preprocessing"] = preprocessing_module
    sys.modules["metric_prediction.data.scaling"] = scaling_module
    legacy_root.data = legacy_data
    legacy_data.preprocessing = preprocessing_module
    legacy_data.scaling = scaling_module

    original_module = FeaturePreprocessor.__module__
    try:
        FeaturePreprocessor.__module__ = "metric_prediction.data.preprocessing"
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "model": DummyTabularRegressor(),
                    "preprocessor": preprocessor,
                    "feature_columns": ["a"],
                },
                handle,
            )
    finally:
        FeaturePreprocessor.__module__ = original_module
        for name, module in previous_aliases.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    restored = load_model_bundle(path)
    assert isinstance(restored["preprocessor"], FeaturePreprocessor)
