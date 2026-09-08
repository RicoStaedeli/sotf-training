from __future__ import annotations

import importlib.util
import unittest

import pandas as pd

from utils.shap_analysis import _resolve_shap_model
from metric_prediction.models import create_regressor

HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


@unittest.skipUnless(HAS_LIGHTGBM, "lightgbm is not installed")
class LightGBMRegressorTests(unittest.TestCase):
    def test_factory_builds_lightgbm_and_model_predicts(self) -> None:
        model = create_regressor(
            {
                "name": "lightgbm",
                "params": {
                    "objective": "regression",
                    "n_estimators": 20,
                    "learning_rate": 0.1,
                    "num_leaves": 15,
                    "min_child_samples": 1,
                    "verbosity": -1,
                    "n_jobs": 1,
                    "random_state": 7,
                },
            }
        )

        X = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [1.0, 1.0, 2.0, 2.0],
            }
        )
        y = pd.Series([1.0, 2.0, 3.0, 4.0])

        fitted = model.fit(X, y)
        predictions = fitted.predict(X)

        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(hasattr(fitted, "feature_importances_"))
        self.assertEqual(len(fitted.feature_importances_), X.shape[1])

    def test_shap_model_resolution_keeps_native_estimator(self) -> None:
        model = create_regressor(
            {
                "name": "lightgbm",
                "params": {
                    "objective": "regression",
                    "n_estimators": 10,
                    "verbosity": -1,
                    "n_jobs": 1,
                    "random_state": 7,
                },
            }
        )

        shap_model = _resolve_shap_model(model)

        self.assertIs(shap_model, model)
        self.assertEqual(type(shap_model).__name__, "LGBMRegressor")


if __name__ == "__main__":
    unittest.main()
