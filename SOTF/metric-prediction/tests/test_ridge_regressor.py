from __future__ import annotations

import unittest

import pandas as pd

from utils.shap_analysis import _resolve_shap_model
from metric_prediction.models import create_regressor


class RidgeRegressorTests(unittest.TestCase):
    def test_factory_builds_ridge_and_model_predicts(self) -> None:
        model = create_regressor(
            {
                "name": "ridge",
                "params": {
                    "alpha": 1.0,
                    "fit_intercept": True,
                    "solver": "auto",
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

    def test_shap_model_resolution_unwraps_underlying_estimator(self) -> None:
        model = create_regressor(
            {
                "name": "ridge",
                "params": {
                    "alpha": 1.0,
                    "fit_intercept": True,
                    "solver": "auto",
                    "random_state": 7,
                },
            }
        )

        shap_model = _resolve_shap_model(model)

        self.assertIsNot(shap_model, model)
        self.assertEqual(type(shap_model).__name__, "Ridge")


if __name__ == "__main__":
    unittest.main()
