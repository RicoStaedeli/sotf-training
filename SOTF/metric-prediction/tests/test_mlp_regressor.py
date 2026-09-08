from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd

from metric_prediction.models import create_regressor


class MLPRegressorTests(unittest.TestCase):
    def test_factory_builds_torch_mlp_and_model_predicts(self) -> None:
        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            rng.normal(size=(96, 4)),
            columns=["f1", "f2", "f3", "f4"],
        )
        target = 2.5 * X["f1"] - 1.25 * X["f2"] + 0.5 * X["f3"] + 0.1 * rng.normal(size=len(X))
        y = pd.Series(target.to_numpy(dtype="float64"))

        model = create_regressor(
            {
                "name": "mlp",
                "params": {
                    "hidden_layer_sizes": [64, 32],
                    "activation": "gelu",
                    "dropout": 0.0,
                    "max_epochs": 250,
                    "patience": 30,
                    "early_stopping": False,
                    "validation_fraction": 0.2,
                    "scale_features": True,
                    "scale_target": True,
                    "random_state": 7,
                },
            }
        )

        fitted = model.fit(X, y)
        predictions = fitted.predict(X)

        rmse = float(np.sqrt(np.mean((predictions - y.to_numpy(dtype="float64")) ** 2)))

        self.assertEqual(predictions.shape, (96,))
        self.assertEqual(predictions.dtype.kind, "f")
        self.assertLess(rmse, 0.35)

    def test_torch_mlp_round_trips_through_pickle(self) -> None:
        rng = np.random.default_rng(11)
        X = pd.DataFrame(
            rng.normal(size=(48, 3)),
            columns=["f1", "f2", "f3"],
        )
        y = pd.Series((1.2 * X["f1"] - 0.7 * X["f2"] + 0.3 * X["f3"]).to_numpy(dtype="float64"))

        model = create_regressor(
            {
                "name": "mlp",
                "params": {
                    "hidden_layer_sizes": [32, 16],
                    "dropout": 0.0,
                    "max_epochs": 120,
                    "early_stopping": True,
                    "patience": 20,
                    "random_state": 11,
                },
            }
        )

        fitted = model.fit(X, y)
        expected = fitted.predict(X)

        restored = pickle.loads(pickle.dumps(fitted))
        actual = restored.predict(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_same_seed_produces_identical_predictions_across_two_fits(self) -> None:
        rng = np.random.default_rng(13)
        X = pd.DataFrame(
            rng.normal(size=(64, 4)),
            columns=["f1", "f2", "f3", "f4"],
        )
        y = pd.Series((1.5 * X["f1"] - 0.8 * X["f2"] + 0.2 * X["f3"]).to_numpy(dtype="float64"))

        model_cfg = {
            "name": "mlp",
            "params": {
                "hidden_layer_sizes": [32, 16],
                "dropout": 0.0,
                "max_epochs": 80,
                "early_stopping": False,
                "validation_fraction": 0.2,
                "random_state": 13,
                "device": "cpu",
            },
        }

        fitted_a = create_regressor(model_cfg).fit(X, y)
        fitted_b = create_regressor(model_cfg).fit(X, y)

        predictions_a = fitted_a.predict(X)
        predictions_b = fitted_b.predict(X)

        np.testing.assert_allclose(predictions_a, predictions_b, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
