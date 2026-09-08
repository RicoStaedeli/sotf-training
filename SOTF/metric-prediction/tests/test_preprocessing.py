from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data.feature_preprocessing import FeaturePreprocessor


class FeaturePreprocessorTests(unittest.TestCase):
    def test_median_strategy_treats_infinite_values_as_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "a": [1.0, np.inf, 3.0],
                "b": [4.0, 5.0, -np.inf],
            }
        )

        transformed = FeaturePreprocessor(strategy="median").fit_transform(frame)

        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertEqual(transformed.loc[1, "a"], 2.0)
        self.assertEqual(transformed.loc[2, "b"], 4.5)

    def test_none_strategy_rejects_infinite_values(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.inf]})

        with self.assertRaisesRegex(ValueError, "Missing or non-finite values remain"):
            FeaturePreprocessor(strategy="none").fit_transform(frame)


if __name__ == "__main__":
    unittest.main()
