from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .base import RegressorProtocol


class RidgeRegressor(RegressorProtocol):
    def __init__(self, **params: Any) -> None:
        """Initialize instance.

        Args:
            **params (Any): Value used by the function.

        Returns:
            None: None.
        """
        self._model = Ridge(**params)
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeRegressor":
        """Fit.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.
            y (pd.Series): pd.Series value used by the function.

        Returns:
            "RidgeRegressor": Result of the function.
        """
        self._model.fit(X, y)
        coefficients = np.asarray(self._model.coef_, dtype="float64").reshape(-1)
        self.feature_importances_ = np.abs(coefficients)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.

        Returns:
            np.ndarray: Result of the function.
        """
        predictions = self._model.predict(X)
        return np.asarray(predictions, dtype="float64")


def build_ridge_regressor(params: Mapping[str, Any]) -> RegressorProtocol:
    """Build ridge regressor.

    Args:
        params (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        RegressorProtocol: Result of the function.
    """
    return RidgeRegressor(**dict(params))
