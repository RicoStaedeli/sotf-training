from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class RegressorProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.
            y (pd.Series): pd.Series value used by the function.

        Returns:
            Any: Result of the function.
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict.

        Args:
            X (pd.DataFrame): pd.DataFrame value used by the function.

        Returns:
            np.ndarray: Result of the function.
        """
        ...
