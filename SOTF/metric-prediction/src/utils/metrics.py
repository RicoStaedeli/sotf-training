from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true (np.ndarray): np.ndarray value used by the function.
        y_pred (np.ndarray): np.ndarray value used by the function.

    Returns:
        Dict[str, float]: Result of the function.
    """
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
