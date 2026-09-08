from __future__ import annotations

from typing import Any, Mapping
from xgboost import XGBRegressor

from .base import RegressorProtocol


def build_xgboost_regressor(params: Mapping[str, Any]) -> RegressorProtocol:
    """Build xgboost regressor.

    Args:
        params (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        RegressorProtocol: Result of the function.
    """

    return XGBRegressor(**dict(params))
