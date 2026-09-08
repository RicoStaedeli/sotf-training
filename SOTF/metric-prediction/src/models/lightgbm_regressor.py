from __future__ import annotations

from typing import Any, Mapping
from lightgbm import LGBMRegressor
from .base import RegressorProtocol


def build_lightgbm_regressor(params: Mapping[str, Any]) -> RegressorProtocol:
    """Build lightgbm regressor.

    Args:
        params (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        RegressorProtocol: Result of the function.
    """
    return LGBMRegressor(**dict(params))  # type: ignore[return-value]
