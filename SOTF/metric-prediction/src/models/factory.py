from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

from .base import RegressorProtocol
from .latent_hierarchical_transformer_regressor import build_latent_hierarchical_transformer_regressor
from .latent_mlp_regressor import build_latent_mlp_regressor
from .latent_window_lstm_regressor import build_latent_window_lstm_regressor
from .lightgbm_regressor import build_lightgbm_regressor
from .ridge_regressor import build_ridge_regressor
from .weight_statistic_mlp_regressor import build_weight_statistic_mlp_regressor
from .xgboost_regressor import build_xgboost_regressor


def create_regressor(
    model_cfg: Mapping[str, Any],
    *,
    feature_metadata: pd.DataFrame | None = None,
) -> RegressorProtocol:
    """Create regressor.

    Args:
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        feature_metadata (pd.DataFrame | None): pd.DataFrame | None value used by the function.

    Returns:
        RegressorProtocol: Result of the function.
    """
    model_name = str(model_cfg["name"]).lower()
    params = deepcopy(dict(model_cfg.get("params", {})))

    if model_name == "lightgbm":
        return build_lightgbm_regressor(params)
    if model_name == "xgboost":
        return build_xgboost_regressor(params)
    if model_name == "ridge":
        return build_ridge_regressor(params)
    if model_name in {"mlp", "mlp_weight_stats", "mlp_weight_statistics", "weight_statistic_mlp"}:
        return build_weight_statistic_mlp_regressor(params)
    if model_name in {"mlp_latent", "latent_mlp"}:
        return build_latent_mlp_regressor(params)
    if model_name in {"latent_window_lstm", "lstm_latent_sequences", "lstm_latent_windows", "latent_windows_lstm"}:
        return build_latent_window_lstm_regressor(params)
    if model_name in {
        "latent_hierarchical_transformer",
        "hierarchical_transformer",
        "latent_window_transformer",
        "latent_windows_transformer",
    }:
        return build_latent_hierarchical_transformer_regressor(params)

    raise ValueError(f"Unsupported model backend '{model_cfg['name']}'.")
