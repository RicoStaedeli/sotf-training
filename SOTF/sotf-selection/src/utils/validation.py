from __future__ import annotations

from data.loading import data_source
from models.predictors import normalize_observed_metric_columns
from utils.settings import OnlineLSTMPredictorConfig, SimulationConfig, XGBoostPredictorConfig


def validate_config(cfg: SimulationConfig) -> None:
    """Validate cross-field simulation settings before any data or model artifacts are loaded.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    source = data_source(cfg)
    if source not in {"prepared_dataset", "latent_forecast"}:
        raise ValueError("data.source must be either 'prepared_dataset' or 'latent_forecast'.")
    if source == "prepared_dataset":
        if not getattr(cfg.data, "pt_path", None) and not getattr(cfg.data, "parquet_path", None):
            raise ValueError("data.source=prepared_dataset requires data.pt_path or data.parquet_path.")
        if bool(getattr(cfg.data, "pt_train_path", None)):
            raise ValueError(
                "data.source=prepared_dataset does not support data.pt_train_path. "
                "Use data.pt_path for the primary split and data.pt_val_path for an optional second split."
            )
    if source == "latent_forecast":
        if not getattr(cfg.data, "target_parquet_path", None) and not getattr(cfg.data, "parquet_path", None):
            raise ValueError("data.source=latent_forecast requires data.target_parquet_path or data.parquet_path.")
        has_single_pt = bool(getattr(cfg.data, "pt_path", None))
        has_train_pt = bool(getattr(cfg.data, "pt_train_path", None))
        has_val_pt = bool(getattr(cfg.data, "pt_val_path", None))
        if has_train_pt and has_single_pt:
            raise ValueError("Cannot set both data.pt_path and data.pt_train_path for data.source=latent_forecast.")
        if has_train_pt and not has_val_pt:
            raise ValueError("data.pt_train_path and data.pt_val_path must be set together.")
        if has_val_pt and not has_train_pt and not has_single_pt:
            raise ValueError("data.pt_val_path requires data.pt_path or data.pt_train_path.")
        if not has_single_pt and not has_train_pt:
            raise ValueError("data.source=latent_forecast requires data.pt_path or data.pt_train_path/data.pt_val_path.")
    target_mode = str(cfg.sequence.target_mode).lower()
    if target_mode not in {"horizon_value", "final_value"}:
        raise ValueError("sequence.target_mode must be either 'horizon_value' or 'final_value'.")
    if target_mode == "horizon_value" and int(cfg.sequence.horizon) <= 0:
        raise ValueError("sequence.horizon must be positive when sequence.target_mode=horizon_value.")
    if int(cfg.selection.start_epoch) < 0:
        raise ValueError("selection.start_epoch must be non-negative.")
    if int(cfg.selection.interval) <= 0:
        raise ValueError("selection.interval must be positive.")
    if not (0.0 < float(cfg.selection.survival_rate) <= 1.0):
        raise ValueError("selection.survival_rate must be in the interval (0, 1].")
    if int(cfg.selection.min_population) <= 0:
        raise ValueError("selection.min_population must be positive.")
    if str(cfg.selection.metric_goal).lower() not in {"maximize", "minimize"}:
        raise ValueError("selection.metric_goal must be either 'maximize' or 'minimize'.")
    if isinstance(cfg.predictor, XGBoostPredictorConfig):
        _validate_online_predictor_common(cfg.predictor, predictor_name="xgboost")
    if isinstance(cfg.predictor, OnlineLSTMPredictorConfig):
        if source != "latent_forecast":
            raise ValueError("predictor=online_lstm requires data.source=latent_forecast.")
        _validate_online_predictor_common(cfg.predictor, predictor_name="online_lstm", min_train_rows_floor=2)


def _validate_online_predictor_common(
    predictor_cfg: XGBoostPredictorConfig | OnlineLSTMPredictorConfig,
    *,
    predictor_name: str,
    min_train_rows_floor: int = 1,
) -> None:
    """Validate shared online-predictor settings."""
    normalize_observed_metric_columns(predictor_cfg.observed_metric_columns)
    if int(predictor_cfg.min_train_rows) < int(min_train_rows_floor):
        comparator = "positive" if int(min_train_rows_floor) == 1 else f"at least {int(min_train_rows_floor)}"
        raise ValueError(f"predictor.min_train_rows must be {comparator} when predictor={predictor_name}.")
    if int(predictor_cfg.min_train_models) <= 0:
        raise ValueError(f"predictor.min_train_models must be positive when predictor={predictor_name}.")
    if str(predictor_cfg.fallback_strategy).lower() != "current_target":
        raise ValueError(
            f"predictor.fallback_strategy must currently be 'current_target' when predictor={predictor_name}."
        )
