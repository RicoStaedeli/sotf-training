from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

import pandas as pd

from data.dataset_splitter import split_dataset
from data.feature_preprocessing import FeaturePreprocessor
from data.weightstatistic_dataset_loader import load_weightstatistic_dataset, load_weightstatistic_split_datasets
from models.factory import create_regressor
from utils.artifacts import write_frame, write_json, write_model_bundle
from utils.logging_utils import setup_logging
from utils.metrics import compute_regression_metrics
from runner.wandb_config import attach_wandb_settings_to_model_config

from .config import ForecastConfig
from .dataset import build_forecast_dataset

log = logging.getLogger(__name__)


def run_forecast_experiment(cfg: ForecastConfig) -> dict[str, Any]:
    """Run forecast experiment.

    Args:
        cfg (ForecastConfig): Configuration object used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    now = datetime.now()
    artifacts_dir = cfg.output.resolve_run_dir(now=now)
    log_filename = cfg.output.log_filename or f"{cfg.output.experiment_name}.log"
    log_file = artifacts_dir / log_filename if bool(cfg.output.save_log_file) else None
    setup_logging(cfg.output.logging_level, log_file=log_file)

    log.info("Starting forecast experiment '%s'", cfg.output.experiment_name)
    log.info("Forecast artifacts directory: %s", artifacts_dir)

    model_cfg = cfg.model.to_model_config()
    model_cfg = attach_wandb_settings_to_model_config(
        model_cfg,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        api_key_file=cfg.wandb.api_key_file,
        run_name=cfg.output.experiment_name,
        config=cfg.to_dict(),
    )
    is_presplit = bool(cfg.data.pt_train_path)
    predictions = None

    if is_presplit:
        train_prepared, val_prepared = load_weightstatistic_split_datasets(
            cfg.data.to_omegaconf(),
            cfg.features.to_omegaconf(),
        )
        model = create_regressor(model_cfg, feature_metadata=train_prepared.feature_metadata)
        train_forecast = build_forecast_dataset(train_prepared, cfg.sequence)
        val_forecast = build_forecast_dataset(val_prepared, cfg.sequence)
        preprocessor = FeaturePreprocessor(strategy=cfg.features.missing_value_fill)
        X_train = preprocessor.fit_transform(train_forecast.features)
        X_val = preprocessor.transform(val_forecast.features)
        metadata_train = train_forecast.metadata
        metadata_val = val_forecast.metadata
        y_train = train_forecast.target
        y_val = val_forecast.target
        feature_names = list(X_train.columns)
        _forecast_feature_matrix_ref = pd.concat([train_forecast.features, val_forecast.features], axis=0)
        _forecast_target_ref = pd.concat([train_forecast.target, val_forecast.target], axis=0)
    else:
        prepared = load_weightstatistic_dataset(cfg.data.to_omegaconf(), cfg.features.to_omegaconf())
        model = create_regressor(model_cfg, feature_metadata=prepared.feature_metadata)
        forecast_dataset = build_forecast_dataset(prepared, cfg.sequence)
        split = split_dataset(
            forecast_dataset,
            validation_size=cfg.train.validation_size,
            random_seed=cfg.train.random_seed,
            split_csv_path=cfg.data.split_csv_path,
        )
        preprocessor = FeaturePreprocessor(strategy=cfg.features.missing_value_fill)
        X_train = preprocessor.fit_transform(split.X_train)
        metadata_train = split.metadata_train
        metadata_val = split.metadata_val
        y_train = split.y_train
        y_val = split.y_val
        feature_names = list(X_train.columns)
        X_val = preprocessor.transform(split.X_val)
        _forecast_feature_matrix_ref = forecast_dataset.features
        _forecast_target_ref = forecast_dataset.target

    log.info(
        "Training forecast model '%s' with %d train rows, %d validation rows, %d features",
        model_cfg["name"],
        len(y_train),
        len(y_val),
        len(feature_names),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    metrics = compute_regression_metrics(y_val.to_numpy(), predictions)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        artifacts_dir / "metrics.json",
        _metrics_payload(
            metrics,
            metadata_train=metadata_train,
            metadata_val=metadata_val,
            n_train=len(y_train),
            n_validation=len(y_val),
            n_features=len(feature_names),
            cfg=cfg,
            presplit=is_presplit,
        ),
    )
    write_json(artifacts_dir / "resolved_config.json", cfg.to_dict())
    write_json(
        artifacts_dir / "used_features.json",
        {
            "count": len(feature_names),
            "features": feature_names,
        },
    )

    if bool(cfg.train.save_predictions):
        prediction_frame = metadata_val.copy()
        prediction_frame["target"] = y_val.to_numpy(dtype="float64")
        prediction_frame["prediction"] = predictions
        prediction_frame["residual"] = prediction_frame["target"] - prediction_frame["prediction"]
        write_frame(artifacts_dir / "validation_predictions.parquet", prediction_frame.reset_index())

    if bool(cfg.train.save_feature_matrix):
        feature_matrix = _forecast_feature_matrix_ref.copy()
        feature_matrix.insert(0, "target", _forecast_target_ref)
        write_frame(artifacts_dir / "forecast_feature_matrix.parquet", feature_matrix.reset_index())

    if bool(cfg.train.save_model):
        write_model_bundle(
            artifacts_dir,
            {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns": feature_names,
                "config": cfg.to_dict(),
            },
        )

    feature_importances = _feature_importance_frame(model, pd.Index(feature_names))
    if feature_importances is not None:
        write_frame(artifacts_dir / "feature_importances.parquet", feature_importances)

    log.info("Forecast experiment finished with metrics: %s", metrics)
    return {
        "metrics": metrics,
        "artifacts_dir": str(artifacts_dir),
    }


def _metrics_payload(
    metrics: dict[str, float],
    *,
    metadata_train: pd.DataFrame,
    metadata_val: pd.DataFrame,
    n_train: int,
    n_validation: int,
    n_features: int,
    cfg: ForecastConfig,
    presplit: bool,
) -> dict[str, Any]:
    """Metrics payload.

    Args:
        metrics (dict[str, float]): dict[str, float] value used by the function.
        metadata_train (pd.DataFrame): pd.DataFrame value used by the function.
        metadata_val (pd.DataFrame): pd.DataFrame value used by the function.
        n_train (int): int value used by the function.
        n_validation (int): int value used by the function.
        n_features (int): int value used by the function.
        cfg (ForecastConfig): Configuration object used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    train_models = int(metadata_train["model_id"].astype(str).nunique())
    val_models = int(metadata_val["model_id"].astype(str).nunique())
    return {
        **metrics,
        "n_train": int(n_train),
        "n_validation": int(n_validation),
        "n_features": int(n_features),
        "n_train_models": train_models,
        "n_validation_models": val_models,
        "target_mode": str(cfg.sequence.target_mode),
        "target_column": str(cfg.data.target_column),
        "horizon": int(cfg.sequence.horizon),
        "max_lag": int(cfg.sequence.max_lag),
        "min_history": int(cfg.sequence.min_history),
        "presplit": bool(presplit),
    }


def _feature_importance_frame(model: Any, columns: pd.Index) -> pd.DataFrame | None:
    """Feature importance frame.

    Args:
        model (Any): Model to process.
        columns (pd.Index): Columns to process.

    Returns:
        pd.DataFrame | None: Result of the function.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    return pd.DataFrame(
        {
            "feature": list(columns),
            "importance": importances,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
