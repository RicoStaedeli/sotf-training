from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.dataset_splitter import split_dataset
from data.feature_preprocessing import FeaturePreprocessor
from data.latent_forecast_dataset_loader import (
    load_latent_forecast_dataset,
    load_latent_forecast_split_datasets,
    load_parquet_only_forecast_dataset,
)
from forecasting.config import SequenceFeatureConfig
from forecasting.dataset import WINDOW_SEQUENCE_COLUMN, build_forecast_dataset, build_latent_window_forecast_dataset
from models.factory import create_regressor
from runner.wandb_config import attach_wandb_to_model_config
from utils.artifacts import write_frame, write_json, write_model_bundle
from utils.metrics import compute_regression_metrics

log = logging.getLogger(__name__)


def run_latent_forecasting_experiment(cfg: DictConfig, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Run latent forecasting experiment.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        output_dir (Path | None): Output directory for artifacts.

    Returns:
        dict[str, Any]: Result of the function.
    """
    log.info("Starting latent-forecast experiment '%s'", cfg.experiment_name)

    is_presplit = bool(cfg.data.get("pt_train_path"))

    sequence_cfg = SequenceFeatureConfig(
        target_mode=str(cfg.sequence.target_mode),
        horizon=int(cfg.sequence.horizon),
        max_lag=int(cfg.sequence.max_lag),
        min_history=int(cfg.sequence.min_history),
        add_delta_features=bool(cfg.sequence.add_delta_features),
        include_time_features=bool(cfg.sequence.get("include_time_features", True)),
        include_current_target=bool(cfg.sequence.include_current_target),
        include_terminal_samples=bool(cfg.sequence.include_terminal_samples),
        observed_metric_columns=[str(c) for c in cfg.sequence.get("observed_metric_columns", [])],
    )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg = attach_wandb_to_model_config(model_cfg, cfg)
    is_window_lstm = _is_latent_window_sequence_model(model_cfg)

    if is_presplit:
        train_prepared, val_prepared = load_latent_forecast_split_datasets(cfg.data)
        if is_window_lstm:
            train_forecast = build_latent_window_forecast_dataset(train_prepared, sequence_cfg)
            val_forecast = build_latent_window_forecast_dataset(val_prepared, sequence_cfg)
            preprocessor = None
            X_train = train_forecast.features
            X_val = val_forecast.features
        else:
            train_forecast = build_forecast_dataset(train_prepared, sequence_cfg)
            val_forecast = build_forecast_dataset(val_prepared, sequence_cfg)
            preprocessor = FeaturePreprocessor(strategy="none")
            X_train = preprocessor.fit_transform(train_forecast.features)
            X_val = preprocessor.transform(val_forecast.features)
        y_train = train_forecast.target
        y_val = val_forecast.target
        metadata_train = train_forecast.metadata
        metadata_val = val_forecast.metadata
        # Keep a reference for save_feature_matrix
        _train_forecast_ref = train_forecast
        _val_forecast_ref = val_forecast
    else:
        if bool(cfg.data.get("pt_path")):
            prepared = load_latent_forecast_dataset(cfg.data)
        else:
            prepared = load_parquet_only_forecast_dataset(cfg.data)
        if is_window_lstm:
            forecast_dataset = build_latent_window_forecast_dataset(prepared, sequence_cfg)
        else:
            forecast_dataset = build_forecast_dataset(prepared, sequence_cfg)
        split = split_dataset(
            forecast_dataset,
            validation_size=float(cfg.train.validation_size),
            random_seed=int(cfg.train.random_seed),
            split_csv_path=cfg.data.get("split_csv_path", None),
        )
        if is_window_lstm:
            preprocessor = None
            X_train = split.X_train
            X_val = split.X_val
        else:
            preprocessor = FeaturePreprocessor(strategy="none")
            X_train = preprocessor.fit_transform(split.X_train)
            X_val = preprocessor.transform(split.X_val)
        y_train = split.y_train
        y_val = split.y_val
        metadata_train = split.metadata_train
        metadata_val = split.metadata_val
        _train_forecast_ref = forecast_dataset
        _val_forecast_ref = None

    feature_names = list(X_train.columns)

    log.info(
        "Training '%s' with %d train rows, %d val rows, %d features",
        model_cfg["name"],
        len(y_train),
        len(y_val),
        len(feature_names),
    )
    model = create_regressor(model_cfg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    metrics = compute_regression_metrics(y_val.to_numpy(), predictions)
    log.info("Validation metrics: %s", metrics)

    artifacts_dir = _artifacts_dir(cfg, output_dir=output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing artifacts to %s", artifacts_dir)

    train_models = int(metadata_train["model_id"].astype(str).nunique())
    val_models = int(metadata_val["model_id"].astype(str).nunique())
    write_json(
        artifacts_dir / "metrics.json",
        {
            **metrics,
            "n_train": len(y_train),
            "n_validation": len(y_val),
            "n_features": len(feature_names),
            "n_train_models": train_models,
            "n_validation_models": val_models,
            "target_mode": str(sequence_cfg.target_mode),
            "target_column": str(cfg.data.target_column),
            "horizon": int(sequence_cfg.horizon),
            "max_lag": int(sequence_cfg.max_lag),
            "min_history": int(sequence_cfg.min_history),
            "presplit": is_presplit,
            "latent_window_mode": is_window_lstm,
        },
    )
    write_json(artifacts_dir / "resolved_model_config.json", model_cfg)
    write_json(
        artifacts_dir / "used_features.json",
        {"count": len(feature_names), "features": feature_names},
    )

    if bool(cfg.train.save_predictions):
        pred_frame = metadata_val.copy()
        pred_frame["target"] = y_val.to_numpy(dtype="float64")
        pred_frame["prediction"] = predictions
        pred_frame["residual"] = pred_frame["target"] - pred_frame["prediction"]
        write_frame(artifacts_dir / "validation_predictions.parquet", pred_frame.reset_index())

    if bool(cfg.train.save_model):
        write_model_bundle(
            artifacts_dir,
            {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns": feature_names,
                "sequence_config": {
                    "target_mode": sequence_cfg.target_mode,
                    "horizon": sequence_cfg.horizon,
                    "max_lag": sequence_cfg.max_lag,
                    "min_history": sequence_cfg.min_history,
                    "add_delta_features": sequence_cfg.add_delta_features,
                    "include_time_features": sequence_cfg.include_time_features,
                    "include_current_target": sequence_cfg.include_current_target,
                },
            },
        )

    if bool(cfg.train.save_feature_matrix):
        if is_presplit:
            combined_features = pd.concat(
                [_train_forecast_ref.features, _val_forecast_ref.features], ignore_index=True
            )
            combined_target = pd.concat(
                [_train_forecast_ref.target, _val_forecast_ref.target], ignore_index=True
            )
            combined_features.insert(0, "target", combined_target.values)
            combined_features = _artifact_feature_frame(combined_features)
            write_frame(artifacts_dir / "forecast_feature_matrix.parquet", combined_features.reset_index())
        else:
            feature_matrix = _train_forecast_ref.features.copy()
            feature_matrix.insert(0, "target", _train_forecast_ref.target)
            feature_matrix = _artifact_feature_frame(feature_matrix)
            write_frame(artifacts_dir / "forecast_feature_matrix.parquet", feature_matrix.reset_index())

    feature_importances = _feature_importance_frame(model, pd.Index(feature_names))
    if feature_importances is not None:
        write_frame(artifacts_dir / "feature_importances.parquet", feature_importances)

    log.info("Latent forecast experiment finished successfully")
    return {"metrics": metrics, "artifacts_dir": str(artifacts_dir)}


def _is_latent_window_sequence_model(model_cfg: dict[str, Any]) -> bool:
    """Return True for any model that consumes variable-length window sequences.

    Both the LSTM and the hierarchical transformer share the same data pipeline
    (``build_latent_window_forecast_dataset``), so both are handled here.
    """
    model_name = str(model_cfg.get("name", "")).lower()
    return model_name in {
        # LSTM variants
        "latent_window_lstm",
        "lstm_latent_sequences",
        "lstm_latent_windows",
        "latent_windows_lstm",
        # Hierarchical transformer variants
        "latent_hierarchical_transformer",
        "hierarchical_transformer",
        "latent_window_transformer",
        "latent_windows_transformer",
    }


# Keep the old name as an alias so external code doesn't break
_is_latent_window_lstm_model = _is_latent_window_sequence_model


def _artifact_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    if WINDOW_SEQUENCE_COLUMN not in features.columns:
        return features
    serializable = features.copy()
    serializable["window_sequence_length"] = serializable[WINDOW_SEQUENCE_COLUMN].map(lambda value: int(value.shape[0]))
    serializable["window_embedding_dim"] = serializable[WINDOW_SEQUENCE_COLUMN].map(lambda value: int(value.shape[1]))
    return serializable.drop(columns=[WINDOW_SEQUENCE_COLUMN])


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
    return pd.DataFrame({"feature": list(columns), "importance": importances}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def _artifacts_dir(cfg: DictConfig, *, output_dir: Path | None = None) -> Path:
    """Artifacts dir.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        output_dir (Path | None): Output directory for artifacts.

    Returns:
        Path: Resolved path.
    """
    if output_dir is None and HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    elif output_dir is None:
        output_dir = Path.cwd()
    return output_dir / str(cfg.train.artifacts_dir)
