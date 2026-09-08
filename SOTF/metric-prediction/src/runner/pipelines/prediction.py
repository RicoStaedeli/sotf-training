from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.dataset_splitter import split_dataset
from data.feature_preprocessing import FeaturePreprocessor
from data.dataset_splitter import DatasetSplit
from data.weightstatistic_dataset_loader import load_weightstatistic_dataset, load_weightstatistic_split_datasets
from models import create_regressor
from runner.tuning import run_ray_tune
from utils.artifacts import write_frame, write_json, write_model_bundle
from utils.metrics import compute_regression_metrics
from utils.shap_analysis import write_shap_feature_contributions
from runner.wandb_config import attach_wandb_to_model_config

log = logging.getLogger(__name__)


def run_prediction_experiment(cfg: DictConfig, *, output_dir: Path | None = None) -> Dict[str, Any]:
    """Run prediction experiment.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        output_dir (Path | None): Output directory for artifacts.

    Returns:
        Dict[str, Any]: Result of the function.
    """
    log.info("Starting experiment pipeline for '%s'", cfg.experiment_name)
    is_presplit = bool(cfg.data.get("pt_train_path"))
    dataset = None
    if is_presplit:
        train_dataset, val_dataset = load_weightstatistic_split_datasets(cfg.data, cfg.features)
        split = DatasetSplit(
            X_train=train_dataset.features,
            X_val=val_dataset.features,
            y_train=train_dataset.target,
            y_val=val_dataset.target,
            metadata_train=train_dataset.metadata,
            metadata_val=val_dataset.metadata,
        )
        log.info(
            "Pre-split weight-statistic datasets loaded: train=%d validation=%d",
            len(split.X_train),
            len(split.X_val),
        )
    else:
        dataset = load_weightstatistic_dataset(cfg.data, cfg.features)
        split = split_dataset(
            dataset,
            validation_size=float(cfg.train.validation_size),
            random_seed=int(cfg.train.random_seed),
            shuffle=bool(cfg.train.shuffle),
            split_csv_path=cfg.data.get("split_csv_path", None),
        )
        log.info(
            "Train/validation split created: train=%d validation=%d",
            len(split.X_train),
            len(split.X_val),
        )

    model_cfg: dict[str, Any] = OmegaConf.to_container(cfg.model, resolve=True)  # type: ignore[assignment]
    tuning_result = None
    if bool(cfg.tune.enabled):
        log.info("Ray Tune enabled; starting hyperparameter search")
        tune_cfg: dict[str, Any] = OmegaConf.to_container(cfg.tune, resolve=True)  # type: ignore[assignment]
        tuning_result = run_ray_tune(
            model_cfg=model_cfg,
            tune_cfg=tune_cfg,
            split=split,
            fill_strategy=str(cfg.features.missing_value_fill),
            output_dir=_artifacts_dir(cfg, output_dir=output_dir),
        )
        model_cfg = deepcopy(model_cfg)
        model_cfg["params"].update(tuning_result["best_params"])
        log.info("Best Ray Tune parameters: %s", tuning_result["best_params"])

    log.info("Fitting preprocessor with missing value strategy '%s'", cfg.features.missing_value_fill)
    preprocessor = FeaturePreprocessor(strategy=str(cfg.features.missing_value_fill))
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    fitted_feature_names = list(X_train.columns)
    _log_feature_names(
        fitted_feature_names,
        enabled=bool(cfg.features.log_feature_names),
        max_console_logged_features=int(cfg.features.max_console_logged_features),
    )

    model_cfg = attach_wandb_to_model_config(model_cfg, cfg)
    log.info("Training model '%s' with %d features", model_cfg["name"], X_train.shape[1])
    model = create_regressor(model_cfg)
    model.fit(X_train, split.y_train)
    log.info("Model fit complete; generating validation predictions")
    predictions = model.predict(X_val)
    metrics = compute_regression_metrics(split.y_val.to_numpy(), predictions)
    log.info("Validation metrics: %s", metrics)

    artifacts_dir = _artifacts_dir(cfg, output_dir=output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing artifacts to %s", artifacts_dir)

    metrics_payload = {
        **metrics,
        "n_train": int(len(split.X_train)),
        "n_validation": int(len(split.X_val)),
        "n_features": int(X_train.shape[1]),
    }
    write_json(artifacts_dir / "metrics.json", metrics_payload)
    write_json(artifacts_dir / "resolved_model_config.json", model_cfg)
    write_json(
        artifacts_dir / "used_features.json",
        {
            "count": len(fitted_feature_names),
            "features": fitted_feature_names,
        },
    )

    if tuning_result is not None:
        write_json(artifacts_dir / "ray_tune_summary.json", tuning_result)

    if bool(cfg.train.save_model):
        write_model_bundle(
            artifacts_dir,
            {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns": fitted_feature_names,
            },
        )

    if bool(cfg.train.save_predictions):
        prediction_frame = split.metadata_val.copy()
        prediction_frame["target"] = split.y_val.to_numpy()
        prediction_frame["prediction"] = predictions
        prediction_frame["residual"] = prediction_frame["target"] - prediction_frame["prediction"]
        write_frame(artifacts_dir / "validation_predictions.parquet", prediction_frame.reset_index())

    if bool(cfg.train.save_feature_matrix):
        if is_presplit:
            feature_frame = pd.concat([split.X_train, split.X_val], axis=0)
            target_series = pd.concat([split.y_train, split.y_val], axis=0)
            feature_frame.insert(0, "target", target_series)
        else:
            assert dataset is not None
            feature_frame = dataset.features.copy()
            feature_frame.insert(0, "target", dataset.target)
        write_frame(artifacts_dir / "feature_matrix.parquet", feature_frame.reset_index())

    feature_importances = _feature_importance_frame(model, X_train.columns)
    if feature_importances is not None:
        write_frame(artifacts_dir / "feature_importances.parquet", feature_importances)

    if bool(cfg.train.save_shap_artifacts):
        write_shap_feature_contributions(
            model=model,
            X=X_val,
            artifacts_dir=artifacts_dir,
            max_samples=int(cfg.train.shap_max_samples),
            max_display=int(cfg.train.shap_max_display),
            random_seed=int(cfg.train.random_seed),
        )

    log.info("Experiment pipeline finished successfully")
    return {
        "metrics": metrics,
        "artifacts_dir": str(artifacts_dir),
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


def _log_feature_names(
    feature_names: list[str],
    *,
    enabled: bool,
    max_console_logged_features: int,
) -> None:
    """Log feature names.

    Args:
        feature_names (list[str]): Feature names to process.
        enabled (bool): bool value used by the function.
        max_console_logged_features (int): int value used by the function.

    Returns:
        None: None.
    """
    if not enabled:
        return

    log.info("Model will fit using %d features", len(feature_names))
    if len(feature_names) <= max_console_logged_features:
        log.info("Used features: %s", feature_names)
        return

    preview = feature_names[:max_console_logged_features]
    log.info(
        "Used features (first %d of %d): %s",
        max_console_logged_features,
        len(feature_names),
        preview,
    )
    log.info("Full feature list is saved to used_features.json")
