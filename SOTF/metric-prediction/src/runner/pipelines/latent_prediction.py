from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.dataset_splitter import split_dataset
from data.feature_preprocessing import FeaturePreprocessor
from data.latent_dataset_loader import load_latent_dataset, load_latent_split_dataset
from models import create_regressor
from utils.artifacts import write_frame, write_json, write_model_bundle
from utils.metrics import compute_regression_metrics
from utils.shap_analysis import write_shap_feature_contributions
from runner.wandb_config import attach_wandb_to_model_config

log = logging.getLogger(__name__)


def run_latent_prediction_experiment(cfg: DictConfig, *, output_dir: Path | None = None) -> Dict[str, Any]:
    """Run latent prediction experiment.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        output_dir (Path | None): Output directory for artifacts.

    Returns:
        Dict[str, Any]: Result of the function.
    """
    log.info("Starting latent-representation experiment '%s'", cfg.experiment_name)

    is_presplit = bool(cfg.data.get("pt_train_path"))

    if is_presplit:
        log.info("Pre-split mode: loading train and val PT files separately")
        split = load_latent_split_dataset(cfg.data)
        log.info(
            "Pre-split datasets loaded: train=%d val=%d",
            len(split.X_train),
            len(split.X_val),
        )
    else:
        dataset = load_latent_dataset(cfg.data)
        split = split_dataset(
            dataset,
            validation_size=float(cfg.train.validation_size),
            random_seed=int(cfg.train.random_seed),
            shuffle=bool(cfg.train.shuffle),
            split_csv_path=cfg.data.get("split_csv_path", None),
        )
        log.info(
            "Train/validation split: train=%d validation=%d",
            len(split.X_train),
            len(split.X_val),
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg = attach_wandb_to_model_config(model_cfg, cfg)

    preprocessor = FeaturePreprocessor(strategy="none")
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    fitted_feature_names = list(X_train.columns)

    log.info("Training model '%s' with %d features", model_cfg["name"], X_train.shape[1])
    model = create_regressor(model_cfg)
    model.fit(X_train, split.y_train)

    predictions = model.predict(X_val)
    metrics = compute_regression_metrics(split.y_val.to_numpy(), predictions)
    log.info("Validation metrics: %s", metrics)

    artifacts_dir = _artifacts_dir(cfg, output_dir=output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing artifacts to %s", artifacts_dir)

    write_json(
        artifacts_dir / "metrics.json",
        {**metrics, "n_train": len(split.X_train), "n_validation": len(split.X_val), "n_features": X_train.shape[1]},
    )
    write_json(artifacts_dir / "resolved_model_config.json", model_cfg)
    write_json(artifacts_dir / "used_features.json", {"count": len(fitted_feature_names), "features": fitted_feature_names})

    if bool(cfg.train.save_model):
        write_model_bundle(
            artifacts_dir,
            {"model": model, "preprocessor": preprocessor, "feature_columns": fitted_feature_names},
        )

    if bool(cfg.train.save_predictions):
        prediction_frame = split.metadata_val.copy()
        prediction_frame["target"] = split.y_val.to_numpy()
        prediction_frame["prediction"] = predictions
        prediction_frame["residual"] = prediction_frame["target"] - prediction_frame["prediction"]
        write_frame(artifacts_dir / "validation_predictions.parquet", prediction_frame.reset_index())

    if bool(cfg.train.save_feature_matrix):
        if is_presplit:
            feature_frame = pd.concat([split.X_train, split.X_val], ignore_index=True)
            target_combined = pd.concat([split.y_train, split.y_val], ignore_index=True)
            feature_frame.insert(0, "target", target_combined.values)
        else:
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

    log.info("Latent experiment finished successfully")
    return {"metrics": metrics, "artifacts_dir": str(artifacts_dir)}


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
