from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from data.weightstatistic_dataset import WeightStatisticDataset
from data.latent_forecast_dataset_loader import (
    load_latent_forecast_dataset,
    load_latent_forecast_split_datasets,
)
from data.weightstatistic_dataset_loader import load_weightstatistic_dataset, load_weightstatistic_pt_dataset
from forecasting.dataset import WINDOW_SEQUENCE_COLUMN, build_forecast_dataset, build_latent_window_forecast_dataset

from models.predictors import (
    ForecastModelPredictor,
    NaivePredictor,
    OnlineHierarchicalTransformerPredictor,
    OnlineLCExtrapolationPredictor,
    OnlineLCNetPredictor,
    OnlineLSTMPredictor,
    OnlineXGBoostPredictor,
    PopulationPredictor,
    load_forecast_model_bundle,
    normalize_observed_metric_columns,
)
from utils.settings import (
    FeatureSelectionConfig,
    ForecastModelPredictorConfig,
    OnlineHierarchicalTransformerPredictorConfig,
    OnlineLCExtrapolationPredictorConfig,
    OnlineLCNetPredictorConfig,
    OnlineLSTMPredictorConfig,
    SequenceFeatureConfig,
    SimulationConfig,
    XGBoostPredictorConfig,
    feature_selection_config_from_mapping,
    sequence_feature_config_from_mapping,
)

log = logging.getLogger(__name__)


def build_predictor_and_effective_configs(
    cfg: SimulationConfig,
) -> tuple[PopulationPredictor, FeatureSelectionConfig, SequenceFeatureConfig, dict[str, Any]]:
    """Build the configured predictor and resolve any bundle-provided feature/sequence settings.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.

    Returns:
        tuple[PopulationPredictor, FeatureSelectionConfig, SequenceFeatureConfig, dict[str, Any]]: Computed predictor and effective configs.
    """
    if isinstance(cfg.predictor, ForecastModelPredictorConfig):
        if cfg.predictor.model_bundle_path is None:
            raise ValueError("predictor.model_bundle_path must be set when predictor.name=forecast_model.")

        bundle = load_forecast_model_bundle(
            cfg.predictor.model_bundle_path,
            used_features_path=cfg.predictor.used_features_path,
        )
        bundle_config = bundle.config
        effective_feature_cfg = cfg.features
        effective_sequence_cfg = cfg.sequence

        if bool(cfg.predictor.use_bundle_feature_config):
            features_mapping = bundle_config.get("features")
            if isinstance(features_mapping, dict):
                effective_feature_cfg = feature_selection_config_from_mapping(features_mapping)
                log.info("Using feature-selection config embedded in forecast model bundle")
            elif data_source(cfg) == "latent_forecast":
                effective_feature_cfg = FeatureSelectionConfig(missing_value_fill="none")
                log.info("Forecast model bundle has no feature config; using latent-forecast feature defaults")
            else:
                require_bundle_mapping(bundle_config, "features")
        if bool(cfg.predictor.use_bundle_sequence_config):
            sequence_mapping = require_bundle_mapping(bundle_config, "sequence")
            effective_sequence_cfg = sequence_feature_config_from_mapping(sequence_mapping)
            log.info("Using sequence config embedded in forecast model bundle")

        validate_bundle_compatibility(cfg, bundle_config, effective_sequence_cfg)
        return ForecastModelPredictor(bundle), effective_feature_cfg, effective_sequence_cfg, bundle_config

    if isinstance(cfg.predictor, XGBoostPredictorConfig):
        observed_metric_columns = normalize_observed_metric_columns(cfg.predictor.observed_metric_columns)
        return (
            OnlineXGBoostPredictor(
                model_params=dict(cfg.predictor.params),
                missing_value_fill=str(cfg.features.missing_value_fill),
                min_train_rows=int(cfg.predictor.min_train_rows),
                min_train_models=int(cfg.predictor.min_train_models),
                fallback_strategy=str(cfg.predictor.fallback_strategy),
                include_observed_metrics=bool(cfg.predictor.include_observed_metrics),
                observed_metric_columns=tuple(observed_metric_columns),
            ),
            cfg.features,
            cfg.sequence,
            {},
        )
    if isinstance(cfg.predictor, OnlineLSTMPredictorConfig):
        observed_metric_columns = normalize_observed_metric_columns(cfg.predictor.observed_metric_columns)
        return (
            OnlineLSTMPredictor(
                model_params=dict(cfg.predictor.params),
                missing_value_fill=str(cfg.features.missing_value_fill),
                min_train_rows=int(cfg.predictor.min_train_rows),
                min_train_models=int(cfg.predictor.min_train_models),
                fallback_strategy=str(cfg.predictor.fallback_strategy),
                include_observed_metrics=bool(cfg.predictor.include_observed_metrics),
                observed_metric_columns=tuple(observed_metric_columns),
            ),
            cfg.features,
            cfg.sequence,
            {},
        )
    if isinstance(cfg.predictor, OnlineHierarchicalTransformerPredictorConfig):
        observed_metric_columns = normalize_observed_metric_columns(cfg.predictor.observed_metric_columns)
        return (
            OnlineHierarchicalTransformerPredictor(
                model_params=dict(cfg.predictor.params),
                missing_value_fill=str(cfg.features.missing_value_fill),
                min_train_rows=int(cfg.predictor.min_train_rows),
                min_train_models=int(cfg.predictor.min_train_models),
                fallback_strategy=str(cfg.predictor.fallback_strategy),
                include_observed_metrics=bool(cfg.predictor.include_observed_metrics),
                observed_metric_columns=tuple(observed_metric_columns),
            ),
            cfg.features,
            cfg.sequence,
            {},
        )

    if isinstance(cfg.predictor, OnlineLCExtrapolationPredictorConfig):
        return (
            OnlineLCExtrapolationPredictor(
                min_points=int(cfg.predictor.min_points),
                fallback_strategy=str(cfg.predictor.fallback_strategy),
            ),
            cfg.features,
            cfg.sequence,
            {},
        )
    if isinstance(cfg.predictor, OnlineLCNetPredictorConfig):
        return (
            OnlineLCNetPredictor(
                model_params=dict(cfg.predictor.params),
                max_history_points=int(cfg.predictor.max_history_points),
                min_train_rows=int(cfg.predictor.min_train_rows),
                min_train_models=int(cfg.predictor.min_train_models),
                fallback_strategy=str(cfg.predictor.fallback_strategy),
            ),
            cfg.features,
            cfg.sequence,
            {},
        )
    return NaivePredictor(strategy=cfg.predictor.strategy), cfg.features, cfg.sequence, {}


def load_simulation_dataset(
    *,
    cfg: SimulationConfig,
    predictor: PopulationPredictor,
    effective_feature_cfg: FeatureSelectionConfig,
    effective_sequence_cfg: SequenceFeatureConfig,
) -> WeightStatisticDataset:
    """Load the prepared tabular or latent-forecast data in the shape required by SOTF.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        predictor (PopulationPredictor): PopulationPredictor value used by the function.
        effective_feature_cfg (FeatureSelectionConfig): Configuration for effective feature.
        effective_sequence_cfg (SequenceFeatureConfig): Effective sequence configuration for forecasting.

    Returns:
        WeightStatisticDataset: Computed simulation dataset.
    """
    if data_source(cfg) == "latent_forecast":
        data_cfg = cfg.data.to_omegaconf()
        if not data_cfg.get("target_parquet_path"):
            data_cfg.target_parquet_path = cfg.data.parquet_path
        if data_cfg.get("pt_train_path"):
            train_prepared, val_prepared = load_latent_forecast_split_datasets(data_cfg)
            prepared = concat_prepared_datasets(train_prepared, val_prepared)
            log.info("Loaded combined latent forecast dataset from %s and %s", data_cfg.pt_train_path, data_cfg.pt_val_path)
        else:
            prepared = load_latent_forecast_dataset(data_cfg)
            log.info("Loaded latent forecast dataset from %s", data_cfg.pt_path)
            if data_cfg.get("pt_val_path"):
                from omegaconf import OmegaConf
                val_data_cfg = OmegaConf.merge(data_cfg, {"pt_path": str(data_cfg.pt_val_path)})
                val_prepared = load_latent_forecast_dataset(val_data_cfg)
                prepared = concat_prepared_datasets(prepared, val_prepared)
                log.info("Concatenated latent forecast val split from %s", data_cfg.pt_val_path)
        prepared = filter_dataset_to_pt_model_ids(prepared, cfg=cfg)
        return filter_dataset_to_latent_valid_model_ids(
            prepared,
            cfg=cfg,
            effective_sequence_cfg=effective_sequence_cfg,
        )

    data_cfg = cfg.data.to_omegaconf()
    data_cfg.metadata_columns = resolved_metadata_columns(cfg=cfg, predictor=predictor)
    if data_cfg.get("pt_path"):
        prepared = load_weightstatistic_pt_dataset(data_cfg, effective_feature_cfg.to_omegaconf())
        log.info("Loaded prepared weight-statistic split from %s", data_cfg.pt_path)
        if data_cfg.get("pt_val_path"):
            from omegaconf import OmegaConf
            val_data_cfg = OmegaConf.merge(data_cfg, {"pt_path": str(data_cfg.pt_val_path)})
            val_prepared = load_weightstatistic_pt_dataset(val_data_cfg, effective_feature_cfg.to_omegaconf())
            prepared = concat_prepared_datasets(prepared, val_prepared)
            log.info("Concatenated weight-statistic val split from %s", data_cfg.pt_val_path)
        return prepared

    data_cfg.cache_dir = str(Path(cfg.data.parquet_path).expanduser().resolve().parent)
    prepared = load_weightstatistic_dataset(data_cfg, effective_feature_cfg.to_omegaconf())
    return filter_dataset_to_pt_model_ids(prepared, cfg=cfg)


def data_source(cfg: SimulationConfig) -> str:
    """Return the normalized data backend name used by validation and loading.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.

    Returns:
        str: Resolved string value.
    """
    return str(getattr(cfg.data, "source", "prepared_dataset")).lower()


def concat_prepared_datasets(*datasets: WeightStatisticDataset) -> WeightStatisticDataset:
    """Concatenate split weight-statistic datasets while rebuilding a clean sample index.

    Args:
        *datasets (WeightStatisticDataset): Datasets to process.

    Returns:
        WeightStatisticDataset: Result of the function.
    """
    if not datasets:
        raise ValueError("At least one prepared dataset is required.")

    features = pd.concat([dataset.features for dataset in datasets], ignore_index=True)
    target = pd.concat([dataset.target for dataset in datasets], ignore_index=True)
    metadata = pd.concat([dataset.metadata for dataset in datasets], ignore_index=True)

    sample_idx = pd.Index(range(len(features)), name="sample_idx")
    features.index = sample_idx
    target.index = sample_idx
    metadata.index = sample_idx
    target.name = datasets[0].target.name

    return WeightStatisticDataset(
        features=features,
        target=target,
        metadata=metadata,
        feature_metadata=datasets[0].feature_metadata.copy(),
    )


def filter_dataset_to_pt_model_ids(
    dataset: WeightStatisticDataset,
    *,
    cfg: SimulationConfig,
) -> WeightStatisticDataset:
    """Restrict a prepared dataset to the model IDs present in one or two PT files.

    Reads model IDs from ``data.model_ids_from_pt_path`` and, when set, also from
    ``data.model_ids_from_pt_val_path``, then keeps only rows whose model ID appears
    in the union of both sets.

    Args:
        dataset (WeightStatisticDataset): Dataset to process.
        cfg (SimulationConfig): Configuration object used by the function.

    Returns:
        WeightStatisticDataset: Result of the function.
    """
    pt_path = getattr(cfg.data, "model_ids_from_pt_path", None)
    if not pt_path:
        return dataset

    normalize_basename = bool(getattr(cfg.data, "id_normalize_basename", True))
    keep_model_ids = load_model_ids_from_pt(pt_path, normalize_basename=normalize_basename)

    pt_val_path = getattr(cfg.data, "model_ids_from_pt_val_path", None)
    if pt_val_path:
        keep_model_ids = keep_model_ids | load_model_ids_from_pt(pt_val_path, normalize_basename=normalize_basename)

    return filter_dataset_to_model_ids(
        dataset,
        keep_model_ids=keep_model_ids,
        normalize_basename=normalize_basename,
        reason=f"PT model IDs from {pt_path}" + (f" + {pt_val_path}" if pt_val_path else ""),
        empty_error=(
            "Filtering prepared dataset by PT model IDs removed all rows. "
            f"Check data.model_ids_from_pt_path={pt_path!r} and data.parquet_path={cfg.data.parquet_path!r}."
        ),
    )


def filter_dataset_to_latent_valid_model_ids(
    dataset: WeightStatisticDataset,
    *,
    cfg: SimulationConfig,
    effective_sequence_cfg: SequenceFeatureConfig,
) -> WeightStatisticDataset:
    """Restrict a dataset to models that remain valid after latent-forecast sequence construction.

    Args:
        dataset (WeightStatisticDataset): Dataset to process.
        cfg (SimulationConfig): Configuration object used by the function.
        effective_sequence_cfg (SequenceFeatureConfig): Effective sequence configuration for forecasting.

    Returns:
        WeightStatisticDataset: Result of the function.
    """
    pt_path = getattr(cfg.data, "model_ids_from_pt_path", None)
    if not pt_path:
        return dataset

    normalize_basename = bool(getattr(cfg.data, "id_normalize_basename", True))
    comparable_model_ids = load_latent_valid_model_ids(
        cfg=cfg,
        effective_sequence_cfg=effective_sequence_cfg,
    )
    return filter_dataset_to_model_ids(
        dataset,
        keep_model_ids=comparable_model_ids,
        normalize_basename=normalize_basename,
        reason=f"latent-valid forecast population from {pt_path}",
        empty_error=(
            "Filtering to the latent-valid forecast population removed all rows. "
            f"Check data.model_ids_from_pt_path={pt_path!r}, data.parquet_path={cfg.data.parquet_path!r}, "
            f"and the effective sequence config {effective_sequence_cfg!r}."
        ),
    )


def load_latent_valid_model_ids(
    *,
    cfg: SimulationConfig,
    effective_sequence_cfg: SequenceFeatureConfig,
) -> set[str]:
    """Derive the comparable model population from the latent forecast dataset under the effective sequence config.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        effective_sequence_cfg (SequenceFeatureConfig): Effective sequence configuration for forecasting.

    Returns:
        set[str]: Set of string values.
    """
    pt_path = getattr(cfg.data, "model_ids_from_pt_path", None)
    if not pt_path:
        return set()

    reference_cfg = cfg.data.to_omegaconf()
    reference_cfg.pt_path = pt_path
    reference_cfg.pt_train_path = None
    reference_cfg.pt_val_path = None
    if not reference_cfg.get("target_parquet_path"):
        reference_cfg.target_parquet_path = cfg.data.parquet_path

    latent_prepared = load_latent_forecast_dataset(reference_cfg)
    latent_forecast_dataset = build_simulation_forecast_dataset(
        latent_prepared,
        effective_sequence_cfg=effective_sequence_cfg,
    )
    valid_model_ids = {
        str(model_id)
        for model_id in latent_forecast_dataset.metadata["model_id"].astype(str).unique().tolist()
        if str(model_id)
    }
    if not valid_model_ids:
        raise ValueError(
            "No latent-valid model IDs remain after applying the effective sequence config. "
            f"Check data.model_ids_from_pt_path={pt_path!r}."
        )
    log.info(
        "Derived latent-valid forecast population from %s: %d models",
        pt_path,
        len(valid_model_ids),
    )
    return valid_model_ids


def build_simulation_forecast_dataset(
    prepared_dataset: WeightStatisticDataset,
    *,
    effective_sequence_cfg: SequenceFeatureConfig,
):
    """Build the forecast dataset shape expected by the simulation and predictor."""
    if WINDOW_SEQUENCE_COLUMN in prepared_dataset.features.columns:
        return build_latent_window_forecast_dataset(prepared_dataset, effective_sequence_cfg)
    return build_forecast_dataset(prepared_dataset, effective_sequence_cfg)


def filter_dataset_to_model_ids(
    dataset: WeightStatisticDataset,
    *,
    keep_model_ids: set[str],
    normalize_basename: bool,
    reason: str,
    empty_error: str,
) -> WeightStatisticDataset:
    """Filter a dataset by model ID while preserving the weight-statistic dataset structure.

    Args:
        dataset (WeightStatisticDataset): Dataset to process.
        keep_model_ids (set[str]): Model IDs to keep.
        normalize_basename (bool): Whether to normalize dataset model IDs to basenames before comparing.
        reason (str): Human-readable reason for logging.
        empty_error (str): Error message if filtering removes all rows.

    Returns:
        WeightStatisticDataset: Result of the function.
    """
    if "model_id" not in dataset.metadata.columns:
        raise ValueError("Filtering by model IDs requires metadata column 'model_id'.")

    model_ids = dataset.metadata["model_id"].astype(str)
    if normalize_basename:
        model_ids = model_ids.map(lambda model_id: Path(str(model_id)).name)

    keep_mask = model_ids.isin(keep_model_ids)
    if not bool(keep_mask.any()):
        raise ValueError(empty_error)

    filtered = WeightStatisticDataset(
        features=dataset.features.loc[keep_mask].copy(),
        target=dataset.target.loc[keep_mask].copy(),
        metadata=dataset.metadata.loc[keep_mask].copy(),
        feature_metadata=dataset.feature_metadata.copy(),
    )
    log.info(
        "Filtered dataset to %s: kept %d / %d rows (%d / %d models)",
        reason,
        len(filtered.features),
        len(dataset.features),
        int(filtered.metadata["model_id"].astype(str).nunique()),
        int(dataset.metadata["model_id"].astype(str).nunique()),
    )
    return filtered


def load_model_ids_from_pt(path: str | Path, *, normalize_basename: bool) -> set[str]:
    """Load unique model IDs from a latent-representation PT artifact.

    Args:
        path (str | Path): Path to the input resource.
        normalize_basename (bool): bool value used by the function.

    Returns:
        set[str]: Set of string values.
    """
    pt_path = Path(path).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Latent model-id source does not exist: {pt_path}")

    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {pt_path} to contain a dict with model IDs.")

    for key in ("model_ids", "model_id", "models", "model"):
        if key not in payload:
            continue
        values = payload[key]
        if hasattr(values, "tolist"):
            values = values.tolist()
        model_ids = [str(value) for value in values]
        if normalize_basename:
            model_ids = [Path(model_id).name for model_id in model_ids]
        unique_model_ids = {model_id for model_id in model_ids if model_id}
        if not unique_model_ids:
            raise ValueError(f"Model-id field '{key}' in {pt_path} is empty.")
        return unique_model_ids

    raise ValueError(f"Could not find a model-id field in {pt_path}; tried model_ids, model_id, models, model.")


def validate_bundle_compatibility(
    cfg: SimulationConfig,
    bundle_config: dict[str, Any],
    effective_sequence_cfg: SequenceFeatureConfig,
) -> None:
    """Ensure a forecast model bundle predicts the same target semantics as the simulation.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        bundle_config (dict[str, Any]): Configuration for bundle.
        effective_sequence_cfg (SequenceFeatureConfig): Configuration for effective sequence.

    Returns:
        None: None.
    """
    resolved_target_mode = str(effective_sequence_cfg.target_mode).lower()
    if resolved_target_mode not in {"horizon_value", "final_value"}:
        raise ValueError(
            "The simulation supports only sequence.target_mode in {'horizon_value', 'final_value'}. "
            f"Resolved sequence.target_mode={effective_sequence_cfg.target_mode!r}."
        )

    requested_target_mode = str(cfg.sequence.target_mode).lower()
    if requested_target_mode != resolved_target_mode:
        raise ValueError(
            "Configured simulation target_mode does not match the forecast model bundle target_mode: "
            f"{cfg.sequence.target_mode!r} vs {effective_sequence_cfg.target_mode!r}."
        )

    if resolved_target_mode == "horizon_value" and int(effective_sequence_cfg.horizon) != int(cfg.sequence.horizon):
        raise ValueError(
            "Configured simulation horizon does not match the forecast model bundle horizon: "
            f"{cfg.sequence.horizon} vs {effective_sequence_cfg.horizon}."
        )

    bundle_target_column = bundle_config.get("data", {}).get("target_column")
    if bundle_target_column is not None and str(bundle_target_column) != str(cfg.data.target_column):
        raise ValueError(
            "Configured simulation target column does not match the forecast model bundle target column: "
            f"{cfg.data.target_column!r} vs {bundle_target_column!r}."
        )


def resolved_metadata_columns(*, cfg: SimulationConfig, predictor: PopulationPredictor) -> list[str]:
    """Add predictor-required metadata columns while preserving user-configured columns.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        predictor (PopulationPredictor): PopulationPredictor value used by the function.

    Returns:
        list[str]: List of string values.
    """
    metadata_columns = [str(column) for column in cfg.data.metadata_columns]
    required_metadata_columns = getattr(predictor, "required_metadata_columns", None)
    if not callable(required_metadata_columns):
        return metadata_columns

    required_columns = required_metadata_columns()
    added_columns = [column for column in required_columns if column not in metadata_columns]
    if added_columns:
        metadata_columns.extend(added_columns)
        log.info("Added predictor-required metadata columns for %s: %s", predictor.name, added_columns)
    return metadata_columns


def used_features_for_predictor(*, predictor: PopulationPredictor, dataset) -> list[str]:
    """Report feature columns actually consumed by predictors that expose that information.

    Args:
        predictor (PopulationPredictor): PopulationPredictor value used by the function.
        dataset: Dataset to process.

    Returns:
        list[str]: List of string values.
    """
    if isinstance(predictor, ForecastModelPredictor):
        return list(predictor.bundle.feature_columns)
    used_feature_columns = getattr(predictor, "used_feature_columns", None)
    if callable(used_feature_columns):
        return used_feature_columns(dataset)
    return []


def require_bundle_mapping(bundle_config: dict[str, Any], key: str) -> dict[str, Any]:
    """Fetch a required mapping from a forecast bundle config with a clear error message.

    Args:
        bundle_config (dict[str, Any]): Configuration for bundle.
        key (str): Key to look up.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    value = bundle_config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Forecast model bundle config is missing a '{key}' mapping.")
    return value
