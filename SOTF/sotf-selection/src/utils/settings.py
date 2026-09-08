from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from forecasting.config import (
    FeatureSelectionConfig,
    PreparedDatasetConfig,
    SequenceFeatureConfig,
)


@dataclass(slots=True)
class SimulationDataConfig(PreparedDatasetConfig):
    source: str = "prepared_dataset"
    pt_path: str | None = None
    pt_train_path: str | None = None
    pt_val_path: str | None = None
    target_parquet_path: str | None = None
    model_ids_from_pt_path: str | None = None
    model_ids_from_pt_val_path: str | None = None
    id_normalize_basename: bool = True
    epoch_from_checkpoint_path: bool = False

    def to_omegaconf(self):
        """Convert omegaconf.

        Returns:
            Any: Result of the function.
        """
        base = {
            "parquet_path": self.parquet_path,
            "pt_path": self.pt_path,
            "pt_val_path": self.pt_val_path,
            "target_parquet_path": self.target_parquet_path,
            "target_column": self.target_column,
            "metadata_columns": list(self.metadata_columns),
            "epoch_filter": list(self.epoch_filter),
            "target_model_min_threshold": self.target_model_min_threshold,
            "model_ids_from_pt_path": self.model_ids_from_pt_path,
            "model_ids_from_pt_val_path": self.model_ids_from_pt_val_path,
        }
        if self.source == "latent_forecast":
            base.update(
                {
                    "pt_path": self.pt_path,
                    "pt_train_path": self.pt_train_path,
                    "pt_val_path": self.pt_val_path,
                    "target_parquet_path": self.target_parquet_path or self.parquet_path,
                    "id_normalize_basename": self.id_normalize_basename,
                    "epoch_from_checkpoint_path": self.epoch_from_checkpoint_path,
                }
            )
        return OmegaConf.create(base)


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    save_to_file: bool = True
    filename: str = "sotf_selection.log"


@dataclass(slots=True)
class SelectionConfig:
    start_epoch: int = 10
    interval: int = 10
    survival_rate: float = 0.5
    min_population: int = 1
    metric_goal: str = "maximize"
    eliminate_models_missing_decision_epoch: bool = True
    top_k_preview: int = 5


@dataclass(slots=True)
class ForecastModelPredictorConfig:
    name: str = "forecast_model"
    model_bundle_path: str | None = None
    used_features_path: str | None = None
    use_bundle_feature_config: bool = True
    use_bundle_sequence_config: bool = True


def _default_xgboost_params() -> dict[str, Any]:
    """Default xgboost params.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    return {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": 42,
    }


def _default_online_lstm_params() -> dict[str, Any]:
    """Default online latent-window LSTM params.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    return {
        "hidden_size": 64,
        "num_layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
        "head_hidden_layer_sizes": [32],
        "activation": "gelu",
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "max_epochs": 150,
        "patience": 20,
        "validation_fraction": 0.15,
        "early_stopping": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 0.00001,
        "gradient_clip_norm": 1.0,
        "scale_features": True,
        "scale_target": True,
        "device": "auto",
        "random_state": 42,
    }


def _default_online_transformer_params() -> dict[str, Any]:
    """Default online hierarchical-transformer params.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    return {
        "d_model": 64,
        "nhead": 4,
        "num_local_layers": 1,
        "num_global_layers": 2,
        "local_window_size": 1,
        "dim_feedforward": 128,
        "dropout": 0.0,
        "head_hidden_layer_sizes": [32],
        "activation": "gelu",
        "use_positional_encoding": True,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "max_epochs": 150,
        "patience": 20,
        "validation_fraction": 0.15,
        "early_stopping": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 0.00001,
        "gradient_clip_norm": 1.0,
        "scale_features": True,
        "scale_target": True,
        "device": "auto",
        "random_state": 42,
    }


def _default_observed_metric_columns() -> list[str]:
    """Default observed metric columns.

    Returns:
        list[str]: List of string values.
    """
    return [
        "current_target",
        "test_acc",
        "train_acc",
        "test_loss",
        "train_loss",
    ]


@dataclass(slots=True)
class XGBoostPredictorConfig:
    name: str = "xgboost"
    params: dict[str, Any] = field(default_factory=_default_xgboost_params)
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    include_observed_metrics: bool = True
    observed_metric_columns: list[str] = field(default_factory=_default_observed_metric_columns)


@dataclass(slots=True)
class OnlineLSTMPredictorConfig:
    name: str = "online_lstm"
    params: dict[str, Any] = field(default_factory=_default_online_lstm_params)
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    include_observed_metrics: bool = True
    observed_metric_columns: list[str] = field(default_factory=_default_observed_metric_columns)


@dataclass(slots=True)
class OnlineHierarchicalTransformerPredictorConfig:
    name: str = "online_hierarchical_transformer"
    params: dict[str, Any] = field(default_factory=_default_online_transformer_params)
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    include_observed_metrics: bool = True
    observed_metric_columns: list[str] = field(default_factory=_default_observed_metric_columns)


@dataclass(slots=True)
class NaivePredictorConfig:
    name: str = "naive"
    strategy: str = "current_target"


def _default_lcnet_params() -> dict[str, Any]:
    """Default LCNet BNN params.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    return {
        "hidden_layer_sizes": [128, 128],
        "activation": "relu",
        "dropout": 0.1,
        "use_layer_norm": False,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 120,
        "patience": 15,
        "validation_fraction": 0.15,
        "early_stopping": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 0.00001,
        "gradient_clip_norm": 1.0,
        "scale_features": True,
        "scale_target": True,
        "random_state": 42,
        "device": "auto",
        "loss": "smooth_l1",
        "mc_samples": 30,
    }


@dataclass(slots=True)
class OnlineLCExtrapolationPredictorConfig:
    name: str = "online_lc_extrapolation"
    min_points: int = 3
    fallback_strategy: str = "current_target"


@dataclass(slots=True)
class OnlineLCNetPredictorConfig:
    name: str = "online_lcnet"
    params: dict[str, Any] = field(default_factory=_default_lcnet_params)
    max_history_points: int = 8
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"


PredictorConfig = (
    ForecastModelPredictorConfig
    | XGBoostPredictorConfig
    | OnlineLSTMPredictorConfig
    | OnlineHierarchicalTransformerPredictorConfig
    | NaivePredictorConfig
    | OnlineLCExtrapolationPredictorConfig
    | OnlineLCNetPredictorConfig
)


@dataclass(slots=True)
class OutputConfig:
    experiment_name: str = "sotf_selection"
    output_root: str = "sotf-selection/outputs"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    run_dir: str | None = None

    def resolve_run_dir(self, *, now) -> Path:
        """Resolve run dir.

        Args:
            now: Timestamp used to construct a run directory.

        Returns:
            Path: Resolved path.
        """
        if self.run_dir:
            return Path(self.run_dir)
        return (
            Path(self.output_root)
            / now.strftime("%Y-%m-%d")
            / self.experiment_name
            / now.strftime("%H-%M-%S-%f")
        )


@dataclass(slots=True)
class SimulationConfig:
    data: SimulationDataConfig
    features: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    sequence: SequenceFeatureConfig = field(default_factory=SequenceFeatureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    predictor: PredictorConfig = field(default_factory=NaivePredictorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert dict.

        Returns:
            dict[str, Any]: Dictionary containing the computed values.
        """
        return {
            "data": asdict(self.data),
            "features": asdict(self.features),
            "sequence": asdict(self.sequence),
            "selection": asdict(self.selection),
            "predictor": asdict(self.predictor),
            "output": {
                "experiment_name": self.output.experiment_name,
                "output_root": self.output.output_root,
                "logging": asdict(self.output.logging),
                "run_dir": self.output.run_dir,
            },
        }


def simulation_config_from_omegaconf(cfg: DictConfig, *, run_dir: str | None = None) -> SimulationConfig:
    """Simulation config from omegaconf.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        run_dir (str | None): Directory for the run.

    Returns:
        SimulationConfig: Result of the function.
    """
    return SimulationConfig(
        data=SimulationDataConfig(
            parquet_path=_optional_str(cfg.data.get("parquet_path", None)),
            target_column=str(cfg.data.target_column),
            metadata_columns=[str(value) for value in cfg.data.metadata_columns],
            epoch_filter=[int(value) for value in cfg.data.get("epoch_filter", [])],
            target_model_min_threshold=_optional_float(cfg.data.get("target_model_min_threshold", None)),
            source=str(cfg.data.get("source", "prepared_dataset")),
            pt_path=_optional_str(cfg.data.get("pt_path", None)),
            pt_train_path=_optional_str(cfg.data.get("pt_train_path", None)),
            pt_val_path=_optional_str(cfg.data.get("pt_val_path", None)),
            target_parquet_path=_optional_str(cfg.data.get("target_parquet_path", None)),
            model_ids_from_pt_path=_optional_str(cfg.data.get("model_ids_from_pt_path", None)),
            model_ids_from_pt_val_path=_optional_str(cfg.data.get("model_ids_from_pt_val_path", None)),
            id_normalize_basename=bool(cfg.data.get("id_normalize_basename", True)),
            epoch_from_checkpoint_path=bool(cfg.data.get("epoch_from_checkpoint_path", False)),
        ),
        features=_feature_selection_config_from_mapping(OmegaConf.to_container(cfg.features, resolve=True)),  # type: ignore[arg-type]
        sequence=_sequence_feature_config_from_mapping(OmegaConf.to_container(cfg.sequence, resolve=True)),  # type: ignore[arg-type]
        selection=SelectionConfig(
            start_epoch=int(cfg.selection.start_epoch),
            interval=int(cfg.selection.interval),
            survival_rate=float(cfg.selection.survival_rate),
            min_population=int(cfg.selection.min_population),
            metric_goal=str(cfg.selection.metric_goal),
            eliminate_models_missing_decision_epoch=bool(cfg.selection.eliminate_models_missing_decision_epoch),
            top_k_preview=int(cfg.selection.top_k_preview),
        ),
        predictor=_predictor_config_from_omegaconf(cfg.predictor),
        output=OutputConfig(
            experiment_name=str(cfg.experiment_name),
            output_root=str(cfg.output_root),
            logging=LoggingConfig(
                level=str(cfg.logging.level),
                save_to_file=bool(cfg.logging.save_to_file),
                filename=str(cfg.logging.filename),
            ),
            run_dir=run_dir,
        ),
    )


def feature_selection_config_from_mapping(mapping: Mapping[str, Any]) -> FeatureSelectionConfig:
    """Feature selection config from mapping.

    Args:
        mapping (Mapping[str, Any]): Mapping to convert.

    Returns:
        FeatureSelectionConfig: Result of the function.
    """
    return _feature_selection_config_from_mapping(mapping)


def sequence_feature_config_from_mapping(mapping: Mapping[str, Any]) -> SequenceFeatureConfig:
    """Sequence feature config from mapping.

    Args:
        mapping (Mapping[str, Any]): Mapping to convert.

    Returns:
        SequenceFeatureConfig: Result of the function.
    """
    return _sequence_feature_config_from_mapping(mapping)


def _feature_selection_config_from_mapping(mapping: Mapping[str, Any]) -> FeatureSelectionConfig:
    """Feature selection config from mapping.

    Args:
        mapping (Mapping[str, Any]): Mapping to convert.

    Returns:
        FeatureSelectionConfig: Result of the function.
    """
    return FeatureSelectionConfig(
        include_model_metrics=bool(mapping.get("include_model_metrics", True)),
        include_layer_metrics=bool(mapping.get("include_layer_metrics", True)),
        metric_families=[str(value) for value in mapping.get("metric_families", [])],
        metrics=[str(value) for value in mapping.get("metrics", [])],
        scopes=[str(value) for value in mapping.get("scopes", [])],
        layer_types=[str(value) for value in mapping.get("layer_types", [])],
        include_feature_names=[str(value) for value in mapping.get("include_feature_names", [])],
        exclude_feature_names=[str(value) for value in mapping.get("exclude_feature_names", [])],
        include_feature_patterns=[str(value) for value in mapping.get("include_feature_patterns", [])],
        exclude_feature_patterns=[str(value) for value in mapping.get("exclude_feature_patterns", [])],
        max_missing_fraction=float(mapping.get("max_missing_fraction", 0.25)),
        drop_constant_features=bool(mapping.get("drop_constant_features", True)),
        missing_value_fill=str(mapping.get("missing_value_fill", "median")),
        sort_columns=bool(mapping.get("sort_columns", True)),
        log_feature_names=bool(mapping.get("log_feature_names", False)),
        max_console_logged_features=int(mapping.get("max_console_logged_features", 50)),
    )


def _sequence_feature_config_from_mapping(mapping: Mapping[str, Any]) -> SequenceFeatureConfig:
    """Sequence feature config from mapping.

    Args:
        mapping (Mapping[str, Any]): Mapping to convert.

    Returns:
        SequenceFeatureConfig: Result of the function.
    """
    return SequenceFeatureConfig(
        target_mode=str(mapping.get("target_mode", "horizon_value")),
        horizon=int(mapping.get("horizon", 5)),
        max_lag=int(mapping.get("max_lag", 3)),
        min_history=int(mapping.get("min_history", 1)),
        add_delta_features=bool(mapping.get("add_delta_features", True)),
        include_time_features=bool(mapping.get("include_time_features", True)),
        include_current_target=bool(mapping.get("include_current_target", False)),
        include_terminal_samples=bool(mapping.get("include_terminal_samples", False)),
        observed_metric_columns=[str(c) for c in mapping.get("observed_metric_columns", [])],
    )


def _predictor_config_from_omegaconf(cfg: DictConfig) -> PredictorConfig:
    """Predictor config from omegaconf.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        PredictorConfig: Result of the function.
    """
    predictor_name = str(cfg.name).lower()
    if predictor_name == "forecast_model":
        return ForecastModelPredictorConfig(
            model_bundle_path=_optional_str(cfg.get("model_bundle_path", None)),
            used_features_path=_optional_str(cfg.get("used_features_path", None)),
            use_bundle_feature_config=bool(cfg.get("use_bundle_feature_config", True)),
            use_bundle_sequence_config=bool(cfg.get("use_bundle_sequence_config", True)),
        )
    if predictor_name == "xgboost":
        return XGBoostPredictorConfig(
            params=_params_mapping_from_cfg(cfg.get("params", {})),
            min_train_rows=int(cfg.get("min_train_rows", 10)),
            min_train_models=int(cfg.get("min_train_models", 2)),
            fallback_strategy=str(cfg.get("fallback_strategy", "current_target")),
            include_observed_metrics=bool(cfg.get("include_observed_metrics", True)),
            observed_metric_columns=_observed_metric_columns_from_cfg(cfg),
        )
    if predictor_name == "online_lstm":
        return OnlineLSTMPredictorConfig(
            params=_params_mapping_from_cfg(cfg.get("params", {})),
            min_train_rows=int(cfg.get("min_train_rows", 10)),
            min_train_models=int(cfg.get("min_train_models", 2)),
            fallback_strategy=str(cfg.get("fallback_strategy", "current_target")),
            include_observed_metrics=bool(cfg.get("include_observed_metrics", True)),
            observed_metric_columns=_observed_metric_columns_from_cfg(cfg),
        )
    if predictor_name == "online_hierarchical_transformer":
        return OnlineHierarchicalTransformerPredictorConfig(
            params=_params_mapping_from_cfg(cfg.get("params", {})),
            min_train_rows=int(cfg.get("min_train_rows", 10)),
            min_train_models=int(cfg.get("min_train_models", 2)),
            fallback_strategy=str(cfg.get("fallback_strategy", "current_target")),
            include_observed_metrics=bool(cfg.get("include_observed_metrics", True)),
            observed_metric_columns=_observed_metric_columns_from_cfg(cfg),
        )
    if predictor_name == "naive":
        return NaivePredictorConfig(
            strategy=str(cfg.get("strategy", "current_target")),
        )
    if predictor_name == "online_lc_extrapolation":
        return OnlineLCExtrapolationPredictorConfig(
            min_points=int(cfg.get("min_points", 3)),
            fallback_strategy=str(cfg.get("fallback_strategy", "current_target")),
        )
    if predictor_name == "online_lcnet":
        return OnlineLCNetPredictorConfig(
            params=_params_mapping_from_cfg(cfg.get("params", {})),
            max_history_points=int(cfg.get("max_history_points", 8)),
            min_train_rows=int(cfg.get("min_train_rows", 10)),
            min_train_models=int(cfg.get("min_train_models", 2)),
            fallback_strategy=str(cfg.get("fallback_strategy", "current_target")),
        )
    raise ValueError(f"Unsupported predictor.name '{cfg.name}'.")


def _observed_metric_columns_from_cfg(cfg: DictConfig) -> list[str]:
    """Resolve observed metric column overrides from an OmegaConf mapping.

    Args:
        cfg (DictConfig): Predictor config mapping.

    Returns:
        list[str]: List of string values.
    """
    raw_metric_columns = cfg.get("observed_metric_columns", _default_observed_metric_columns())
    if isinstance(raw_metric_columns, (str, bytes)):
        return [str(raw_metric_columns)]
    return [str(value) for value in raw_metric_columns]


def _params_mapping_from_cfg(value: Any) -> dict[str, Any]:
    """Convert predictor params from OmegaConf containers to plain Python types.

    Args:
        value (Any): Raw params value.

    Returns:
        dict[str, Any]: Dictionary containing only plain Python container types.
    """
    if isinstance(value, DictConfig):
        resolved = OmegaConf.to_container(value, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("predictor.params must resolve to a mapping.")
        return dict(resolved)
    return dict(value)


def _optional_float(value: Any) -> float | None:
    """Optional float.

    Args:
        value (Any): Value to process.

    Returns:
        float | None: Result of the function.
    """
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    """Optional str.

    Args:
        value (Any): Value to process.

    Returns:
        str | None: Result of the function.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None
