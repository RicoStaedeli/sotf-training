from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class PreparedDatasetConfig:
    parquet_path: str | None = None
    pt_train_path: str | None = None
    pt_val_path: str | None = None
    target_column: str = "test_acc"
    metadata_columns: list[str] = field(default_factory=lambda: ["model_id", "path", "epoch", "hyperparameter_key"])
    epoch_filter: list[int] = field(default_factory=list)
    target_model_min_threshold: float | None = None
    split_csv_path: str | None = None

    def to_omegaconf(self):
        """To omegaconf.

        Returns:
            Any: Result of the function.
        """
        return OmegaConf.create(
            {
                "parquet_path": self.parquet_path,
                "pt_train_path": self.pt_train_path,
                "pt_val_path": self.pt_val_path,
                "target_column": self.target_column,
                "metadata_columns": list(self.metadata_columns),
                "epoch_filter": list(self.epoch_filter),
                "target_model_min_threshold": self.target_model_min_threshold,
                "split_csv_path": self.split_csv_path,
            }
        )


@dataclass(slots=True)
class FeatureSelectionConfig:
    include_model_metrics: bool = True
    include_layer_metrics: bool = True
    metric_families: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    layer_types: list[str] = field(default_factory=list)
    include_feature_names: list[str] = field(default_factory=list)
    exclude_feature_names: list[str] = field(default_factory=list)
    include_feature_patterns: list[str] = field(default_factory=list)
    exclude_feature_patterns: list[str] = field(default_factory=list)
    max_missing_fraction: float = 0.25
    drop_constant_features: bool = True
    missing_value_fill: str = "median"
    sort_columns: bool = True
    log_feature_names: bool = False
    max_console_logged_features: int = 50

    def to_omegaconf(self):
        """To omegaconf.

        Returns:
            Any: Result of the function.
        """
        return OmegaConf.create(asdict(self))


@dataclass(slots=True)
class SequenceFeatureConfig:
    target_mode: str = "final_value"
    horizon: int = 5
    max_lag: int = 3
    min_history: int = 1
    add_delta_features: bool = True
    include_time_features: bool = True
    include_current_target: bool = False
    include_terminal_samples: bool = False
    observed_metric_columns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainConfig:
    validation_size: float = 0.2
    random_seed: int = 42
    deterministic: bool = False
    disable_tf32: bool = False
    save_model: bool = True
    save_predictions: bool = True
    save_feature_matrix: bool = False


@dataclass(slots=True)
class ModelConfig:
    name: str = "xgboost"
    params: dict[str, Any] = field(
        default_factory=lambda: {
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
    )

    def to_model_config(self) -> dict[str, Any]:
        """To model config.

        Returns:
            dict[str, Any]: Result of the function.
        """
        return {
            "name": str(self.name),
            "params": dict(self.params),
        }


@dataclass(slots=True)
class OutputConfig:
    experiment_name: str = "xgboost_forecast"
    output_root: str = "metric-prediction/outputs_forecasting"
    logging_level: str = "INFO"
    save_log_file: bool = True
    log_filename: str | None = None
    run_dir: str | None = None

    def resolve_run_dir(self, *, now) -> Path:
        """Resolve run dir.

        Args:
            now (Any): Value used by the function.

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
class WandbConfig:
    project: str | None = None
    group: str | None = None
    api_key_file: str | None = None


@dataclass(slots=True)
class ForecastConfig:
    data: PreparedDatasetConfig
    features: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    sequence: SequenceFeatureConfig = field(default_factory=SequenceFeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            dict[str, Any]: Result of the function.
        """
        return asdict(self)


def forecast_config_from_omegaconf(cfg: DictConfig) -> ForecastConfig:
    """Forecast config from omegaconf.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        ForecastConfig: Result of the function.
    """
    raw_target_model_min_threshold = cfg.data.get("target_model_min_threshold", None)
    return ForecastConfig(
        data=PreparedDatasetConfig(
            parquet_path=_optional_cfg_str(cfg.data, "parquet_path"),
            pt_train_path=_optional_cfg_str(cfg.data, "pt_train_path"),
            pt_val_path=_optional_cfg_str(cfg.data, "pt_val_path"),
            target_column=str(cfg.data.target_column),
            metadata_columns=[str(value) for value in cfg.data.metadata_columns],
            epoch_filter=[int(value) for value in cfg.data.get("epoch_filter", [])],
            target_model_min_threshold=(
                None
                if raw_target_model_min_threshold is None
                else float(raw_target_model_min_threshold)
            ),
            split_csv_path=_optional_cfg_str(cfg.data, "split_csv_path"),
        ),
        features=FeatureSelectionConfig(
            include_model_metrics=bool(cfg.features.include_model_metrics),
            include_layer_metrics=bool(cfg.features.include_layer_metrics),
            metric_families=[str(value) for value in cfg.features.metric_families],
            metrics=[str(value) for value in cfg.features.metrics],
            scopes=[str(value) for value in cfg.features.scopes],
            layer_types=[str(value) for value in cfg.features.layer_types],
            include_feature_names=[str(value) for value in cfg.features.include_feature_names],
            exclude_feature_names=[str(value) for value in cfg.features.exclude_feature_names],
            include_feature_patterns=[str(value) for value in cfg.features.include_feature_patterns],
            exclude_feature_patterns=[str(value) for value in cfg.features.exclude_feature_patterns],
            max_missing_fraction=float(cfg.features.max_missing_fraction),
            drop_constant_features=bool(cfg.features.drop_constant_features),
            missing_value_fill=str(cfg.features.missing_value_fill),
            sort_columns=bool(cfg.features.sort_columns),
            log_feature_names=bool(cfg.features.log_feature_names),
            max_console_logged_features=int(cfg.features.max_console_logged_features),
        ),
        sequence=SequenceFeatureConfig(
            target_mode=str(cfg.sequence.target_mode),
            horizon=int(cfg.sequence.horizon),
            max_lag=int(cfg.sequence.max_lag),
            min_history=int(cfg.sequence.min_history),
            add_delta_features=bool(cfg.sequence.add_delta_features),
            include_time_features=bool(cfg.sequence.get("include_time_features", True)),
            include_current_target=bool(cfg.sequence.include_current_target),
            include_terminal_samples=bool(cfg.sequence.include_terminal_samples),
            observed_metric_columns=[str(c) for c in cfg.sequence.get("observed_metric_columns", [])],
        ),
        train=TrainConfig(
            validation_size=float(cfg.train.validation_size),
            random_seed=int(cfg.train.random_seed),
            deterministic=bool(cfg.train.get("deterministic", False)),
            disable_tf32=bool(cfg.train.get("disable_tf32", False)),
            save_model=bool(cfg.train.save_model),
            save_predictions=bool(cfg.train.save_predictions),
            save_feature_matrix=bool(cfg.train.save_feature_matrix),
        ),
        model=ModelConfig(
            name=str(cfg.model.name),
            params=OmegaConf.to_container(cfg.model.params, resolve=True),
        ),
        output=OutputConfig(
            experiment_name=str(cfg.experiment_name),
            output_root=str(cfg.train.artifacts_root),
            logging_level=str(cfg.logging.level),
            save_log_file=bool(cfg.logging.save_to_file),
            log_filename=str(cfg.logging.filename),
            run_dir=None,
        ),
        wandb=WandbConfig(
            project=_optional_cfg_str(cfg, "wandb_project"),
            group=_optional_cfg_str(cfg, "wandb_group"),
            api_key_file=_optional_cfg_str(cfg, "wandb_api_key_file"),
        ),
    )


def _optional_cfg_str(cfg: DictConfig, key: str) -> str | None:
    """Optional cfg str.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        key (str): str value used by the function.

    Returns:
        str | None: Result of the function.
    """
    value = cfg.get(key, None)
    if value is None:
        return None
    value = str(value).strip()
    return value or None
