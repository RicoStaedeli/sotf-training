from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.lcnet import LCNetRegressor, build_lcnet_features
from baselines.learning_curve_extrapolation import infer_target_bounds
from data.dataset_splitter import split_dataset
from data.weightstatistic_dataset_loader import load_weightstatistic_dataset, load_weightstatistic_split_datasets
from forecasting.config import (
    FeatureSelectionConfig,
    PreparedDatasetConfig,
    SequenceFeatureConfig,
    TrainConfig,
)
from forecasting.dataset import build_forecast_dataset, build_forecast_target_dataset
from utils.artifacts import write_frame, write_json
from utils.metrics import compute_regression_metrics


DEFAULT_OUTPUT_ROOT = Path("metric-prediction/outputs/baselines/lcnet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an LCNet-style Bayesian neural-network learning-curve baseline.",
    )
    parser.add_argument("--batch-config", type=Path, default=None)
    parser.add_argument(
        "-cn",
        "--config-name",
        type=str,
        default=None,
        help="Baseline YAML config name or path, resolved relative to metric-prediction/src/baselines/configs.",
    )
    parser.add_argument("--train-pt", type=Path, default=None)
    parser.add_argument("--val-pt", type=Path, default=None)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--target-mode", choices=["final_value", "horizon_value"], default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--min-history", type=int, default=None)
    parser.add_argument("--epoch-filter", type=int, nargs="*", default=None)
    parser.add_argument("--target-model-min-threshold", type=float, default=None)
    parser.add_argument("--split-csv-path", type=Path, default=None)
    parser.add_argument("--validation-size", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-history-points", type=int, default=None)
    parser.add_argument("--hidden-layer-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=_parse_batch_size, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfgs = resolve_baseline_configs(args)
    if args.dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create({"runs": cfgs})))
        return 0

    results = []
    for index, cfg in enumerate(cfgs, start=1):
        print(f"[{index}/{len(cfgs)}] {cfg['experiment_name']}")
        result = run_lcnet_baseline(cfg)
        result_row = {
            "experiment_name": str(cfg["experiment_name"]),
            "batch_name": str(cfg.get("batch_name", "")),
            "target_mode": str(cfg["target_mode"]),
            "horizon": int(cfg.get("horizon", 5)),
            "r2": float(result["metrics"]["r2"]),
            "rmse": float(result["metrics"]["rmse"]),
            "mae": float(result["metrics"]["mae"]),
            "artifacts_dir": str(result["artifacts_dir"]),
        }
        results.append(result_row)
        for key in ("r2", "rmse", "mae"):
            print(f"  {key}: {result['metrics'][key]:.6f}")
        print(f"  artifacts: {result['artifacts_dir']}")

    if results:
        summary_path = write_run_summary(results, output_root=Path(str(cfgs[0].get("output_root", DEFAULT_OUTPUT_ROOT))))
        print(f"Summary: {summary_path}")
    return 0


def run_lcnet_baseline(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = PreparedDatasetConfig(
        parquet_path=_optional_str(cfg.get("parquet_path")),
        pt_train_path=_optional_str(cfg.get("pt_train_path")),
        pt_val_path=_optional_str(cfg.get("pt_val_path")),
        target_column=str(cfg["target_column"]),
        metadata_columns=list(cfg.get("metadata_columns", ["model_id", "path", "epoch", "hyperparameter_key"])),
        epoch_filter=[int(value) for value in cfg.get("epoch_filter", [])],
        target_model_min_threshold=cfg.get("target_model_min_threshold"),
        split_csv_path=_optional_str(cfg.get("split_csv_path")),
    )
    feature_cfg = FeatureSelectionConfig(
        include_model_metrics=False,
        include_layer_metrics=False,
        max_missing_fraction=1.0,
        drop_constant_features=False,
    )
    sequence_cfg = SequenceFeatureConfig(
        target_mode=str(cfg["target_mode"]),
        horizon=int(cfg.get("horizon", 5)),
        max_lag=0,
        min_history=int(cfg.get("min_history", 1)),
        add_delta_features=False,
        include_time_features=False,
        include_current_target=False,
        include_terminal_samples=False,
    )
    train_cfg = TrainConfig(
        validation_size=float(cfg.get("validation_size", 0.2)),
        random_seed=int(cfg.get("random_seed", 42)),
        save_predictions=bool(cfg.get("save_predictions", True)),
        save_feature_matrix=False,
        save_model=False,
    )

    is_presplit = bool(data_cfg.pt_train_path and data_cfg.pt_val_path)
    if is_presplit:
        train_prepared, val_prepared = load_weightstatistic_split_datasets(
            data_cfg.to_omegaconf(),
            feature_cfg.to_omegaconf(),
        )
        train_targets = build_forecast_target_dataset(train_prepared, sequence_cfg)
        val_targets = build_forecast_target_dataset(val_prepared, sequence_cfg)
        metadata_train = train_targets.metadata
        metadata_val = val_targets.metadata
        y_train = train_targets.target
        y_val = val_targets.target
        full_metadata_train = train_prepared.metadata.copy()
        full_metadata_val = val_prepared.metadata.copy()
        train_target_values = train_prepared.target
        val_target_values = val_prepared.target
        all_target_values = pd.concat([train_prepared.target, val_prepared.target], axis=0)
    else:
        prepared = load_weightstatistic_dataset(data_cfg.to_omegaconf(), feature_cfg.to_omegaconf())
        target_dataset = build_forecast_dataset(prepared, sequence_cfg)
        split = split_dataset(
            target_dataset,
            validation_size=train_cfg.validation_size,
            random_seed=train_cfg.random_seed,
            split_csv_path=data_cfg.split_csv_path,
        )
        metadata_train = split.metadata_train
        metadata_val = split.metadata_val
        y_train = split.y_train
        y_val = split.y_val
        full_metadata_train = prepared.metadata.copy()
        full_metadata_val = prepared.metadata.copy()
        train_target_values = prepared.target
        val_target_values = prepared.target
        all_target_values = prepared.target

    target_column = data_cfg.target_column
    full_metadata_train = _attach_target(full_metadata_train, train_target_values, target_column=target_column)
    full_metadata_val = _attach_target(full_metadata_val, val_target_values, target_column=target_column)
    bounds = infer_target_bounds(
        pd.to_numeric(all_target_values, errors="coerce").to_numpy(dtype="float64"),
        target_column=target_column,
    )

    max_history_points = int(cfg.get("max_history_points", 8))
    X_train = build_lcnet_features(
        full_metadata=full_metadata_train,
        forecast_metadata=metadata_train,
        target_column=target_column,
        target_mode=sequence_cfg.target_mode,
        horizon=sequence_cfg.horizon,
        max_history_points=max_history_points,
    )
    X_val = build_lcnet_features(
        full_metadata=full_metadata_val,
        forecast_metadata=metadata_val,
        target_column=target_column,
        target_mode=sequence_cfg.target_mode,
        horizon=sequence_cfg.horizon,
        max_history_points=max_history_points,
    )

    model_params = _model_params(cfg)
    model = LCNetRegressor(**model_params)
    model.fit(X_train, y_train)
    prediction_distribution = model.predict_distribution(X_val)
    predictions = np.clip(prediction_distribution.mean, bounds[0], bounds[1])
    prediction_std = prediction_distribution.std
    metrics = compute_regression_metrics(y_val.to_numpy(dtype="float64"), predictions)

    artifacts_dir = resolve_artifacts_dir(cfg)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **metrics,
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_features": int(X_train.shape[1]),
        "n_train_models": int(metadata_train["model_id"].astype(str).nunique()),
        "n_validation_models": int(metadata_val["model_id"].astype(str).nunique()),
        "target_mode": sequence_cfg.target_mode,
        "target_column": target_column,
        "horizon": int(sequence_cfg.horizon),
        "max_lag": 0,
        "min_history": int(sequence_cfg.min_history),
        "presplit": bool(is_presplit),
        "baseline": "lcnet_bayesian_nn",
        "target_lower_bound": float(bounds[0]),
        "target_upper_bound": float(bounds[1]),
        "max_history_points": max_history_points,
        "mc_samples": int(model_params["mc_samples"]),
        "training_epochs": int(len(model.training_history_)),
    }
    resolved_config = {
        "experiment_name": str(cfg["experiment_name"]),
        "model": {
            "name": "lcnet_bayesian_nn",
            "params": {"max_history_points": max_history_points, **model_params},
        },
        "data": {
            "parquet_path": data_cfg.parquet_path,
            "pt_train_path": data_cfg.pt_train_path,
            "pt_val_path": data_cfg.pt_val_path,
            "target_column": target_column,
            "metadata_columns": data_cfg.metadata_columns,
            "epoch_filter": data_cfg.epoch_filter,
            "target_model_min_threshold": data_cfg.target_model_min_threshold,
            "split_csv_path": data_cfg.split_csv_path,
        },
        "sequence": {
            "target_mode": sequence_cfg.target_mode,
            "horizon": int(sequence_cfg.horizon),
            "max_lag": 0,
            "min_history": int(sequence_cfg.min_history),
            "include_terminal_samples": False,
        },
        "train": {
            "validation_size": train_cfg.validation_size,
            "random_seed": train_cfg.random_seed,
            "save_predictions": train_cfg.save_predictions,
        },
        "output": {
            "experiment_name": str(cfg["experiment_name"]),
            "output_root": str(cfg["output_root"]),
            "run_dir": str(artifacts_dir),
        },
    }

    write_json(artifacts_dir / "metrics.json", metrics_payload)
    write_json(artifacts_dir / "resolved_config.json", resolved_config)
    write_json(artifacts_dir / "training_history.json", model.training_history_)
    if train_cfg.save_predictions:
        prediction_frame = metadata_val.copy()
        prediction_frame["target"] = y_val.to_numpy(dtype="float64")
        prediction_frame["prediction"] = predictions
        prediction_frame["prediction_std"] = prediction_std
        prediction_frame["residual"] = prediction_frame["target"] - prediction_frame["prediction"]
        write_frame(artifacts_dir / "validation_predictions.parquet", prediction_frame.reset_index())

    return {"metrics": metrics, "artifacts_dir": str(artifacts_dir)}


def resolve_baseline_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config_name is not None:
        config_path = resolve_config_name(args.config_name)
        cfgs = _configs_from_multi_run_config(config_path)
        return [_apply_cli_overrides(cfg, args) for cfg in cfgs]
    return [resolve_baseline_config(args)]


def resolve_baseline_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "target_column": "test_acc",
        "target_mode": "final_value",
        "horizon": 5,
        "min_history": 1,
        "metadata_columns": ["model_id", "path", "epoch", "hyperparameter_key"],
        "epoch_filter": [],
        "target_model_min_threshold": None,
        "split_csv_path": None,
        "validation_size": 0.2,
        "random_seed": 42,
        "max_history_points": 8,
        "save_predictions": True,
        "output_root": str(args.output_root),
    }
    config.update(_default_model_params())
    if args.batch_config is not None:
        config.update(_config_from_batch(args.batch_config))
    return _validate_resolved_config(_apply_cli_overrides(config, args))


def resolve_artifacts_dir(cfg: Mapping[str, Any]) -> Path:
    if cfg.get("run_dir"):
        return Path(str(cfg["run_dir"]))
    output_root = Path(str(cfg.get("output_root", DEFAULT_OUTPUT_ROOT)))
    target_mode = str(cfg.get("target_mode", "unknown"))
    batch_name = str(cfg.get("batch_name", "single_dataset"))
    experiment_name = str(cfg["experiment_name"])
    return output_root / target_mode / _path_safe(batch_name) / "results" / _path_safe(experiment_name) / "artifacts"


def resolve_config_name(config_name: str) -> Path:
    path = Path(config_name).expanduser()
    candidates = [path]
    if path.suffix not in {".yaml", ".yml"}:
        candidates.append(path.with_suffix(".yaml"))
    config_root = Path(__file__).resolve().parent / "configs"
    candidates.extend(config_root / candidate for candidate in list(candidates))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve baseline config '{config_name}'. Tried: {tried}")


def write_run_summary(results: list[dict[str, Any]], *, output_root: Path) -> Path:
    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "lcnet.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    return summary_path


def _config_from_batch(path: Path) -> dict[str, Any]:
    raw = OmegaConf.load(path)
    common = _flatten_overrides(raw.get("common_overrides", {}))
    batch_name = str(raw.get("batch_name", path.stem))
    return {
        "batch_name": batch_name,
        "pt_train_path": common.get("data.pt_train_path"),
        "pt_val_path": common.get("data.pt_val_path"),
        "parquet_path": common.get("data.parquet_path"),
        "target_column": common.get("data.target_column", "test_acc"),
        "target_mode": common.get("sequence.target_mode", "final_value"),
        "horizon": int(common.get("sequence.horizon", 5)),
        "min_history": int(common.get("sequence.min_history", 1)),
        "metadata_columns": list(common.get("data.metadata_columns", ["model_id", "path", "epoch", "hyperparameter_key"])),
        "epoch_filter": list(common.get("data.epoch_filter", [])),
        "target_model_min_threshold": common.get("data.target_model_min_threshold"),
        "split_csv_path": common.get("data.split_csv_path"),
        "validation_size": float(common.get("train.validation_size", 0.2)),
        "random_seed": int(common.get("train.random_seed", 42)),
        "experiment_name": f"{batch_name}_lcnet_{common.get('sequence.target_mode', 'final_value')}",
    }


def _configs_from_multi_run_config(path: Path) -> list[dict[str, Any]]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Baseline config must be a mapping: {path}")

    defaults = dict(raw.get("defaults", {}))
    output_root = raw.get("output_root", defaults.get("output_root", DEFAULT_OUTPUT_ROOT))
    target_modes = raw.get("target_modes", None)
    if target_modes is None:
        target_modes = [
            {
                "target_mode": defaults.get("target_mode", "final_value"),
                "horizon": defaults.get("horizon", 5),
                "min_history": defaults.get("min_history", 1),
            }
        ]
    datasets = raw.get("datasets", [])
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not datasets:
        raise ValueError(f"Baseline config must define a non-empty 'datasets' list: {path}")

    configs: list[dict[str, Any]] = []
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise ValueError("Each dataset entry must be a mapping.")
        for mode_cfg in target_modes:
            if not isinstance(mode_cfg, Mapping):
                raise ValueError("Each target_modes entry must be a mapping.")
            merged = {
                "target_column": "test_acc",
                "target_mode": "final_value",
                "horizon": 5,
                "min_history": 1,
                "metadata_columns": ["model_id", "path", "epoch", "hyperparameter_key"],
                "epoch_filter": [],
                "target_model_min_threshold": None,
                "split_csv_path": None,
                "validation_size": 0.2,
                "random_seed": 42,
                "max_history_points": 8,
                "save_predictions": True,
                "output_root": str(output_root),
                **_default_model_params(),
                **defaults,
                **dict(dataset),
                **dict(mode_cfg),
            }
            dataset_name = str(merged.get("name", merged.get("batch_name", "dataset")))
            merged["batch_name"] = str(merged.get("batch_name", dataset_name))
            merged["experiment_name"] = str(
                merged.get("experiment_name", f"{dataset_name}_lcnet_{merged['target_mode']}")
            )
            if "pt_train_path" not in merged and "dataset_path" in merged:
                dataset_path = Path(str(merged["dataset_path"]))
                merged["pt_train_path"] = str(dataset_path / "weight_statistics_train.pt")
                merged["pt_val_path"] = str(dataset_path / "weight_statistics_val.pt")
            configs.append(_validate_resolved_config(merged))
    return configs


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    cli_overrides = {
        "pt_train_path": args.train_pt,
        "pt_val_path": args.val_pt,
        "parquet_path": args.parquet,
        "target_column": args.target_column,
        "target_mode": args.target_mode,
        "horizon": args.horizon,
        "min_history": args.min_history,
        "epoch_filter": args.epoch_filter,
        "target_model_min_threshold": args.target_model_min_threshold,
        "split_csv_path": args.split_csv_path,
        "validation_size": args.validation_size,
        "random_seed": args.random_seed,
        "experiment_name": args.experiment_name,
        "run_dir": args.run_dir,
        "output_root": args.output_root if args.output_root != DEFAULT_OUTPUT_ROOT else None,
        "max_history_points": args.max_history_points,
        "hidden_layer_sizes": args.hidden_layer_sizes,
        "dropout": args.dropout,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "mc_samples": args.mc_samples,
        "device": args.device,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    return _validate_resolved_config(config)


def _validate_resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    if "experiment_name" not in config or not config["experiment_name"]:
        dataset_name = str(config.get("batch_name", "dataset"))
        config["experiment_name"] = f"{dataset_name}_lcnet_{config['target_mode']}"
    if not config.get("pt_train_path") and not config.get("parquet_path"):
        raise ValueError("Provide either dataset paths, train/validation PT paths, or a parquet path.")
    if bool(config.get("pt_train_path")) != bool(config.get("pt_val_path")):
        raise ValueError("Both train and validation PT paths are required for pre-split mode.")
    return config


def _model_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = _default_model_params()
    for key in list(params):
        if key in cfg:
            params[key] = cfg[key]
    params["random_state"] = int(cfg.get("random_seed", params["random_state"]))
    return params


def _default_model_params() -> dict[str, Any]:
    return {
        "hidden_layer_sizes": [128, 128],
        "activation": "relu",
        "dropout": 0.1,
        "use_layer_norm": False,
        "batch_size": 512,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "max_epochs": 120,
        "patience": 15,
        "validation_fraction": 0.15,
        "early_stopping": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 1.0e-5,
        "gradient_clip_norm": 1.0,
        "scale_features": True,
        "scale_target": True,
        "random_state": 42,
        "device": "auto",
        "loss": "smooth_l1",
        "mc_samples": 30,
        "deterministic": False,
        "disable_tf32": False,
    }


def _attach_target(metadata: pd.DataFrame, target: pd.Series, *, target_column: str) -> pd.DataFrame:
    out = metadata.copy()
    if target_column not in out.columns:
        out[target_column] = target.astype("float64")
    return out


def _flatten_overrides(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        parsed = OmegaConf.from_dotlist([str(item) for item in raw])
        return _dot_mapping(OmegaConf.to_container(parsed, resolve=True))
    raise ValueError(f"Unsupported common_overrides format: {type(raw).__name__}")


def _dot_mapping(mapping: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_dot_mapping(value, prefix=dotted))
        else:
            out[dotted] = value
    return out


def _path_safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_batch_size(value: str) -> int | str:
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    return int(text)


if __name__ == "__main__":
    raise SystemExit(main())
