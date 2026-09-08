from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from utils.config_utils import (
    dataset_id_from_identity_path,
    dataset_slug_from_path,
    register_omegaconf_resolvers,
    resolve_dataset_identity_path,
    sanitize_path_component,
)

from models.predictors import load_forecast_model_bundle
from simulation.runner import run_simulation
from utils.settings import simulation_config_from_omegaconf

register_omegaconf_resolvers()

_DEFAULT_OUTPUT_ROOT = Path("sotf-selection") / "outputs"
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@dataclass(frozen=True)
class ForecastSourceConfig:
    results_root: Path
    run_name_template: str = "ds{dataset_id}_forecast_horizon_{horizon}_all_families"


@dataclass(frozen=True)
class PredictorBatchConfig:
    name: str
    strategy: str | None = None
    use_bundle_feature_config: bool = True
    use_bundle_sequence_config: bool = True
    forecast_source: ForecastSourceConfig | None = None


@dataclass(frozen=True)
class ScheduledRun:
    experiment_name: str
    overrides: tuple[str, ...]
    predictor: PredictorBatchConfig | None = None


@dataclass(frozen=True)
class SOTFBatchConfig:
    batch_name: str
    output_root: Path
    common_overrides: tuple[str, ...]
    predictor: PredictorBatchConfig
    runs: tuple[ScheduledRun, ...]
    dry_run: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        argparse.ArgumentParser: Computed parser.
    """
    parser = argparse.ArgumentParser(
        prog="sotf-select-batch",
        description="Run a SOTF batch experiment from a YAML config.",
    )
    parser.add_argument("batch_config", type=Path, help="Path to the SOTF batch YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved runs without executing them.")
    return parser


def main() -> None:
    """Run the function.

    Returns:
        None: None.
    """
    args = build_parser().parse_args()
    run_batch(args.batch_config, dry_run=bool(args.dry_run))


def run_batch(batch_config_path: str | Path, *, dry_run: bool) -> None:
    """Run batch.

    Args:
        batch_config_path (str | Path): Path to the batch config.
        dry_run (bool): Whether to print commands without executing them.

    Returns:
        None: None.
    """
    cfg = load_batch_config(batch_config_path)
    cfg = SOTFBatchConfig(
        batch_name=cfg.batch_name,
        output_root=cfg.output_root,
        common_overrides=cfg.common_overrides,
        predictor=cfg.predictor,
        runs=cfg.runs,
        dry_run=bool(dry_run),
    )
    run_batch_config(cfg, config_source_path=Path(batch_config_path))


def run_batch_config(cfg: SOTFBatchConfig, *, config_source_path: Path | None = None) -> None:
    """Run a parsed SOTF batch/experiment plan.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.
        config_source_path (Path | None): Path to the config source.

    Returns:
        None: None.
    """
    validate_batch_config(cfg)
    if not cfg.dry_run:
        copied_config_path = copy_batch_config(config_source_path, cfg)
        print(f"Saved batch config to {copied_config_path}")

    failures: list[tuple[str, int]] = []
    for index, run in enumerate(cfg.runs, start=1):
        run_cfg = build_run_config(cfg, run=run)
        run_dir = build_run_dir(cfg, run_cfg=run_cfg, now=datetime.now())
        print(f"[{index}/{len(cfg.runs)}] {run.experiment_name}")
        print(f"  run_dir={run_dir}")
        if cfg.dry_run:
            print("  overrides:")
            for override in build_run_overrides(cfg, run=run):
                print(f"    {override}")
            continue

        try:
            _prepare_run_dir(run_dir)
            simulation_cfg = simulation_config_from_omegaconf(run_cfg, run_dir=str(run_dir))
            result = run_simulation(simulation_cfg)
            _print_simulation_summary(result)
        except Exception as exc:  # noqa: BLE001 - keep batch execution moving across independent runs.
            failures.append((run.experiment_name, 1))
            print(f"  Failed {run.experiment_name}: {exc}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if failures:
        summary = ", ".join(f"{name} ({code})" for name, code in failures)
        raise SystemExit(f"Batch execution failed: {summary}")


def _print_simulation_summary(result: dict[str, Any]) -> None:
    """Print a compact summary for a completed simulation run.

    Args:
        result (dict[str, Any]): Result returned by the simulation runner.

    Returns:
        None: None.
    """
    summary = result["summary"]
    print("  Simulation summary:")
    print(f"    predictor: {summary['predictor']}")
    print(f"    rounds_executed: {summary['rounds_executed']}")
    print(f"    initial_population: {summary['initial_population']}")
    print(f"    final_population: {summary['final_population']}")
    print(f"    retained_oracle_best: {summary['retained_oracle_best']}")
    if summary["oracle_top_k_recall"] is not None:
        print(f"    oracle_top_k_recall: {summary['oracle_top_k_recall']:.6f}")
    if summary["mean_survivor_rank_gap"] is not None:
        print(f"    mean_survivor_rank_gap: {summary['mean_survivor_rank_gap']:.6f}")
    if summary["best_survivor_model_id"] is not None:
        print(f"    best_survivor_model_id: {summary['best_survivor_model_id']}")
        print(f"    best_survivor_final_target: {summary['best_survivor_final_target']:.6f}")
    print(f"  Artifacts: {result['artifacts_dir']}")


def load_batch_config(path: str | Path) -> SOTFBatchConfig:
    """Load batch config.

    Args:
        path (str | Path): Path to the input resource.

    Returns:
        SOTFBatchConfig: Computed batch config.
    """
    cfg = OmegaConf.load(Path(path))
    if isinstance(cfg, ListConfig):
        raise ValueError("Batch config must be a YAML mapping.")
    return parse_batch_config(cfg)


def parse_batch_config(cfg: DictConfig | dict[str, Any]) -> SOTFBatchConfig:
    """Parse either a legacy batch YAML or a Hydra-packaged experiment plan.

    Args:
        cfg (DictConfig | dict[str, Any]): Configuration object used by the function.

    Returns:
        SOTFBatchConfig: Computed batch config.
    """
    raw_cfg = _extract_plan_mapping(OmegaConf.to_container(cfg, resolve=True))
    if not isinstance(raw_cfg, dict):
        raise ValueError("Batch config must be a YAML mapping.")

    batch_name = str(raw_cfg.get("batch_name", "")).strip()
    if not batch_name:
        raise ValueError("Batch config must define a non-empty 'batch_name'.")

    common_overrides = _coerce_override_list(raw_cfg.get("common_overrides", {}), field_name="common_overrides")
    output_root = Path(str(raw_cfg.get("output_root", _DEFAULT_OUTPUT_ROOT)))
    predictor = _parse_predictor(raw_cfg.get("predictor"))

    raw_runs = raw_cfg.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Batch config must define a non-empty 'runs' list.")

    runs = tuple(_parse_run(run_cfg, common_overrides=common_overrides) for run_cfg in raw_runs)
    return SOTFBatchConfig(
        batch_name=batch_name,
        output_root=output_root,
        common_overrides=common_overrides,
        predictor=predictor,
        runs=runs,
        dry_run=bool(raw_cfg.get("dry_run", False)),
    )


def _extract_plan_mapping(raw_cfg: Any) -> Any:
    """Extract plan mapping.

    Args:
        raw_cfg (Any): Configuration for raw.

    Returns:
        Any: Result of the function.
    """
    if not isinstance(raw_cfg, dict):
        return raw_cfg
    if "runs" in raw_cfg:
        return raw_cfg
    experiments_cfg = raw_cfg.get("experiments")
    if isinstance(experiments_cfg, dict) and "runs" in experiments_cfg:
        return experiments_cfg
    return raw_cfg


def _parse_predictor(raw_value: Any) -> PredictorBatchConfig:
    """Parse predictor.

    Args:
        raw_value (Any): Any value used by the function.

    Returns:
        PredictorBatchConfig: Computed predictor.
    """
    if not isinstance(raw_value, dict):
        raise ValueError("Batch config must define a 'predictor' mapping.")

    name = str(raw_value.get("name", "")).strip().lower()
    if name not in {
        "forecast_model",
        "naive",
        "xgboost",
        "online_lstm",
        "online_hierarchical_transformer",
        "online_lc_extrapolation",
        "online_lcnet",
    }:
        raise ValueError(
            "predictor.name must be one of 'forecast_model', 'naive', 'xgboost', "
            "'online_lstm', 'online_hierarchical_transformer', 'online_lc_extrapolation', "
            "or 'online_lcnet'."
        )

    if name == "naive":
        strategy = str(raw_value.get("strategy", "current_target")).strip()
        return PredictorBatchConfig(name=name, strategy=strategy)
    if name in {"xgboost", "online_lstm", "online_hierarchical_transformer", "online_lc_extrapolation", "online_lcnet"}:
        return PredictorBatchConfig(name=name)

    source_cfg = raw_value.get("forecast_source")
    forecast_source = None
    if source_cfg is not None:
        if not isinstance(source_cfg, dict):
            raise ValueError("predictor.forecast_source must be a mapping when provided.")
        results_root = Path(str(source_cfg.get("results_root", "")).strip())
        if not str(results_root):
            raise ValueError("predictor.forecast_source.results_root must be set when forecast_source is provided.")
        run_name_template = str(source_cfg.get("run_name_template", ForecastSourceConfig.run_name_template)).strip()
        if not run_name_template:
            raise ValueError("predictor.forecast_source.run_name_template must be non-empty.")
        forecast_source = ForecastSourceConfig(
            results_root=results_root,
            run_name_template=run_name_template,
        )

    return PredictorBatchConfig(
        name=name,
        use_bundle_feature_config=bool(raw_value.get("use_bundle_feature_config", True)),
        use_bundle_sequence_config=bool(raw_value.get("use_bundle_sequence_config", True)),
        forecast_source=forecast_source,
    )


def _parse_run(raw_run: Any, common_overrides: tuple[str, ...] = ()) -> ScheduledRun:
    """Parse run.

    Args:
        raw_run (Any): Any value used by the function.
        common_overrides (tuple[str, ...]): Common overrides from the batch config, used to
            satisfy the dataset identity requirement when not set per-run.

    Returns:
        ScheduledRun: Computed run.
    """
    if not isinstance(raw_run, dict):
        raise ValueError("Each batch run must be a mapping.")

    experiment_name = str(raw_run.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ValueError("Each batch run must define a non-empty 'experiment_name'.")

    overrides = _coerce_override_list(raw_run.get("overrides", []), field_name=f"runs[{experiment_name}].overrides")
    if _dataset_identity_path_from_overrides((*common_overrides, *overrides)) is None:
        raise ValueError(
            f"Run '{experiment_name}' must define a dataset identity via data.pt_path, "
            "data.parquet_path, or data.target_parquet_path (in run overrides or common_overrides)."
        )
    predictor = None
    raw_predictor = raw_run.get("predictor")
    if raw_predictor is not None:
        predictor = _parse_predictor(raw_predictor)
    return ScheduledRun(experiment_name=experiment_name, overrides=overrides, predictor=predictor)


def build_run_config(cfg: SOTFBatchConfig, *, run: ScheduledRun) -> DictConfig:
    """Compose the simulation config for one scheduled run.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.
        run (ScheduledRun): Run metadata used by the function.

    Returns:
        DictConfig: Composed Hydra configuration for the run.
    """
    return _compose_sotf_config(build_run_overrides(cfg, run=run))


def build_run_overrides(cfg: SOTFBatchConfig, *, run: ScheduledRun) -> tuple[str, ...]:
    """Build Hydra overrides for one scheduled run.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.
        run (ScheduledRun): Run metadata used by the function.

    Returns:
        tuple[str, ...]: Hydra override strings.
    """
    predictor = run.predictor or cfg.predictor
    combined_overrides = (*cfg.common_overrides, *run.overrides)
    if any(override.startswith("predictor=") for override in combined_overrides):
        raise ValueError("Do not override predictor in common_overrides or run overrides; use a predictor block.")
    if any(override.startswith("hydra.run.dir=") for override in combined_overrides):
        raise ValueError("Do not override hydra.run.dir in batch configs; it is controlled by the batch runner.")
    if any(override.startswith("experiment_name=") for override in combined_overrides):
        raise ValueError("Do not override experiment_name in overrides; use the run-level experiment_name field.")

    dataset_path = _dataset_identity_path_from_overrides(combined_overrides)
    if dataset_path is None:
        raise ValueError("Run overrides do not define a dataset identity path.")
    overrides = list(combined_overrides)
    overrides.append(_format_mapping_override("predictor", predictor.name))

    if predictor.name == "naive":
        overrides.append(_format_mapping_override("predictor.strategy", predictor.strategy))
    elif predictor.name == "forecast_model":
        explicit_model_bundle_path = _extract_optional_override(combined_overrides, "predictor.model_bundle_path")
        explicit_used_features_path = _extract_optional_override(combined_overrides, "predictor.used_features_path")
        model_bundle_path, used_features_path = resolve_forecast_artifacts(
            cfg=cfg,
            predictor=predictor,
            combined_overrides=combined_overrides,
            dataset_path=dataset_path,
        )
        if explicit_model_bundle_path is None:
            overrides.append(_format_mapping_override("predictor.model_bundle_path", str(model_bundle_path)))
        if explicit_used_features_path is None and used_features_path is not None:
            overrides.append(_format_mapping_override("predictor.used_features_path", str(used_features_path)))
        overrides.extend(
            [
                _format_mapping_override("predictor.use_bundle_feature_config", predictor.use_bundle_feature_config),
                _format_mapping_override("predictor.use_bundle_sequence_config", predictor.use_bundle_sequence_config),
            ]
        )

    return (_format_mapping_override("experiment_name", run.experiment_name), *overrides)


def _compose_sotf_config(overrides: tuple[str, ...]) -> DictConfig:
    """Compose the base SOTF simulation config with run-level overrides.

    Args:
        overrides (tuple[str, ...]): Hydra overrides to apply.

    Returns:
        DictConfig: Composed config.
    """
    if GlobalHydra.instance().is_initialized():
        return compose(config_name="config", overrides=list(overrides))

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def copy_batch_config(batch_config_path: str | Path | None, cfg: SOTFBatchConfig) -> Path:
    """Copy batch config.

    Args:
        batch_config_path (str | Path | None): Path to the batch config.
        cfg (SOTFBatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    destination_dir = batch_files_root(cfg)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if batch_config_path is None:
        destination_path = destination_dir / f"{sanitize_path_component(cfg.batch_name)}.yaml"
        destination_path.write_text(OmegaConf.to_yaml(_batch_config_to_omegaconf(cfg), resolve=False), encoding="utf-8")
        return destination_path.resolve()

    source_path = Path(batch_config_path)
    destination_path = destination_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path.resolve()


def _batch_config_to_omegaconf(cfg: SOTFBatchConfig) -> DictConfig:
    """Resolve config to omegaconf.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.

    Returns:
        DictConfig: Result of the function.
    """
    return OmegaConf.create(
        {
            "batch_name": cfg.batch_name,
            "output_root": str(cfg.output_root),
            "dry_run": cfg.dry_run,
            "common_overrides": list(cfg.common_overrides),
            "predictor": _predictor_config_to_mapping(cfg.predictor),
            "runs": [
                {
                    "experiment_name": run.experiment_name,
                    "overrides": list(run.overrides),
                    "predictor": None if run.predictor is None else _predictor_config_to_mapping(run.predictor),
                }
                for run in cfg.runs
            ],
        }
    )


def _predictor_config_to_mapping(predictor: PredictorBatchConfig) -> dict[str, Any]:
    """Convert predictor batch config to a serializable mapping.

    Args:
        predictor (PredictorBatchConfig): Predictor batch config.

    Returns:
        dict[str, Any]: Serializable config mapping.
    """
    mapping: dict[str, Any] = {
        "name": predictor.name,
        "strategy": predictor.strategy,
        "use_bundle_feature_config": predictor.use_bundle_feature_config,
        "use_bundle_sequence_config": predictor.use_bundle_sequence_config,
    }
    if predictor.forecast_source is not None:
        mapping["forecast_source"] = {
            "results_root": str(predictor.forecast_source.results_root),
            "run_name_template": predictor.forecast_source.run_name_template,
        }
    return mapping


def batch_root(cfg: SOTFBatchConfig) -> Path:
    """Resolve root.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    parts = [sanitize_path_component(p) for p in cfg.batch_name.split("/") if p.strip()]
    return Path(cfg.output_root).joinpath("batch_experiments", *parts)


def batch_files_root(cfg: SOTFBatchConfig) -> Path:
    """Resolve files root.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    return batch_root(cfg) / "batch_files"


def batch_results_root(cfg: SOTFBatchConfig) -> Path:
    """Resolve results root.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    return batch_root(cfg) / "results"


def build_run_dir(cfg: SOTFBatchConfig, *, run_cfg: DictConfig, now: datetime) -> Path:
    """Build the output directory for one in-process simulation run.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.
        run_cfg (DictConfig): Composed run configuration.
        now (datetime): Ignored legacy argument kept for call-site compatibility.

    Returns:
        Path: Resolved run directory.
    """
    dataset_path = _dataset_identity_path_from_cfg(run_cfg)
    dataset_id = dataset_id_from_identity_path(dataset_path)
    if dataset_id is None:
        raise ValueError(f"Could not extract dataset id from dataset identity path: {dataset_path}")
    dataset_label = f"dataset_{dataset_id}_{dataset_slug_from_path(dataset_path)}"
    return (
        batch_results_root(cfg)
        / sanitize_path_component(run_cfg.predictor.name)
        / sanitize_path_component(run_cfg.data.target_column)
        / sanitize_path_component(run_cfg.sequence.target_mode)
        / dataset_label
        / sanitize_path_component(run_cfg.experiment_name)
    )


def _prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)


def latest_run_dir(base_dir: Path) -> Path:
    """Resolve run dir.

    Args:
        base_dir (Path): Directory for the base.

    Returns:
        Path: Resolved path.
    """
    if not base_dir.is_dir():
        raise ValueError(f"Missing forecast results directory: {base_dir}")
    if any((base_dir / filename).exists() for filename in ("model_bundle.pt", "model_bundle.pkl")):
        return base_dir
    matches = sorted(path for path in base_dir.iterdir() if path.is_dir())
    if not matches:
        raise ValueError(f"No run directories found in {base_dir}")
    return matches[-1]


def validate_batch_config(cfg: SOTFBatchConfig) -> None:
    """Validate batch config.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    for run in cfg.runs:
        predictor = run.predictor or cfg.predictor
        if predictor.name != "forecast_model":
            continue
        combined_overrides = (*cfg.common_overrides, *run.overrides)
        dataset_path = _dataset_identity_path_from_overrides(combined_overrides)
        if dataset_path is None:
            raise ValueError(
                f"Run '{run.experiment_name}' must define a dataset identity via data.pt_path, "
                "data.parquet_path, or data.target_parquet_path."
            )
        model_bundle_path, used_features_path = resolve_forecast_artifacts(
            cfg=cfg,
            predictor=predictor,
            combined_overrides=combined_overrides,
            dataset_path=dataset_path,
        )
        try:
            load_forecast_model_bundle(model_bundle_path, used_features_path=used_features_path)
        except Exception as exc:
            raise ValueError(
                "Forecast batch config is not compatible with SOTF forecast_model predictor. "
                f"Run '{run.experiment_name}' resolves to model bundle '{model_bundle_path}', "
                f"which failed validation: {exc}"
            ) from exc


def resolve_forecast_artifacts(
    *,
    cfg: SOTFBatchConfig,
    predictor: PredictorBatchConfig | None = None,
    combined_overrides: tuple[str, ...] | list[str],
    dataset_path: Path,
) -> tuple[Path, Path | None]:
    """Resolve forecast artifacts.

    Args:
        cfg (SOTFBatchConfig): Configuration object used by the function.
        predictor (PredictorBatchConfig | None): PredictorBatchConfig | None value used by the function.
        combined_overrides (tuple[str, ...] | list[str]): tuple[str, ...] | list[str] value used by the function.
        dataset_path (Path): Path to the dataset.

    Returns:
        tuple[Path, Path | None]: Computed forecast artifacts.
    """
    predictor = predictor or cfg.predictor
    explicit_model_bundle_path = _extract_optional_override(combined_overrides, "predictor.model_bundle_path")
    explicit_used_features_path = _extract_optional_override(combined_overrides, "predictor.used_features_path")
    if explicit_model_bundle_path is not None:
        model_bundle_path = Path(explicit_model_bundle_path).expanduser().resolve()
        used_features_path = (
            None
            if explicit_used_features_path is None
            else Path(explicit_used_features_path).expanduser().resolve()
        )
        return model_bundle_path, used_features_path

    if predictor.forecast_source is None:
        raise ValueError(
            "forecast_model batches must define either explicit predictor.model_bundle_path overrides "
            "or a predictor.forecast_source block."
        )

    run_name_template = str(predictor.forecast_source.run_name_template)
    run_name_kwargs: dict[str, Any] = {}
    if "{dataset_id" in run_name_template:
        dataset_id = dataset_id_from_identity_path(dataset_path)
        if dataset_id is None:
            raise ValueError(f"Could not extract dataset id from dataset identity path: {dataset_path}")
        run_name_kwargs["dataset_id"] = dataset_id
    if "{horizon" in run_name_template:
        run_name_kwargs["horizon"] = int(_extract_required_override(combined_overrides, "sequence.horizon"))

    forecast_run_dir = latest_run_dir(
        predictor.forecast_source.results_root
        / run_name_template.format(**run_name_kwargs)
    )
    return _resolve_model_bundle_path(forecast_run_dir), (forecast_run_dir / "used_features.json").resolve()


def _resolve_model_bundle_path(run_dir: Path) -> Path:
    """Resolve model bundle path.

    Args:
        run_dir (Path): Directory for the run.

    Returns:
        Path: Resolved path.
    """
    for filename in ("model_bundle.pt", "model_bundle.pkl"):
        path = run_dir / filename
        if path.exists():
            return path.resolve()
    return (run_dir / "model_bundle.pkl").resolve()


def _extract_required_override(overrides: tuple[str, ...] | list[str], key: str) -> str:
    """Extract required override.

    Args:
        overrides (tuple[str, ...] | list[str]): tuple[str, ...] | list[str] value used by the function.
        key (str): Key to look up.

    Returns:
        str: Resolved string value.
    """
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            return override.split("=", 1)[1]
    raise ValueError(f"Required override '{key}' is missing.")


def _extract_optional_override(overrides: tuple[str, ...] | list[str], key: str) -> str | None:
    """Extract optional override.

    Args:
        overrides (tuple[str, ...] | list[str]): tuple[str, ...] | list[str] value used by the function.
        key (str): Key to look up.

    Returns:
        str | None: Result of the function.
    """
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            return override.split("=", 1)[1]
    return None


def _dataset_identity_path_from_overrides(overrides: tuple[str, ...] | list[str]) -> Path | None:
    """Resolve the dataset identity path from Hydra overrides."""
    resolved = resolve_dataset_identity_path(
        source=_extract_optional_override(overrides, "data.source"),
        pt_path=_extract_optional_override(overrides, "data.pt_path"),
        parquet_path=_extract_optional_override(overrides, "data.parquet_path"),
        target_parquet_path=_extract_optional_override(overrides, "data.target_parquet_path"),
    )
    if not resolved:
        return None
    return Path(resolved)


def _dataset_identity_path_from_cfg(run_cfg: DictConfig) -> Path:
    """Resolve the dataset identity path from a composed run config."""
    resolved = resolve_dataset_identity_path(
        source=run_cfg.data.get("source", "prepared_dataset"),
        pt_path=run_cfg.data.get("pt_path", None),
        parquet_path=run_cfg.data.get("parquet_path", None),
        target_parquet_path=run_cfg.data.get("target_parquet_path", None),
    )
    if not resolved:
        raise ValueError("Composed run config is missing a dataset identity path.")
    return Path(resolved)


def _coerce_override_list(raw_value: Any, *, field_name: str) -> tuple[str, ...]:
    """Coerce override list.

    Args:
        raw_value (Any): Any value used by the function.
        field_name (str): str value used by the function.

    Returns:
        tuple[str, ...]: Result of the function.
    """
    if isinstance(raw_value, list):
        return tuple(str(item) for item in raw_value)
    if isinstance(raw_value, dict):
        return tuple(_format_mapping_override(key, value) for key, value in raw_value.items())
    raise ValueError(f"'{field_name}' must be either a YAML list of override strings or a mapping of Hydra-style keys.")


def _format_mapping_override(key: Any, value: Any) -> str:
    """Format mapping override.

    Args:
        key (Any): Key to look up.
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    return f"{key}={_serialize_override_value(value)}"


def _serialize_override_value(value: Any) -> str:
    """Serialize override value.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value and all(character not in value for character in " ,:=[]{}"):
            return value
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_override_value(item) for item in value) + "]"
    if isinstance(value, dict):
        serialized_items = [
            f"{item_key}:{_serialize_override_value(item_value)}"
            for item_key, item_value in value.items()
        ]
        return "{" + ",".join(serialized_items) + "}"
    raise ValueError(f"Unsupported override value type: {type(value).__name__}")


if __name__ == "__main__":
    main()
