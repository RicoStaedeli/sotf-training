from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from runner.experiments import run_single_experiment
from utils.config_utils import register_omegaconf_resolvers, sanitize_path_component

register_omegaconf_resolvers()

_DEFAULT_OUTPUT_ROOT = Path("metric-prediction") / "outputs"
_FORECAST_OUTPUT_ROOT = Path("metric-prediction") / "outputs_forecasting"
_BATCH_RESULTS_SUBDIR = "results"
_FORECAST_MODES = {"forecasting", "latent_forecast", "latent_windows_forecast", "latent_windows_transformer_forecast"}


@dataclass(frozen=True)
class ScheduledExperiment:
    experiment_name: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class BatchConfig:
    batch_name: str | None
    output_root: Path
    common_overrides: tuple[str, ...]
    runs: tuple[ScheduledExperiment, ...]
    dry_run: bool = False
    batch_root_subdir: str | None = "batch_experiments"
    batch_results_subdir: str | None = _BATCH_RESULTS_SUBDIR
    output_root_explicit: bool = False


def build_parser(*, prog: str, description: str) -> argparse.ArgumentParser:
    """Build parser.

    Args:
        prog (str): str value used by the function.
        description (str): str value used by the function.

    Returns:
        argparse.ArgumentParser: Result of the function.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("batch_config", type=Path, help="Path to the YAML batch config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved runs without executing them.",
    )
    return parser


def run_batch(
    *,
    batch_config_path: str | Path,
    dry_run: bool,
    batch_results_subdir: str | None = _BATCH_RESULTS_SUBDIR,
) -> None:
    """Run batch.

    Args:
        batch_config_path (str | Path): str | Path value used by the function.
        dry_run (bool): bool value used by the function.
        batch_results_subdir (str | None): str | None value used by the function.

    Returns:
        None: None.
    """
    cfg = load_batch_config(batch_config_path)
    cfg = BatchConfig(
        batch_name=cfg.batch_name,
        output_root=cfg.output_root,
        common_overrides=cfg.common_overrides,
        runs=cfg.runs,
        dry_run=bool(dry_run or cfg.dry_run),
        batch_root_subdir=cfg.batch_root_subdir,
        batch_results_subdir=batch_results_subdir,
        output_root_explicit=cfg.output_root_explicit,
    )
    run_batch_config(cfg, config_source_path=Path(batch_config_path))


def run_batch_config(
    cfg: BatchConfig,
    *,
    config_source_path: Path | None = None,
    raw_config: DictConfig | None = None,
) -> None:
    """Run batch config.

    Args:
        cfg (BatchConfig): Configuration object used by the function.
        config_source_path (Path | None): Path | None value used by the function.
        raw_config (DictConfig | None): DictConfig | None value used by the function.

    Returns:
        None: None.
    """
    validate_batch_config(cfg)
    if cfg.batch_name and not cfg.dry_run:
        copied_config_path = copy_batch_config(config_source_path, cfg, raw_config=raw_config)
        print(f"Saved batch config to {copied_config_path}")

    failures: list[tuple[str, int]] = []
    for index, run in enumerate(cfg.runs, start=1):
        try:
            run_cfg = build_run_config(cfg, run=run)
            run_dir = build_run_dir(cfg, run=run, run_cfg=run_cfg, now=datetime.now())
        except Exception as exc:  # noqa: BLE001 - keep reporting all invalid run entries where possible.
            failures.append((run.experiment_name, 1))
            print(f"[{index}/{len(cfg.runs)}] {run.experiment_name}")
            print(f"  Failed to resolve run config: {exc}")
            continue

        print(f"[{index}/{len(cfg.runs)}] {run.experiment_name}")
        print(f"  run_dir={run_dir}")
        if cfg.dry_run:
            print("  overrides:")
            for override in build_run_overrides(cfg, run=run):
                print(f"    {override}")
            continue

        try:
            _prepare_run_dir(run_dir)
            _write_run_hydra_config(run_cfg, run_dir=run_dir)
            run_single_experiment(run_cfg, output_dir=run_dir)
            print(f"  Completed {run.experiment_name}")
            continue
        except Exception as exc:  # noqa: BLE001 - batches should continue across independent runs.
            failures.append((run.experiment_name, 1))
            print(f"  Failed {run.experiment_name}: {exc}")

    if failures:
        summary = ", ".join(f"{name} ({code})" for name, code in failures)
        raise SystemExit(f"Batch execution failed: {summary}")


def load_batch_config(path: str | Path) -> BatchConfig:
    """Load batch config.

    Args:
        path (str | Path): Path to process.

    Returns:
        BatchConfig: Result of the function.
    """
    cfg = OmegaConf.load(Path(path))
    if isinstance(cfg, ListConfig):
        raise ValueError("Batch config must be a YAML mapping.")
    return parse_batch_config(cfg)


def parse_batch_config(cfg: DictConfig | dict[str, Any]) -> BatchConfig:
    """Parse batch config.

    Args:
        cfg (DictConfig | dict[str, Any]): Configuration object used by the function.

    Returns:
        BatchConfig: Result of the function.
    """
    raw_cfg = cfg if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_cfg, dict):
        raise ValueError("Batch config must be a YAML mapping.")

    common_overrides = _coerce_override_list(raw_cfg.get("common_overrides", []), field_name="common_overrides")
    raw_runs = raw_cfg.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Batch config must define a non-empty 'runs' list.")

    output_root_explicit = raw_cfg.get("output_root") is not None
    output_root = raw_cfg.get("output_root")
    if output_root is None:
        output_root = _default_output_root(common_overrides)

    raw_batch_root_subdir = raw_cfg.get("batch_root_subdir", "batch_experiments")
    batch_root_subdir = str(raw_batch_root_subdir).strip() if raw_batch_root_subdir else None

    return BatchConfig(
        batch_name=_parse_batch_name(raw_cfg.get("batch_name")),
        output_root=Path(str(output_root)),
        common_overrides=common_overrides,
        runs=tuple(_parse_run(run_cfg) for run_cfg in raw_runs),
        dry_run=bool(raw_cfg.get("dry_run", False)),
        batch_root_subdir=batch_root_subdir,
        output_root_explicit=output_root_explicit,
    )


def build_run_config(cfg: BatchConfig, *, run: ScheduledExperiment) -> DictConfig:
    """Build run config.

    Args:
        cfg (BatchConfig): Configuration object used by the function.
        run (ScheduledExperiment): ScheduledExperiment value used by the function.

    Returns:
        DictConfig: Result of the function.
    """
    return _compose_metric_config(build_run_overrides(cfg, run=run))


def build_run_overrides(cfg: BatchConfig, *, run: ScheduledExperiment) -> tuple[str, ...]:
    """Build run overrides.

    Args:
        cfg (BatchConfig): Configuration object used by the function.
        run (ScheduledExperiment): ScheduledExperiment value used by the function.

    Returns:
        tuple[str, ...]: Result of the function.
    """
    combined_overrides = (*cfg.common_overrides, *run.overrides)
    if any(override.startswith("experiment_name=") for override in combined_overrides):
        raise ValueError("Use the run-level 'experiment_name' field instead of overriding experiment_name directly.")
    if any(override.startswith("hydra.run.dir=") for override in combined_overrides):
        raise ValueError("Do not override hydra.run.dir in batch configs; it is controlled by the batch runner.")
    return (_format_mapping_override("experiment_name", run.experiment_name), *combined_overrides)


def build_run_dir(
    cfg: BatchConfig,
    *,
    run: ScheduledExperiment,
    run_cfg: DictConfig,
    now: datetime,
) -> Path:
    """Build run dir.

    Args:
        cfg (BatchConfig): Configuration object used by the function.
        run (ScheduledExperiment): ScheduledExperiment value used by the function.
        run_cfg (DictConfig): DictConfig value used by the function.
        now (datetime): datetime value used by the function.

    Returns:
        Path: Resolved path.
    """
    if cfg.batch_name:
        return batch_results_root(cfg) / sanitize_path_component(run.experiment_name)

    experiment_name = sanitize_path_component(str(run_cfg.experiment_name))
    if str(run_cfg.mode) in _FORECAST_MODES:
        return Path(cfg.output_root) / now.strftime("%Y-%m-%d") / experiment_name / now.strftime("%H-%M-%S-%f")
    return (
        Path(cfg.output_root)
        / "single_experiments"
        / now.strftime("%Y-%m-%d")
        / experiment_name
        / now.strftime("%H-%M-%S-%f")
    )


def copy_batch_config(
    batch_config_path: str | Path | None,
    cfg: BatchConfig,
    *,
    raw_config: DictConfig | None = None,
) -> Path:
    """Copy batch config.

    Args:
        batch_config_path (str | Path | None): str | Path | None value used by the function.
        cfg (BatchConfig): Configuration object used by the function.
        raw_config (DictConfig | None): DictConfig | None value used by the function.

    Returns:
        Path: Resolved path.
    """
    if not cfg.batch_name:
        raise ValueError("Cannot copy a batch config without a batch_name.")

    destination_dir = batch_files_root(cfg)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if batch_config_path is not None:
        source_path = Path(batch_config_path)
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        return destination_path.resolve()

    destination_path = destination_dir / f"{sanitize_path_component(cfg.batch_name)}.yaml"
    source_cfg = raw_config if raw_config is not None else _batch_config_to_omegaconf(cfg)
    destination_path.write_text(OmegaConf.to_yaml(source_cfg, resolve=False), encoding="utf-8")
    return destination_path.resolve()


def batch_root(cfg: BatchConfig) -> Path:
    """Batch root.

    Args:
        cfg (BatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    if not cfg.batch_name:
        raise ValueError("Cannot build a batch directory without a batch_name.")
    if cfg.batch_root_subdir:
        return Path(cfg.output_root) / cfg.batch_root_subdir / sanitize_path_component(cfg.batch_name)
    return Path(cfg.output_root) / sanitize_path_component(cfg.batch_name)


def batch_files_root(cfg: BatchConfig) -> Path:
    """Batch files root.

    Args:
        cfg (BatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    return batch_root(cfg) / "batch_files"


def batch_results_root(cfg: BatchConfig) -> Path:
    """Batch results root.

    Args:
        cfg (BatchConfig): Configuration object used by the function.

    Returns:
        Path: Resolved path.
    """
    root = batch_root(cfg)
    if not cfg.batch_results_subdir:
        return root
    return root / cfg.batch_results_subdir


def validate_batch_config(cfg: BatchConfig) -> None:
    """Validate batch config.

    Args:
        cfg (BatchConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    for run in cfg.runs:
        build_run_overrides(cfg, run=run)


def _compose_metric_config(overrides: tuple[str, ...]) -> DictConfig:
    """Compose metric config.

    Args:
        overrides (tuple[str, ...]): tuple[str, ...] value used by the function.

    Returns:
        DictConfig: Result of the function.
    """
    if GlobalHydra.instance().is_initialized():
        return compose(config_name="config", overrides=list(overrides))

    with initialize_config_module(config_module="config.configs", version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def _parse_run(raw_run: Any) -> ScheduledExperiment:
    """Parse run.

    Args:
        raw_run (Any): Value used by the function.

    Returns:
        ScheduledExperiment: Result of the function.
    """
    if not isinstance(raw_run, dict):
        raise ValueError("Each batch run must be a mapping.")

    experiment_name = str(raw_run.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ValueError("Each batch run must define a non-empty 'experiment_name'.")

    overrides = _coerce_override_list(raw_run.get("overrides", []), field_name=f"runs[{experiment_name}].overrides")
    return ScheduledExperiment(experiment_name=experiment_name, overrides=overrides)


def _parse_batch_name(raw_batch_name: Any) -> str | None:
    """Parse batch name.

    Args:
        raw_batch_name (Any): Value used by the function.

    Returns:
        str | None: Result of the function.
    """
    if raw_batch_name is None:
        return None
    batch_name = str(raw_batch_name).strip()
    if not batch_name:
        raise ValueError("If provided, 'batch_name' must be a non-empty string.")
    return batch_name


def _coerce_override_list(raw_value: Any, *, field_name: str) -> tuple[str, ...]:
    """Coerce override list.

    Args:
        raw_value (Any): Raw value to process.
        field_name (str): Field name used for validation.

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
        key (Any): Value used by the function.
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
        serialized_items = [f"{item_key}:{_serialize_override_value(item_value)}" for item_key, item_value in value.items()]
        return "{" + ",".join(serialized_items) + "}"
    raise ValueError(f"Unsupported override value type: {type(value).__name__}")


def _default_output_root(common_overrides: tuple[str, ...]) -> Path:
    """Default output root.

    Args:
        common_overrides (tuple[str, ...]): tuple[str, ...] value used by the function.

    Returns:
        Path: Resolved path.
    """
    mode_override = next((override for override in common_overrides if override.startswith("mode=")), "")
    mode = mode_override.removeprefix("mode=")
    if mode in _FORECAST_MODES:
        return _FORECAST_OUTPUT_ROOT
    return _DEFAULT_OUTPUT_ROOT


def _batch_config_to_omegaconf(cfg: BatchConfig) -> DictConfig:
    """Batch config to omegaconf.

    Args:
        cfg (BatchConfig): Configuration object used by the function.

    Returns:
        DictConfig: Result of the function.
    """
    return OmegaConf.create(
        {
            "batch_name": cfg.batch_name,
            "output_root": str(cfg.output_root),
            "dry_run": cfg.dry_run,
            "common_overrides": list(cfg.common_overrides),
            "runs": [
                {
                    "experiment_name": run.experiment_name,
                    "overrides": list(run.overrides),
                }
                for run in cfg.runs
            ],
        }
    )


def _prepare_run_dir(run_dir: Path) -> None:
    """Prepare run dir.

    Args:
        run_dir (Path): Path value used by the function.

    Returns:
        None: None.
    """
    if run_dir.exists():
        shutil.rmtree(run_dir)


def _write_run_hydra_config(run_cfg: DictConfig, *, run_dir: Path) -> None:
    """Write run hydra config.

    Args:
        run_cfg (DictConfig): DictConfig value used by the function.
        run_dir (Path): Path value used by the function.

    Returns:
        None: None.
    """
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(OmegaConf.to_yaml(run_cfg, resolve=False), encoding="utf-8")
