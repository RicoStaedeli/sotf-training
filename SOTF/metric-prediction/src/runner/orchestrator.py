from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import hydra
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from runner.batch_runner import (
    BatchConfig,
    ScheduledExperiment,
    parse_batch_config,
    run_batch_config,
)
from runner.experiments import run_single_experiment
from utils.config_utils import register_omegaconf_resolvers

register_omegaconf_resolvers()

ExperimentRun = ScheduledExperiment
ExperimentPlan = BatchConfig


@hydra.main(version_base=None, config_path="pkg://config.configs", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra main.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    if _is_plan_config(cfg):
        run_plan(cfg)
        return

    run_single(cfg)


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    plan_config_name = _plan_config_name_from_args(sys.argv[1:])
    if plan_config_name is None:
        _hydra_main()
        return

    cfg = _compose_plan_config(plan_config_name, overrides=_plan_overrides_from_args(sys.argv[1:]))
    run_batch_config(
        _parse_plan(cfg),
        config_source_path=_config_source_path(plan_config_name),
        raw_config=cfg,
    )


def run_single(cfg: DictConfig) -> dict[str, Any]:
    """Run single.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    result = run_single_experiment(cfg)
    _print_result(result)
    return result


def run_plan(cfg: DictConfig) -> None:
    """Run plan.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    plan = _parse_plan(cfg)
    config_name = HydraConfig.get().job.config_name or "experiment"
    run_batch_config(plan, config_source_path=_config_source_path(config_name), raw_config=cfg)


def _is_plan_config(cfg: DictConfig) -> bool:
    """Is plan config.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        bool: Result of the function.
    """
    return "runs" in cfg


def _plan_config_name_from_args(args: list[str]) -> str | None:
    """Plan config name from args.

    Args:
        args (list[str]): list[str] value used by the function.

    Returns:
        str | None: Result of the function.
    """
    for index, arg in enumerate(args):
        if arg in {"-cn", "--config-name"} and index + 1 < len(args):
            return _plan_config_name(args[index + 1])
        if arg.startswith("--config-name="):
            return _plan_config_name(arg.split("=", maxsplit=1)[1])
    return None


def _plan_config_name(config_name: str) -> str | None:
    """Plan config name.

    Args:
        config_name (str): str value used by the function.

    Returns:
        str | None: Result of the function.
    """
    return config_name if config_name.startswith("experiments/") else None


def _plan_overrides_from_args(args: list[str]) -> list[str]:
    """Plan overrides from args.

    Args:
        args (list[str]): list[str] value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    overrides: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-cn", "--config-name"}:
            skip_next = True
            continue
        if arg.startswith("--config-name="):
            continue
        overrides.append(arg)
    return overrides


def _compose_plan_config(config_name: str, *, overrides: list[str]) -> DictConfig:
    """Compose plan config.

    Args:
        config_name (str): str value used by the function.
        overrides (list[str]): list[str] value used by the function.

    Returns:
        DictConfig: Result of the function.
    """
    with initialize_config_module(config_module="config.configs", version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _parse_plan(cfg: DictConfig) -> ExperimentPlan:
    """Parse plan.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        ExperimentPlan: Result of the function.
    """
    return parse_batch_config(cfg)


def _config_source_path(config_name: str) -> Path | None:
    """Config source path.

    Args:
        config_name (str): str value used by the function.

    Returns:
        Path | None: Result of the function.
    """
    relative_config_path = Path(config_name)
    if relative_config_path.suffix not in {".yaml", ".yml"}:
        relative_config_path = relative_config_path.with_suffix(".yaml")
    candidate = Path(__file__).resolve().parents[1] / "config" / "configs" / relative_config_path
    return candidate if candidate.exists() else None


def _print_result(result: dict[str, Any]) -> None:
    """Print result.

    Args:
        result (dict[str, Any]): dict[str, Any] value used by the function.

    Returns:
        None: None.
    """
    print("Validation metrics:")
    for metric_name, metric_value in result["metrics"].items():
        print(f"  {metric_name}: {metric_value:.6f}")
    print(f"Artifacts: {result['artifacts_dir']}")


if __name__ == "__main__":
    main()
