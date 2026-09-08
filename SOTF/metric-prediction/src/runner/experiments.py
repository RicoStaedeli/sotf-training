from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.logging_utils import setup_logging
from runner.pipelines.prediction import run_prediction_experiment
from forecasting.config import forecast_config_from_omegaconf
from forecasting.pipeline import run_forecast_experiment
from runner.pipelines.latent_prediction import run_latent_prediction_experiment
from runner.pipelines.latent_forecasting import run_latent_forecasting_experiment

_MODES = ("prediction", "forecasting", "latent", "latent_forecast", "latent_windows_forecast", "latent_windows_transformer_forecast")

log = logging.getLogger(__name__)


def run_single_experiment(cfg: DictConfig, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Run single experiment.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        output_dir (Path | None): Output directory for artifacts.

    Returns:
        dict[str, Any]: Result of the function.
    """
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    log_file = output_dir / str(cfg.logging.filename) if bool(cfg.logging.save_to_file) else None
    setup_logging(str(cfg.logging.level), log_file=log_file)

    mode = str(cfg.mode)
    log.info("Starting %s run: experiment=%s output=%s", mode, cfg.experiment_name, output_dir)

    result = _run_mode(cfg, output_dir=output_dir)
    log.info("Run finished with metrics: %s", result["metrics"])
    return result


def _run_mode(cfg: DictConfig, *, output_dir: Path) -> dict[str, Any]:
    """Run mode.

    Args:
        cfg (DictConfig): Configuration object used by the function.
        output_dir (Path): Output directory for artifacts.

    Returns:
        dict[str, Any]: Result of the function.
    """
    mode = str(cfg.mode)
    if mode == "prediction":
        return run_prediction_experiment(cfg, output_dir=output_dir)
    
    if mode == "forecasting":
        forecast_cfg = forecast_config_from_omegaconf(cfg)
        forecast_cfg.output.run_dir = str(output_dir)
        return run_forecast_experiment(forecast_cfg)
    
    if mode == "latent":
        return run_latent_prediction_experiment(cfg, output_dir=output_dir)
    
    if mode in {"latent_forecast", "latent_windows_forecast", "latent_windows_transformer_forecast"}:
        return run_latent_forecasting_experiment(cfg, output_dir=output_dir)

    raise ValueError(f"Unknown mode '{mode}'. Expected one of: {', '.join(_MODES)}")
