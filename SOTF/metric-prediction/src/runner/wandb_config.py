from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

_TORCH_MLP_MODELS = {
    "mlp",
    "mlp_weight_stats",
    "mlp_weight_statistics",
    "weight_statistic_mlp",
    "mlp_latent",
    "latent_mlp",
    "latent_window_lstm",
    "lstm_latent_sequences",
    "lstm_latent_windows",
    "latent_windows_lstm",
}


def attach_wandb_to_model_config(model_cfg: Mapping[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """Attach wandb to model config.

    Args:
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    return attach_wandb_settings_to_model_config(
        model_cfg,
        project=_optional_cfg_str(cfg, "wandb_project"),
        group=_optional_cfg_str(cfg, "wandb_group"),
        api_key_file=_optional_cfg_str(cfg, "wandb_api_key_file"),
        run_name=str(cfg.experiment_name),
        config=_wandb_run_config(model_cfg, cfg),
    )


def attach_wandb_settings_to_model_config(
    model_cfg: Mapping[str, Any],
    *,
    project: str | None,
    group: str | None,
    api_key_file: str | None,
    run_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Attach wandb settings to model config.

    Args:
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        project (str | None): str | None value used by the function.
        group (str | None): str | None value used by the function.
        api_key_file (str | None): str | None value used by the function.
        run_name (str): str value used by the function.
        config (dict[str, Any]): dict[str, Any] value used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    enriched = deepcopy(dict(model_cfg))
    model_name = str(enriched.get("name", "")).lower()
    if model_name not in _TORCH_MLP_MODELS:
        return enriched

    params = deepcopy(dict(enriched.get("params", {})))
    params = _attach_torch_runtime_settings(params, config)

    if project is None:
        enriched["params"] = params
        return enriched

    params.update(
        {
            "wandb_project": project,
            "wandb_group": group,
            "wandb_api_key_file": api_key_file,
            "wandb_run_name": run_name,
            "wandb_config": config,
        }
    )
    enriched["params"] = params
    return enriched


def _attach_torch_runtime_settings(
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach train-level torch runtime settings to torch model params."""
    enriched = deepcopy(dict(params))
    train_cfg = config.get("train", {})
    if not isinstance(train_cfg, Mapping):
        return enriched

    if "deterministic" in train_cfg and "deterministic" not in enriched:
        enriched["deterministic"] = bool(train_cfg["deterministic"])
    if "disable_tf32" in train_cfg and "disable_tf32" not in enriched:
        enriched["disable_tf32"] = bool(train_cfg["disable_tf32"])
    return enriched


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


def _wandb_run_config(model_cfg: Mapping[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """Wandb run config.

    Args:
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        dict[str, Any]: Result of the function.
    """
    return {
        "experiment_name": str(cfg.experiment_name),
        "mode": str(cfg.mode),
        "model": deepcopy(dict(model_cfg)),
        "train": OmegaConf.to_container(cfg.train, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "features": OmegaConf.to_container(cfg.features, resolve=True),
    }
