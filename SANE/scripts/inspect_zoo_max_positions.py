#!/usr/bin/env python3
"""Inspect position-index ranges required by a model zoo/tokenization config.

Example:
    uv run --active python scripts/inspect_zoo_max_positions.py \
        config/5_cifar10_Resnet18_width_2-256.yaml \
        --max-checkpoints null
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm import tqdm

import sane
from sane.data.config_schema import (
    AugmentationConfig,
    CachedWindowedDatasetConfig,
    TokenizerConfig,
    ZooDatasetConfig,
    from_dict,
)
from sane.data.dataset_factory import DatasetFactory, _ref_checkpoint
from sane.data.datasets import CachedWindowedDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(sane.__file__).parent.parent.parent


def _config_name(config_arg: str, config_dir: Path) -> str:
    path = Path(config_arg)
    if path.suffix in {".yaml", ".yml"}:
        resolved = path.resolve()
        try:
            return resolved.relative_to(config_dir.resolve()).with_suffix("").as_posix()
        except ValueError:
            return path.with_suffix("").as_posix()
    return config_arg


def _load_experiment_config(config_arg: str) -> dict[str, Any]:
    config_dir = _repo_root() / "config"
    config_name = _config_name(config_arg, config_dir)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _load_named_yaml(group: str, name: str) -> dict[str, Any]:
    path = _repo_root() / "config" / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config '{name}' not found at {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _resolve_checkpoints_config(cfg: Any) -> ZooDatasetConfig | list[ZooDatasetConfig]:
    if isinstance(cfg, list):
        return [_resolve_checkpoints_config(item) for item in cfg]  # type: ignore[list-item]
    if isinstance(cfg, str):
        cfg = _load_named_yaml("data/checkpoints_dataset", cfg)

    augmentations = cfg.get("augmentations")
    if isinstance(augmentations, dict):
        augmentations = from_dict(AugmentationConfig, augmentations)
    return from_dict(ZooDatasetConfig, cfg, augmentations=augmentations)


def _each_checkpoint_config(cfg: Any):
    if isinstance(cfg, list):
        for item in cfg:
            yield from _each_checkpoint_config(item)
    else:
        yield cfg


def _set_max_checkpoints(cfg: Any, value: int | None) -> None:
    for item in _each_checkpoint_config(cfg):
        item.max_checkpoints = value


def _set_epochs(cfg: Any, epochs: list[int]) -> None:
    for item in _each_checkpoint_config(cfg):
        item.epoch_idx = epochs


def _discover_all_epoch_indices(root_dir: str) -> list[int]:
    root = Path(root_dir).expanduser().resolve()
    counts = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        epoch_dirs = [
            path
            for path in model_dir.iterdir()
            if path.is_dir() and (path / "checkpoints").exists()
        ]
        if epoch_dirs:
            counts.append(len(epoch_dirs))
    if not counts:
        raise RuntimeError(f"No model checkpoint directories found below {root}")
    max_epochs = max(counts)
    min_epochs = min(counts)
    if min_epochs != max_epochs:
        log.warning(
            "Model directories have different checkpoint counts (min=%d, max=%d). "
            "Using indices 0..%d; shorter models will be skipped by ZooDataset.",
            min_epochs,
            max_epochs,
            max_epochs - 1,
        )
    return list(range(max_epochs))


def _set_all_epochs(cfg: Any) -> None:
    epoch_lists = []
    for item in _each_checkpoint_config(cfg):
        epochs = _discover_all_epoch_indices(item.root_dir)
        item.epoch_idx = epochs
        epoch_lists.append(epochs)

    first = epoch_lists[0]
    if any(epochs != first for epochs in epoch_lists):
        raise ValueError("Different zoos have different all-epoch ranges; use --epochs explicitly.")


def _parse_max_checkpoints(value: str | None) -> int | None:
    if value is None or value.lower() in {"null", "none", "0"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-checkpoints must be positive, null, none, or 0")
    return parsed


def _parse_num_windows(value: str | None) -> int | Literal["auto"] | None:
    if value is None:
        return None
    if value.lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--num-windows must be positive or 'auto'")
    return parsed


def _update_max(
    max_pos: torch.Tensor | None,
    max_source: list[tuple[int, int]] | None,
    positions: torch.Tensor,
    checkpoint_idx: int,
    window_idx: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    reduce_dims = tuple(range(positions.ndim - 1))
    batch_max = torch.amax(positions, dim=reduce_dims).to(dtype=torch.int64, device="cpu")
    if max_pos is None:
        return batch_max, [(checkpoint_idx, window_idx) for _ in range(batch_max.numel())]

    assert max_source is not None
    for dim, value in enumerate(batch_max.tolist()):
        if value > int(max_pos[dim]):
            max_pos[dim] = value
            max_source[dim] = (checkpoint_idx, window_idx)
    return max_pos, max_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute required learned-position max_positions for a model zoo."
    )
    parser.add_argument(
        "config",
        help="Hydra experiment config, e.g. config/5_cifar10_Resnet18_width_2-256.yaml",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        help="Override checkpoint epoch indices to scan.",
    )
    parser.add_argument(
        "--all-epochs",
        action="store_true",
        help="Scan all epoch indices available in the zoo.",
    )
    parser.add_argument(
        "--max-checkpoints",
        default="null",
        help="Maximum checkpoints to scan. Default: null, i.e. scan the complete configured zoo.",
    )
    parser.add_argument(
        "--num-windows",
        type=_parse_num_windows,
        help="Override tokens_dataset.num_windows_per_model, e.g. 128 or auto.",
    )
    parser.add_argument(
        "--window-strategy",
        choices=("consecutive", "distributed"),
        help="Override tokens_dataset.window_distribution_strategy.",
    )
    parser.add_argument(
        "--allow-overlapping-windows",
        action="store_true",
        help="Override tokens_dataset.allow_overlapping_windows to True.",
    )
    args = parser.parse_args()

    if args.epochs and args.all_epochs:
        parser.error("--epochs and --all-epochs are mutually exclusive")

    cfg = _load_experiment_config(args.config)
    data_cfg = cfg["data"]
    checkpoint_cfg = _resolve_checkpoints_config(data_cfg["checkpoints_dataset"])
    _set_max_checkpoints(checkpoint_cfg, _parse_max_checkpoints(args.max_checkpoints))
    if args.epochs:
        _set_epochs(checkpoint_cfg, args.epochs)
    elif args.all_epochs:
        _set_all_epochs(checkpoint_cfg)

    tokens_cfg = dict(data_cfg["tokens_dataset"])
    if args.num_windows is not None:
        tokens_cfg["num_windows_per_model"] = args.num_windows
        tokens_cfg["num_windows_per_layer"] = None
    if args.window_strategy is not None:
        tokens_cfg["window_distribution_strategy"] = args.window_strategy
    if args.allow_overlapping_windows:
        tokens_cfg["allow_overlapping_windows"] = True
    tokens_cfg["cache_mode"] = "none"
    tokens_cfg["preload_cache"] = False
    tokens_cfg["cache_dir"] = None

    factory = DatasetFactory()
    checkpoints_dataset = factory._load_checkpoints_dataset(checkpoint_cfg)
    tokenizer = factory._build_tokenizer(
        from_dict(TokenizerConfig, data_cfg["tokenizer"]),
        _ref_checkpoint(checkpoints_dataset),
    )
    tokens_dataset = CachedWindowedDataset(
        checkpoints_dataset=checkpoints_dataset,
        tokenizer=tokenizer,
        split_name="all",
        dataset_name=getattr(checkpoints_dataset, "name", "zoo"),
        window_size=tokens_cfg["window_size"],
        num_windows_per_model=tokens_cfg["num_windows_per_model"],
        num_windows_per_layer=tokens_cfg.get("num_windows_per_layer"),
        pad_windows=tokens_cfg["pad_windows"],
        allow_overlapping_windows=tokens_cfg["allow_overlapping_windows"],
        window_distribution_strategy=tokens_cfg["window_distribution_strategy"],
        cache_mode=tokens_cfg["cache_mode"],
        cache_dir=tokens_cfg["cache_dir"],
        preload_cache=tokens_cfg["preload_cache"],
    )

    log.info(
        "Scanning %d checkpoints with window_size=%s, num_windows_per_model=%s, "
        "num_windows_per_layer=%s, pad_windows=%s, strategy=%s, overlapping=%s",
        len(checkpoints_dataset),
        tokens_cfg["window_size"],
        tokens_cfg["num_windows_per_model"],
        tokens_cfg.get("num_windows_per_layer"),
        tokens_cfg["pad_windows"],
        tokens_cfg["window_distribution_strategy"],
        tokens_cfg["allow_overlapping_windows"],
    )

    max_pos: torch.Tensor | None = None
    max_source: list[tuple[int, int]] | None = None
    max_windows = 0
    total_windows = 0
    for checkpoint_idx in tqdm(
        range(len(checkpoints_dataset)),
        desc="Scanning checkpoints",
        unit="ckpt",
        dynamic_ncols=True,
    ):
        windows = tokens_dataset.__getmodel__(checkpoint_idx)
        max_windows = max(max_windows, len(windows))
        total_windows += len(windows)
        for window_idx, (_, _, positions) in enumerate(windows):
            max_pos, max_source = _update_max(
                max_pos,
                max_source,
                positions,
                checkpoint_idx,
                window_idx,
            )

    if max_pos is None or max_source is None:
        raise RuntimeError("No position tensors were scanned.")

    required = [int(value) + 1 for value in max_pos.tolist()]
    print("\nObserved position maxima:", max_pos.tolist())
    print("Required max_positions:", required)
    print("Max source per dimension:", max_source)
    print(f"Scanned checkpoints: {len(checkpoints_dataset)}")
    print(f"Scanned windows: {total_windows} (max per checkpoint: {max_windows})")


if __name__ == "__main__":
    main()
