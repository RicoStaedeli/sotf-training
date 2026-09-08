#!/usr/bin/env python3
"""Encode all dataset splits with a trained SANEAutoEncoder and save latent z vectors.

Usage:
    python scripts/export_latent.py <sane_checkpoint_path> [output_dir]

Outputs (inside <output_dir>/encoded_checkpoints/):
    <dataset_name>_latent_representations_train.pt
    <dataset_name>_latent_representations_val.pt
    <dataset_name>_latent_representations_test.pt

Use ``--mode windows`` to save per-checkpoint window embeddings instead of the
default mean-pooled checkpoint embedding.

Use --cache-dir to override the token cache directory stored in the checkpoint
config, which is useful when the original cache path is not writable.
"""
import argparse
import logging
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from sane.trainer.trainer import _load_trial_config
from sane.trainer.contrastive_trainer import SANEContrastiveTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_checkpoint_path(full_path_str: str) -> dict:
    full_path = Path(full_path_str).resolve()
    checkpoint_pt = full_path
    checkpoint_dir = checkpoint_pt.parent
    training_run_dir = checkpoint_dir.parent
    return {
        "checkpoint_pt": checkpoint_pt,
        "checkpoint_dir": checkpoint_dir,
        "trial_dir": training_run_dir,
        "experiment_root": training_run_dir.parent,
    }


def _get_model_id(dataset, checkpoint_idx: int) -> str:
    """Return a string identifier for the checkpoint at checkpoint_idx."""
    checkpoint_ds = dataset.checkpoints_dataset
    if hasattr(checkpoint_ds, "indices") and hasattr(checkpoint_ds, "dataset"):
        actual_idx = checkpoint_ds.indices[checkpoint_idx]
        underlying = checkpoint_ds.dataset
        if hasattr(underlying, "get_model_id"):
            return underlying.get_model_id(actual_idx)
        return str(actual_idx)
    if hasattr(checkpoint_ds, "get_model_id"):
        return checkpoint_ds.get_model_id(checkpoint_idx)
    return str(checkpoint_idx)


def _get_checkpoint_info(dataset, checkpoint_idx: int) -> tuple[str, int]:
    """Return (checkpoint_path_str, epoch_idx) for the checkpoint at checkpoint_idx."""
    checkpoint_ds = dataset.checkpoints_dataset
    is_subset = hasattr(checkpoint_ds, "indices") and hasattr(checkpoint_ds, "dataset")
    underlying_ds = checkpoint_ds.dataset if is_subset else checkpoint_ds
    actual_idx = checkpoint_ds.indices[checkpoint_idx] if is_subset else checkpoint_idx
    ckpt_info_list = getattr(underlying_ds, "_ZooDataset__ckpt_info_list", None)
    if ckpt_info_list is not None:
        info = ckpt_info_list[actual_idx]
        return str(info.state_dict_path), int(info.epoch_idx)
    return f"checkpoint_{checkpoint_idx}", -1


def _get_position_limits(sane_ae: torch.nn.Module) -> list[int] | None:
    """Return learned position embedding limits if the model exposes them."""
    encoder = getattr(sane_ae, "encoder", None)
    position_embedding = getattr(encoder, "position_embedding", None)
    max_positions = getattr(position_embedding, "max_positions", None)
    if max_positions is None:
        return None
    return [int(value) for value in max_positions]


def _validate_batch_positions(
    batch_pos: torch.Tensor,
    *,
    sane_ae: torch.nn.Module,
    split_name: str,
    checkpoint_idx: int,
    checkpoint_path: str,
    model_id: str,
    epoch: int,
    window_start: int,
    window_end: int,
) -> None:
    """Fail early with a clear error if positions exceed learned embedding ranges."""
    max_positions = _get_position_limits(sane_ae)
    if max_positions is None:
        return

    if batch_pos.ndim != 3:
        raise ValueError(
            f"Expected batch_pos to have shape [B, N, D], got {tuple(batch_pos.shape)}"
        )
    if batch_pos.shape[-1] != len(max_positions):
        raise ValueError(
            f"Expected {len(max_positions)} position dimensions, got {batch_pos.shape[-1]}"
        )

    mins = torch.amin(batch_pos, dim=(0, 1)).to(dtype=torch.int64).cpu()
    maxs = torch.amax(batch_pos, dim=(0, 1)).to(dtype=torch.int64).cpu()
    invalid_dims = [
        dim
        for dim, (min_value, max_value, limit) in enumerate(zip(mins.tolist(), maxs.tolist(), max_positions))
        if min_value < 0 or max_value >= limit
    ]
    if not invalid_dims:
        return

    invalid_mask = torch.zeros(batch_pos.shape[:2], dtype=torch.bool)
    dim_messages: list[str] = []
    for dim in invalid_dims:
        dim_tensor = batch_pos[:, :, dim]
        limit = max_positions[dim]
        dim_invalid = (dim_tensor < 0) | (dim_tensor >= limit)
        invalid_mask |= dim_invalid
        invalid_values = dim_tensor[dim_invalid].to(dtype=torch.int64).cpu().flatten()
        unique_values = torch.unique(invalid_values)
        shown_values = unique_values[:8].tolist()
        suffix = "..." if unique_values.numel() > 8 else ""
        dim_messages.append(
            f"dim={dim} min={int(mins[dim])} max={int(maxs[dim])} limit={limit} invalid_values={shown_values}{suffix}"
        )

    invalid_indices = torch.nonzero(invalid_mask, as_tuple=False).cpu()
    example_messages = []
    for batch_idx, token_idx in invalid_indices[:5].tolist():
        example_messages.append(
            f"(window_in_batch={batch_idx}, token_idx={token_idx}, pos={batch_pos[batch_idx, token_idx].tolist()})"
        )
    examples = ", ".join(example_messages) if example_messages else "none"

    raise ValueError(
        "Position indices exceed learned embedding limits during latent export. "
        f"split={split_name} checkpoint_idx={checkpoint_idx} epoch={epoch} "
        f"window_range=[{window_start}, {window_end}) model_id={model_id} "
        f"checkpoint_path={checkpoint_path} batch_pos_min={mins.tolist()} "
        f"batch_pos_max={maxs.tolist()} max_positions={max_positions} "
        f"violations={' | '.join(dim_messages)} examples={examples}"
    )


def encode_split(
    sane_ae: torch.nn.Module,
    dataset,
    encode_batch_size: int = 64,
    mode: Literal["average", "windows", "token"] = "average",
    split_name: str = "unknown",
    chunk_size: int = 0,
    chunk_dir: Path | None = None,
) -> tuple[list[str], list[str], list[int], np.ndarray | list[torch.Tensor] | list[Path]]:
    """Encode every checkpoint in a dataset split.

    Returns (model_ids, checkpoint_paths, epochs, embeddings) where embeddings[i] is the
    latent vector for model_ids[i].  One row per checkpoint — duplicates in model_ids are
    possible when multiple epochs of the same model are loaded (model_id is the model
    directory, not epoch-specific).

    In ``average`` mode, checkpoints that span multiple windows are averaged across
    windows. In ``windows`` mode, every checkpoint keeps its full ``[n_windows, latent_dim]``
    tensor. Uses sane_ae.forward_embeddings (encoder + embedder) to obtain one vector
    per window. In ``token`` mode, every checkpoint keeps all raw encoder token vectors as
    a flat sequence ``[n_windows * N, latent_dim]``, bypassing the embedder entirely.

    When ``chunk_size > 0`` and ``chunk_dir`` is set (and mode is ``token`` or ``windows``),
    embeddings are flushed to ``chunk_dir/chunk_NNNN.pt`` every ``chunk_size`` checkpoints
    to bound peak RAM usage.  The fourth return value is then a ``list[Path]`` of chunk
    file paths instead of an in-memory list of tensors.
    """
    device = next(sane_ae.parameters()).device
    use_amp = device.type == "cuda"
    sane_ae.eval()

    n_checkpoints = len(dataset.checkpoints_dataset)
    model_ids: list[str] = []
    checkpoint_paths: list[str] = []
    epoch_indices: list[int] = []
    embeddings: list[np.ndarray] | list[torch.Tensor] = []

    use_chunking = chunk_size > 0 and chunk_dir is not None and mode in ("windows", "token")
    chunk_paths: list[Path] = []
    current_chunk: list[torch.Tensor] = []
    current_chunk_ids: list[str] = []
    current_chunk_ckpts: list[str] = []
    current_chunk_epochs: list[int] = []
    chunk_counter = 0

    def _flush_chunk() -> None:
        nonlocal chunk_counter
        if not current_chunk:
            return
        chunk_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        path = chunk_dir / f"chunk_{chunk_counter:04d}.pt"
        torch.save({
            "mode": mode,
            "model_ids": current_chunk_ids[:],
            "checkpoint_paths": current_chunk_ckpts[:],
            "epoch": current_chunk_epochs[:],
            "embeddings": list(current_chunk),
        }, path)
        log.info(
            "Chunk %d: saved %d checkpoints → %s",
            chunk_counter, len(current_chunk), path.name,
        )
        chunk_paths.append(path)
        chunk_counter += 1
        current_chunk.clear()
        current_chunk_ids.clear()
        current_chunk_ckpts.clear()
        current_chunk_epochs.clear()

    with torch.no_grad():
        pbar = tqdm(range(n_checkpoints), desc="Encoding checkpoints", unit="ckpt", dynamic_ncols=True)
        for ckpt_idx in pbar:
            model_id = _get_model_id(dataset, ckpt_idx)
            ckpt_path, epoch = _get_checkpoint_info(dataset, ckpt_idx)
            model_ids.append(model_id)
            checkpoint_paths.append(ckpt_path)
            epoch_indices.append(epoch)
            pbar.set_postfix(model=Path(model_id).name if "/" in model_id else model_id)

            windows = dataset.__getmodel__(ckpt_idx)

            tokens_list, pos_list = [], []
            for window in windows:
                tokens, _, pos = window
                if tokens.dim() == 3:
                    tokens, pos = tokens[0], pos[0]
                tokens_list.append(tokens)
                pos_list.append(pos)

            window_embeddings = []
            n_batches = (len(tokens_list) + encode_batch_size - 1) // encode_batch_size
            for batch_idx, start in enumerate(range(0, len(tokens_list), encode_batch_size)):
                batch_tokens = torch.stack(tokens_list[start : start + encode_batch_size])
                batch_pos = torch.stack(pos_list[start : start + encode_batch_size])
                _validate_batch_positions(
                    batch_pos,
                    sane_ae=sane_ae,
                    split_name=split_name,
                    checkpoint_idx=ckpt_idx,
                    checkpoint_path=ckpt_path,
                    model_id=model_id,
                    epoch=epoch,
                    window_start=start,
                    window_end=min(start + encode_batch_size, len(tokens_list)),
                )
                batch_tokens = batch_tokens.to(device)
                batch_pos = batch_pos.to(device)
                if n_batches > 1:
                    pbar.set_postfix(model=Path(model_id).name if "/" in model_id else model_id,
                                     windows=f"{batch_idx + 1}/{n_batches}")
                with autocast(enabled=use_amp, device_type=device.type):
                    if mode == "token":
                        z = sane_ae.encoder(batch_tokens, batch_pos)  # (B, N, latent_dim)
                        window_embeddings.append(z.cpu().float().flatten(0, 1))  # (B*N, latent_dim)
                    else:
                        z_emb = sane_ae.forward_embeddings(batch_tokens, batch_pos)  # (B, latent_dim)
                        window_embeddings.append(z_emb.cpu().float())

            all_embeddings = torch.cat(window_embeddings, dim=0)
            if use_chunking:
                current_chunk.append(all_embeddings)
                current_chunk_ids.append(model_id)
                current_chunk_ckpts.append(ckpt_path)
                current_chunk_epochs.append(epoch)
                if len(current_chunk) >= chunk_size:
                    _flush_chunk()
            elif mode in ("windows", "token"):
                embeddings.append(all_embeddings)
            else:
                embeddings.append(all_embeddings.mean(dim=0).numpy())

    if use_chunking:
        _flush_chunk()
        return model_ids, checkpoint_paths, epoch_indices, chunk_paths

    if mode in ("windows", "token"):
        return model_ids, checkpoint_paths, epoch_indices, embeddings
    return model_ids, checkpoint_paths, epoch_indices, np.stack(embeddings)


def _log_split_summary(dataset, split_name: str) -> None:
    """Log model-zoo stats before encoding: unique models, checkpoints, and epoch indices."""
    checkpoint_ds = dataset.checkpoints_dataset

    # Resolve the underlying dataset when wrapped in a Subset
    is_subset = hasattr(checkpoint_ds, "indices") and hasattr(checkpoint_ds, "dataset")
    underlying_ds = checkpoint_ds.dataset if is_subset else checkpoint_ds

    n_checkpoints = len(checkpoint_ds)

    # Collect all (model_id, checkpoint_path, epoch) tuples
    ckpt_info_list = getattr(underlying_ds, "_ZooDataset__ckpt_info_list", None)

    model_to_ckpts: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for i in range(n_checkpoints):
        model_id = _get_model_id(dataset, i)
        if ckpt_info_list is not None:
            actual_idx = checkpoint_ds.indices[i] if is_subset else i
            info = ckpt_info_list[actual_idx]
            model_to_ckpts[model_id].append((info.state_dict_path, info.epoch_idx))
        else:
            model_to_ckpts[model_id].append((f"checkpoint_{i}", -1))

    unique_models = sorted(model_to_ckpts)
    n_unique = len(unique_models)
    all_epochs = sorted({ep for ckpts in model_to_ckpts.values() for _, ep in ckpts})

    log.info(
        "%s split — model zoo: %d unique models, %d checkpoints total, epoch indices: %s",
        split_name.upper(), n_unique, n_checkpoints, all_epochs,
    )


def _resolve_checkpoints_dataset_str(cfg_name: str) -> dict:
    """Expand a checkpoints_dataset shorthand string to its full config dict."""
    import sane
    from omegaconf import OmegaConf

    config_path = (
        Path(sane.__file__).parent.parent.parent
        / "config"
        / "data"
        / "checkpoints_dataset"
        / f"{cfg_name}.yaml"
    )
    if not config_path.exists():
        raise ValueError(
            f"Checkpoints dataset config '{cfg_name}' not found at: {config_path}"
        )
    return OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)  # type: ignore[return-value]


def _expand_checkpoints_dataset(params: dict) -> None:
    """Resolve any string shorthand in params["data"]["checkpoints_dataset"] in-place.

    When checkpoints_dataset is stored as a config name (e.g. "stl10_cnn_large_uniform")
    rather than a full dict, expand it to the resolved dict so that downstream patch
    functions (_override_epoch_idx, _override_max_checkpoints, etc.) can mutate it.
    """
    data = params.get("data")
    if not isinstance(data, dict):
        return
    chkpt_cfg = data.get("checkpoints_dataset")
    if isinstance(chkpt_cfg, str):
        data["checkpoints_dataset"] = _resolve_checkpoints_dataset_str(chkpt_cfg)
    elif isinstance(chkpt_cfg, list):
        data["checkpoints_dataset"] = [
            _resolve_checkpoints_dataset_str(entry) if isinstance(entry, str) else entry
            for entry in chkpt_cfg
        ]


def _checkpoint_dataset_entries(params: dict) -> list[dict]:
    chkpt_cfg = params.get("data", {}).get("checkpoints_dataset")
    if isinstance(chkpt_cfg, dict):
        return [chkpt_cfg]
    if isinstance(chkpt_cfg, list):
        return [entry for entry in chkpt_cfg if isinstance(entry, dict)]
    return []


def _discover_full_epoch_model_dirs(root_dir: str) -> tuple[list[int], list[Path]]:
    """Return max epoch indices and model dirs that contain the full max epoch range."""
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model zoo root does not exist: {root}")

    model_epoch_counts: dict[Path, int] = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        epoch_dirs = [
            path
            for path in model_dir.iterdir()
            if path.is_dir() and (path / "checkpoints").exists()
        ]
        if epoch_dirs:
            model_epoch_counts[model_dir] = len(epoch_dirs)

    if not model_epoch_counts:
        raise RuntimeError(f"No model directories found in {root_dir}")

    min_epochs = min(model_epoch_counts.values())
    max_epochs = max(model_epoch_counts.values())
    full_length_model_dirs = [
        model_dir
        for model_dir, epoch_count in model_epoch_counts.items()
        if epoch_count == max_epochs
    ]

    if min_epochs != max_epochs:
        log.warning(
            "Model directories have different numbers of checkpoints (min=%d, max=%d). "
            "Using only the %d / %d model directories with %d checkpoints.",
            min_epochs,
            max_epochs,
            len(full_length_model_dirs),
            len(model_epoch_counts),
            max_epochs,
        )

    return list(range(max_epochs)), full_length_model_dirs


def _build_filtered_zoo_root(model_dirs: list[Path]) -> tempfile.TemporaryDirectory:
    tmp_dir = tempfile.TemporaryDirectory(prefix="sane_eval_full_epoch_zoo_")
    filtered_root = Path(tmp_dir.name)
    for model_dir in model_dirs:
        (filtered_root / model_dir.name).symlink_to(model_dir, target_is_directory=True)
    return tmp_dir


def _configure_all_epochs(params: dict) -> tuple[list[int], list[tempfile.TemporaryDirectory]]:
    """Use all epochs from full-length model dirs and patch zoo roots accordingly."""
    temp_roots: list[tempfile.TemporaryDirectory] = []
    epoch_indices_by_root: list[list[int]] = []

    for entry in _checkpoint_dataset_entries(params):
        root_dir = entry.get("root_dir")
        if not root_dir:
            continue

        epoch_indices, model_dirs = _discover_full_epoch_model_dirs(root_dir)
        filtered_root = _build_filtered_zoo_root(model_dirs)
        temp_roots.append(filtered_root)
        entry["root_dir"] = filtered_root.name
        entry["epoch_idx"] = epoch_indices
        epoch_indices_by_root.append(epoch_indices)

        log.info(
            "--all-epochs: %s -> %s (%d full-length models, %d epochs)",
            root_dir,
            filtered_root.name,
            len(model_dirs),
            len(epoch_indices),
        )

    if not epoch_indices_by_root:
        raise ValueError("Cannot use --all-epochs: no checkpoint dataset root_dir found in config.")

    first = epoch_indices_by_root[0]
    if any(indices != first for indices in epoch_indices_by_root):
        raise ValueError(
            "--all-epochs discovered different epoch ranges across checkpoint datasets. "
            "Use --epochs explicitly for this multi-zoo setup."
        )

    return first, temp_roots


def _override_max_checkpoints(params: dict, value: int | None) -> None:
    """Patch max_checkpoints in every checkpoints_dataset entry inside params."""
    chkpt_cfg = params.get("data", {}).get("checkpoints_dataset")
    if isinstance(chkpt_cfg, dict):
        chkpt_cfg["max_checkpoints"] = value
    elif isinstance(chkpt_cfg, list):
        for entry in chkpt_cfg:
            if isinstance(entry, dict):
                entry["max_checkpoints"] = value


def _override_epoch_idx(params: dict, epochs: list[int]) -> None:
    """Patch epoch_idx in every checkpoints_dataset entry inside params."""
    chkpt_cfg = params.get("data", {}).get("checkpoints_dataset")
    if isinstance(chkpt_cfg, dict):
        chkpt_cfg["epoch_idx"] = epochs
    elif isinstance(chkpt_cfg, list):
        for entry in chkpt_cfg:
            if isinstance(entry, dict):
                entry["epoch_idx"] = epochs


def _override_cache_dir(params: dict, cache_dir: str) -> None:
    """Patch tokens_dataset.cache_dir inside params."""
    tokens_cfg = params.get("data", {}).get("tokens_dataset")
    if not isinstance(tokens_cfg, dict):
        raise ValueError("Expected data.tokens_dataset to be a mapping in the checkpoint config")
    tokens_cfg["cache_dir"] = str(Path(cache_dir).expanduser())


def _override_cache_preload(params: dict, enabled: bool) -> None:
    """Patch tokens_dataset.preload_cache inside params."""
    tokens_cfg = params.get("data", {}).get("tokens_dataset")
    if not isinstance(tokens_cfg, dict):
        raise ValueError("Expected data.tokens_dataset to be a mapping in the checkpoint config")
    tokens_cfg["preload_cache"] = enabled


def _mirror_tokens_hash_config(params: dict, key: str, value) -> None:
    """Mirror token export overrides into data_config_for_hash when present."""
    tokens_cfg = params.get("data", {}).get("tokens_dataset")
    if not isinstance(tokens_cfg, dict):
        return

    hash_cfg = tokens_cfg.get("data_config_for_hash")
    if not isinstance(hash_cfg, dict):
        return

    hash_tokens_cfg = hash_cfg.get("tokens_dataset")
    if isinstance(hash_tokens_cfg, dict):
        hash_tokens_cfg[key] = value


def _override_num_windows(params: dict, value: int | Literal["auto"]) -> None:
    """Patch tokens_dataset.num_windows_per_model inside params."""
    tokens_cfg = params.get("data", {}).get("tokens_dataset")
    if not isinstance(tokens_cfg, dict):
        raise ValueError("Expected data.tokens_dataset to be a mapping in the checkpoint config")
    tokens_cfg["num_windows_per_model"] = value
    # num_windows_per_layer takes precedence over num_windows_per_model. Clear it so
    # the CLI override has the effect users asked for.
    tokens_cfg["num_windows_per_layer"] = None
    _mirror_tokens_hash_config(params, "num_windows_per_model", value)
    _mirror_tokens_hash_config(params, "num_windows_per_layer", None)


def _override_cache_mode(params: dict, cache_mode: Literal["none", "per_model", "per_window"]) -> None:
    """Patch tokens_dataset.cache_mode inside params."""
    tokens_cfg = params.get("data", {}).get("tokens_dataset")
    if not isinstance(tokens_cfg, dict):
        raise ValueError("Expected data.tokens_dataset to be a mapping in the checkpoint config")
    tokens_cfg["cache_mode"] = cache_mode
    _mirror_tokens_hash_config(params, "cache_mode", cache_mode)


def _override_window_distribution_strategy(
    params: dict,
    strategy: Literal["consecutive", "distributed"],
) -> None:
    """Patch tokens_dataset.window_distribution_strategy inside params."""
    tokens_cfg = params.get("data", {}).get("tokens_dataset")
    if not isinstance(tokens_cfg, dict):
        raise ValueError("Expected data.tokens_dataset to be a mapping in the checkpoint config")
    tokens_cfg["window_distribution_strategy"] = strategy
    _mirror_tokens_hash_config(params, "window_distribution_strategy", strategy)


def _parse_num_windows(value: str) -> int | Literal["auto"]:
    if value.lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--num-windows must be a positive integer or 'auto'")
    return parsed


def _disable_drop_nan(params: dict) -> int:
    """Disable checkpoint NaN filtering for export to avoid full upfront scans."""
    disabled = 0
    for entry in _checkpoint_dataset_entries(params):
        if entry.get("drop_nan"):
            entry["drop_nan"] = False
            disabled += 1
    return disabled


def _disable_checkpoint_augmentations(params: dict) -> int:
    """Remove checkpoint augmentations for latent export.

    Latent export only uses the canonical view, so keeping augmentation enabled makes
    cache generation much slower without changing the saved embeddings.
    """
    chkpt_cfg = params.get("data", {}).get("checkpoints_dataset")
    removed = 0
    if isinstance(chkpt_cfg, dict):
        if chkpt_cfg.pop("augmentations", None) is not None:
            removed += 1
    elif isinstance(chkpt_cfg, list):
        for entry in chkpt_cfg:
            if isinstance(entry, dict) and entry.pop("augmentations", None) is not None:
                removed += 1
    return removed


def _dataset_name_from_config(params: dict) -> str:
    """Extract a human-readable dataset name from the trainer config."""
    chkpt_cfg = params.get("data", {}).get("checkpoints_dataset")
    if isinstance(chkpt_cfg, str):
        return chkpt_cfg
    if isinstance(chkpt_cfg, dict):
        root = chkpt_cfg.get("root_dir", "")
        return Path(root).name if root else "dataset"
    if isinstance(chkpt_cfg, list) and chkpt_cfg:
        first = chkpt_cfg[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            root = first.get("root_dir", "")
            return Path(root).name if root else "dataset"
    return "dataset"


def _merge_and_save_chunks(
    chunk_paths: list[Path],
    out_path: Path,
    mode: str,
    model_ids: list[str],
    checkpoint_paths: list[str],
    epoch_indices: list[int],
) -> None:
    """Load chunk files produced during encoding, concatenate embeddings, save one .pt, remove chunks."""
    merged_embeddings: list[torch.Tensor] = []
    for chunk_path in chunk_paths:
        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        merged_embeddings.extend(chunk_data["embeddings"])
        del chunk_data

    torch.save(
        {
            "mode": mode,
            "model_ids": model_ids,
            "checkpoint_paths": checkpoint_paths,
            "epoch": epoch_indices,
            "embeddings": merged_embeddings,
        },
        out_path,
    )

    chunk_dir = chunk_paths[0].parent if chunk_paths else None
    for chunk_path in chunk_paths:
        chunk_path.unlink(missing_ok=True)
    if chunk_dir is not None and chunk_dir.exists():
        try:
            chunk_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode dataset splits with a trained SANEAutoEncoder.")
    parser.add_argument("checkpoint_path", help="Path to the SANE .pt checkpoint file")
    parser.add_argument("output_dir", nargs="?", help="Output directory (default: experiment root)")
    parser.add_argument("-mode", "--mode", choices=("average", "windows", "token"), default="average",
                        help="Save one mean embedding per checkpoint (average), all window embeddings (windows), or all raw encoder token vectors (token).")
    parser.add_argument("--epochs", nargs="+", type=int, metavar="N",
                        help="Epoch indices to encode (e.g. --epochs 24 25). Overrides the config.")
    parser.add_argument("--all-epochs", action="store_true",
                        help=(
                            "Encode the maximum available epoch range, filtering to model "
                            "directories that contain the full range."
                        ))
    parser.add_argument("--max-checkpoints", metavar="N", default=None,
                        help="Override max_checkpoints in config. Use 'null' or '0' for no limit.")
    parser.add_argument("--cache-dir", metavar="PATH",
                        help="Override data.tokens_dataset.cache_dir from the checkpoint config.")
    parser.add_argument("--num-windows", "--num-windows-per-model", dest="num_windows",
                        type=_parse_num_windows, metavar="N|auto",
                        help=(
                            "Override data.tokens_dataset.num_windows_per_model. "
                            "Use a positive integer for faster sampled export, or 'auto' for full coverage."
                        ))
    parser.add_argument("--window-strategy", choices=("consecutive", "distributed"),
                        help="Override data.tokens_dataset.window_distribution_strategy.")
    parser.add_argument("--cache-mode", choices=("none", "per_model", "per_window"), default="none",
                        help=(
                            "Override data.tokens_dataset.cache_mode. For latent export, "
                            "'none' avoids disk cache I/O for one-off exports (default), "
                            "while 'per_model' reuses one cache file per checkpoint."
                        ))
    parser.add_argument("--preload-cache", action="store_true",
                        help="Preload the token cache before encoding. By default export generates cache entries lazily.")
    parser.add_argument("--encode-batch-size", type=int, default=64, metavar="N",
                        help="Number of windows encoded per GPU forward pass (default: 64).")
    args = parser.parse_args()
    if args.encode_batch_size <= 0:
        parser.error("--encode-batch-size must be a positive integer")

    t_start = datetime.now()
    info = parse_checkpoint_path(args.checkpoint_path)
    base_out_dir = Path(args.output_dir) if args.output_dir else info["experiment_root"]

    log.info("Loading SANE checkpoint: %s", info["checkpoint_pt"])
    checkpoint = torch.load(info["checkpoint_pt"], map_location="cpu", weights_only=False)
    log.info("Checkpoint loaded — parsing config")
    params = checkpoint.get("config") or _load_trial_config(info["trial_dir"])
    _expand_checkpoints_dataset(params)

    params.pop("downstream_tasks", None)
    trainer_cfg = params.get("trainer")
    if isinstance(trainer_cfg, dict):
        trainer_cfg.pop("downstream_tasks", None)
    dataset_name = _dataset_name_from_config(params)
    log.info("Encoding mode: %s", args.mode)

    if args.epochs and args.all_epochs:
        parser.error("--epochs and --all-epochs are mutually exclusive")
    temp_zoo_roots: list[tempfile.TemporaryDirectory] = []
    if args.all_epochs:
        epochs, temp_zoo_roots = _configure_all_epochs(params)
        log.info("--all-epochs: found %d epochs, using indices %s", len(epochs), epochs)
        if args.max_checkpoints is None:
            _override_max_checkpoints(params, None)
            log.info("--all-epochs: disabled config max_checkpoints so full epoch histories are not truncated")
    elif args.epochs:
        _override_epoch_idx(params, args.epochs)
        log.info("Overriding epoch_idx from CLI: %s", args.epochs)

    if args.max_checkpoints is not None:
        limit = None if args.max_checkpoints.lower() in ("null", "none", "0") else int(args.max_checkpoints)
        _override_max_checkpoints(params, limit)
        log.info("Overriding max_checkpoints: %s", limit)

    if args.cache_dir is not None:
        _override_cache_dir(params, args.cache_dir)
        log.info("Overriding token cache directory: %s", params["data"]["tokens_dataset"]["cache_dir"])

    if args.num_windows is not None:
        _override_num_windows(params, args.num_windows)
        log.info("Overriding num_windows_per_model: %s", args.num_windows)

    if args.window_strategy is not None:
        _override_window_distribution_strategy(params, args.window_strategy)
        log.info("Overriding window_distribution_strategy: %s", args.window_strategy)

    if args.cache_mode is not None:
        _override_cache_mode(params, args.cache_mode)
        log.info("Overriding token cache mode: %s", args.cache_mode)

    disabled_drop_nan = _disable_drop_nan(params)
    if disabled_drop_nan:
        log.info(
            "Disabled drop_nan checkpoint filtering for latent export (%d config block%s)",
            disabled_drop_nan,
            "" if disabled_drop_nan == 1 else "s",
        )

    _override_cache_preload(params, args.preload_cache)
    if args.preload_cache:
        log.info("Enabled eager token-cache preload")
    else:
        log.info("Disabled eager token-cache preload for latent export; cache will be generated lazily")

    removed_augmentations = _disable_checkpoint_augmentations(params)
    if removed_augmentations:
        log.info(
            "Disabled checkpoint augmentations for latent export (%d config block%s)",
            removed_augmentations,
            "" if removed_augmentations == 1 else "s",
        )

    try:
        log.info("Setting up SANEContrastiveTrainer")
        trainer = SANEContrastiveTrainer()
        trainer.setup(params)
        log.info("Trainer setup complete")

        model_state = checkpoint["model"]
        if any(k.startswith("module.") for k in model_state):
            model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
            log.info("Stripped 'module.' prefix from DDP checkpoint")

        sane_ae = trainer.sane_ae
        sane_ae.load_state_dict(model_state)
        sane_ae.eval()
        log.info("Model weights loaded and set to eval mode")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sane_ae = sane_ae.to(device)
        log.info("Encoding on device: %s", device)

        log.info("Dataset name: %s", dataset_name)

        splits = {
            "train": trainer.trainset,
            "val": trainer.valset,
            "test": trainer.testset,
        }

        for split_name, dataset in splits.items():
            if dataset is None:
                log.info("Skipping %s split (not available)", split_name)
                continue
            out_dir = base_out_dir / f"{dataset_name}_encoded_checkpoints"
            out_dir.mkdir(parents=True, exist_ok=True)

            n_ckpts = len(dataset.checkpoints_dataset)
            n_windows = len(dataset)
            log.info("--- %s split: %d checkpoints, %d windows total ---", split_name.upper(), n_ckpts, n_windows)
            _log_split_summary(dataset, split_name)

            use_chunking = args.mode in ("windows", "token")
            suffix = "windows" if args.mode == "windows" else "tokens"
            chunk_dir_for_split = (
                out_dir / f"{dataset_name}_latent_representations_{split_name}_{suffix}_chunks"
                if use_chunking else None
            )
            t_split = datetime.now()
            model_ids, checkpoint_paths, epoch_indices, embeddings = encode_split(
                sane_ae,
                dataset,
                encode_batch_size=args.encode_batch_size,
                mode=args.mode,
                split_name=split_name,
                chunk_size=100 if use_chunking else 0,
                chunk_dir=chunk_dir_for_split,
            )
            elapsed = (datetime.now() - t_split).total_seconds()
            if use_chunking:
                chunk_paths_out: list[Path] = embeddings  # type: ignore[assignment]
                if args.mode == "windows":
                    # Per-checkpoint window tensors are small — merge into one file.
                    filename = f"{dataset_name}_latent_representations_{split_name}_windows.pt"
                    pt_path = out_dir / filename
                    log.info(
                        "%s split done in %.1fs — merging %d chunks (%d checkpoints) → %s",
                        split_name, elapsed, len(chunk_paths_out), len(model_ids), filename,
                    )
                    _merge_and_save_chunks(chunk_paths_out, pt_path, args.mode, model_ids, checkpoint_paths, epoch_indices)
                    log.info("Saved: %s", pt_path)
                    continue
                else:
                    # token mode: per-checkpoint token tensors can be hundreds of GB in
                    # total — merging into a single file would OOM.  Keep chunk files and
                    # save a manifest that records where each checkpoint lives.
                    log.info(
                        "%s split done in %.1fs — %d checkpoints saved as %d chunks in %s",
                        split_name, elapsed, len(model_ids), len(chunk_paths_out), chunk_dir_for_split,
                    )
                    payload = {
                        "mode": args.mode,
                        "model_ids": model_ids,
                        "checkpoint_paths": checkpoint_paths,
                        "epoch": epoch_indices,
                        "chunk_files": [p.name for p in chunk_paths_out],
                        "chunk_dir": str(chunk_dir_for_split),
                        "total_checkpoints": len(model_ids),
                    }
                    filename = f"{dataset_name}_latent_representations_{split_name}_tokens_manifest.pt"
            elif args.mode == "windows":
                window_counts = [tensor.shape[0] for tensor in embeddings]
                latent_dim = embeddings[0].shape[-1] if embeddings else 0
                log.info(
                    "%s split done in %.1fs — saved %d checkpoint window tensors (latent_dim=%d, windows min/max=%d/%d)",
                    split_name,
                    elapsed,
                    len(embeddings),
                    latent_dim,
                    min(window_counts, default=0),
                    max(window_counts, default=0),
                )
                payload = {
                    "mode": args.mode,
                    "model_ids": model_ids,
                    "checkpoint_paths": checkpoint_paths,
                    "epoch": epoch_indices,
                    "embeddings": embeddings,
                }
                filename = f"{dataset_name}_latent_representations_{split_name}_windows.pt"
            elif args.mode == "token":
                token_counts = [tensor.shape[0] for tensor in embeddings]
                latent_dim = embeddings[0].shape[-1] if embeddings else 0
                log.info(
                    "%s split done in %.1fs — saved %d checkpoint token tensors (latent_dim=%d, tokens min/max=%d/%d)",
                    split_name,
                    elapsed,
                    len(embeddings),
                    latent_dim,
                    min(token_counts, default=0),
                    max(token_counts, default=0),
                )
                payload = {
                    "mode": args.mode,
                    "model_ids": model_ids,
                    "checkpoint_paths": checkpoint_paths,
                    "epoch": epoch_indices,
                    "embeddings": embeddings,
                }
                filename = f"{dataset_name}_latent_representations_{split_name}_tokens.pt"
            else:
                log.info("%s split done in %.1fs — embedding matrix: %s (checkpoints x latent_dim)",
                         split_name, elapsed, embeddings.shape)
                payload = {
                    "mode": args.mode,
                    "model_ids": model_ids,
                    "checkpoint_paths": checkpoint_paths,
                    "epoch": epoch_indices,
                    "embeddings": torch.from_numpy(embeddings),
                }
                filename = f"{dataset_name}_latent_representations_{split_name}.pt"

            pt_path = out_dir / filename
            torch.save(payload, pt_path)
            log.info("Saved: %s", pt_path)
    finally:
        for temp_root in temp_zoo_roots:
            temp_root.cleanup()

    total_elapsed = (datetime.now() - t_start).total_seconds()
    log.info("All splits encoded in %.1fs total", total_elapsed)


if __name__ == "__main__":
    main()
