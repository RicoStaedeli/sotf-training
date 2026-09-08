#!/usr/bin/env python3
"""Analyse sparse-tokenizer padding and mask structure for a given experiment config.

Walks through every step of how SparseTokenizer converts raw model weights into
fixed-size tokens, creates intra-token zero-padding, and builds the boolean mask
that MSEReconstructionLoss uses to compute compression_ratio.

Steps
-----
1.  Load & resolve the Hydra experiment config.
2.  Load checkpoints from the model zoo.
3.  Build the tokenizer and flatten one checkpoint's state dict.
4.  Per-layer token & mask arithmetic (token_factor, pad_right, mask_ratio).
5.  Per-layer mask heatmap  →  saved to output dir.
6.  Full-model tokenisation summary (total real vs. padding elements).
7.  Simulate one batched window and compute compression_ratio exactly as
    MSEReconstructionLoss does.
8.  Window-level mask heatmap with layer boundaries  →  saved to output dir.
9.  Multi-model comparison table (mask_ratio, compression_ratio, test_acc).
10. Summary bar charts  →  saved to output dir.

Example
-------
    uv run --active python scripts/analyze_tokenizer_masks.py config/5_cifar10_Resnet18_width_2-256.yaml --n-models 10 --output-dir /tmp/tokenizer_analysis
"""

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import sane
from sane.data.datasets.zoo_dataset import ZooDataset
from sane.data.tokenizers.dense import DenseTokenizer
from sane.data.tokenizers.sparse import SparseTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirrors inspect_zoo_max_positions.py conventions)
# ---------------------------------------------------------------------------

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
    name = _config_name(config_arg, config_dir)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=name)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _parse_width(model_id: str) -> int | None:
    """Extract model_width=N from a zoo model directory path."""
    m = re.search(r"model_width=(\d+)", model_id)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------

def step1_load_config(config_arg: str, zoo_path_override: str | None) -> dict[str, Any]:
    """Load and resolve the Hydra experiment config, then extract relevant fields."""
    log.info("=" * 70)
    log.info("STEP 1 — Load & resolve Hydra config: %s", config_arg)
    log.info("=" * 70)

    config = _load_experiment_config(config_arg)

    tok_cfg     = config["data"]["tokenizer"]
    zoo_cfg     = config["data"]["checkpoints_dataset"]
    ds_cfg      = config["data"]["tokens_dataset"]

    # Hydra resolves the checkpoints_dataset config group only when listed in
    # defaults:; configs that set it as a plain string in the data: block get
    # the group name instead of the loaded content — resolve it manually.
    if isinstance(zoo_cfg, str):
        zoo_yaml = _repo_root() / "config" / "data" / "checkpoints_dataset" / f"{zoo_cfg}.yaml"
        zoo_cfg = OmegaConf.to_container(OmegaConf.load(zoo_yaml), resolve=True)

    params = {
        "config_name":  Path(config_arg).stem,
        "tokensize":    tok_cfg["tokensize"],
        "tok_class":    tok_cfg["tokenizer_class"],
        "tok_mode":     tok_cfg.get("mode", "full_model"),
        "tok_padding":  tok_cfg.get("padding", "zero"),
        "window_size":  ds_cfg["window_size"],
        "latent_dim":   config["model"]["latent_dim"],
        "epoch_idx":    zoo_cfg.get("epoch_idx", -1),
        "zoo_path":     zoo_path_override or zoo_cfg["root_dir"],
    }
    params["threshold"] = params["latent_dim"] / params["tokensize"]

    log.info("  config_name  = %s", params["config_name"])
    log.info("  zoo_path     = %s", params["zoo_path"])
    log.info("  tokenizer    = %s  (tokensize=%d, mode=%s, padding=%s)",
             params["tok_class"], params["tokensize"], params["tok_mode"], params["tok_padding"])
    log.info("  window_size  = %d", params["window_size"])
    log.info("  latent_dim   = %d", params["latent_dim"])
    log.info("  epoch_idx    = %s", params["epoch_idx"])
    log.info("  threshold (compression_ratio=1): mask_ratio = %d/%d = %.4f",
             params["latent_dim"], params["tokensize"], params["threshold"])
    return params


def step2_load_zoo(params: dict, n_models: int, model_idx: int) -> tuple[ZooDataset, Any]:
    """Load the model zoo and pick one checkpoint for detailed inspection."""
    log.info("=" * 70)
    log.info("STEP 2 — Load zoo  (n_models=%d, inspect model_idx=%d)", n_models, model_idx)
    log.info("=" * 70)

    epoch_idx = params["epoch_idx"]
    epoch_idx_list = epoch_idx if isinstance(epoch_idx, list) else [epoch_idx]

    zoo = ZooDataset(
        root_dir=params["zoo_path"],
        epoch_idx=epoch_idx_list,
        max_checkpoints=n_models,
    )
    log.info("  Zoo loaded: %d checkpoints", len(zoo))

    checkpoint = zoo[model_idx]
    state_dict = checkpoint.model.state_dict()

    log.info("  Model type : %s", type(checkpoint.model).__name__)
    if checkpoint.metadata:
        eval_res = checkpoint.metadata.get("eval_results", {})
        if "test_acc" in eval_res:
            log.info("  test_acc   = %.4f", eval_res["test_acc"])

    log.info("  %-50s %-30s %10s", "Layer key", "Shape", "Params")
    log.info("  " + "-" * 93)
    total = 0
    for k, v in state_dict.items():
        n = v.numel()
        total += n
        log.info("  %-50s %-30s %10s", k, str(tuple(v.shape)), f"{n:,}")
    log.info("  " + "-" * 93)
    log.info("  %-50s %-30s %10s", "Total", "", f"{total:,}")

    return zoo, checkpoint


def step3_flatten(params: dict, state_dict: dict) -> tuple[Any, list]:
    """Build the tokenizer and flatten the state dict into per-layer 2-D matrices."""
    log.info("=" * 70)
    log.info("STEP 3 — Build tokenizer and flatten state dict")
    log.info("=" * 70)
    log.info(
        "  flatten() reshapes each weight tensor to (C_out, C_in*kH*kW) "
        "and appends the bias as a column → (C_out, F+1)"
    )

    TokenizerClass = SparseTokenizer if params["tok_class"] == "sparse" else DenseTokenizer
    tokenizer = TokenizerClass(
        tokensize=params["tokensize"],
        device="cpu",
        mode=params["tok_mode"],
        padding=params["tok_padding"],
    )
    log.info("  Using %s", TokenizerClass.__name__)

    flattened = tokenizer.flatten(state_dict)
    weight_keys = [k for k in state_dict if "weight" in k]

    log.info("  %-6s %-45s %-18s %8s", "Layer", "Key", "Flat shape", "F+bias")
    log.info("  " + "-" * 80)
    for i, (w, key) in enumerate(zip(flattened, weight_keys)):
        has_bias = key.replace("weight", "bias") in state_dict
        log.info("  %-6d %-45s %-18s %8d%s",
                 i, key, str(tuple(w.shape)), w.shape[1], "  +bias" if has_bias else "")

    return tokenizer, flattened


def step4_mask_arithmetic(params: dict, flattened: list) -> None:
    """Print per-layer token_factor, pad_right, and mask_ratio.

    For each layer, the tokenizer pads each channel row (length F) to the nearest
    multiple of tokensize:
        token_factor = ceil(F / tokensize)
        pad_right    = token_factor * tokensize - F
    The mask marks only the first F positions as True; the rest are padding.
    """
    log.info("=" * 70)
    log.info("STEP 4 — Per-layer token & mask arithmetic")
    log.info("=" * 70)

    ts  = params["tokensize"]
    thr = params["threshold"]

    log.info(
        "  tokensize=%d  latent_dim=%d  threshold=%.4f\n"
        "  (compression_ratio < 1 when mask_ratio < threshold)",
        ts, params["latent_dim"], thr,
    )
    log.info("  %3s %6s %6s %11s %10s %8s %12s  status",
             "L", "C_out", "F", "tok_factor", "pad_right", "#tokens", "mask_ratio")
    log.info("  " + "-" * 78)

    for i, w in enumerate(flattened):
        C_out, F = w.shape
        tok_factor = math.ceil(F / ts)
        pad_right  = tok_factor * ts - F
        n_tokens   = C_out * tok_factor
        mr         = F / (tok_factor * ts)
        flag = "dense" if mr >= thr else f"sparse ({mr:.3f} < {thr:.3f})"
        log.info("  %3d %6d %6d %11d %10d %8d %12.4f  %s",
                 i, C_out, F, tok_factor, pad_right, n_tokens, mr, flag)


def step5_layer_stats(tokenizer: Any, flattened: list) -> list:
    """Log per-layer mask stats (tokens, mask_ratio) and return layer_results."""
    log.info("=" * 70)
    log.info("STEP 5 — Per-layer mask statistics")
    log.info("=" * 70)

    layer_results = tokenizer.slice_by_layers(flattened)
    for i, (tok, msk, _) in enumerate(layer_results):
        real  = msk.sum().item()
        total = msk.numel()
        log.info("  Layer %2d: %d tokens, mask_ratio=%.4f  (%d/%d real)",
                 i, tok.shape[0], real / total, real, total)
    return layer_results


def step6_full_model_stats(params: dict, tokenizer: Any, state_dict: dict) -> tuple:
    """Tokenise the full model and report overall real-vs-padding counts."""
    log.info("=" * 70)
    log.info("STEP 6 — Full-model tokenisation summary")
    log.info("=" * 70)
    log.info(
        "  mode='full_model' concatenates all layer tokens into one sequence "
        "(total_tokens, tokensize)."
    )

    tokens_all, mask_all, pos_all = tokenizer.tokenize(state_dict)
    real   = mask_all.sum().item()
    total  = mask_all.numel()

    log.info("  tokens shape : %s", tuple(tokens_all.shape))
    log.info("  mask   shape : %s", tuple(mask_all.shape))
    log.info("  pos    shape : %s", tuple(pos_all.shape))
    log.info("  Total elements  : %10s", f"{total:,}")
    log.info("  Real  (True)    : %10s  (%.1f%%)", f"{real:,}", 100 * real / total)
    log.info("  Padding (False) : %10s  (%.1f%%)", f"{total-real:,}", 100 * (total - real) / total)
    log.info("  mask_ratio      : %.4f", real / total)

    return tokens_all, mask_all


def step7_window_compression(
    params: dict,
    tokens_all: torch.Tensor,
    mask_all: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Simulate one batched window and compute compression_ratio as MSEReconstructionLoss does.

    compression_ratio = mask.sum() / z.numel()
                      = mask_ratio × (tokensize / latent_dim)

    compression_ratio > 1  ↔  latent is smaller than the real input values  (true compression)
    compression_ratio < 1  ↔  latent is larger than the real input values   (expansion)
    """
    log.info("=" * 70)
    log.info("STEP 7 — One window: compression_ratio (batch_size=%d)", batch_size)
    log.info("=" * 70)

    ts          = params["tokensize"]
    ws          = params["window_size"]
    latent_dim  = params["latent_dim"]
    n_tok       = tokens_all.shape[0]

    if n_tok >= ws:
        w_tok = tokens_all[:ws]
        w_msk = mask_all[:ws]
        log.info("  Using first %d of %d tokens", ws, n_tok)
    else:
        pad_n = ws - n_tok
        w_tok = torch.cat([tokens_all, torch.zeros(pad_n, ts)])
        w_msk = torch.cat([mask_all,   torch.zeros(pad_n, ts, dtype=torch.bool)])
        log.info("  Model has %d tokens — padded %d to reach window_size=%d", n_tok, pad_n, ws)

    # Batched tensors exactly as seen by the loss function
    mask = w_msk.unsqueeze(0).expand(batch_size, -1, -1)   # (B, W, T)
    z    = torch.zeros(batch_size, ws, latent_dim)           # (B, W, D)

    mask_ratio        = mask.float().mean().item()
    compression_ratio = mask.sum().item() / z.numel()

    log.info("  x    shape : %s   (batch, window_size, tokensize)", (batch_size, ws, ts))
    log.info("  mask shape : %s", tuple(mask.shape))
    log.info("  z    shape : %s   (batch, window_size, latent_dim)", tuple(z.shape))
    log.info("  mask.sum() = %s", f"{mask.sum().item():,}")
    log.info("  z.numel()  = %s", f"{z.numel():,}")
    log.info("  mask_ratio        = %.4f  (%.1f%% real)", mask_ratio, 100 * mask_ratio)
    verdict = "> 1  compressed" if compression_ratio > 1 else "< 1  latent LARGER than input"
    log.info("  compression_ratio = %.4f  %s", compression_ratio, verdict)
    log.info(
        "  threshold = latent_dim/tokensize = %d/%d = %.4f",
        latent_dim, ts, params["threshold"],
    )
    log.info(
        "  Formula check: mask_ratio × (tokensize/latent_dim) = %.4f × %.4f = %.4f",
        mask_ratio, ts / latent_dim, mask_ratio * ts / latent_dim,
    )

    return w_msk, mask, mask_ratio, compression_ratio


def step8_reconstruction_mask_ratio(
    params: dict,
    mask: torch.Tensor,
    z: torch.Tensor,
) -> None:
    """Report mask_ratio and compression_ratio exactly as MSEReconstructionLoss does.

    Mirrors src/sane/loss/reconstruction.py lines 122-123:
        mask_ratio        = mask.sum() / mask.numel()
        compression_ratio = mask.sum() / z.numel()
    """
    log.info("=" * 70)
    log.info("STEP 8 — mask_ratio as computed in MSEReconstructionLoss")
    log.info("=" * 70)
    log.info(
        "  Replicating reconstruction.py:\n"
        "    mask_ratio        = mask.sum() / mask.numel()\n"
        "    compression_ratio = mask.sum() / z.numel()"
    )

    mask_bool = mask.to(dtype=torch.bool)
    mask_sum  = mask_bool.sum()
    mask_numel = mask_bool.numel()
    z_numel   = z.numel()

    mask_ratio        = mask_sum / mask_numel
    compression_ratio = mask_sum / z_numel

    log.info("  mask shape       : %s  (batch, window_size, tokensize)", tuple(mask_bool.shape))
    log.info("  z    shape       : %s  (batch, window_size, latent_dim)", tuple(z.shape))
    log.info("  mask.sum()       = %s", f"{mask_sum.item():,}")
    log.info("  mask.numel()     = %s", f"{mask_numel:,}")
    log.info("  z.numel()        = %s", f"{z_numel:,}")
    log.info("  mask_ratio       = %.6f  (%.2f%% real elements)",
             mask_ratio.item(), 100 * mask_ratio.item())
    verdict = "> 1  compressed" if compression_ratio.item() > 1 else "< 1  latent LARGER than input"
    log.info("  compression_ratio= %.6f  %s", compression_ratio.item(), verdict)
    log.info(
        "  (threshold for compression_ratio=1: mask_ratio must exceed %d/%d = %.4f)",
        params["latent_dim"], params["tokensize"], params["threshold"],
    )


def step_compare_widths(params: dict, batch_size: int = 4) -> list[dict]:
    """Load one checkpoint per model_width from the zoo and log tokenisation stats.

    Scans the zoo directory for subdirectories containing 'model_width=N' in their name,
    picks the first (alphabetically) directory per unique width, loads one checkpoint
    (first available epoch) for each, and logs mask_ratio / compression_ratio.
    """
    log.info("=" * 70)
    log.info("STEP — Width comparison")
    log.info("=" * 70)

    zoo_path    = Path(params["zoo_path"])
    epoch_idx   = params["epoch_idx"]
    epoch_list  = epoch_idx if isinstance(epoch_idx, list) else [epoch_idx]
    first_epoch = [epoch_list[0]]

    # One directory per unique width (first alphabetically)
    width_dirs: dict[int, Path] = {}
    for d in sorted(zoo_path.iterdir()):
        if not d.is_dir():
            continue
        w = _parse_width(d.name)
        if w is not None and w not in width_dirs:
            width_dirs[w] = d

    widths = sorted(width_dirs.keys())
    log.info("  Widths found in zoo: %s", widths)
    log.info("  Loading one checkpoint per width (epoch %s)", first_epoch)

    zoo = ZooDataset(
        root_dir=str(zoo_path),
        epoch_idx=first_epoch,
        max_checkpoints=len(widths),
    )
    log.info("  Zoo: %d checkpoints loaded", len(zoo))

    # Map each checkpoint index to its width via the model directory name
    idx_to_width: dict[int, int] = {}
    idx_to_path: dict[int, str] = {}
    for i in range(len(zoo)):
        model_id = zoo.get_model_id(i)
        w = _parse_width(model_id)
        if w is not None:
            idx_to_width[i] = w
            idx_to_path[i] = model_id

    log.info("  Selected model directories:")
    for i in sorted(idx_to_width, key=lambda k: idx_to_width[k]):
        log.info("    width=%3d  %s", idx_to_width[i], idx_to_path[i])

    ts         = params["tokensize"]
    ws         = params["window_size"]
    latent_dim = params["latent_dim"]

    rows: list[dict] = []

    for i in sorted(idx_to_width, key=lambda k: idx_to_width[k]):
        w    = idx_to_width[i]
        ckpt = zoo[i]
        sd   = ckpt.model.state_dict()

        log.info("")
        log.info("*" * 70)
        log.info("* model_width = %d  (%s)", w, idx_to_path[i])
        log.info("*" * 70)

        tokenizer, flattened = step3_flatten(params, sd)
        step4_mask_arithmetic(params, flattened)
        step5_layer_stats(tokenizer, flattened)

        tokens_all, mask_all, _ = tokenizer.tokenize(sd)

        n_tok      = tokens_all.shape[0]
        real       = mask_all.sum().item()
        total      = mask_all.numel()
        mask_ratio = real / total

        if n_tok >= ws:
            w_msk = mask_all[:ws]
        else:
            pad_n = ws - n_tok
            w_msk = torch.cat([mask_all, torch.zeros(pad_n, ts, dtype=torch.bool)])

        window_mr = w_msk.float().mean().item()
        compr     = window_mr * ts / latent_dim

        batched_mask = w_msk.unsqueeze(0).expand(batch_size, -1, -1)
        z            = torch.zeros(batch_size, ws, latent_dim)
        step8_reconstruction_mask_ratio(params, batched_mask, z)

        max_F     = max(fl.shape[1] for fl in flattened)
        n_params  = sum(v.numel() for v in sd.values())

        eval_res = (ckpt.metadata or {}).get("eval_results", {})
        test_acc = eval_res.get("test_acc", float("nan"))

        rows.append({
            "width": w, "params": n_params, "tokens": n_tok,
            "max_F": max_F, "mask_ratio": mask_ratio,
            "window_mask_ratio": window_mr, "compression_ratio": compr,
            "test_acc": test_acc,
        })

    log.info("")
    log.info("=" * 70)
    log.info("WIDTH COMPARISON SUMMARY")
    log.info("=" * 70)
    log.info("  %5s %10s %8s %6s %12s %13s %10s  status",
             "width", "params", "tokens", "max_F", "mask_ratio", "compr_ratio", "test_acc")
    log.info("  " + "-" * 82)
    for r in rows:
        flag = "compressed" if r["compression_ratio"] > 1 else "latent > input"
        log.info("  %5d %10s %8d %6d %12.4f %13.4f %10.4f  %s",
                 r["width"], f"{r['params']:,}", r["tokens"], r["max_F"],
                 r["mask_ratio"], r["compression_ratio"], r["test_acc"], flag)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Experiment config — path to a YAML file (e.g. config/5_cifar10_Resnet18_width_2-256.yaml) "
             "or a bare config name (e.g. 5_cifar10_Resnet18_width_2-256).",
    )
    parser.add_argument(
        "--zoo-path",
        default=None,
        help="Override the zoo root_dir from the config (useful when running locally "
             "and the config points to a cluster path).",
    )
    parser.add_argument(
        "--model-idx",
        type=int,
        default=0,
        help="Index of the checkpoint to inspect in detail (default: 0).",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=10,
        help="Number of checkpoints to load for the multi-model comparison (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Simulated batch size used when computing compression_ratio (default: 4).",
    )
    parser.add_argument(
        "--compare-widths",
        action="store_true",
        help="Instead of inspecting one checkpoint, load one model per width from the zoo "
             "and produce a cross-width comparison of mask_ratio and compression_ratio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tokenizer_analysis"),
        help="Directory where the log file is saved (default: outputs/tokenizer_analysis).",
    )
    return parser.parse_args()


def _setup_log_file(out_dir: Path) -> None:
    log_path = out_dir / "run.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)
    log.info("Log file: %s", log_path)


def main() -> None:
    args = parse_args()

    out_dir = args.output_dir / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_log_file(out_dir)

    # Step 1 — config (always)
    params = step1_load_config(args.config, args.zoo_path)

    if args.compare_widths:
        step_compare_widths(params, batch_size=args.batch_size)
        log.info("Done.")
        return

    # Step 2 — zoo + single checkpoint
    _, checkpoint = step2_load_zoo(params, args.n_models, args.model_idx)
    state_dict = checkpoint.model.state_dict()

    # Step 3 — tokenizer + flatten
    tokenizer, flattened = step3_flatten(params, state_dict)

    # Step 4 — per-layer arithmetic
    step4_mask_arithmetic(params, flattened)

    # Step 5 — per-layer mask statistics
    step5_layer_stats(tokenizer, flattened)

    # Step 6 — full-model stats
    tokens_all, mask_all = step6_full_model_stats(params, tokenizer, state_dict)

    # Step 7 — window compression_ratio
    w_msk, mask, mask_ratio, compression_ratio = step7_window_compression(
        params, tokens_all, mask_all, args.batch_size
    )

    # Step 8 — mask_ratio exactly as MSEReconstructionLoss computes it
    ws         = params["window_size"]
    latent_dim = params["latent_dim"]
    z = torch.zeros(args.batch_size, ws, latent_dim)
    step8_reconstruction_mask_ratio(params, mask, z)

    log.info("Done.")


if __name__ == "__main__":
    main()
