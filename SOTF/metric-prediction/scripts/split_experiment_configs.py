#!/usr/bin/env python3
"""
Reorganize forecasting experiment configs into per-model-type subdirectories.

latent_averaged/dsN.yaml      → latent_averaged/{xgboost,ridge,mlp}/dsN.yaml
latent_tokens/final_value/    → latent_tokens/final_value/lstm/
latent_tokens/horizon/        → latent_tokens/horizon/lstm/  (transformer/ already ok)
latent_windows/final_value/   → latent_windows/final_value/lstm/
latent_windows/horizon/       → latent_windows/horizon/lstm/
weight_statistics/final_value/→ weight_statistics/final_value/xgboost/
weight_statistics/horizon/    → weight_statistics/horizon/xgboost/
"""

import yaml
import shutil
from pathlib import Path

FORECASTING = Path(
    "/Users/ricostadeli/Documents/HSG/thesis/ms-thesis"
    "/metric-prediction/src/config/configs/experiments/forecasting"
)

MODEL_TYPE_MAP = {
    "xgboost": "xgboost",
    "ridge": "ridge",
    "mlp_latent": "mlp",
    "mlp": "mlp",
    "lstm_latent_sequences": "lstm",
    "hierarchical_transformer": "transformer",
}

INDENT = "  "


def _model_type_from_run(run, common_model=None):
    overrides = run.get("overrides") or {}
    model = overrides.get("model", common_model)
    return MODEL_TYPE_MAP.get(model, "other")


def _yaml_header(src_name, model_type, description_lines):
    lines = [
        "# @package _global_",
        f"# Split from {src_name} — {model_type} runs only.",
    ]
    lines += [""] + description_lines
    return "\n".join(lines) + "\n\n"


def _dump_run(run):
    """Dump a single run dict as YAML list item."""
    lines = yaml.dump([run], default_flow_style=False, sort_keys=False).splitlines()
    return "\n".join(lines) + "\n"


def _build_file(data, runs_subset, output_root_override):
    """Return YAML string for a split batch file."""
    top = {
        "hydra": data["hydra"],
        "dry_run": data.get("dry_run", False),
    }
    meta = {}
    for k in ("batch_name", "output_root", "batch_root_subdir", "evaluation"):
        if k in data:
            meta[k] = data[k]
    if output_root_override:
        meta["output_root"] = output_root_override

    body = {
        "common_overrides": data.get("common_overrides", {}),
        "runs": runs_subset,
    }

    parts = [
        yaml.dump(top, default_flow_style=False, sort_keys=False),
        yaml.dump(meta, default_flow_style=False, sort_keys=False),
        yaml.dump(body, default_flow_style=False, sort_keys=False),
    ]
    return "\n".join(parts)


def split_latent_averaged(src_path: Path, dry_run=False):
    """Split a latent_averaged/dsN.yaml into xgboost/, ridge/, mlp/ subdirs."""
    with open(src_path) as f:
        data = yaml.safe_load(f)

    runs = data.get("runs") or []
    by_model: dict[str, list] = {}
    for run in runs:
        mt = _model_type_from_run(run)
        by_model.setdefault(mt, []).append(run)

    parent = src_path.parent
    for model_type, model_runs in sorted(by_model.items()):
        out_dir = parent / model_type
        out_path = out_dir / src_path.name
        base_root = data.get("output_root", "")
        new_root = f"{base_root}/{model_type}" if base_root else model_type

        new_data = dict(data)
        new_data["output_root"] = new_root
        new_data["runs"] = model_runs

        content = (
            f"# @package _global_\n"
            f"# Split from {src_path.name} — {model_type} runs only.\n\n"
            + yaml.dump(
                {k: new_data[k] for k in new_data if k != "runs"},
                default_flow_style=False,
                sort_keys=False,
            )
            + yaml.dump({"runs": model_runs}, default_flow_style=False, sort_keys=False)
        )

        if dry_run:
            print(f"[dry] would write {out_path}  ({len(model_runs)} runs)")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content)
            print(f"written {out_path}  ({len(model_runs)} runs)")


def move_to_model_subdir(src_path: Path, model_subdir: str, dry_run=False):
    """Move a YAML batch file into a model-type subdirectory, updating output_root."""
    with open(src_path) as f:
        data = yaml.safe_load(f)

    out_dir = src_path.parent / model_subdir
    out_path = out_dir / src_path.name

    base_root = data.get("output_root", "")
    new_root = f"{base_root}/{model_subdir}" if base_root else model_subdir
    data["output_root"] = new_root

    content = (
        f"# @package _global_\n"
        f"# Moved to {model_subdir}/ subdirectory.\n\n"
        + yaml.dump(
            {k: data[k] for k in data if k != "runs"},
            default_flow_style=False,
            sort_keys=False,
        )
        + yaml.dump({"runs": data.get("runs", [])}, default_flow_style=False, sort_keys=False)
    )

    if dry_run:
        print(f"[dry] would move {src_path} → {out_path}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        src_path.unlink()
        print(f"moved {src_path.name} → {out_path}")


def process_all(dry_run=False):
    # 1. Split latent_averaged/
    averaged_dir = FORECASTING / "latent_averaged"
    for yaml_path in sorted(averaged_dir.glob("*.yaml")):
        split_latent_averaged(yaml_path, dry_run=dry_run)

    # 2. Move latent_tokens/final_value/*.yaml → lstm/
    for yaml_path in sorted((FORECASTING / "latent_tokens" / "final_value").glob("*.yaml")):
        move_to_model_subdir(yaml_path, "lstm", dry_run=dry_run)

    # 3. Move latent_tokens/horizon/*.yaml → lstm/  (skip transformer/ — already ok)
    for yaml_path in sorted((FORECASTING / "latent_tokens" / "horizon").glob("*.yaml")):
        move_to_model_subdir(yaml_path, "lstm", dry_run=dry_run)

    # 4. Move latent_windows/final_value/*.yaml → lstm/
    for yaml_path in sorted((FORECASTING / "latent_windows" / "final_value").glob("*.yaml")):
        move_to_model_subdir(yaml_path, "lstm", dry_run=dry_run)

    # 5. Move latent_windows/horizon/*.yaml → lstm/
    for yaml_path in sorted((FORECASTING / "latent_windows" / "horizon").glob("*.yaml")):
        move_to_model_subdir(yaml_path, "lstm", dry_run=dry_run)

    # 6. Move weight_statistics/final_value/*.yaml → xgboost/
    for yaml_path in sorted((FORECASTING / "weight_statistics" / "final_value").glob("*.yaml")):
        move_to_model_subdir(yaml_path, "xgboost", dry_run=dry_run)

    # 7. Move weight_statistics/horizon/*.yaml → xgboost/
    for yaml_path in sorted((FORECASTING / "weight_statistics" / "horizon").glob("*.yaml")):
        move_to_model_subdir(yaml_path, "xgboost", dry_run=dry_run)


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        print("=== DRY RUN ===")
    process_all(dry_run=dry)
    if not dry:
        print("\nDone. Original latent_averaged/dsN.yaml files still present — delete manually after review.")
