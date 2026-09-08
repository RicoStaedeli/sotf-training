#!/usr/bin/env python3
"""Log model size and architecture information for a metric-prediction experiment.

Usage:
    uv run python metric-prediction/scripts/log_model_info.py -cn <experiment_name>
    uv run python metric-prediction/scripts/log_model_info.py -cn <experiment_name> --outputs-dir metric-prediction/outputs
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metric-prediction" / "src"))


# ---------------------------------------------------------------------------
# Experiment lookup
# ---------------------------------------------------------------------------


def find_experiment(outputs_root: Path, experiment_name: str) -> tuple[Path, Path]:
    """Return (experiment_root, bundle_path).

    Handles two layouts:
      - <experiment_name>/model_bundle.{pkl,pt}
      - <experiment_name>/artifacts/model_bundle.{pkl,pt}
    """
    all_bundles = sorted(outputs_root.rglob("model_bundle.pkl")) + sorted(outputs_root.rglob("model_bundle.pt"))
    for bundle_path in all_bundles:
        parent = bundle_path.parent
        if parent.name == experiment_name:
            return parent, bundle_path
        if parent.name == "artifacts" and parent.parent.name == experiment_name:
            return parent.parent, bundle_path

    # Fall back: scan resolved_config.json / resolved_model_config.json
    for cfg_name in ("resolved_config.json", "resolved_model_config.json"):
        for cfg_path in outputs_root.rglob(cfg_name):
            try:
                cfg = json.loads(cfg_path.read_text())
                exp = cfg.get("output", {}).get("experiment_name") or cfg.get("experiment_name")
                if exp != experiment_name:
                    continue
                exp_dir = cfg_path.parent
                for bundle_name in ("model_bundle.pkl", "model_bundle.pt"):
                    candidate = exp_dir / bundle_name
                    if candidate.exists():
                        return exp_dir, candidate
            except Exception:
                pass

    raise FileNotFoundError(
        f"No experiment found for '{experiment_name}' under {outputs_root}"
    )


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def _count_torch_params(network: Any) -> dict[str, int]:
    total = sum(p.numel() for p in network.parameters())
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def _torch_summary_str(network: Any, input_data: Any) -> str:
    """Return torchinfo summary as a string, falling back gracefully."""
    try:
        import torchinfo
        buf = io.StringIO()
        summary = torchinfo.summary(
            network,
            input_data=input_data,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
        )
        return str(summary)
    except Exception as exc:
        return f"torchinfo summary unavailable: {exc}"


def _network_device(network: Any) -> Any:
    import torch
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _dummy_mlp_input(regressor: Any) -> Any:
    import torch
    scaler = getattr(regressor, "_feature_scaler", None)
    if scaler is not None and hasattr(scaler, "mean"):
        input_dim = len(scaler.mean)
    else:
        input_dim = 1
    device = _network_device(regressor._network)
    return torch.zeros(1, input_dim, device=device)


def _dummy_seq_input(regressor: Any) -> list[Any]:
    # LSTM and Transformer both take a single dict positional arg.
    # Wrapping in a list makes torchinfo call model(dict) instead of model(**dict).
    import torch
    input_dim = getattr(regressor, "_input_dim", 1) or 1
    scalar_dim = getattr(regressor, "_scalar_dim", 0) or 0
    seq_len = 5
    device = _network_device(regressor._network)
    return [{
        "sequences": torch.zeros(1, seq_len, input_dim, device=device),
        "lengths": torch.tensor([seq_len], device=device),
        "scalar_features": torch.zeros(1, scalar_dim, device=device),
    }]


def introspect_torch_regressor(regressor: Any, model_class: str) -> dict[str, Any]:
    network = getattr(regressor, "_network", None)
    if network is None:
        return {"error": "no _network found (model not fitted?)"}

    info: dict[str, Any] = {"architecture": repr(network)}
    info.update(_count_torch_params(network))

    if "lstm" in model_class.lower() or "transformer" in model_class.lower():
        dummy = _dummy_seq_input(regressor)
    else:
        dummy = _dummy_mlp_input(regressor)

    info["torchinfo_summary"] = _torch_summary_str(network, dummy)
    return info


def introspect_xgboost(model: Any) -> dict[str, Any]:
    params = model.get_params()
    info: dict[str, Any] = {"params": params}
    booster = None
    try:
        booster = model.get_booster()
    except Exception:
        pass
    if booster is not None:
        try:
            dump = booster.get_dump()
            info["num_trees"] = len(dump)
        except Exception:
            pass
        try:
            info["num_features"] = booster.num_features()
        except Exception:
            pass
    return info


def introspect_lightgbm(model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"params": model.get_params()}
    try:
        info["num_trees"] = model.booster_.num_trees()
    except Exception:
        pass
    try:
        info["num_features"] = model.n_features_
    except Exception:
        pass
    return info


def introspect_ridge(regressor: Any) -> dict[str, Any]:
    inner = getattr(regressor, "_model", regressor)
    info: dict[str, Any] = {}
    try:
        info["alpha"] = float(inner.alpha)
    except Exception:
        pass
    try:
        coef = inner.coef_
        info["num_features"] = int(coef.shape[-1])
        info["num_nonzero_coefficients"] = int((coef != 0).sum())
    except Exception:
        pass
    return info


def introspect_model(model: Any, model_class: str) -> dict[str, Any]:
    lower = model_class.lower()
    if any(k in lower for k in ("mlp", "lstm", "transformer")):
        return introspect_torch_regressor(model, model_class)
    if "xgb" in lower or "xgboost" in lower:
        return introspect_xgboost(model)
    if "lgbm" in lower or "lightgbm" in lower:
        return introspect_lightgbm(model)
    if "ridge" in lower:
        return introspect_ridge(model)
    return {"repr": repr(model)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_CONFIGS_ROOT = ROOT / "metric-prediction" / "src" / "config" / "configs"


def resolve_experiment_names(config_name: str) -> list[str]:
    """Return a list of experiment names from a config name.

    Accepts either:
    - A single experiment name (e.g. ``ds1_latent_xgb_horizon_5``)
    - A Hydra config path to a batch YAML (e.g. ``experiments/forecasting/latent_tokens/horizon/transformer/ds2``)
    - A direct path to a YAML file
    """
    # Direct YAML file path
    candidate = Path(config_name)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")

    yaml_path: Path | None = None
    if candidate.exists():
        yaml_path = candidate
    else:
        # Try resolving as a Hydra config path relative to the configs root
        hydra_candidate = _CONFIGS_ROOT / candidate
        if hydra_candidate.exists():
            yaml_path = hydra_candidate

    if yaml_path is not None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(yaml_path)
        raw = OmegaConf.to_container(cfg, resolve=False)
        if isinstance(raw, dict) and "runs" in raw:
            names = [str(run["experiment_name"]) for run in raw["runs"] if "experiment_name" in run]
            if not names:
                raise ValueError(f"Batch config {yaml_path} has a 'runs' list but no experiment_name entries.")
            return names
        raise ValueError(f"{yaml_path} does not look like a batch config (no 'runs' key).")

    # Plain experiment name
    return [config_name]


def process_experiment(experiment_name: str, outputs_root: Path) -> bool:
    """Process a single experiment. Returns True on success."""
    print(f"Searching for experiment '{experiment_name}' under {outputs_root} ...")
    exp_dir, bundle_path = find_experiment(outputs_root, experiment_name)
    print(f"Found: {exp_dir} (bundle: {bundle_path.name})")

    if bundle_path.suffix == ".pkl":
        with open(bundle_path, "rb") as fh:
            bundle = pickle.load(fh)
    else:
        import torch
        bundle = torch.load(bundle_path, weights_only=False)

    model = bundle["model"]
    model_class = type(model).__name__
    feature_columns: list[str] = bundle.get("feature_columns", [])

    resolved_cfg: dict[str, Any] = {}
    for cfg_name in ("resolved_config.json", "resolved_model_config.json"):
        cfg_path = bundle_path.parent / cfg_name
        if cfg_path.exists():
            resolved_cfg = json.loads(cfg_path.read_text())
            break

    print(f"Model type: {model_class}")
    model_details = introspect_model(model, model_class)

    report: dict[str, Any] = {
        "experiment_name": experiment_name,
        "experiment_dir": str(exp_dir),
        "model_class": model_class,
        "num_features": len(feature_columns),
        "model_config": resolved_cfg.get("model", {}),
        "details": model_details,
    }

    json_out = exp_dir / "model_info.json"
    json_out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Written: {json_out}")

    lines = [
        f"Experiment : {experiment_name}",
        f"Directory  : {exp_dir}",
        f"Model class: {model_class}",
        f"Features   : {len(feature_columns)}",
        "",
    ]
    if "total_parameters" in model_details:
        lines += [
            f"Total parameters    : {model_details['total_parameters']:,}",
            f"Trainable parameters: {model_details['trainable_parameters']:,}",
            "",
        ]
    if "num_trees" in model_details:
        lines.append(f"Number of trees: {model_details['num_trees']}")
    if "num_features" in model_details:
        lines.append(f"Number of features (model): {model_details['num_features']}")
    if "alpha" in model_details:
        lines.append(f"Alpha (regularisation): {model_details['alpha']}")
    if "params" in model_details:
        lines += ["", "Hyperparameters:"]
        for k, v in model_details["params"].items():
            lines.append(f"  {k}: {v}")
    if "architecture" in model_details:
        lines += ["", "Architecture:", model_details["architecture"]]
    if "torchinfo_summary" in model_details:
        lines += ["", "torchinfo summary:", model_details["torchinfo_summary"]]

    body = "\n".join(lines) + "\n"
    (exp_dir / "model_info.log").write_text(body)
    print(f"Written: {exp_dir / 'model_info.log'}")

    if "total_parameters" in model_details:
        print(f"Total parameters: {model_details['total_parameters']:,}")
        print(f"Trainable parameters: {model_details['trainable_parameters']:,}")
    elif "num_trees" in model_details:
        print(f"Number of trees: {model_details['num_trees']}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Log model info for a metric-prediction experiment.")
    parser.add_argument(
        "-cn", "--config-name", required=True, metavar="EXPERIMENT_OR_BATCH",
        help=(
            "Either a single experiment name, a Hydra batch config path "
            "(e.g. experiments/forecasting/latent_tokens/horizon/transformer/ds2), "
            "or a direct path to a batch YAML file."
        ),
    )
    parser.add_argument("--outputs-dir", default=None, metavar="DIR",
                        help="Root outputs directory to search (default: metric-prediction/outputs)")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_dir) if args.outputs_dir else ROOT / "metric-prediction" / "outputs"
    experiment_names = resolve_experiment_names(args.config_name)

    failures: list[str] = []
    for i, name in enumerate(experiment_names, 1):
        if len(experiment_names) > 1:
            print(f"\n[{i}/{len(experiment_names)}] {name}")
        try:
            process_experiment(name, outputs_root)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append(name)

    if len(experiment_names) > 1:
        print(f"\n{'All' if not failures else len(experiment_names) - len(failures)}/{len(experiment_names)} experiments completed.")
    if failures:
        print("Failed:", ", ".join(failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
