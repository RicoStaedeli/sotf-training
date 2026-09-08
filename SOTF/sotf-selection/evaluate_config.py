from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOTF_SRC = PROJECT_ROOT / "sotf-selection" / "src"
METRIC_PREDICTION_SRC = PROJECT_ROOT / "metric-prediction" / "src"

if str(SOTF_SRC) not in sys.path:
    sys.path.insert(0, str(SOTF_SRC))
if str(METRIC_PREDICTION_SRC) not in sys.path:
    sys.path.insert(1, str(METRIC_PREDICTION_SRC))

from evaluation.config_runner import run_from_config, run_from_multi_dataset_config

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "src" / "config" / "evaluation"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven SOTF evaluation. "
            "Pass a YAML config file path or name (resolved in configs/evaluation/)."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to evaluation YAML config, or bare filename resolved in configs/evaluation/.",
    )
    parser.add_argument(
        "-cn", "--config-name",
        dest="config_name",
        default=None,
        metavar="NAME",
        help="Config name or path (alternative to positional argument).",
    )
    return parser.parse_args(argv)


def resolve_config_path(name: str) -> Path:
    """Resolve a config name or path to an absolute Path.

    Resolution order:
    1. Absolute path as given.
    2. Relative to cwd.
    3. Relative to configs/evaluation/ next to this script.
    4. Same as (3) with .yaml extension appended.
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = _DEFAULT_CONFIG_DIR / p
    if candidate.exists():
        return candidate.resolve()
    if not name.endswith((".yaml", ".yml")):
        candidate_yaml = _DEFAULT_CONFIG_DIR / (name + ".yaml")
        if candidate_yaml.exists():
            return candidate_yaml.resolve()
    raise SystemExit(
        f"Config not found: {name!r}\n"
        f"Default config directory: {_DEFAULT_CONFIG_DIR}"
    )


if __name__ == "__main__":
    args = parse_args()
    config_name = args.config_name or args.config
    if not config_name:
        raise SystemExit(
            "Provide a config file.\n"
            "Usage: python evaluate_config.py ds3.yaml\n"
            "   or: python evaluate_config.py -cn ds3\n"
            f"Configs live in: {_DEFAULT_CONFIG_DIR}"
        )
    config_path = resolve_config_path(config_name)
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if isinstance(raw, dict) and "datasets" in raw:
        raise SystemExit(run_from_multi_dataset_config(config_path))
    raise SystemExit(run_from_config(config_path))
