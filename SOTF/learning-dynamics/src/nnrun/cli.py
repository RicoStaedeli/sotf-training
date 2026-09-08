#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# --- bootstrap: allow running as a script without installing the package ---
# This makes "src" importable, so "import nnrun..." works.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]          # .../src
_REPO_ROOT = _SRC_DIR.parent              # repo root
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


from nnrun.utils.logging import setup_logging
from nnrun.io.registry import load_dataset_registry
from nnrun.io.jsonl import load_dataset_long_df
from nnrun.reporting.report import build_report


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find repo root by walking up until we see a 'configs' and 'src' folder.
    """
    p = (start or Path.cwd()).resolve()
    for _ in range(20):
        if (p / "src").exists() and (p / "configs").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return _REPO_ROOT


def cmd_cache(args: argparse.Namespace) -> None:
    reg = load_dataset_registry(args.datasets)

    # Support: --dataset all
    # Support: --dataset ds1,ds2
    if args.dataset.strip().lower() == "all":
        dataset_names = list(reg.keys())
    else:
        dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]

    # Validate
    unknown = [d for d in dataset_names if d not in reg]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Available: {list(reg)}")

    # Cache each
    for ds_name in dataset_names:
        spec = reg[ds_name]
        df = load_dataset_long_df(
            dataset_name=spec.name,
            root_dir=spec.root_dir,
            cache_dir=args.cache_dir,
            force_rebuild=args.force,
        )
        print(f"Cached dataset '{spec.name}': rows={len(df)} cols={len(df.columns)}")


def cmd_report(args: argparse.Namespace) -> None:
    pdf_path = build_report(args.config, datasets_registry_path=args.datasets)
    print(str(pdf_path))


def main() -> None:
    repo_root = find_repo_root()
    default_log_file = repo_root / "outputs" / "logs" / "nnrun.log"

    parser = argparse.ArgumentParser(prog="nnrun", description="NN run analysis/reporting")
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--log-file", default=str(default_log_file))

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("cache", help="Build parquet cache for one dataset")
    p_cache.add_argument("--datasets", default=str(repo_root /  "configs" / "datasets.yaml"))
    p_cache.add_argument("--dataset", required=True)
    p_cache.add_argument("--cache-dir", default=str(repo_root / "outputs" / "cache"))
    p_cache.add_argument("--force", action="store_true")
    p_cache.set_defaults(func=cmd_cache)

    p_report = sub.add_parser("report", help="Build per-analysis PDF reports from config")
    p_report.add_argument("--datasets", default=None,
                          help="Path to datasets registry YAML. Required only for weight analyses that need raw checkpoints.")
    p_report.add_argument("--config", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    setup_logging(args.loglevel, log_file=args.log_file)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    args.func(args)


if __name__ == "__main__":
    main()
