from __future__ import annotations

from pathlib import Path

from runner.batch_runner import (
    build_parser,
    build_run_config,
    build_run_dir,
    build_run_overrides,
    copy_batch_config as _copy_batch_config,
    load_batch_config,
    run_batch,
)

_BATCH_RESULTS_SUBDIR = "results"


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    args = build_parser(
        prog="metric-predict-batch",
        description="Run metric-prediction experiments (prediction/forecasting/latent) from a batch config.",
    ).parse_args()
    run_batch(
        batch_config_path=args.batch_config,
        dry_run=bool(args.dry_run),
        batch_results_subdir=_BATCH_RESULTS_SUBDIR,
    )


def copy_batch_config(batch_config_path: str | Path, batch_name: str) -> Path:
    """Copy batch config.

    Args:
        batch_config_path (str | Path): str | Path value used by the function.
        batch_name (str): str value used by the function.

    Returns:
        Path: Resolved path.
    """
    cfg = load_batch_config(batch_config_path)
    cfg = type(cfg)(
        batch_name=batch_name,
        output_root=cfg.output_root,
        common_overrides=cfg.common_overrides,
        runs=cfg.runs,
        dry_run=cfg.dry_run,
        batch_results_subdir=_BATCH_RESULTS_SUBDIR,
        output_root_explicit=cfg.output_root_explicit,
    )
    return _copy_batch_config(
        batch_config_path,
        cfg,
    )
