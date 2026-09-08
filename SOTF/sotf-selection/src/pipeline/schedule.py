from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from pipeline.batch import (
    SOTFBatchConfig,
    _DEFAULT_OUTPUT_ROOT,
    _coerce_override_list,
    _parse_predictor,
    _parse_run,
    run_batch_config,
)

# Resolved relative batch_config references against this directory.
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "config" / "experiments"


@dataclass(frozen=True)
class SOTFSweepSet:
    label: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class SOTFScheduleConfig:
    schedule_name: str
    output_root: Path
    base_overrides: tuple[str, ...]
    sweeps: tuple[SOTFSweepSet, ...]
    predictor: Any  # PredictorBatchConfig
    runs: tuple[Any, ...]  # tuple[ScheduledRun, ...]
    dry_run: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sotf-select-schedule",
        description="Run a SOTF schedule — one experiment set across multiple SOTF parameter sweeps.",
    )
    parser.add_argument("schedule_config", type=Path, help="Path to the SOTF schedule YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved runs without executing them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_schedule(args.schedule_config, dry_run=bool(args.dry_run))


def load_schedule_config(path: str | Path) -> SOTFScheduleConfig:
    cfg = OmegaConf.load(Path(path))
    if isinstance(cfg, ListConfig):
        raise ValueError("Schedule config must be a YAML mapping.")
    return parse_schedule_config(cfg)


def parse_schedule_config(cfg: DictConfig | dict[str, Any]) -> SOTFScheduleConfig:
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_cfg, dict):
        raise ValueError("Schedule config must be a YAML mapping.")

    schedule_name = str(raw_cfg.get("schedule_name", "")).strip()
    if not schedule_name:
        raise ValueError("Schedule config must define a non-empty 'schedule_name'.")

    output_root = Path(str(raw_cfg.get("output_root", _DEFAULT_OUTPUT_ROOT)))

    raw_sweeps = raw_cfg.get("sotf_sweeps")
    if not isinstance(raw_sweeps, list) or not raw_sweeps:
        raise ValueError("Schedule config must define a non-empty 'sotf_sweeps' list.")
    sweeps = tuple(_parse_sweep_set(s) for s in raw_sweeps)

    # Load the experiment definition: either from a referenced batch template or inline.
    batch_config_ref = raw_cfg.get("batch_config")
    if batch_config_ref is not None:
        base_overrides, predictor, runs = _load_batch_template(str(batch_config_ref))
    else:
        base_overrides = _coerce_override_list(
            raw_cfg.get("base_overrides", raw_cfg.get("common_overrides", {})),
            field_name="base_overrides",
        )
        predictor = _parse_predictor(raw_cfg.get("predictor"))
        raw_runs = raw_cfg.get("runs")
        if not isinstance(raw_runs, list) or not raw_runs:
            raise ValueError(
                "Schedule config must define either a 'batch_config' reference or an inline 'runs' list."
            )
        runs = tuple(_parse_run(run, common_overrides=base_overrides) for run in raw_runs)

    return SOTFScheduleConfig(
        schedule_name=schedule_name,
        output_root=output_root,
        base_overrides=base_overrides,
        sweeps=sweeps,
        predictor=predictor,
        runs=runs,
        dry_run=bool(raw_cfg.get("dry_run", False)),
    )


def _load_batch_template(
    batch_config_ref: str,
) -> tuple[tuple[str, ...], Any, tuple[Any, ...]]:
    """Load base_overrides, predictor, and runs from a batch template file.

    The path is resolved relative to the experiments config directory first;
    if not found there, it is resolved relative to the current working directory.
    """
    candidate = _EXPERIMENTS_DIR / batch_config_ref
    if not candidate.exists():
        candidate = Path(batch_config_ref)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Batch template not found: '{batch_config_ref}' "
            f"(tried {_EXPERIMENTS_DIR / batch_config_ref} and {Path(batch_config_ref).resolve()})"
        )

    raw_template = OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
    if not isinstance(raw_template, dict):
        raise ValueError(f"Batch template must be a YAML mapping: {candidate}")

    base_overrides = _coerce_override_list(
        raw_template.get("common_overrides", raw_template.get("base_overrides", {})),
        field_name="common_overrides",
    )
    predictor = _parse_predictor(raw_template.get("predictor"))

    raw_runs = raw_template.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Batch template must define a non-empty 'runs' list: {candidate}")

    runs = tuple(_parse_run(run, common_overrides=base_overrides) for run in raw_runs)
    return base_overrides, predictor, runs


def _parse_sweep_set(raw_sweep: Any) -> SOTFSweepSet:
    if not isinstance(raw_sweep, dict):
        raise ValueError("Each sotf_sweep entry must be a mapping.")
    label = str(raw_sweep.get("label", "")).strip()
    if not label:
        raise ValueError("Each sotf_sweep entry must define a non-empty 'label'.")
    overrides = _coerce_override_list(
        raw_sweep.get("overrides", {}),
        field_name=f"sotf_sweeps[{label}].overrides",
    )
    return SOTFSweepSet(label=label, overrides=overrides)


def schedule_to_batch_configs(schedule: SOTFScheduleConfig) -> list[SOTFBatchConfig]:
    """Expand a schedule into one SOTFBatchConfig per sweep."""
    configs = []
    for sweep in schedule.sweeps:
        batch_name = f"{schedule.schedule_name}/{sweep.label}"
        common_overrides = (*schedule.base_overrides, *sweep.overrides)
        configs.append(
            SOTFBatchConfig(
                batch_name=batch_name,
                output_root=schedule.output_root,
                common_overrides=common_overrides,
                predictor=schedule.predictor,
                runs=schedule.runs,
                dry_run=schedule.dry_run,
            )
        )
    return configs


def run_schedule(schedule_config_path: str | Path, *, dry_run: bool) -> None:
    schedule = load_schedule_config(schedule_config_path)
    schedule = SOTFScheduleConfig(
        schedule_name=schedule.schedule_name,
        output_root=schedule.output_root,
        base_overrides=schedule.base_overrides,
        sweeps=schedule.sweeps,
        predictor=schedule.predictor,
        runs=schedule.runs,
        dry_run=bool(dry_run),
    )
    batch_configs = schedule_to_batch_configs(schedule)
    total = len(batch_configs)
    print(f"Schedule '{schedule.schedule_name}': {total} sweep(s)")
    for i, batch_cfg in enumerate(batch_configs, start=1):
        print(f"\n=== Sweep [{i}/{total}]: {batch_cfg.batch_name} ===")
        run_batch_config(batch_cfg, config_source_path=None)


if __name__ == "__main__":
    main()
