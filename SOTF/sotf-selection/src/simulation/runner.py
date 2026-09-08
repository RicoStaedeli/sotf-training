from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import logging
import re
from typing import Any

from utils.artifacts import write_frame, write_json
from utils.logging_utils import setup_logging

from data.loading import (
    build_simulation_forecast_dataset,
    build_predictor_and_effective_configs,
    load_simulation_dataset,
    used_features_for_predictor,
)
from simulation.rounds import final_metric_by_model, simulate_rounds
from utils.outputs import build_run_manifest, summary_payload
from utils.settings import SimulationConfig
from utils.validation import validate_config

log = logging.getLogger(__name__)


def run_simulation(cfg: SimulationConfig) -> dict[str, Any]:
    """Run one survival-of-the-fittest simulation and persist all result artifacts.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    now = datetime.now()
    artifacts_dir = cfg.output.resolve_run_dir(now=now)
    log_file = artifacts_dir / cfg.output.logging.filename if bool(cfg.output.logging.save_to_file) else None
    setup_logging(cfg.output.logging.level, log_file=log_file)

    validate_config(cfg)
    predictor, effective_feature_cfg, effective_sequence_cfg, bundle_config = build_predictor_and_effective_configs(cfg)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _log_run_start(cfg, predictor_name=predictor.name, effective_sequence_cfg=effective_sequence_cfg, artifacts_dir=artifacts_dir)

    prepared = load_simulation_dataset(
        cfg=cfg,
        predictor=predictor,
        effective_feature_cfg=effective_feature_cfg,
        effective_sequence_cfg=effective_sequence_cfg,
    )
    forecast_dataset = build_simulation_forecast_dataset(
        prepared,
        effective_sequence_cfg=effective_sequence_cfg,
    )
    used_features = used_features_for_predictor(predictor=predictor, dataset=forecast_dataset)
    final_metrics = final_metric_by_model(prepared)
    round_records, round_summary, survivor_frame = simulate_rounds(
        dataset=forecast_dataset,
        predictor=predictor,
        selection_cfg=cfg.selection,
        final_metrics=final_metrics,
    )

    checkpoint_stride = _infer_checkpoint_stride(cfg)
    summary = summary_payload(
        cfg=cfg,
        predictor_name=predictor.name,
        round_records=round_records,
        round_summary=round_summary,
        survivor_frame=survivor_frame,
        final_metrics=final_metrics,
        checkpoint_stride=checkpoint_stride,
    )
    resolved_config = {
        "requested": cfg.to_dict(),
        "effective": {
            "features": asdict(effective_feature_cfg),
            "sequence": asdict(effective_sequence_cfg),
            "predictor": predictor.name,
        },
        "forecast_model_bundle_config": bundle_config,
    }
    run_manifest = build_run_manifest(
        cfg=cfg,
        predictor_name=predictor.name,
        effective_sequence_cfg=effective_sequence_cfg,
        artifacts_dir=artifacts_dir,
    )

    _write_artifacts(
        artifacts_dir=artifacts_dir,
        resolved_config=resolved_config,
        run_manifest=run_manifest,
        summary=summary,
        used_features=used_features,
        round_records=round_records,
        round_summary=round_summary,
        survivor_frame=survivor_frame,
    )
    _log_run_finish(summary)
    return {
        "artifacts_dir": str(artifacts_dir),
        "summary": summary,
    }


def _log_run_start(cfg: SimulationConfig, *, predictor_name: str, effective_sequence_cfg, artifacts_dir) -> None:
    """Log the high-level settings that make a simulation run identifiable.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        predictor_name (str): str value used by the function.
        effective_sequence_cfg: Configuration for effective sequence.
        artifacts_dir: Directory for the artifacts.

    Returns:
        None: None.
    """
    log.info("Starting survival-of-the-fittest simulation '%s'", cfg.output.experiment_name)
    log.info("Simulation artifacts directory: %s", artifacts_dir)
    log.info("Predictor backend: %s", predictor_name)
    log.info(
        "Selection policy: start_epoch=%d, interval=%d, survival_rate=%.3f, min_population=%d, metric_goal=%s",
        cfg.selection.start_epoch,
        cfg.selection.interval,
        cfg.selection.survival_rate,
        cfg.selection.min_population,
        cfg.selection.metric_goal,
    )
    log.info(
        "Effective sequence config: target_mode=%s, horizon=%d, max_lag=%d, min_history=%d",
        effective_sequence_cfg.target_mode,
        effective_sequence_cfg.horizon,
        effective_sequence_cfg.max_lag,
        effective_sequence_cfg.min_history,
    )


def _write_artifacts(
    *,
    artifacts_dir,
    resolved_config: dict[str, Any],
    run_manifest: dict[str, Any],
    summary: dict[str, Any],
    used_features: list[str],
    round_records,
    round_summary,
    survivor_frame,
) -> None:
    """Persist the JSON and parquet outputs that downstream scripts expect.

    Args:
        artifacts_dir: Directory for the artifacts.
        resolved_config (dict[str, Any]): Configuration for resolved.
        run_manifest (dict[str, Any]): dict[str, Any] value used by the function.
        summary (dict[str, Any]): dict[str, Any] value used by the function.
        used_features (list[str]): list[str] value used by the function.
        round_records: Round records value used by the function.
        round_summary: Round summary value used by the function.
        survivor_frame: Survivor frame value used by the function.

    Returns:
        None: None.
    """
    write_json(artifacts_dir / "resolved_config.json", resolved_config)
    write_json(artifacts_dir / "run_manifest.json", run_manifest)
    write_json(artifacts_dir / "simulation_summary.json", summary)
    write_json(
        artifacts_dir / "used_features.json",
        {
            "count": len(used_features),
            "features": used_features,
        },
    )
    write_frame(artifacts_dir / "round_details.parquet", round_records)
    write_frame(artifacts_dir / "round_summary.parquet", round_summary)
    write_frame(artifacts_dir / "final_survivors.parquet", survivor_frame)


def _infer_checkpoint_stride(cfg: SimulationConfig) -> int:
    """Return the number of real training epochs represented by each checkpoint index in the PT file.

    For datasets where only every N-th epoch is encoded (e.g. latent representations), the epoch
    index stored in the PT file (0, 1, 2, …) corresponds to real training epochs 0, N, 2N, …
    This stride is needed to express compute savings in real-epoch units so that experiments with
    different checkpoint granularities are comparable.

    When epoch_from_checkpoint_path=True the epoch values are already real epoch numbers derived
    from the checkpoint paths, so no additional stride scaling is needed.
    """
    if getattr(cfg.data, "epoch_from_checkpoint_path", False):
        return 1
    try:
        import torch

        pt_path = getattr(cfg.data, "pt_path", None)
        if not pt_path:
            return 1
        payload = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        paths: list = payload.get("checkpoint_paths", [])
        real_epochs: list[int] = []
        for p in paths[:10]:
            m = re.search(r"checkpoint_0*(\d+)", str(p))
            if m:
                real_epochs.append(int(m.group(1)))
        unique_sorted = sorted(set(real_epochs))
        if len(unique_sorted) < 2:
            return 1
        stride = unique_sorted[1] - unique_sorted[0]
        return max(1, stride)
    except Exception:
        return 1


def _log_run_finish(summary: dict[str, Any]) -> None:
    """Log the final population size and the best surviving model when available.

    Args:
        summary (dict[str, Any]): dict[str, Any] value used by the function.

    Returns:
        None: None.
    """
    log.info(
        "Simulation finished after %d rounds with %d/%d survivors",
        int(summary["rounds_executed"]),
        int(summary["final_population"]),
        int(summary["initial_population"]),
    )
    if summary["best_survivor_model_id"] is not None:
        log.info(
            "Best remaining model: %s (final target=%.6f)",
            summary["best_survivor_model_id"],
            summary["best_survivor_final_target"],
        )
