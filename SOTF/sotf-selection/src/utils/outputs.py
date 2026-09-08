from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation.rounds import sort_for_goal
from utils.config_utils import dataset_id_from_identity_path, dataset_slug_from_path, resolve_dataset_identity_path
from utils.settings import SequenceFeatureConfig, SimulationConfig


def build_run_manifest(
    *,
    cfg: SimulationConfig,
    predictor_name: str,
    effective_sequence_cfg: SequenceFeatureConfig,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Build a compact machine-readable description of the simulation run.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        predictor_name (str): str value used by the function.
        effective_sequence_cfg (SequenceFeatureConfig): Configuration for effective sequence.
        artifacts_dir (Path): Directory for the artifacts.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    dataset_path = Path(
        resolve_dataset_identity_path(
            source=getattr(cfg.data, "source", "prepared_dataset"),
            pt_path=getattr(cfg.data, "pt_path", None),
            parquet_path=getattr(cfg.data, "parquet_path", None),
            target_parquet_path=getattr(cfg.data, "target_parquet_path", None),
        )
    )
    dataset_name = dataset_slug_from_path(dataset_path)
    dataset_id = dataset_id_from_identity_path(dataset_path)

    return {
        "run_dir": str(artifacts_dir),
        "experiment_name": cfg.output.experiment_name,
        "predictor": predictor_name,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "target_column": cfg.data.target_column,
        "target_mode": effective_sequence_cfg.target_mode,
        "horizon": int(effective_sequence_cfg.horizon),
        "selection": {
            "start_epoch": int(cfg.selection.start_epoch),
            "interval": int(cfg.selection.interval),
            "survival_rate": float(cfg.selection.survival_rate),
            "min_population": int(cfg.selection.min_population),
            "metric_goal": str(cfg.selection.metric_goal),
        },
    }


def summary_payload(
    *,
    cfg: SimulationConfig,
    predictor_name: str,
    round_records: pd.DataFrame,
    round_summary: pd.DataFrame,
    survivor_frame: pd.DataFrame,
    final_metrics: pd.DataFrame,
    checkpoint_stride: int = 1,
) -> dict[str, Any]:
    """Summarize simulation quality and survival outcomes for downstream evaluation.

    Args:
        cfg (SimulationConfig): Configuration object used by the function.
        predictor_name (str): str value used by the function.
        round_records (pd.DataFrame): pd.DataFrame value used by the function.
        round_summary (pd.DataFrame): pd.DataFrame value used by the function.
        survivor_frame (pd.DataFrame): pd.DataFrame value used by the function.
        final_metrics (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        dict[str, Any]: Dictionary containing the computed values.
    """
    initial_population = int(final_metrics["model_id"].astype(str).nunique())
    final_population = int(len(survivor_frame))
    oracle_ranking = sort_for_goal(final_metrics, value_column="final_target", metric_goal=cfg.selection.metric_goal)
    best_survivor = sort_for_goal(survivor_frame, value_column="final_target", metric_goal=cfg.selection.metric_goal)
    oracle_top_k_recall = None
    mean_survivor_rank_gap = None
    if final_population > 0:
        oracle_top_k_ids = set(oracle_ranking.head(final_population)["model_id"].astype(str))
        retained_ids = set(survivor_frame["model_id"].astype(str))
        oracle_top_k_recall = float(len(oracle_top_k_ids.intersection(retained_ids)) / final_population)
        if {"survivor_rank", "full_population_final_rank"}.issubset(survivor_frame.columns):
            rank_gap = (
                survivor_frame["full_population_final_rank"].to_numpy(dtype="float64")
                - survivor_frame["survivor_rank"].to_numpy(dtype="float64")
            )
            mean_survivor_rank_gap = float(rank_gap.mean())

    mean_abs_error = None
    if "predicted_future_target" in round_records.columns:
        valid_prediction_rows = round_records.dropna(subset=["predicted_future_target", "actual_future_target"])
        if not valid_prediction_rows.empty:
            mean_abs_error = float(
                np.abs(
                    valid_prediction_rows["actual_future_target"].to_numpy(dtype="float64")
                    - valid_prediction_rows["predicted_future_target"].to_numpy(dtype="float64")
                ).mean()
            )

    return {
        "predictor": predictor_name,
        "target_column": cfg.data.target_column,
        "decision_horizon": int(cfg.sequence.horizon),
        "selection_interval": int(cfg.selection.interval),
        "checkpoint_stride": int(checkpoint_stride),
        "rounds_executed": int(len(round_summary)),
        "initial_population": initial_population,
        "final_population": final_population,
        "oracle_best_model_id": None if oracle_ranking.empty else str(oracle_ranking.iloc[0]["model_id"]),
        "oracle_best_final_target": None if oracle_ranking.empty else float(oracle_ranking.iloc[0]["final_target"]),
        "best_survivor_model_id": None if best_survivor.empty else str(best_survivor.iloc[0]["model_id"]),
        "best_survivor_final_target": None if best_survivor.empty else float(best_survivor.iloc[0]["final_target"]),
        "retained_oracle_best": (
            False
            if oracle_ranking.empty or best_survivor.empty
            else str(oracle_ranking.iloc[0]["model_id"]) in set(survivor_frame["model_id"].astype(str))
        ),
        "oracle_top_k_recall": oracle_top_k_recall,
        "mean_survivor_rank_gap": mean_survivor_rank_gap,
        "mean_absolute_prediction_error": mean_abs_error,
    }
