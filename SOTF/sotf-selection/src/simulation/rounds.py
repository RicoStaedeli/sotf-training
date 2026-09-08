from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from models.predictors import PopulationPredictor

log = logging.getLogger(__name__)


def simulate_rounds(
    *,
    dataset,
    predictor: PopulationPredictor,
    selection_cfg,
    final_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all selection rounds and return detailed rows, per-round summaries, and final survivors.

    Args:
        dataset: Dataset to process.
        predictor (PopulationPredictor): PopulationPredictor value used by the function.
        selection_cfg: Configuration for selection.
        final_metrics (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Result of the function.
    """
    metadata = dataset.metadata.copy()
    target = dataset.target.copy()

    working = metadata.copy()
    working["model_id"] = working["model_id"].astype(str)
    working["epoch"] = pd.to_numeric(working["epoch"], errors="coerce")
    working = working.dropna(subset=["epoch"]).copy()
    working["current_target"] = pd.to_numeric(working["current_target"], errors="coerce")
    working["actual_future_target"] = target.loc[working.index].to_numpy(dtype="float64")

    epoch_values = decision_epochs(
        working["epoch"],
        start_epoch=int(selection_cfg.start_epoch),
        interval=int(selection_cfg.interval),
    )
    alive_ids = set(working["model_id"].astype(str).unique())
    observed_until_by_model: dict[str, float] = {}
    round_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    if not epoch_values:
        log.warning("No decision epochs matched the configured selection schedule.")

    for round_idx, decision_epoch in enumerate(epoch_values, start=1):
        alive_before = len(alive_ids)
        if alive_before <= int(selection_cfg.min_population):
            log.info("Stopping before round %d because the population is already at the minimum size", round_idx)
            break

        advance_observed_history(
            working=working,
            alive_ids=alive_ids,
            decision_epoch=float(decision_epoch),
            observed_until_by_model=observed_until_by_model,
        )

        epoch_mask = np.isclose(working["epoch"].to_numpy(dtype="float64"), float(decision_epoch))
        candidate = working.loc[epoch_mask & working["model_id"].isin(alive_ids)].copy()
        candidate_ids = set(candidate["model_id"].astype(str))
        missing_ids = sorted(alive_ids.difference(candidate_ids))

        if candidate.empty:
            log.info("Skipping decision epoch %s because no alive models have a checkpoint there", decision_epoch)
            continue

        predicted, prediction_details = predict_round(
            predictor=predictor,
            dataset=dataset,
            candidate=candidate,
            observed_until_by_model=observed_until_by_model,
        )
        candidate["predicted_future_target"] = predicted
        candidate["round_index"] = round_idx
        candidate["decision_epoch"] = float(decision_epoch)
        candidate["population_before"] = alive_before
        for detail_key, detail_value in prediction_details.items():
            candidate[detail_key] = detail_value

        target_population_after = min(
            alive_before,
            max(
                int(selection_cfg.min_population),
                int(math.floor(alive_before * float(selection_cfg.survival_rate))),
            ),
        )
        scored_keep_count = min(len(candidate), target_population_after)
        ranked = sort_for_goal(candidate, value_column="predicted_future_target", metric_goal=selection_cfg.metric_goal)
        keep_ids = set(ranked.head(scored_keep_count)["model_id"].astype(str))
        candidate["selected"] = candidate["model_id"].astype(str).isin(keep_ids)
        candidate["elimination_reason"] = np.where(candidate["selected"], "survived", "ranked_out")
        candidate["predicted_rank"] = rank_series(
            candidate["model_id"].astype(str),
            candidate["predicted_future_target"],
            selection_cfg.metric_goal,
        )
        candidate["actual_future_rank"] = rank_series(
            candidate["model_id"].astype(str),
            candidate["actual_future_target"],
            selection_cfg.metric_goal,
        )

        oracle_ranked = sort_for_goal(candidate, value_column="actual_future_target", metric_goal=selection_cfg.metric_goal)
        oracle_keep_ids = set(oracle_ranked.head(scored_keep_count)["model_id"].astype(str))
        candidate["oracle_selected"] = candidate["model_id"].astype(str).isin(oracle_keep_ids)
        candidate["oracle_best_model"] = candidate["model_id"].astype(str).eq(str(oracle_ranked.iloc[0]["model_id"]))

        if bool(selection_cfg.eliminate_models_missing_decision_epoch):
            retained_missing_ids = set(sorted(missing_ids)[: max(0, target_population_after - len(keep_ids))])
        else:
            retained_missing_ids = set(missing_ids)
        alive_after_ids = set(keep_ids).union(retained_missing_ids)
        candidate_records = candidate.reset_index().rename(columns={"index": "sample_idx"})
        missing_frame = missing_decision_frame(
            round_idx=round_idx,
            decision_epoch=decision_epoch,
            missing_ids=missing_ids,
            population_before=alive_before,
            retained_missing_ids=retained_missing_ids,
            selection_cfg=selection_cfg,
        )
        if not missing_frame.empty:
            round_frame = pd.concat([candidate_records, missing_frame], ignore_index=True, sort=False)
        else:
            round_frame = candidate_records
        alive_ids = alive_after_ids

        if "sample_idx" not in round_frame.columns:
            round_frame = round_frame.reset_index().rename(columns={"index": "sample_idx"})
        round_frame["population_after"] = len(alive_ids)
        round_frame = round_frame.merge(final_metrics, on="model_id", how="left")
        round_frame["retained_after_round"] = round_frame["model_id"].astype(str).isin(alive_ids)
        round_frames.append(round_frame)

        round_summary = {
            "round_index": round_idx,
            "decision_epoch": float(decision_epoch),
            "population_before": alive_before,
            "candidate_models": int(len(candidate_ids)),
            "missing_models": int(len(missing_ids)),
            "kept_models": int(len(alive_after_ids)),
            "scored_kept_models": int(len(keep_ids)),
            "retained_missing_models": int(len(retained_missing_ids)),
            "target_population_after": int(target_population_after),
            "population_after": int(len(alive_ids)),
            "best_predicted_model_id": str(ranked.iloc[0]["model_id"]),
            "best_predicted_value": float(ranked.iloc[0]["predicted_future_target"]),
            "oracle_best_model_id": str(oracle_ranked.iloc[0]["model_id"]),
            "oracle_best_future_value": float(oracle_ranked.iloc[0]["actual_future_target"]),
        }
        round_summary.update(prediction_details)
        summary_rows.append(round_summary)
        log_round_summary(
            round_idx=round_idx,
            decision_epoch=decision_epoch,
            round_frame=round_frame,
            summary_row=round_summary,
            top_k=int(selection_cfg.top_k_preview),
            metric_goal=str(selection_cfg.metric_goal),
        )

        if len(alive_ids) <= int(selection_cfg.min_population):
            log.info("Stopping after round %d because the minimum population size was reached", round_idx)
            break

    round_records = concat_or_empty(round_frames)
    round_summary = pd.DataFrame(summary_rows)
    ranked_final_metrics = sort_for_goal(
        final_metrics,
        value_column="final_target",
        metric_goal=selection_cfg.metric_goal,
    ).copy()
    ranked_final_metrics["full_population_final_rank"] = np.arange(1, len(ranked_final_metrics) + 1, dtype="int64")
    survivor_frame = (
        ranked_final_metrics.loc[ranked_final_metrics["model_id"].astype(str).isin(alive_ids)]
        .reset_index(drop=True)
        .copy()
    )
    survivor_frame["survivor_rank"] = np.arange(1, len(survivor_frame) + 1, dtype="int64")
    survivor_frame.insert(0, "survived", True)
    return round_records, round_summary, survivor_frame


def missing_decision_frame(
    *,
    round_idx: int,
    decision_epoch: float,
    missing_ids: list[str],
    population_before: int,
    retained_missing_ids: set[str],
    selection_cfg,
) -> pd.DataFrame:
    """Represent alive models that did not have a checkpoint at a decision epoch.

    Args:
        round_idx (int): int value used by the function.
        decision_epoch (float): float value used by the function.
        missing_ids (list[str]): list[str] value used by the function.
        population_before (int): int value used by the function.
        retained_missing_ids (set[str]): set[str] value used by the function.
        selection_cfg: Configuration for selection.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if not missing_ids:
        return pd.DataFrame()

    selected_flags = [model_id in retained_missing_ids for model_id in missing_ids]
    elimination_reason = [
        "missing_decision_epoch_kept" if selected else "missing_decision_epoch"
        for selected in selected_flags
    ]
    return pd.DataFrame(
        {
            "sample_idx": pd.Series([pd.NA] * len(missing_ids), dtype="Int64"),
            "model_id": missing_ids,
            "epoch": [float(decision_epoch)] * len(missing_ids),
            "current_target": [np.nan] * len(missing_ids),
            "actual_future_target": [np.nan] * len(missing_ids),
            "predicted_future_target": [np.nan] * len(missing_ids),
            "round_index": [round_idx] * len(missing_ids),
            "decision_epoch": [float(decision_epoch)] * len(missing_ids),
            "population_before": [population_before] * len(missing_ids),
            "selected": selected_flags,
            "elimination_reason": elimination_reason,
            "predicted_rank": [pd.NA] * len(missing_ids),
            "actual_future_rank": [pd.NA] * len(missing_ids),
            "oracle_selected": [False] * len(missing_ids),
            "oracle_best_model": [False] * len(missing_ids),
            "retained_after_round": [model_id in retained_missing_ids for model_id in missing_ids],
        }
    )


def predict_round(
    *,
    predictor: PopulationPredictor,
    dataset,
    candidate: pd.DataFrame,
    observed_until_by_model: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Call a predictor using its round-aware API when available, otherwise use plain features.

    Args:
        predictor (PopulationPredictor): PopulationPredictor value used by the function.
        dataset: Dataset to process.
        candidate (pd.DataFrame): pd.DataFrame value used by the function.
        observed_until_by_model (dict[str, float]): dict[str, float] value used by the function.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Result of the function.
    """
    predict_for_round = getattr(predictor, "predict_for_round", None)
    if callable(predict_for_round):
        return predict_for_round(
            dataset=dataset,
            candidate=candidate,
            observed_until_by_model=observed_until_by_model,
        )

    return predictor.predict(features=dataset.features.loc[candidate.index], metadata=candidate), {}


def advance_observed_history(
    *,
    working: pd.DataFrame,
    alive_ids: set[str],
    decision_epoch: float,
    observed_until_by_model: dict[str, float],
) -> None:
    """Track the latest epoch observable for each alive model before prediction.

    Args:
        working (pd.DataFrame): pd.DataFrame value used by the function.
        alive_ids (set[str]): set[str] value used by the function.
        decision_epoch (float): float value used by the function.
        observed_until_by_model (dict[str, float]): dict[str, float] value used by the function.

    Returns:
        None: None.
    """
    observed_rows = working.loc[working["epoch"] <= float(decision_epoch), ["model_id", "epoch"]].copy()
    if observed_rows.empty:
        return

    latest_observed = observed_rows.groupby("model_id", sort=False)["epoch"].max().to_dict()
    for model_id in alive_ids:
        observed_epoch = latest_observed.get(model_id)
        if observed_epoch is None:
            continue
        previous_epoch = observed_until_by_model.get(model_id, float("-inf"))
        observed_until_by_model[model_id] = max(previous_epoch, float(observed_epoch))


def final_metric_by_model(prepared) -> pd.DataFrame:
    """Extract each model's final observed target value for oracle comparisons.

    Args:
        prepared: Prepared value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    frame = prepared.metadata.copy()
    frame["model_id"] = frame["model_id"].astype(str)
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["target"] = prepared.target.to_numpy(dtype="float64")
    frame = frame.dropna(subset=["model_id", "epoch", "target"]).copy()
    frame = frame.sort_values(["model_id", "epoch"], kind="stable")
    final_rows = frame.groupby("model_id", sort=False).tail(1).copy()
    return final_rows.loc[:, ["model_id", "epoch", "target"]].rename(
        columns={"epoch": "final_epoch", "target": "final_target"}
    )


def decision_epochs(epoch_series: pd.Series, *, start_epoch: int, interval: int) -> list[float]:
    """Return dataset epochs that match the configured selection schedule.

    Args:
        epoch_series (pd.Series): pd.Series value used by the function.
        start_epoch (int): int value used by the function.
        interval (int): int value used by the function.

    Returns:
        list[float]: Result of the function.
    """
    unique_epochs = sorted(float(value) for value in pd.unique(epoch_series.dropna()))
    if interval <= 0:
        raise ValueError("selection.interval must be positive.")
    return [
        epoch
        for epoch in unique_epochs
        if epoch >= float(start_epoch) and math.isclose((epoch - float(start_epoch)) % float(interval), 0.0, abs_tol=1e-9)
    ]


def sort_for_goal(frame: pd.DataFrame, *, value_column: str, metric_goal: str) -> pd.DataFrame:
    """Sort rows so the best value for the metric goal appears first.

    Args:
        frame (pd.DataFrame): Data frame to process.
        value_column (str): str value used by the function.
        metric_goal (str): str value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    ascending = str(metric_goal).lower() == "minimize"
    return frame.sort_values(
        by=[value_column, "model_id"],
        ascending=[ascending, True],
        kind="stable",
        ignore_index=True,
    )


def rank_series(model_ids: pd.Series, values: pd.Series, metric_goal: str) -> pd.Series:
    """Rank model ids according to values and return ranks aligned to the input order.

    Args:
        model_ids (pd.Series): pd.Series value used by the function.
        values (pd.Series): Values to process.
        metric_goal (str): str value used by the function.

    Returns:
        pd.Series: Computed series.
    """
    frame = pd.DataFrame({"model_id": model_ids.astype(str), "value": pd.to_numeric(values, errors="coerce")})
    ranked = sort_for_goal(frame, value_column="value", metric_goal=metric_goal)
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype="int64")
    return model_ids.astype(str).map(ranked.set_index("model_id")["rank"])


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate round frames or return an empty details frame with the expected schema.

    Args:
        frames (list[pd.DataFrame]): list[pd.DataFrame] value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if not frames:
        return pd.DataFrame(
            columns=[
                "sample_idx",
                "model_id",
                "epoch",
                "current_target",
                "actual_future_target",
                "predicted_future_target",
                "round_index",
                "decision_epoch",
                "population_before",
                "population_after",
                "selected",
                "elimination_reason",
                "predicted_rank",
                "actual_future_rank",
                "oracle_selected",
                "oracle_best_model",
                "retained_after_round",
                "final_epoch",
                "final_target",
            ]
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def log_round_summary(
    *,
    round_idx: int,
    decision_epoch: float,
    round_frame: pd.DataFrame,
    summary_row: dict[str, Any],
    top_k: int,
    metric_goal: str,
) -> None:
    """Log compact round diagnostics plus a top-k preview for human inspection.

    Args:
        round_idx (int): int value used by the function.
        decision_epoch (float): float value used by the function.
        round_frame (pd.DataFrame): pd.DataFrame value used by the function.
        summary_row (dict[str, Any]): dict[str, Any] value used by the function.
        top_k (int): int value used by the function.
        metric_goal (str): str value used by the function.

    Returns:
        None: None.
    """
    log.info(
        "Round %d | epoch=%s | before=%d | candidates=%d | missing=%d | scored_kept=%d | retained_missing=%d | target_after=%d | after=%d",
        round_idx,
        format_epoch(decision_epoch),
        summary_row["population_before"],
        summary_row["candidate_models"],
        summary_row["missing_models"],
        summary_row["scored_kept_models"],
        summary_row["retained_missing_models"],
        summary_row["target_population_after"],
        summary_row["population_after"],
    )

    # if round_frame.empty or top_k <= 0:
    #     return

    # top_ranked = sort_for_goal(
    #     round_frame.dropna(subset=["predicted_future_target"]),
    #     value_column="predicted_future_target",
    #     metric_goal=metric_goal,
    # ).head(top_k)
    # if top_ranked.empty:
    #     return

    # preview = top_ranked.loc[
    #     :,
    #     ["model_id", "current_target", "predicted_future_target", "actual_future_target", "selected"],
    # ].to_string(index=False)
    # log.info("Round %d top predicted models:\n%s", round_idx, preview)


def format_epoch(epoch: float) -> str:
    """Render integer-like epochs without decimals while preserving fractional epochs.

    Args:
        epoch (float): float value used by the function.

    Returns:
        str: Resolved string value.
    """
    return str(int(epoch)) if math.isclose(epoch, round(epoch)) else f"{epoch:.6g}"
