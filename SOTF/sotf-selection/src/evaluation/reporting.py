from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .shared import dataframe_to_markdown, format_rank, predictor_display_name, safe_ratio

def write_summary_report(
    *,
    dataset_summary: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    predictor_comparison: pd.DataFrame,
    round_metrics: pd.DataFrame,
    output_path: Path,
    filters: dict[str, Any],
    evaluation_id: str | None = None,
) -> None:
    """Write summary report.

    Args:
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        predictor_summary (pd.DataFrame): pd.DataFrame value used by the function.
        predictor_comparison (pd.DataFrame): pd.DataFrame value used by the function.
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        filters (dict[str, Any]): dict[str, Any] value used by the function.

    Returns:
        None: None.
    """
    if dataset_summary.empty:
        output_path.write_text("No dataset summaries available.\n", encoding="utf-8")
        return

    best_recall = dataset_summary.sort_values("oracle_top_k_recall", ascending=False).head(3)
    worst_recall = dataset_summary.sort_values("oracle_top_k_recall", ascending=True).head(3)
    best_accuracy = dataset_summary.sort_values("decision_accuracy_weighted", ascending=False).head(3)
    worst_false_kill = dataset_summary.sort_values("false_kill_rate_weighted", ascending=False).head(3)

    overall = {
        "run_count": int(len(dataset_summary)),
        "dataset_count": int(dataset_summary["dataset_label"].nunique()),
        "mean_oracle_top_k_recall": float(dataset_summary["oracle_top_k_recall"].mean()),
        "mean_decision_accuracy": float(dataset_summary["decision_accuracy_weighted"].mean()),
        "mean_keep_precision": float(dataset_summary["keep_precision_weighted"].mean()),
        "mean_keep_recall": float(dataset_summary["keep_recall_weighted"].mean()),
        "mean_roc_auc": float(dataset_summary["roc_auc_weighted"].mean()),
        "mean_false_kill_rate": float(dataset_summary["false_kill_rate_weighted"].mean()),
        "mean_prediction_r2": float(dataset_summary["prediction_r2_weighted"].mean()),
        "mean_prediction_rmse": float(dataset_summary["prediction_rmse_weighted"].mean()),
        "oracle_best_retained_rate": float(dataset_summary["retained_oracle_best"].mean()),
        "mean_best_survivor_regret": float(dataset_summary["best_survivor_regret"].mean()),
    }

    lines = [
        "# SOTF Evaluation Summary",
        "",
        *([f"- Evaluation ID: `{evaluation_id}`"] if evaluation_id else []),
        f"- Filters: `{json.dumps(filters, sort_keys=True)}`",
        f"- Runs evaluated: {overall['run_count']}",
        f"- Unique datasets covered: {overall['dataset_count']}",
        f"- Mean final population recall: {overall['mean_oracle_top_k_recall']:.3f}",
        f"- Mean decision accuracy: {overall['mean_decision_accuracy']:.3f}",
        f"- Mean keep precision: {overall['mean_keep_precision']:.3f}",
        f"- Mean keep recall: {overall['mean_keep_recall']:.3f}",
        f"- Mean ROC AUC: {overall['mean_roc_auc']:.3f}",
        f"- Mean false kill rate: {overall['mean_false_kill_rate']:.3f}",
        f"- Mean prediction R^2: {overall['mean_prediction_r2']:.3f}",
        f"- Mean prediction RMSE: {overall['mean_prediction_rmse']:.3f}",
        f"- Oracle-best retained rate: {overall['oracle_best_retained_rate']:.3f}",
        f"- Gap to true best: {overall['mean_best_survivor_regret']:.3f}",
        "",
        "## Predictor Summary",
        dataframe_to_markdown(
            predictor_summary,
            [
                "predictor",
                "dataset_count",
                "mean_oracle_top_k_recall",
                "mean_decision_accuracy_weighted",
                "mean_keep_precision_weighted",
                "mean_keep_recall_weighted",
                "mean_kill_precision_weighted",
                "mean_kill_recall_weighted",
                "mean_false_kill_rate_weighted",
                "mean_roc_auc_weighted",
                "mean_prediction_r2_weighted",
                "mean_prediction_rmse_weighted",
                "oracle_best_retained_rate",
                "mean_best_survivor_regret",
                "mean_best_survivor_rank_percentile",
            ],
        ),
        "",
        "## Best Final Population Recall",
        dataframe_to_markdown(
            best_recall,
            ["predictor", "dataset_label", "oracle_top_k_recall", "retained_oracle_best", "best_survivor_full_rank", "best_survivor_regret"],
        ),
        "",
        "## Worst Final Population Recall",
        dataframe_to_markdown(
            worst_recall,
            ["predictor", "dataset_label", "oracle_top_k_recall", "retained_oracle_best", "best_survivor_full_rank", "best_survivor_regret"],
        ),
        "",
        "## Best Round-Level Decision Accuracy",
        dataframe_to_markdown(
            best_accuracy,
            [
                "predictor",
                "dataset_label",
                "decision_accuracy_weighted",
                "keep_precision_weighted",
                "keep_recall_weighted",
                "kill_precision_weighted",
                "false_kill_rate_weighted",
            ],
        ),
        "",
        "## Highest False Kill Rate",
        dataframe_to_markdown(
            worst_false_kill,
            ["predictor", "dataset_label", "false_kill_rate_weighted", "false_keep_rate_weighted", "oracle_top_k_recall"],
        ),
    ]

    if not predictor_comparison.empty:
        lines.extend(
            [
                "",
                "## Predictor Comparison vs Baseline",
                dataframe_to_markdown(
                    predictor_comparison,
                    [
                        "dataset_label",
                        "comparison_predictor",
                        "baseline_predictor",
                        "comparison_oracle_top_k_recall",
                        "baseline_oracle_top_k_recall",
                        "delta_oracle_top_k_recall",
                        "comparison_best_survivor_regret",
                        "baseline_best_survivor_regret",
                        "delta_best_survivor_regret",
                        "comparison_false_kill_rate_weighted",
                        "baseline_false_kill_rate_weighted",
                    ],
                ),
            ]
        )

    if not round_metrics.empty:
        lines.extend(
            [
                "",
                "## Mean Metrics By Round",
                dataframe_to_markdown(
                    round_metrics.groupby(["predictor", "round_index"], as_index=False)
                    .agg(
                        decision_accuracy=("decision_accuracy", "mean"),
                        keep_precision=("keep_precision", "mean"),
                        keep_recall=("keep_recall", "mean"),
                        roc_auc=("roc_auc", "mean"),
                        false_kill_rate=("false_kill_rate", "mean"),
                        false_keep_rate=("false_keep_rate", "mean"),
                        prediction_r2=("prediction_r2", "mean"),
                        prediction_rmse=("prediction_rmse", "mean"),
                        rank_alignment=("rank_alignment", "mean"),
                    )
                    .sort_values(["predictor", "round_index"]),
                    [
                        "predictor",
                        "round_index",
                        "decision_accuracy",
                        "keep_precision",
                        "keep_recall",
                        "roc_auc",
                        "false_kill_rate",
                        "false_keep_rate",
                        "prediction_r2",
                        "prediction_rmse",
                        "rank_alignment",
                    ],
                ),
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_precision_recall_summary(predictor_summary: pd.DataFrame) -> None:
    """Print precision recall summary.

    Args:
        predictor_summary (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        None: None.
    """
    if predictor_summary.empty:
        return

    columns = [
        "predictor",
        "mean_keep_precision_weighted",
        "mean_keep_recall_weighted",
        "mean_roc_auc_weighted",
    ]
    available_columns = [column for column in columns if column in predictor_summary.columns]
    if len(available_columns) != len(columns):
        return

    printable = predictor_summary.loc[:, columns].rename(
        columns={
            "mean_keep_precision_weighted": "precision",
            "mean_keep_recall_weighted": "recall",
            "mean_roc_auc_weighted": "roc_auc",
        }
    )
    print("Keep-class precision/recall by predictor:")
    print(printable.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
