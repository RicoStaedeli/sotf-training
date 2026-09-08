from __future__ import annotations

import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15

from .shared import (
    apply_figure_titles,
    build_experiment_subtitle,
    build_model_zoo_subtitle,
    build_round_metric_table,
    clip_negative_r2_for_plot,
    contains_finite_values,
    dataset_sort_key,
    format_rank,
    predictor_display_name,
    predictor_style,
    resolve_baseline_and_comparison_predictors,
    resolve_comparison_predictor_order,
    resolve_comparison_predictors,
    round_metric_plot_max,
    sanitize_file_stem,
    shared_remaining_population_trace,
    unique_predictor_name,
)


def _save_figure(fig, output_path: Path, *, dpi: int = 250) -> None:
    """Save a figure as PNG and mirror it as PDF in a parallel pdf/ directory.

    Expects output_path to live under a directory named 'png'. Replaces that
    component with 'pdf' to derive the PDF destination. Falls back to placing
    the PDF in a sibling 'pdf/' directory when no 'png' component is found.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    parts = list(output_path.parts)
    for i in range(len(parts) - 2, -1, -1):  # skip the filename itself
        if parts[i] == "png":
            parts[i] = "pdf"
            break
    else:
        parts.insert(-1, "pdf")
    pdf_path = Path(*parts).with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _pdf_dir_for_output_dir(output_dir: Path) -> Path:
    """Return the PDF mirror directory for an output directory."""
    output_dir = Path(output_dir)
    parts = list(output_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "png":
            parts[i] = "pdf"
            return Path(*parts)
    return output_dir / "pdf"


_PREDICTOR_CATEGORY_COLORS: dict[str, str] = {
    "baseline": "#a8dadc",  # light teal — naive / functional metric baseline
    "online": "#f4a261",    # warm orange — online refitting predictors (xgboost, online_lstm)
    "offline": "#1d3557",   # dark navy   — offline pre-fitted forecast models
}
_ONLINE_PREDICTORS: frozenset[str] = frozenset({"xgboost", "online_lstm", "online_hierarchical_transformer"})
_OFFLINE_PREDICTORS: frozenset[str] = frozenset({"forecast_model"})
_CATEGORY_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "online": "Online",
    "offline": "Offline",
}


def _predictor_category(predictor_name: str) -> str:
    """Map a predictor name to its display category (baseline / online / offline)."""
    norm = str(predictor_name).strip().lower()
    if norm in _ONLINE_PREDICTORS:
        return "online"
    if norm in _OFFLINE_PREDICTORS:
        return "offline"
    return "baseline"


def add_bottom_right_legend(
    axis,
    *,
    handles: list[Line2D] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Place a legend inside the bottom-right corner of an axis."""
    legend_kwargs = {
        "frameon": False,
        "loc": "lower right",
    }
    if title:
        legend_kwargs["title"] = title
    if handles is None and labels is None:
        axis.legend(**legend_kwargs)
    elif labels is None:
        axis.legend(handles=handles, **legend_kwargs)
    else:
        axis.legend(handles, labels, **legend_kwargs)


def reserve_bottom_legend_band(axis, *, bar_centers: np.ndarray, bottom_padding: float, top_padding: float = 0.6) -> None:
    """Reserve empty vertical space below horizontal bars for an in-axis legend."""
    axis.set_ylim(float(np.max(bar_centers)) + bottom_padding, float(np.min(bar_centers)) - top_padding)


def bottom_right_legend_padding(entry_count: int, *, has_title: bool = False) -> float:
    """Compute a safe empty band for an in-axis bottom-right legend."""
    count = max(1, int(entry_count))
    padding = 0.8 + (0.8 * count) + (0.7 if has_title else 0.0)
    if has_title:
        return max(3.0, padding)
    return padding


def render_dataset_dashboard(
    dataset_summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render dataset dashboard.

    Args:
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if dataset_summary.empty:
        return

    df = dataset_summary.sort_values("dataset_number").reset_index(drop=True)
    labels = df["dataset_label"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    recall_colors = np.where(df["retained_oracle_best"].to_numpy(dtype=bool), "#2a9d8f", "#c1121f")
    axes[0, 0].bar(x, df["oracle_top_k_recall"], color=recall_colors, edgecolor="black", linewidth=0.6)
    axes[0, 0].set_title("Final Population Recall")
    axes[0, 0].set_ylabel("Final population recall")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 0].grid(axis="y", alpha=0.25)

    bar_width = 0.38
    axes[0, 1].bar(
        x - bar_width / 2,
        df["decision_accuracy_weighted"],
        width=bar_width,
        color="#457b9d",
        edgecolor="black",
        linewidth=0.6,
        label="Decision accuracy",
    )
    axes[0, 1].bar(
        x + bar_width / 2,
        df["kill_precision_weighted"],
        width=bar_width,
        color="#e9c46a",
        edgecolor="black",
        linewidth=0.6,
        label="Kill precision",
    )
    axes[0, 1].set_title("Round-Level Kill Quality")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(frameon=False, loc="lower left")

    axes[1, 0].bar(
        x - bar_width / 2,
        df["false_kill_rate_weighted"],
        width=bar_width,
        color="#d62828",
        edgecolor="black",
        linewidth=0.6,
        label="False kill rate",
    )
    axes[1, 0].bar(
        x + bar_width / 2,
        df["false_keep_rate_weighted"],
        width=bar_width,
        color="#6c757d",
        edgecolor="black",
        linewidth=0.6,
        label="False keep rate",
    )
    axes[1, 0].set_title("Where Decisions Went Wrong")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(frameon=False, loc="upper right")

    axes[1, 1].bar(
        x - bar_width / 2,
        df["best_survivor_rank_percentile"],
        width=bar_width,
        color="#2a9d8f",
        edgecolor="black",
        linewidth=0.6,
        label="Best survivor percentile",
    )
    axes[1, 1].bar(
        x + bar_width / 2,
        df["median_survivor_rank_percentile"],
        width=bar_width,
        color="#8ab17d",
        edgecolor="black",
        linewidth=0.6,
        label="Median survivor percentile",
    )
    axes[1, 1].set_title("Final Survivor Quality")
    axes[1, 1].set_ylabel("Percentile in full population")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(frameon=False, loc="lower left")

    layout_top = apply_figure_titles(fig, title=title, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_round_heatmaps(
    round_metrics: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render round heatmaps.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if round_metrics.empty:
        return

    heatmaps = [
        ("decision_accuracy", "Decision Accuracy", "YlGnBu", 0.0, 1.0),
        ("roc_auc", "ROC AUC", "YlGnBu", 0.0, 1.0),
        ("false_kill_rate", "False Kill Rate", "OrRd", 0.0, 1.0),
        ("rank_alignment", "Rank Alignment", "PuBu", -1.0, 1.0),
    ]

    fig, axes = plt.subplots(len(heatmaps), 1, figsize=(14, 19))
    if len(heatmaps) == 1:
        axes = [axes]

    for axis, (metric_column, panel_title, cmap, vmin, vmax) in zip(axes, heatmaps, strict=True):
        table = build_round_metric_table(round_metrics, metric_column=metric_column)
        render_heatmap(
            axis=axis,
            table=table,
            title=panel_title,
            cmap_name=cmap,
            vmin=vmin,
            vmax=vmax,
            value_format="{:.2f}",
        )

    layout_top = apply_figure_titles(fig, title=title, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_prediction_quality_heatmaps(
    round_metrics: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render prediction quality heatmaps.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if round_metrics.empty:
        return

    heatmaps: list[tuple[pd.DataFrame, str, str, float | None, float | None, str]] = []

    r2_table = clip_negative_r2_for_plot(build_round_metric_table(round_metrics, metric_column="prediction_r2"))
    if contains_finite_values(r2_table):
        heatmaps.append((r2_table, r"Prediction $R^2$", "viridis", 0.0, 1.0, "{:.2f}"))

    rmse_table = build_round_metric_table(round_metrics, metric_column="prediction_rmse")
    if contains_finite_values(rmse_table):
        heatmaps.append(
            (
                rmse_table,
                "Prediction RMSE",
                "viridis_r",
                0.0,
                round_metric_plot_max(rmse_table, default=0.3),
                "{:.3f}",
            )
        )

    if not heatmaps:
        return

    fig, axes = plt.subplots(len(heatmaps), 1, figsize=(14, 6.0 * len(heatmaps) + 2.0))
    if len(heatmaps) == 1:
        axes = [axes]

    for axis, (table, panel_title, cmap, vmin, vmax, value_format) in zip(axes, heatmaps, strict=True):
        render_heatmap(
            axis=axis,
            table=table,
            title=panel_title,
            cmap_name=cmap,
            vmin=vmin,
            vmax=vmax,
            value_format=value_format,
        )

    layout_top = apply_figure_titles(fig, title=title, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_dataset_small_multiples(
    round_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render dataset small multiples.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if round_metrics.empty or dataset_summary.empty:
        return

    dataset_order = dataset_summary.sort_values("dataset_number")["dataset_label"].tolist()
    column_count = 3
    row_count = math.ceil(len(dataset_order) / column_count)
    fig, axes = plt.subplots(row_count, column_count, figsize=(16, 3.8 * row_count), sharex=False, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(row_count, column_count)
    flat_axes = list(axes_array.flat)

    for axis in flat_axes[len(dataset_order) :]:
        axis.axis("off")

    for axis, dataset_label in zip(flat_axes, dataset_order, strict=False):
        round_df = round_metrics[round_metrics["dataset_label"] == dataset_label].sort_values("round_index")
        summary_row = dataset_summary.loc[dataset_summary["dataset_label"] == dataset_label].iloc[0]

        axis.plot(round_df["decision_epoch"], round_df["decision_accuracy"], marker="o", color="#1d3557", label="Decision acc")
        axis.plot(round_df["decision_epoch"], round_df["false_kill_rate"], marker="s", color="#d62828", label="False kill")
        axis.plot(round_df["decision_epoch"], round_df["false_keep_rate"], marker="^", color="#6c757d", label="False keep")
        remaining_axis = axis.twinx()
        remaining_axis.step(
            round_df["decision_epoch"],
            round_df["policy_keep_count"],
            where="post",
            color="#8d99ae",
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
            label="Number of models remaining",
        )
        remaining_axis.set_ylabel("Remaining", color="#8d99ae")
        remaining_axis.tick_params(axis="y", colors="#8d99ae", labelsize=8)
        axis.set_title(
            f"{dataset_label}  recall={summary_row['oracle_top_k_recall']:.2f}  best_rank={format_rank(summary_row['best_survivor_full_rank'])}"
        )
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Decision epoch")
        axis.set_ylabel("Rate")

    handles = [
        Line2D([0], [0], color="#1d3557", marker="o", linewidth=2.0, label="Decision acc"),
        Line2D([0], [0], color="#d62828", marker="s", linewidth=2.0, label="False kill"),
        Line2D([0], [0], color="#6c757d", marker="^", linewidth=2.0, label="False keep"),
        Line2D([0], [0], color="#8d99ae", linestyle="--", linewidth=1.6, label="Number of models remaining"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    layout_top = apply_figure_titles(fig, title=title, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
    fig.tight_layout(rect=(0, 0.04, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_roc_curve_overview(
    roc_curve_points: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render roc curve overview.

    Args:
        roc_curve_points (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if roc_curve_points.empty:
        return

    dataset_order = (
        roc_curve_points.loc[:, ["dataset_label", "dataset_number"]]
        .drop_duplicates()
        .sort_values("dataset_number")["dataset_label"]
        .tolist()
    )
    column_count = 3
    row_count = math.ceil(len(dataset_order) / column_count)
    fig, axes = plt.subplots(row_count, column_count, figsize=(15, 4.2 * row_count), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(row_count, column_count)
    flat_axes = list(axes_array.flat)

    for axis in flat_axes[len(dataset_order) :]:
        axis.axis("off")

    for axis, dataset_label in zip(flat_axes, dataset_order, strict=False):
        dataset_curve = roc_curve_points[roc_curve_points["dataset_label"] == dataset_label].sort_values(["fpr", "tpr"])
        if dataset_curve.empty:
            axis.axis("off")
            continue
        roc_auc = float(dataset_curve["roc_auc"].iloc[0])
        positive_count = int(dataset_curve["positive_count"].iloc[0])
        negative_count = int(dataset_curve["negative_count"].iloc[0])
        axis.plot(
            dataset_curve["fpr"],
            dataset_curve["tpr"],
            color="#1d3557",
            linewidth=2.0,
            label=f"AUC={roc_auc:.3f}",
        )
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#6c757d", linestyle=":", linewidth=1.2)
        axis.set_title(f"{dataset_label}  pos={positive_count} neg={negative_count}")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="lower right")

    layout_top = apply_figure_titles(fig, title=title, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_roc_curves_by_dataset(
    roc_curve_points: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render roc curves by dataset.

    Args:
        roc_curve_points (pd.DataFrame): pd.DataFrame value used by the function.
        output_dir (Path): Directory where generated outputs are written.
        title_prefix (str): str value used by the function.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if roc_curve_points.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = (
        roc_curve_points.loc[:, ["dataset_label", "dataset_number"]]
        .drop_duplicates()
        .sort_values("dataset_number")["dataset_label"]
        .tolist()
    )

    for dataset_label in dataset_order:
        dataset_curve = roc_curve_points[roc_curve_points["dataset_label"] == dataset_label].sort_values(["fpr", "tpr"])
        if dataset_curve.empty:
            continue

        fig, axis = plt.subplots(figsize=(7, 6))
        plot_single_roc_curve(axis=axis, curve=dataset_curve, color="#1d3557")
        positive_count = int(dataset_curve["positive_count"].iloc[0])
        negative_count = int(dataset_curve["negative_count"].iloc[0])
        plot_subtitle = f"{subtitle} | pos={positive_count} neg={negative_count}" if subtitle else f"pos={positive_count} neg={negative_count}"
        set_axes_title(
            axis,
            title=f"{title_prefix} - {dataset_label}",
            subtitle=plot_subtitle,
            secondary_subtitle=secondary_subtitle,
        )
        fig.tight_layout()
        _save_figure(fig, output_dir / f"{sanitize_file_stem(dataset_label)}_roc_curve.png")


def render_roc_curve_comparison_overview(
    *,
    roc_curve_points: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    output_path: Path,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render roc curve comparison overview.

    Args:
        roc_curve_points (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if roc_curve_points.empty or dataset_summary.empty:
        return

    predictor_order = resolve_comparison_predictor_order(dataset_summary["predictor"])
    if len(predictor_order) < 2:
        return

    required_predictors = set(predictor_order)
    shared_datasets = (
        dataset_summary.groupby("dataset_label")["predictor"]
        .agg(lambda values: required_predictors.issubset(set(values.astype(str))))
    )
    shared_labels = shared_datasets[shared_datasets].index.tolist()
    if not shared_labels:
        return

    plot_points = roc_curve_points[
        roc_curve_points["dataset_label"].isin(shared_labels)
        & roc_curve_points["predictor"].isin(required_predictors)
    ].copy()
    if plot_points.empty:
        return

    dataset_order = (
        plot_points.loc[:, ["dataset_label", "dataset_number"]]
        .drop_duplicates()
        .sort_values("dataset_number")["dataset_label"]
        .tolist()
    )
    column_count = 4
    row_count = math.ceil(len(dataset_order) / column_count)
    fig, axes = plt.subplots(row_count, column_count, figsize=(20, 4.0 * row_count), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(row_count, column_count)
    flat_axes = list(axes_array.flat)

    for axis in flat_axes[len(dataset_order) :]:
        axis.axis("off")

    for axis, dataset_label in zip(flat_axes, dataset_order, strict=False):
        dataset_points = plot_points[plot_points["dataset_label"] == dataset_label].copy()
        for predictor_index, predictor_name in enumerate(predictor_order):
            predictor_curve = dataset_points[dataset_points["predictor"] == predictor_name].sort_values(["fpr", "tpr"])
            if predictor_curve.empty:
                continue
            style = predictor_style(predictor_name, role_index=predictor_index)
            roc_auc = float(predictor_curve["roc_auc"].iloc[0])
            axis.plot(
                predictor_curve["fpr"],
                predictor_curve["tpr"],
                color=style["color"],
                linewidth=2.0,
                label=f"{style['label']} AUC={roc_auc:.3f}",
            )
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#6c757d", linestyle=":", linewidth=1.2)
        axis.set_title(dataset_label)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="lower right", fontsize=8)

    model_title = (
        " vs ".join(predictor_display_name(name) for name in predictor_order)
        if len(predictor_order) <= 3
        else ", ".join(predictor_display_name(name) for name in predictor_order)
    )
    layout_top = apply_figure_titles(
        fig,
        title=f"ROC Curves: {model_title}",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_roc_curve_comparison_by_dataset(
    *,
    roc_curve_points: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    output_dir: Path,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render roc curve comparison by dataset.

    Args:
        roc_curve_points (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        output_dir (Path): Directory where generated outputs are written.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if roc_curve_points.empty or dataset_summary.empty:
        return

    predictor_order = resolve_comparison_predictor_order(dataset_summary["predictor"])
    if len(predictor_order) < 2:
        return

    required_predictors = set(predictor_order)
    shared_datasets = (
        dataset_summary.groupby("dataset_label")["predictor"]
        .agg(lambda values: required_predictors.issubset(set(values.astype(str))))
    )
    shared_labels = shared_datasets[shared_datasets].index.tolist()
    if not shared_labels:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_points = roc_curve_points[
        roc_curve_points["dataset_label"].isin(shared_labels)
        & roc_curve_points["predictor"].isin(required_predictors)
    ].copy()
    if plot_points.empty:
        return

    dataset_order = (
        plot_points.loc[:, ["dataset_label", "dataset_number"]]
        .drop_duplicates()
        .sort_values("dataset_number")["dataset_label"]
        .tolist()
    )
    model_title = (
        " vs ".join(predictor_display_name(name) for name in predictor_order)
        if len(predictor_order) <= 3
        else ", ".join(predictor_display_name(name) for name in predictor_order)
    )

    for dataset_label in dataset_order:
        dataset_points = plot_points[plot_points["dataset_label"] == dataset_label].copy()
        if dataset_points.empty:
            continue

        fig, axis = plt.subplots(figsize=(7.5, 6))
        for predictor_index, predictor_name in enumerate(predictor_order):
            predictor_curve = dataset_points[dataset_points["predictor"] == predictor_name].sort_values(["fpr", "tpr"])
            if predictor_curve.empty:
                continue
            style = predictor_style(predictor_name, role_index=predictor_index)
            plot_single_roc_curve(
                axis=axis,
                curve=predictor_curve,
                color=style["color"],
                label_prefix=style["label"],
                draw_reference=predictor_index == 0,
            )

        set_axes_title(axis, title=dataset_label, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
        fig.tight_layout()
        _save_figure(fig, output_dir / f"{sanitize_file_stem(dataset_label)}_roc_curve_comparison.png")


def plot_single_roc_curve(
    *,
    axis,
    curve: pd.DataFrame,
    color: str,
    label_prefix: str | None = None,
    draw_reference: bool = True,
) -> None:
    """Plot single roc curve.

    Args:
        axis: Matplotlib axis to update.
        curve (pd.DataFrame): pd.DataFrame value used by the function.
        color (str): str value used by the function.
        label_prefix (str | None): str | None value used by the function.
        draw_reference (bool): bool value used by the function.

    Returns:
        None: None.
    """
    roc_auc = float(curve["roc_auc"].iloc[0])
    label = f"AUC={roc_auc:.3f}" if label_prefix is None else f"{label_prefix} AUC={roc_auc:.3f}"
    axis.plot(curve["fpr"], curve["tpr"], color=color, linewidth=2.0, label=label)
    if draw_reference:
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#6c757d", linestyle=":", linewidth=1.2)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")


def set_axes_title(axis, *, title: str, subtitle: str, secondary_subtitle: str = "") -> None:
    """Set axes title.

    Args:
        axis: Matplotlib axis to update.
        title (str): Title text for the output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    axis.set_title(title, pad=40 if secondary_subtitle else 26)
    if subtitle and secondary_subtitle:
        axis.text(
            0.5,
            1.035,
            subtitle,
            ha="center",
            va="bottom",
            transform=axis.transAxes,
            fontsize=9,
            color="#555555",
        )
        axis.text(
            0.5,
            1.005,
            secondary_subtitle,
            ha="center",
            va="bottom",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#555555",
        )
        return
    if subtitle:
        axis.text(
            0.5,
            1.005,
            subtitle,
            ha="center",
            va="bottom",
            transform=axis.transAxes,
            fontsize=9,
            color="#555555",
        )
    elif secondary_subtitle:
        axis.text(
            0.5,
            1.005,
            secondary_subtitle,
            ha="center",
            va="bottom",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#555555",
        )


def render_predictor_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    *,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render predictor comparison.

    Args:
        comparison_df (pd.DataFrame): Data frame containing comparison values.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if comparison_df.empty:
        return

    df = comparison_df.sort_values("dataset_number").reset_index(drop=True)
    labels = df["dataset_label"].tolist()
    x = np.arange(len(df))
    bar_width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    comparison_specs = [
        ("forecast_oracle_top_k_recall", "naive_oracle_top_k_recall", "Final Population Recall", (0.0, 1.05)),
        ("forecast_decision_accuracy_weighted", "naive_decision_accuracy_weighted", "Decision Accuracy", (0.0, 1.05)),
        ("forecast_false_kill_rate_weighted", "naive_false_kill_rate_weighted", "False Kill Rate", (0.0, 1.05)),
        (
            "forecast_best_survivor_rank_percentile",
            "naive_best_survivor_rank_percentile",
            "Best Survivor Percentile",
            (0.0, 1.05),
        ),
    ]

    for axis, (forecast_col, naive_col, panel_title, y_limits) in zip(axes.flat, comparison_specs, strict=True):
        axis.bar(
            x - bar_width / 2,
            df[forecast_col],
            width=bar_width,
            color="#1d3557",
            edgecolor="black",
            linewidth=0.6,
            label=predictor_display_name("forecast_model"),
        )
        axis.bar(
            x + bar_width / 2,
            df[naive_col],
            width=bar_width,
            color="#a8dadc",
            edgecolor="black",
            linewidth=0.6,
            label=predictor_display_name("naive"),
        )
        axis.set_title(panel_title)
        axis.set_ylim(*y_limits)
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_xlabel("Dataset")
    axes[1, 1].set_xlabel("Dataset")
    axes[0, 0].legend(frameon=False, loc="lower left")
    layout_top = apply_figure_titles(
        fig,
        title=f"{predictor_display_name('forecast_model')} vs {predictor_display_name('naive')}",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_best_survivor_regret_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    *,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render best survivor regret comparison.

    Args:
        comparison_df (pd.DataFrame): Data frame containing comparison values.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    if comparison_df.empty:
        return

    predictors = pd.Index(comparison_df.get("comparison_predictor", pd.Series(dtype="object"))).append(
        pd.Index(comparison_df.get("baseline_predictor", pd.Series(dtype="object")))
    )
    predictor_order = resolve_comparison_predictor_order(predictors)
    if len(predictor_order) < 2:
        return

    comparison_rows = comparison_df.loc[:, ["dataset_number", "dataset_label", "comparison_predictor", "comparison_best_survivor_regret"]].rename(
        columns={
            "comparison_predictor": "predictor",
            "comparison_best_survivor_regret": "best_survivor_regret",
        }
    )
    baseline_rows = comparison_df.loc[:, ["dataset_number", "dataset_label", "baseline_predictor", "baseline_best_survivor_regret"]].rename(
        columns={
            "baseline_predictor": "predictor",
            "baseline_best_survivor_regret": "best_survivor_regret",
        }
    )
    plot_df = pd.concat([comparison_rows, baseline_rows], ignore_index=True).drop_duplicates(
        subset=["dataset_number", "dataset_label", "predictor"]
    )
    if plot_df.empty:
        return

    dataset_order_df = plot_df.loc[:, ["dataset_number", "dataset_label"]].drop_duplicates().sort_values("dataset_number")
    labels = dataset_order_df["dataset_label"].tolist()
    y = np.arange(len(labels))
    predictor_count = len(predictor_order)
    group_height = 0.8
    bar_height = group_height / max(1, predictor_count)
    styles = {
        predictor_name: predictor_style(predictor_name, role_index=index)
        for index, predictor_name in enumerate(predictor_order)
    }

    fig_height = max(6.0, 0.5 * len(labels) + 2.0)
    fig, axis = plt.subplots(figsize=(9, fig_height))
    for predictor_index, predictor_name in enumerate(predictor_order):
        style = styles[predictor_name]
        predictor_values = (
            plot_df[plot_df["predictor"] == predictor_name]
            .set_index("dataset_label")
            .reindex(labels)["best_survivor_regret"]
            .clip(lower=0)
            .to_numpy(dtype="float64")
        )
        offset = (predictor_index - (predictor_count - 1) / 2.0) * bar_height
        axis.barh(
            y + offset,
            predictor_values,
            height=bar_height * 0.95,
            color=style["color"],
            edgecolor="black",
            linewidth=0.6,
            label=style["label"],
        )

    axis.set_title("Gap to True Best Final Model")
    axis.set_xlabel("Gap to true best")
    axis.set_ylabel("Dataset")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    bar_centers = np.concatenate(
        [
            y + (predictor_index - (predictor_count - 1) / 2.0) * bar_height
            for predictor_index in range(predictor_count)
        ]
    )
    reserve_bottom_legend_band(
        axis,
        bar_centers=bar_centers,
        bottom_padding=bottom_right_legend_padding(predictor_count),
    )
    axis.set_xlim(left=0)
    axis.grid(axis="x", alpha=0.25)
    add_bottom_right_legend(axis)

    layout_top = apply_figure_titles(
        fig,
        title=f"Best Survivor Gap",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def build_best_survivor_regret_by_experiment(dataset_summary: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build experiment-level best survivor regret summary for plotting.

    Args:
        dataset_summary (pd.DataFrame): Data frame containing per-run summaries.

    Returns:
        tuple[pd.DataFrame, list[str]]: Aggregated experiment summary and predictor order.
    """
    required_columns = {"experiment_name", "predictor", "best_survivor_regret"}
    if dataset_summary.empty or not required_columns.issubset(dataset_summary.columns):
        return pd.DataFrame(), []

    select_cols = ["experiment_name", "predictor", "best_survivor_regret"]
    if "experiment_order" in dataset_summary.columns:
        select_cols.append("experiment_order")
    if "compute_savings_fraction" in dataset_summary.columns:
        select_cols.append("compute_savings_fraction")
    plot_df = dataset_summary.loc[:, select_cols].copy()
    plot_df = plot_df.dropna(subset=["experiment_name", "predictor"]).reset_index(drop=True)
    if plot_df.empty:
        return pd.DataFrame(), []

    plot_df["experiment_name"] = plot_df["experiment_name"].astype(str)
    plot_df["predictor"] = plot_df["predictor"].astype(str)

    first_seen_order = (
        plot_df.loc[:, ["experiment_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(first_seen_order=lambda frame: np.arange(len(frame), dtype="int64"))
    )
    if "experiment_order" in plot_df.columns:
        plot_df["experiment_order"] = pd.to_numeric(plot_df["experiment_order"], errors="coerce")
        experiment_order = (
            plot_df.groupby("experiment_name", as_index=False, sort=False)["experiment_order"]
            .min()
            .merge(first_seen_order, on="experiment_name", how="right")
        )
        known_orders = experiment_order["experiment_order"].dropna()
        fallback_start = int(known_orders.max()) + 1 if not known_orders.empty else 0
        experiment_order["experiment_order"] = experiment_order["experiment_order"].fillna(
            fallback_start + experiment_order["first_seen_order"]
        )
        experiment_order = experiment_order.drop(columns=["first_seen_order"])
    else:
        experiment_order = first_seen_order.rename(columns={"first_seen_order": "experiment_order"})
    agg_spec: dict = {
        "mean_best_survivor_regret": ("best_survivor_regret", "mean"),
        "dataset_count": ("best_survivor_regret", "size"),
    }
    if "compute_savings_fraction" in plot_df.columns:
        agg_spec["compute_savings_fraction"] = ("compute_savings_fraction", "mean")
    aggregated = (
        plot_df.groupby(["experiment_name", "predictor"], as_index=False, sort=False)
        .agg(**agg_spec)
        .merge(experiment_order, on="experiment_name", how="left")
        .sort_values(["experiment_order", "predictor"], kind="stable")
        .reset_index(drop=True)
    )
    predictor_order = resolve_comparison_predictor_order(aggregated["predictor"])
    if not predictor_order:
        predictor_order = sorted(aggregated["predictor"].dropna().astype(str).unique().tolist())
    return aggregated, predictor_order


def _compute_savings_title_suffix(plot_rows: pd.DataFrame) -> str:
    """Return '≈40% compute saved' when all experiments share the same savings value, else ''."""
    if "compute_savings_fraction" not in plot_rows.columns:
        return ""
    values = pd.to_numeric(plot_rows["compute_savings_fraction"], errors="coerce").dropna()
    if values.empty:
        return ""
    unique_rounded = values.round(2).unique()
    if len(unique_rounded) > 1:
        log.warning(
            "Experiments in this batch have different compute savings fractions: %s. "
            "Expected all experiments to share the same selection configuration.",
            [f"{v:.2f}" for v in sorted(unique_rounded)],
        )
        return ""
    return f"≈{unique_rounded[0] * 100:.0f}% compute saved"


def render_best_survivor_regret_comparison_by_experiment(
    dataset_summary: pd.DataFrame,
    output_path: Path,
    *,
    subtitle: str,
    secondary_subtitle: str = "",
) -> Path | None:
    """Render a single experiment-level best survivor regret comparison plot.

    Args:
        dataset_summary (pd.DataFrame): Data frame containing per-run summaries.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.

    Returns:
        Path | None: Output path when a plot is generated.
    """
    aggregated, predictor_order = build_best_survivor_regret_by_experiment(dataset_summary)
    if aggregated.empty or not predictor_order:
        return None

    plot_rows = (
        aggregated.sort_values(["experiment_order", "predictor"], kind="stable")
        .drop_duplicates(subset=["experiment_name"], keep="first")
        .assign(
            baseline_sort=lambda frame: (~frame["predictor"].astype(str).eq("naive")).astype("int64"),
        )
        .sort_values(
            ["baseline_sort", "experiment_order", "experiment_name"],
            ascending=[True, True, True],
            kind="stable",
        )
        .drop(columns=["baseline_sort"])
        .reset_index(drop=True)
    )
    if plot_rows.empty:
        return None

    labels = plot_rows["experiment_name"].astype(str).tolist()
    y = np.arange(len(labels))
    bar_categories = [_predictor_category(p) for p in plot_rows["predictor"].astype(str)]
    colors = [_PREDICTOR_CATEGORY_COLORS[cat] for cat in bar_categories]

    savings_suffix = _compute_savings_title_suffix(plot_rows)
    effective_subtitle = f"{subtitle} | {savings_suffix.strip()}" if subtitle and savings_suffix else (savings_suffix.strip() or subtitle)

    fig_height = max(6.0, 0.5 * len(labels) + 2.0)
    fig, axis = plt.subplots(figsize=(11, fig_height))
    axis.barh(
        y,
        plot_rows["mean_best_survivor_regret"].clip(lower=0).to_numpy(dtype="float64"),
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    axis.set_title("Gap to True Best Final Model")
    axis.set_xlabel("Gap to true best")
    axis.set_ylabel("Experiment")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.grid(axis="x", alpha=0.25)

    # Build legend — only show categories that are present in the data
    present_categories = dict.fromkeys(bar_categories)  # ordered, deduplicated
    legend_order = [cat for cat in ("baseline", "online", "offline") if cat in present_categories]
    if legend_order:
        legend_handles = [
            Patch(
                facecolor=_PREDICTOR_CATEGORY_COLORS[cat],
                edgecolor="black",
                linewidth=0.6,
                label=_CATEGORY_LABELS[cat],
            )
            for cat in legend_order
        ]
        reserve_bottom_legend_band(
            axis,
            bar_centers=y,
            bottom_padding=bottom_right_legend_padding(len(legend_handles)),
        )
        add_bottom_right_legend(
            axis,
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
        )

    layout_top = apply_figure_titles(
        fig,
        title="Best Survivor Gap by Experiment",
        subtitle=effective_subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)
    return output_path


def render_round_metric_comparison_outputs(
    *,
    round_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Render round metric comparison outputs.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        output_dir (Path): Directory where generated outputs are written.

    Returns:
        None: None.
    """
    if round_metrics.empty or dataset_summary.empty:
        return

    predictor_order = resolve_comparison_predictor_order(round_metrics["predictor"])
    if len(predictor_order) < 2:
        return

    specs = [
        {
            "metric_column": "decision_accuracy",
            "metric_label": "Decision Accuracy",
            "file_stem": "decision_accuracy",
            "y_limits": (0.0, 1.05),
        },
        {
            "metric_column": "roc_auc",
            "metric_label": "ROC AUC",
            "file_stem": "roc_auc",
            "y_limits": (0.0, 1.05),
        },
        {
            "metric_column": "false_kill_rate",
            "metric_label": "False Kill Rate",
            "file_stem": "false_kill_rate",
            "y_limits": (0.0, 1.05),
        },
        {
            "metric_column": "rank_alignment",
            "metric_label": "Rank Alignment",
            "file_stem": "rank_alignment",
            "y_limits": (-1.0, 1.05),
        },
    ]

    by_dataset_dir = output_dir / "comparison_by_dataset"
    by_dataset_dir.mkdir(parents=True, exist_ok=True)
    subtitle = build_experiment_subtitle(round_metrics)
    secondary_subtitle = build_model_zoo_subtitle(round_metrics)

    for spec in specs:
        render_metric_comparison_by_dataset(
            round_metrics=round_metrics,
            dataset_summary=dataset_summary,
            metric_column=spec["metric_column"],
            metric_label=spec["metric_label"],
            file_stem=spec["file_stem"],
            y_limits=spec["y_limits"],
            output_dir=by_dataset_dir,
            subtitle=subtitle,
            secondary_subtitle=secondary_subtitle,
        )
        render_metric_comparison_overview(
            round_metrics=round_metrics,
            dataset_summary=dataset_summary,
            metric_column=spec["metric_column"],
            metric_label=spec["metric_label"],
            y_limits=spec["y_limits"],
            output_path=output_dir / f"20_{spec['file_stem']}_comparison_overview.png",
            subtitle=subtitle,
            secondary_subtitle=secondary_subtitle,
        )


def render_metric_comparison_by_dataset(
    *,
    round_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    file_stem: str,
    y_limits: tuple[float, float],
    output_dir: Path,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render metric comparison by dataset.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        metric_column (str): str value used by the function.
        metric_label (str): str value used by the function.
        file_stem (str): str value used by the function.
        y_limits (tuple[float, float]): tuple[float, float] value used by the function.
        output_dir (Path): Directory where generated outputs are written.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    comparison_rounds = shared_predictor_round_metrics(round_metrics, dataset_summary)
    if comparison_rounds.empty:
        return

    for dataset_label, dataset_df in comparison_rounds.groupby("dataset_label", sort=False):
        fig, axis = plt.subplots(figsize=(8, 5))
        render_predictor_lineplot(
            axis=axis,
            dataset_df=dataset_df,
            metric_column=metric_column,
            metric_label=metric_label,
            y_limits=y_limits,
            title=f"{metric_label}: {dataset_label}",
            show_legend=True,
            subtitle=subtitle,
            secondary_subtitle=secondary_subtitle,
            show_metric_axis_label=True,
            show_remaining_axis_label=True,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _save_figure(fig, output_dir / f"{dataset_label}_{file_stem}.png")


def render_metric_comparison_overview(
    *,
    round_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    y_limits: tuple[float, float],
    output_path: Path,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render metric comparison overview.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.
        metric_column (str): str value used by the function.
        metric_label (str): str value used by the function.
        y_limits (tuple[float, float]): tuple[float, float] value used by the function.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.

    Returns:
        None: None.
    """
    comparison_rounds = shared_predictor_round_metrics(round_metrics, dataset_summary)
    if comparison_rounds.empty:
        return

    predictor_order = resolve_comparison_predictor_order(comparison_rounds["predictor"])
    if len(predictor_order) < 2:
        return

    dataset_order = (
        comparison_rounds.loc[:, ["dataset_label", "dataset_number"]]
        .drop_duplicates()
        .sort_values("dataset_number")
        ["dataset_label"]
        .tolist()
    )
    column_count = 4
    row_count = math.ceil(len(dataset_order) / column_count)
    fig, axes = plt.subplots(row_count, column_count, figsize=(20, 3.8 * row_count), sharex=False, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(row_count, column_count)
    flat_axes = list(axes_array.flat)

    for axis in flat_axes[len(dataset_order) :]:
        axis.axis("off")

    for index, (axis, dataset_label) in enumerate(zip(flat_axes, dataset_order, strict=False)):
        dataset_df = comparison_rounds[comparison_rounds["dataset_label"] == dataset_label].copy()
        show_metric_axis_label = index % column_count == 0
        show_remaining_axis_label = (index % column_count == column_count - 1) or (index == len(dataset_order) - 1)
        render_predictor_lineplot(
            axis=axis,
            dataset_df=dataset_df,
            metric_column=metric_column,
            metric_label=metric_label,
            y_limits=y_limits,
            title=dataset_label,
            show_legend=False,
            subtitle="",
            show_metric_axis_label=show_metric_axis_label,
            show_remaining_axis_label=show_remaining_axis_label,
        )

    handles = []
    for predictor_index, predictor_name in enumerate(predictor_order):
        style = predictor_style(predictor_name, role_index=predictor_index)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                label=style["label"],
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#6c757d",
            linestyle="--",
            linewidth=1.7,
            alpha=0.85,
            label="Number of models remaining",
        )
    )
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    model_title = (
        " vs ".join(predictor_display_name(name) for name in predictor_order)
        if len(predictor_order) <= 3
        else ", ".join(predictor_display_name(name) for name in predictor_order)
    )
    layout_top = apply_figure_titles(
        fig,
        title=f"{metric_label}: {model_title}",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0.04, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


def render_predictor_lineplot(
    *,
    axis,
    dataset_df: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    y_limits: tuple[float, float],
    title: str,
    show_legend: bool,
    subtitle: str,
    show_metric_axis_label: bool,
    show_remaining_axis_label: bool,
    secondary_subtitle: str = "",
) -> None:
    """Render predictor lineplot.

    Args:
        axis: Matplotlib axis to update.
        dataset_df (pd.DataFrame): Data frame containing dataset values.
        metric_column (str): str value used by the function.
        metric_label (str): str value used by the function.
        y_limits (tuple[float, float]): tuple[float, float] value used by the function.
        title (str): Title text for the output.
        show_legend (bool): bool value used by the function.
        subtitle (str): Subtitle text for the output.
        show_metric_axis_label (bool): bool value used by the function.
        show_remaining_axis_label (bool): bool value used by the function.

    Returns:
        None: None.
    """
    axis_label_color = "black"
    axis_label_fontsize = 10
    predictor_order = resolve_comparison_predictor_order(dataset_df["predictor"])
    if len(predictor_order) < 2:
        predictor_order = sorted(dataset_df["predictor"].dropna().astype(str).unique())
    else:
        available = set(dataset_df["predictor"].dropna().astype(str))
        predictor_order = [name for name in predictor_order if name in available]
    styles = {
        predictor_name: predictor_style(predictor_name, role_index=index)
        for index, predictor_name in enumerate(predictor_order)
    }
    remaining_axis = axis.twinx()
    for predictor_name in predictor_order:
        predictor_df = dataset_df[dataset_df["predictor"] == predictor_name].sort_values("decision_epoch")
        if predictor_df.empty:
            continue
        style = styles[predictor_name]
        axis.plot(
            predictor_df["decision_epoch"],
            predictor_df[metric_column],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    remaining_trace, remaining_warning = shared_remaining_population_trace(dataset_df)
    if not remaining_trace.empty:
        remaining_axis.step(
            remaining_trace["decision_epoch"],
            remaining_trace["policy_keep_count"],
            where="post",
            color="#6c757d",
            linestyle="--",
            linewidth=1.7,
            alpha=0.85,
            label="Number of models remaining",
        )
        if np.isclose(y_limits[0], 0.0):
            remaining_max = float(remaining_trace["policy_keep_count"].max())
            remaining_axis.set_ylim(0.0, max(1.0, remaining_max * 1.05))

    set_axes_title(axis, title=title, subtitle=subtitle, secondary_subtitle=secondary_subtitle)
    axis.set_ylim(*y_limits)
    axis.grid(alpha=0.25)
    axis.set_xlabel("Decision epoch")
    if show_metric_axis_label:
        axis.set_ylabel(metric_label, color=axis_label_color, fontsize=axis_label_fontsize)
    if remaining_warning:
        axis.text(
            0.02,
            0.98,
            remaining_warning,
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
            color="#c1121f",
            fontweight="bold",
        )
    remaining_axis.tick_params(axis="y", colors="#6c757d", labelsize=8)
    if show_remaining_axis_label:
        remaining_axis.set_ylabel("Remaining models", color=axis_label_color, fontsize=axis_label_fontsize)
    if show_legend:
        metric_handles, metric_labels = axis.get_legend_handles_labels()
        remaining_handles, remaining_labels = remaining_axis.get_legend_handles_labels()
        axis.legend(metric_handles + remaining_handles, metric_labels + remaining_labels, frameon=False, loc="best")


def shared_predictor_round_metrics(round_metrics: pd.DataFrame, dataset_summary: pd.DataFrame) -> pd.DataFrame:
    """Shared predictor round metrics.

    Args:
        round_metrics (pd.DataFrame): pd.DataFrame value used by the function.
        dataset_summary (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    predictor_order = resolve_comparison_predictor_order(dataset_summary["predictor"])
    if len(predictor_order) < 2:
        return pd.DataFrame()
    required_predictors = set(predictor_order)
    shared_datasets = (
        dataset_summary.groupby("dataset_label")["predictor"]
        .agg(lambda values: required_predictors.issubset(set(values.astype(str))))
    )
    shared_labels = shared_datasets[shared_datasets].index.tolist()
    if not shared_labels:
        return pd.DataFrame()

    filtered = round_metrics[
        round_metrics["dataset_label"].isin(shared_labels)
        & round_metrics["predictor"].isin(sorted(required_predictors))
    ].copy()
    filtered["predictor"] = pd.Categorical(filtered["predictor"], categories=predictor_order, ordered=True)
    filtered = filtered.sort_values(["dataset_number", "predictor", "round_index"]).reset_index(drop=True)
    filtered["predictor"] = filtered["predictor"].astype(str)
    return filtered


def _relative_luminance(red: float, green: float, blue: float) -> float:
    """Return WCAG relative luminance for an sRGB color."""
    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    r_lin = linearize(red)
    g_lin = linearize(green)
    b_lin = linearize(blue)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _heatmap_annotation_color(value: float, *, cmap, norm) -> str:
    """Choose readable annotation text color for a heatmap cell value."""
    red, green, blue, _alpha = cmap(norm(value))
    return "black" if _relative_luminance(red, green, blue) > 0.20 else "white"


def render_heatmap(
    *,
    axis,
    table: pd.DataFrame,
    title: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    value_format: str,
    annotate_values: bool = False,
) -> None:
    """Render heatmap.

    Args:
        axis: Matplotlib axis to update.
        table (pd.DataFrame): pd.DataFrame value used by the function.
        title (str): Title text for the output.
        cmap_name (str): str value used by the function.
        vmin (float | None): float | None value used by the function.
        vmax (float | None): float | None value used by the function.
        value_format (str): str value used by the function.
        annotate_values (bool): bool value used by the function.

    Returns:
        None: None.
    """
    if table.empty:
        axis.axis("off")
        axis.set_title(f"{title} (no data)")
        return

    matrix = table.to_numpy(dtype="float64")
    matrix = np.clip(matrix, vmin, vmax)
    masked = np.ma.masked_invalid(matrix)
    # When vmax is unset and data is uniform at vmin (e.g. all-zero gap),
    # matplotlib expands symmetrically below vmin. Fall back to 1.0 in that case.
    effective_vmax = vmax
    if effective_vmax is None:
        finite = matrix[np.isfinite(matrix)]
        data_max = float(finite.max()) if finite.size > 0 else 0.0
        floor = vmin if vmin is not None else 0.0
        effective_vmax = data_max if data_max > floor else 1.0
    image = axis.imshow(masked, aspect="auto", cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=effective_vmax)
    axis.set_title(title)
    axis.set_xticks(np.arange(table.shape[1]), [str(value) for value in table.columns])
    axis.set_yticks(np.arange(table.shape[0]), list(table.index))
    axis.set_xlabel("Decision epoch")
    axis.set_ylabel("Dataset")

    if annotate_values:
        for row_index in range(table.shape[0]):
            for column_index in range(table.shape[1]):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=_heatmap_annotation_color(value, cmap=image.cmap, norm=image.norm),
                )

    plt.colorbar(image, ax=axis, fraction=0.03, pad=0.02)


def render_population_accuracy_boxplots(
    latest_runs: pd.DataFrame,
    output_path: Path,
    *,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Render per-epoch boxplots of the remaining model accuracy for each experiment.

    One panel per run. The x-axis is the decision epoch; the y-axis shows the
    distribution (boxplot) of the target metric across all models present at that round.

    Args:
        latest_runs (pd.DataFrame): One row per run, must contain `round_details_path`.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.
        secondary_subtitle (str): Optional secondary subtitle.
    """
    if latest_runs.empty or "round_details_path" not in latest_runs.columns:
        return

    ordered_frames = _load_ordered_run_details(latest_runs)
    if not ordered_frames:
        return

    n_runs = len(ordered_frames)
    column_count = min(3, n_runs)
    row_count = math.ceil(n_runs / column_count)

    fig, axes = plt.subplots(row_count, column_count, figsize=(6 * column_count, 4.5 * row_count))
    axes_array = np.atleast_1d(axes).reshape(row_count, column_count)
    flat_axes = list(axes_array.flat)

    for axis in flat_axes[n_runs:]:
        axis.axis("off")

    for axis, run_df in zip(flat_axes, ordered_frames):
        if run_df.empty:
            axis.axis("off")
            continue
        target_col = str(run_df["target_column"].iloc[0]) if "target_column" in run_df.columns else "test_acc"
        if target_col not in run_df.columns:
            for candidate in ("test_acc", "train_acc"):
                if candidate in run_df.columns:
                    target_col = candidate
                    break
            else:
                axis.axis("off")
                continue

        epochs = sorted(run_df["decision_epoch"].dropna().unique())
        if not epochs:
            axis.axis("off")
            continue

        box_data = [run_df.loc[run_df["decision_epoch"] == ep, target_col].dropna().to_numpy(dtype="float64") for ep in epochs]
        valid_epochs = [ep for ep, d in zip(epochs, box_data) if len(d) > 0]
        box_data = [d for d in box_data if len(d) > 0]
        if not box_data:
            axis.axis("off")
            continue

        epoch_gap = float(valid_epochs[1] - valid_epochs[0]) if len(valid_epochs) > 1 else 5.0
        box_width = epoch_gap * 0.55

        axis.boxplot(
            box_data,
            positions=valid_epochs,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor="#a8dadc", edgecolor="black", linewidth=0.8),
            medianprops=dict(color="#1d3557", linewidth=2.0),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            flierprops=dict(marker=".", color="black", markersize=3, alpha=0.5),
        )

        exp_name = str(run_df["experiment_name"].iloc[0]) if "experiment_name" in run_df.columns else ""
        pred_name = str(run_df["predictor"].iloc[0]) if "predictor" in run_df.columns else ""
        panel_title = f"{exp_name} ({predictor_display_name(pred_name)})" if exp_name and pred_name else (exp_name or pred_name)

        axis.set_title(panel_title, fontsize=9)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(target_col.replace("_", " "))
        axis.set_xticks(valid_epochs)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)

    layout_top = apply_figure_titles(
        fig,
        title="Remaining Population Accuracy Distribution",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


_SENSITIVITY_PARAM_PAIRS: list[tuple[str, str]] = [
    ("selection_survival_rate", "selection_start_epoch"),
    ("selection_survival_rate", "selection_interval"),
    ("selection_start_epoch", "selection_interval"),
]

_SENSITIVITY_PARAM_LABELS: dict[str, str] = {
    "selection_start_epoch": "Start Epoch",
    "selection_interval": "Interval",
    "selection_survival_rate": "Survival Rate",
}

_SENSITIVITY_METRIC_STYLES: dict[str, tuple[str, float | None, float | None]] = {
    "best_survivor_regret": ("YlGnBu_r", 0.0, None),  # lower=better → dark at 0
    "oracle_top_k_recall": ("YlGnBu", 0.0, 1.0),       # higher=better → dark at 1
    "compute_savings_fraction": ("YlGnBu", 0.0, 1.0),  # higher=better → dark at 1
    "decision_accuracy_weighted": ("YlGnBu", 0.0, 1.0), # higher=better → dark at 1
    "false_kill_rate_weighted": ("OrRd_r", 0.0, 1.0),  # lower=better → dark at 0
}

_SENSITIVITY_METRIC_LABELS: dict[str, str] = {
    "best_survivor_regret": "Best Survivor Gap (↓)",
    "oracle_top_k_recall": "Final Population Recall (↑)",
    "compute_savings_fraction": "Compute Savings",
    "decision_accuracy_weighted": "Decision Accuracy",
    "false_kill_rate_weighted": "False Kill Rate",
}


def render_sotf_parameter_sensitivity(
    cross_batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: list[str] | None = None,
    subtitle: str = "",
    secondary_subtitle: str = "",
    baseline_predictor: str = "online_lc_extrapolation",
) -> list[Path]:
    """Render 2-D sensitivity heatmaps comparing predictors against a baseline.

    When the baseline predictor (default: ``online_lc_extrapolation``, or
    ``naive`` if passed explicitly) is present each cell shows
    (experiment − baseline), coloured with a diverging colormap centred at zero
    (green = better than baseline).  The baseline experiment itself is omitted
    from the per-experiment loop because it is the reference.

    Falls back to absolute metric heatmaps (one per experiment, no delta) when
    the chosen baseline predictor does not exist in the data.

    Produces one figure per non-baseline experiment.  Each figure is a grid:
      rows    = parameter pairs (survival_rate × start_epoch,
                                 survival_rate × interval,
                                 start_epoch   × interval)
      columns = metrics (default: best_survivor_regret, oracle_top_k_recall)

    Args:
        cross_batch_summary: Pooled dataset_summary DataFrame from all batches.
        output_dir: Directory to write PNG (and mirrored PDF) files.
        metrics: Column names to show; defaults to best_survivor_regret and
                 oracle_top_k_recall.
        subtitle: Subtitle line shown on every figure.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of output paths written.
    """
    _BASELINE_PREDICTOR = baseline_predictor

    if cross_batch_summary.empty:
        return []

    if metrics is None:
        metrics = ["best_survivor_regret", "oracle_top_k_recall"]

    param_cols = ["selection_start_epoch", "selection_interval", "selection_survival_rate"]
    available_metrics = [m for m in metrics if m in cross_batch_summary.columns]
    missing_params = [c for c in param_cols if c not in cross_batch_summary.columns]
    if not available_metrics or missing_params:
        log.warning(
            "Parameter sensitivity: skipping — unavailable metrics=%s or missing params=%s",
            [m for m in metrics if m not in cross_batch_summary.columns],
            missing_params,
        )
        return []

    varying = [
        col for col in param_cols
        if cross_batch_summary[col].dropna().nunique() > 1
    ]
    if len(varying) < 2:
        log.info("Parameter sensitivity: fewer than 2 parameters vary across batches — skipping.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "20_parameter_sensitivity_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    pairs = [(py, px) for py, px in _SENSITIVITY_PARAM_PAIRS if py in varying and px in varying]
    if not pairs:
        return []

    # Determine whether we can compare against the chosen baseline predictor.
    predictors_present = set(cross_batch_summary["predictor"].dropna().astype(str))
    use_delta = _BASELINE_PREDICTOR in predictors_present

    baseline_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    if use_delta:
        baseline_df = cross_batch_summary[
            cross_batch_summary["predictor"].astype(str) == _BASELINE_PREDICTOR
        ].copy()
        for param_y, param_x in pairs:
            for metric in available_metrics:
                agg = (
                    baseline_df
                    .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                    .mean()
                    .dropna(subset=[metric])
                )
                if not agg.empty:
                    baseline_tables[(param_y, param_x, metric)] = agg.pivot(
                        index=param_y, columns=param_x, values=metric
                    )

    candidate_df = cross_batch_summary
    if use_delta:
        # Exclude the baseline experiment itself — it is the reference.
        candidate_df = cross_batch_summary[
            cross_batch_summary["predictor"].astype(str) != _BASELINE_PREDICTOR
        ]

    experiments = (
        candidate_df
        .loc[:, ["experiment_name", "predictor"]]
        .drop_duplicates("experiment_name")
        .sort_values("experiment_name")
        .itertuples(index=False)
    )
    _baseline_label = predictor_display_name(_BASELINE_PREDICTOR)

    # Absolute (non-baseline-relative) companion plots are only meaningful
    # in addition to the delta view — when there's no baseline, the main
    # loop below already renders absolute heatmaps directly.
    absolute_overview_path = _pdf_dir_for_output_dir(output_dir) / "20_parameter_sensitivity_absolute_overview.pdf"
    absolute_overview_pdf: PdfPages | None = None
    absolute_overview_written = False

    def _render_absolute_companion(exp_df: pd.DataFrame, experiment_name: str) -> None:
        nonlocal absolute_overview_pdf, absolute_overview_written
        savings_tables, final_pop_tables = _build_sensitivity_overlay_tables(exp_df, pairs)
        fig = _build_parameter_sensitivity_figure(
            exp_df,
            pairs,
            available_metrics,
            experiment_name=experiment_name,
            use_delta=False,
            baseline_tables={},
            savings_tables=savings_tables,
            final_pop_tables=final_pop_tables,
            subtitle=subtitle,
        )
        if absolute_overview_pdf is None:
            absolute_overview_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_overview_pdf = PdfPages(absolute_overview_path)
        absolute_overview_pdf.savefig(fig, bbox_inches="tight")
        absolute_overview_written = True
        out = output_dir / f"20_parameter_sensitivity_absolute_{sanitize_file_stem(experiment_name)}.png"
        _save_figure(fig, out)
        output_paths.append(out)
        log.info("Wrote absolute parameter sensitivity plot: %s", out)

    try:
        for row in experiments:
            experiment_name = str(row.experiment_name)
            exp_df = cross_batch_summary[
                cross_batch_summary["experiment_name"].astype(str) == experiment_name
            ].copy()
            if exp_df.empty:
                continue

            savings_tables, final_pop_tables = _build_sensitivity_overlay_tables(exp_df, pairs)

            fig = _build_parameter_sensitivity_figure(
                exp_df,
                pairs,
                available_metrics,
                experiment_name=experiment_name,
                use_delta=use_delta,
                baseline_tables=baseline_tables,
                savings_tables=savings_tables,
                final_pop_tables=final_pop_tables,
                subtitle=subtitle,
                baseline_label=_baseline_label,
            )
            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True
            out = output_dir / f"20_parameter_sensitivity_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote parameter sensitivity plot: %s", out)

            # Additionally render the same grid without the baseline
            # comparison — absolute gap-to-best-survivor instead of a delta.
            if use_delta:
                _render_absolute_companion(exp_df, experiment_name)

        # The baseline itself is excluded from the delta loop
        # above (it is the reference); still give it an absolute companion
        # plot so the comparison set is complete.
        if use_delta:
            for experiment_name, baseline_exp_df in baseline_df.groupby(
                baseline_df["experiment_name"].astype(str), sort=False
            ):
                _render_absolute_companion(baseline_exp_df.copy(), str(experiment_name))
    finally:
        if overview_pdf is not None:
            overview_pdf.close()
        if absolute_overview_pdf is not None:
            absolute_overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote parameter sensitivity overview PDF: %s", overview_path)
    if absolute_overview_written:
        output_paths.append(absolute_overview_path)
        log.info("Wrote absolute parameter sensitivity overview PDF: %s", absolute_overview_path)

    return output_paths


def _build_sensitivity_overlay_tables(
    exp_df: pd.DataFrame, pairs: list[tuple[str, str]]
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    """Build the compute-savings and final-population overlay tables shared by delta/absolute figures."""
    savings_tables: dict[tuple[str, str], pd.DataFrame] = {}
    if "compute_savings_fraction" in exp_df.columns:
        for param_y, param_x in pairs:
            agg_s = (
                exp_df
                .groupby([param_y, param_x], as_index=False, sort=True)["compute_savings_fraction"]
                .mean()
                .dropna(subset=["compute_savings_fraction"])
            )
            if not agg_s.empty:
                savings_tables[(param_y, param_x)] = agg_s.pivot(
                    index=param_y, columns=param_x, values="compute_savings_fraction"
                )

    final_pop_tables: dict[tuple[str, str], pd.DataFrame] = {}
    if "final_population" in exp_df.columns:
        for param_y, param_x in pairs:
            agg_p = (
                exp_df
                .groupby([param_y, param_x], as_index=False, sort=True)["final_population"]
                .mean()
                .dropna(subset=["final_population"])
            )
            if not agg_p.empty:
                final_pop_tables[(param_y, param_x)] = agg_p.pivot(
                    index=param_y, columns=param_x, values="final_population"
                )

    return savings_tables, final_pop_tables


def _build_parameter_sensitivity_figure(
    exp_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    available_metrics: list[str],
    *,
    experiment_name: str,
    use_delta: bool,
    baseline_tables: dict[tuple[str, str, str], pd.DataFrame],
    savings_tables: dict[tuple[str, str], pd.DataFrame],
    final_pop_tables: dict[tuple[str, str], pd.DataFrame],
    subtitle: str,
    baseline_label: str = "LC Extrap.",
):
    """Build one parameter-sensitivity figure (delta-vs-baseline or absolute) for a single experiment."""
    n_rows = len(pairs)
    n_cols = len(available_metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes_grid = np.array([[axes]])
    elif n_rows == 1:
        axes_grid = np.array([np.atleast_1d(axes)])
    elif n_cols == 1:
        axes_grid = np.atleast_2d(np.atleast_1d(axes)).T
    else:
        axes_grid = np.atleast_2d(axes)

    for row_idx, (param_y, param_x) in enumerate(pairs):
        for col_idx, metric in enumerate(available_metrics):
            axis = axes_grid[row_idx, col_idx]

            agg = (
                exp_df
                .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                .mean()
                .dropna(subset=[metric])
            )
            if agg.empty:
                axis.axis("off")
                axis.set_title(f"{_sensitivity_metric_label(metric)} (no data)")
                continue

            exp_table = agg.pivot(index=param_y, columns=param_x, values=metric)

            if use_delta:
                baseline_table = baseline_tables.get((param_y, param_x, metric))
                if baseline_table is None:
                    axis.axis("off")
                    axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no {baseline_label} baseline)")
                    continue
                display_table = (exp_table - baseline_table).dropna(how="all").dropna(axis=1, how="all")
                if display_table.empty:
                    axis.axis("off")
                    axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no overlap)")
                    continue
                delta_array = display_table.to_numpy(dtype="float64")
                max_abs = float(np.nanmax(np.abs(delta_array)))
                if max_abs < 1e-9:
                    max_abs = 1e-6
                cmap_name = _sensitivity_delta_cmap(metric)
                render_heatmap(
                    axis=axis,
                    table=display_table,
                    title=f"Δ {_sensitivity_metric_label(metric)}",
                    cmap_name=cmap_name,
                    vmin=-max_abs,
                    vmax=max_abs,
                    value_format="{:+.3f}",
                    annotate_values=True,
                )
            else:
                display_table = exp_table
                cmap_name, vmin, vmax = _sensitivity_metric_style(metric)
                render_heatmap(
                    axis=axis,
                    table=display_table,
                    title=_sensitivity_metric_label(metric),
                    cmap_name=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    value_format="{:.3f}",
                    annotate_values=True,
                )
            axis.set_xlabel(_sensitivity_param_label(param_x))
            axis.set_ylabel(_sensitivity_param_label(param_y))

            savings_table = savings_tables.get((param_y, param_x))
            if savings_table is not None:
                _annotate_compute_savings_overlay(
                    axis,
                    savings_table,
                    row_labels=display_table.index,
                    col_labels=display_table.columns,
                    value_table=display_table,
                )

            final_pop_table = final_pop_tables.get((param_y, param_x))
            if final_pop_table is not None:
                _annotate_final_population_overlay(
                    axis,
                    final_pop_table,
                    row_labels=display_table.index,
                    col_labels=display_table.columns,
                    value_table=display_table,
                )

    if use_delta:
        fig_title = f"Δ vs {baseline_label} — {experiment_name}"
        fig_secondary = f"top-right = compute savings · bottom-left = final population · green = better than {baseline_label}"
    else:
        fig_title = f"SOTF Parameter Sensitivity (Absolute) — {experiment_name}"
        fig_secondary = "top-right = compute savings · bottom-left = final population"
    layout_top = apply_figure_titles(
        fig,
        title=fig_title,
        subtitle=subtitle,
        secondary_subtitle=fig_secondary,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    return fig


def _sensitivity_metric_style(metric: str) -> tuple[str, float | None, float | None]:
    return _SENSITIVITY_METRIC_STYLES.get(metric, ("viridis", None, None))


def _sensitivity_metric_label(name: str) -> str:
    return _SENSITIVITY_METRIC_LABELS.get(name, name.replace("_", " ").title())


def _sensitivity_param_label(name: str) -> str:
    return _SENSITIVITY_PARAM_LABELS.get(name, name.replace("_", " ").title())


_SENSITIVITY_HIGHER_IS_BETTER: dict[str, bool] = {
    "best_survivor_regret": False,
    "oracle_top_k_recall": True,
    "compute_savings_fraction": True,
    "decision_accuracy_weighted": True,
    "false_kill_rate_weighted": False,
}


def _sensitivity_delta_cmap(metric: str) -> str:
    """Diverging colormap for delta plots: green always means improvement over baseline."""
    return "RdYlGn" if _SENSITIVITY_HIGHER_IS_BETTER.get(metric, True) else "RdYlGn_r"


def render_sotf_parameter_sensitivity_delta(
    cross_batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: list[str] | None = None,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    """Render delta sensitivity heatmaps: improvement over naive baseline per parameter pair.

    Same grid layout as render_sotf_parameter_sensitivity, but each cell shows
    (experiment − baseline), coloured with a diverging colormap centred at zero:
      green  = better than baseline
      red    = worse than baseline
      white  = no difference

    One figure per non-baseline experiment.  The naive predictor experiments are
    used as the baseline reference.

    Args:
        cross_batch_summary: Pooled dataset_summary DataFrame from all batches.
        output_dir: Directory to write PNG (and mirrored PDF) files.
        metrics: Column names to compare; defaults to best_survivor_regret and
                 oracle_top_k_recall.
        subtitle: Subtitle line shown on every figure.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of output paths written.
    """
    if cross_batch_summary.empty:
        return []

    if metrics is None:
        metrics = ["best_survivor_regret", "oracle_top_k_recall"]

    param_cols = ["selection_start_epoch", "selection_interval", "selection_survival_rate"]
    available_metrics = [m for m in metrics if m in cross_batch_summary.columns]
    missing_params = [c for c in param_cols if c not in cross_batch_summary.columns]
    if not available_metrics or missing_params:
        return []

    varying = [
        col for col in param_cols
        if cross_batch_summary[col].dropna().nunique() > 1
    ]
    if len(varying) < 2:
        return []

    resolved = resolve_baseline_and_comparison_predictors(cross_batch_summary["predictor"])
    if resolved is None:
        log.info("Parameter sensitivity delta: no naive baseline predictor found — skipping.")
        return []
    _, baseline_predictor = resolved

    baseline_df = cross_batch_summary[
        cross_batch_summary["predictor"].astype(str) == baseline_predictor
    ].copy()
    if baseline_df.empty:
        return []

    pairs = [(py, px) for py, px in _SENSITIVITY_PARAM_PAIRS if py in varying and px in varying]
    if not pairs:
        return []

    # Pre-aggregate baseline once per (param_pair, metric)
    baseline_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for param_y, param_x in pairs:
        for metric in available_metrics:
            agg = (
                baseline_df
                .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                .mean()
                .dropna(subset=[metric])
            )
            if not agg.empty:
                baseline_tables[(param_y, param_x, metric)] = agg.pivot(
                    index=param_y, columns=param_x, values=metric
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "21_parameter_sensitivity_delta_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    comparison_experiments = (
        cross_batch_summary[
            cross_batch_summary["predictor"].astype(str) != baseline_predictor
        ]
        .loc[:, ["experiment_name", "predictor"]]
        .drop_duplicates("experiment_name")
        .sort_values("experiment_name")
        .itertuples(index=False)
    )

    try:
        for exp_row in comparison_experiments:
            experiment_name = str(exp_row.experiment_name)
            predictor = str(exp_row.predictor)
            exp_df = cross_batch_summary[
                cross_batch_summary["experiment_name"].astype(str) == experiment_name
            ].copy()
            if exp_df.empty:
                continue

            n_rows = len(pairs)
            n_cols = len(available_metrics)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes_grid = np.array([[axes]])
            elif n_rows == 1:
                axes_grid = np.array([np.atleast_1d(axes)])
            elif n_cols == 1:
                axes_grid = np.atleast_2d(np.atleast_1d(axes)).T
            else:
                axes_grid = np.atleast_2d(axes)

            for row_idx, (param_y, param_x) in enumerate(pairs):
                for col_idx, metric in enumerate(available_metrics):
                    axis = axes_grid[row_idx, col_idx]

                    baseline_table = baseline_tables.get((param_y, param_x, metric))
                    if baseline_table is None:
                        axis.axis("off")
                        axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no baseline)")
                        continue

                    agg = (
                        exp_df
                        .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                        .mean()
                        .dropna(subset=[metric])
                    )
                    if agg.empty:
                        axis.axis("off")
                        axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no data)")
                        continue

                    exp_table = agg.pivot(index=param_y, columns=param_x, values=metric)
                    delta_table = (exp_table - baseline_table).dropna(how="all").dropna(axis=1, how="all")
                    if delta_table.empty:
                        axis.axis("off")
                        axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no overlap)")
                        continue

                    delta_array = delta_table.to_numpy(dtype="float64")
                    max_abs = float(np.nanmax(np.abs(delta_array)))
                    if max_abs < 1e-9:
                        max_abs = 1e-6
                    cmap_name = _sensitivity_delta_cmap(metric)
                    render_heatmap(
                        axis=axis,
                        table=delta_table,
                        title=f"Δ {_sensitivity_metric_label(metric)}",
                        cmap_name=cmap_name,
                        vmin=-max_abs,
                        vmax=max_abs,
                        value_format="{:+.3f}",
                        annotate_values=True,
                    )
                    axis.set_xlabel(_sensitivity_param_label(param_x))
                    axis.set_ylabel(_sensitivity_param_label(param_y))

            layout_top = apply_figure_titles(
                fig,
                title=f"Parameter Sensitivity vs Baseline — {experiment_name}",
                subtitle=subtitle,
                secondary_subtitle=secondary_subtitle,
            )
            fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True
            out = output_dir / f"21_parameter_sensitivity_delta_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote parameter sensitivity delta plot: %s", out)
    finally:
        if overview_pdf is not None:
            overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote parameter sensitivity delta overview PDF: %s", overview_path)

    return output_paths


def render_sotf_compute_savings_sensitivity(
    cross_batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    """Render compute-savings sensitivity heatmaps across SOTF hyperparameters (plot 22).

    Produces one figure per non-baseline experiment.  Each figure has a grid of heatmaps:
      rows    = parameter pairs (survival_rate × start_epoch,
                                 survival_rate × interval,
                                 start_epoch   × interval)
      columns = [Absolute compute savings | Δ vs naive baseline]

    The left column shows the mean compute_savings_fraction across all datasets averaged
    over the third (off-axis) parameter.  The right column shows the signed delta
    (experiment − baseline), coloured with a diverging RdYlGn colormap so that green
    means more savings than the naive strategy.

    If no naive baseline is present the right column is omitted and a single-column
    layout (absolute only) is produced for every experiment.

    Args:
        cross_batch_summary: Pooled dataset_summary DataFrame from all batches.
        output_dir: Directory to write PNG (and mirrored PDF) files.
        subtitle: Subtitle line shown on every figure.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of output paths written.
    """
    metric = "compute_savings_fraction"

    if cross_batch_summary.empty or metric not in cross_batch_summary.columns:
        log.info("Compute savings sensitivity: column %r not found — skipping.", metric)
        return []

    param_cols = ["selection_start_epoch", "selection_interval", "selection_survival_rate"]
    missing_params = [c for c in param_cols if c not in cross_batch_summary.columns]
    if missing_params:
        log.warning("Compute savings sensitivity: missing param columns %s — skipping.", missing_params)
        return []

    varying = [
        col for col in param_cols
        if cross_batch_summary[col].dropna().nunique() > 1
    ]
    if len(varying) < 2:
        log.info("Compute savings sensitivity: fewer than 2 parameters vary — skipping.")
        return []

    pairs = [(py, px) for py, px in _SENSITIVITY_PARAM_PAIRS if py in varying and px in varying]
    if not pairs:
        return []

    resolved = resolve_baseline_and_comparison_predictors(cross_batch_summary["predictor"])
    if resolved is None:
        # No naive baseline found — show absolute values for all experiments.
        non_baseline_names = (
            cross_batch_summary
            .loc[:, ["experiment_name", "predictor"]]
            .drop_duplicates("experiment_name")
            .sort_values("experiment_name")
        )
        baseline_predictor = None
        baseline_df = pd.DataFrame()
    else:
        _, baseline_predictor = resolved
        baseline_df = cross_batch_summary[
            cross_batch_summary["predictor"].astype(str) == baseline_predictor
        ].copy()
        non_baseline_names = (
            cross_batch_summary[
                cross_batch_summary["predictor"].astype(str) != baseline_predictor
            ]
            .loc[:, ["experiment_name", "predictor"]]
            .drop_duplicates("experiment_name")
            .sort_values("experiment_name")
        )

    # Pre-build baseline pivot tables once.
    baseline_tables: dict[tuple[str, str], pd.DataFrame] = {}
    if not baseline_df.empty:
        for param_y, param_x in pairs:
            agg = (
                baseline_df
                .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                .mean()
                .dropna(subset=[metric])
            )
            if not agg.empty:
                baseline_tables[(param_y, param_x)] = agg.pivot(
                    index=param_y, columns=param_x, values=metric
                )

    has_delta = bool(baseline_tables)
    n_cols = 2 if has_delta else 1
    cmap_name, vmin, vmax = _sensitivity_metric_style(metric)
    metric_label = _sensitivity_metric_label(metric)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "22_compute_savings_sensitivity_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    try:
        for exp_row in non_baseline_names.itertuples(index=False):
            experiment_name = str(exp_row.experiment_name)
            predictor = str(exp_row.predictor)
            exp_df = cross_batch_summary[
                cross_batch_summary["experiment_name"].astype(str) == experiment_name
            ].copy()
            if exp_df.empty:
                continue

            n_rows = len(pairs)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes_grid = np.array([[axes]])
            elif n_rows == 1:
                axes_grid = np.array([np.atleast_1d(axes)])
            elif n_cols == 1:
                axes_grid = np.atleast_2d(np.atleast_1d(axes)).T
            else:
                axes_grid = np.atleast_2d(axes)

            for row_idx, (param_y, param_x) in enumerate(pairs):
                agg = (
                    exp_df
                    .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                    .mean()
                    .dropna(subset=[metric])
                )

                # --- Column 0: absolute compute savings ---
                abs_axis = axes_grid[row_idx, 0]
                if agg.empty:
                    abs_axis.axis("off")
                    abs_axis.set_title(f"{metric_label} (no data)")
                else:
                    table = agg.pivot(index=param_y, columns=param_x, values=metric)
                    render_heatmap(
                        axis=abs_axis,
                        table=table,
                        title=metric_label,
                        cmap_name=cmap_name,
                        vmin=vmin,
                        vmax=vmax,
                        value_format="{:.1%}",
                        annotate_values=True,
                    )
                    abs_axis.set_xlabel(_sensitivity_param_label(param_x))
                    abs_axis.set_ylabel(_sensitivity_param_label(param_y))

                # --- Column 1: delta vs baseline ---
                if not has_delta:
                    continue
                delta_axis = axes_grid[row_idx, 1]
                baseline_table = baseline_tables.get((param_y, param_x))
                if baseline_table is None or agg.empty:
                    delta_axis.axis("off")
                    delta_axis.set_title(f"Δ {metric_label} (no baseline)")
                    continue

                exp_table = agg.pivot(index=param_y, columns=param_x, values=metric)
                delta_table = (exp_table - baseline_table).dropna(how="all").dropna(axis=1, how="all")
                if delta_table.empty:
                    delta_axis.axis("off")
                    delta_axis.set_title(f"Δ {metric_label} (no overlap)")
                    continue

                delta_array = delta_table.to_numpy(dtype="float64")
                max_abs = float(np.nanmax(np.abs(delta_array)))
                if max_abs < 1e-9:
                    max_abs = 1e-6
                render_heatmap(
                    axis=delta_axis,
                    table=delta_table,
                    title=f"Δ {metric_label} vs {predictor_display_name(baseline_predictor)}",
                    cmap_name="RdYlGn",
                    vmin=-max_abs,
                    vmax=max_abs,
                    value_format="{:+.1%}",
                    annotate_values=True,
                )
                delta_axis.set_xlabel(_sensitivity_param_label(param_x))
                delta_axis.set_ylabel(_sensitivity_param_label(param_y))

            layout_top = apply_figure_titles(
                fig,
                title=f"Compute Savings Sensitivity — {experiment_name}",
                subtitle=subtitle,
                secondary_subtitle=secondary_subtitle,
            )
            fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True
            out = output_dir / f"22_compute_savings_sensitivity_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote compute savings sensitivity plot: %s", out)
    finally:
        if overview_pdf is not None:
            overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote compute savings sensitivity overview PDF: %s", overview_path)

    return output_paths


# ---------------------------------------------------------------------------
# Plot B (24): Delta sensitivity heatmap with compute savings overlay
# ---------------------------------------------------------------------------

def _overlay_text_color(axis, r_label, c_label, row_labels, col_labels, value_table: pd.DataFrame | None) -> str:
    """Return black or white depending on the luminance of the heatmap cell at (r_label, c_label)."""
    images = axis.get_images()
    if not images or value_table is None:
        return "#333333"
    img = images[0]
    try:
        cell_val = float(value_table.loc[r_label, c_label])
    except (KeyError, TypeError, ValueError):
        return "#333333"
    return _heatmap_annotation_color(cell_val, cmap=img.cmap, norm=img.norm)


def _annotate_final_population_overlay(
    axis,
    pop_table: pd.DataFrame,
    row_labels,
    col_labels,
    *,
    value_table: pd.DataFrame | None = None,
) -> None:
    """Add mean final population count in the bottom-left corner of each heatmap cell."""
    for r_idx, r_label in enumerate(row_labels):
        for c_idx, c_label in enumerate(col_labels):
            try:
                val = float(pop_table.loc[r_label, c_label])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isnan(val):
                continue
            color = _overlay_text_color(axis, r_label, c_label, row_labels, col_labels, value_table)
            axis.text(
                c_idx - 0.35,
                r_idx + 0.45,
                f"n={int(round(val))}",
                ha="left", va="bottom",
                fontsize=16,
                fontweight="bold",
                color=color,
                alpha=0.85,
                style="italic",
            )


def _annotate_compute_savings_overlay(
    axis,
    savings_table: pd.DataFrame,
    row_labels,
    col_labels,
    *,
    value_table: pd.DataFrame | None = None,
) -> None:
    """Add small compute savings % text in the top-right corner of each heatmap cell."""
    for r_idx, r_label in enumerate(row_labels):
        for c_idx, c_label in enumerate(col_labels):
            try:
                val = float(savings_table.loc[r_label, c_label])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isnan(val):
                continue
            color = _overlay_text_color(axis, r_label, c_label, row_labels, col_labels, value_table)
            axis.text(
                c_idx + 0.35,
                r_idx - 0.45,
                f"{val:.0%}",
                ha="right", va="top",
                fontsize=16,
                fontweight="bold",
                color=color,
                alpha=0.85,
                style="italic",
            )


def render_sotf_parameter_sensitivity_delta_with_compute(
    cross_batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: list[str] | None = None,
    subtitle: str = "",
    secondary_subtitle: str = "",
    show_title: bool = True,
) -> list[Path]:
    """Delta sensitivity heatmap with compute savings overlaid in each cell (plot 24).

    Identical layout to plot 21 (render_sotf_parameter_sensitivity_delta), but
    each cell is additionally annotated with the experiment's mean
    compute_savings_fraction for that parameter combination in small italic text
    (top-right corner).  This lets readers judge whether a green cell (better
    quality than baseline) comes with high or low compute savings — exposing the
    confound between late start_epoch (more survivors → better recall) and genuine
    prediction quality.

    Args:
        cross_batch_summary: Pooled dataset_summary DataFrame from all batches.
        output_dir: Directory to write PNG (and mirrored PDF) files.
        metrics: Column names to compare; defaults to best_survivor_regret and
                 oracle_top_k_recall.
        subtitle: Subtitle line shown on every figure.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of output paths written.
    """
    if cross_batch_summary.empty:
        return []

    if metrics is None:
        metrics = ["best_survivor_regret", "oracle_top_k_recall"]

    param_cols = ["selection_start_epoch", "selection_interval", "selection_survival_rate"]
    available_metrics = [m for m in metrics if m in cross_batch_summary.columns]
    missing_params = [c for c in param_cols if c not in cross_batch_summary.columns]
    if not available_metrics or missing_params:
        return []

    varying = [
        col for col in param_cols
        if cross_batch_summary[col].dropna().nunique() > 1
    ]
    if len(varying) < 2:
        return []

    _LC_EXTRAP_PREDICTOR = "online_lc_extrapolation"
    predictors_present = set(cross_batch_summary["predictor"].dropna().astype(str))
    if _LC_EXTRAP_PREDICTOR in predictors_present:
        baseline_predictor = _LC_EXTRAP_PREDICTOR
        baseline_label = "LC Extrap."
    else:
        resolved = resolve_baseline_and_comparison_predictors(cross_batch_summary["predictor"])
        if resolved is None:
            log.info("Parameter sensitivity delta+compute: no LC Extrap. or naive baseline found — skipping.")
            return []
        _, baseline_predictor = resolved
        baseline_label = "naive"

    baseline_df = cross_batch_summary[
        cross_batch_summary["predictor"].astype(str) == baseline_predictor
    ].copy()
    if baseline_df.empty:
        return []

    pairs = [(py, px) for py, px in _SENSITIVITY_PARAM_PAIRS if py in varying and px in varying]
    if not pairs:
        return []

    has_savings = "compute_savings_fraction" in cross_batch_summary.columns

    # Pre-aggregate baseline tables per (param_pair, metric).
    baseline_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for param_y, param_x in pairs:
        for metric in available_metrics:
            agg = (
                baseline_df
                .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                .mean()
                .dropna(subset=[metric])
            )
            if not agg.empty:
                baseline_tables[(param_y, param_x, metric)] = agg.pivot(
                    index=param_y, columns=param_x, values=metric
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "24_parameter_sensitivity_delta_compute_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    comparison_experiments = (
        cross_batch_summary[
            cross_batch_summary["predictor"].astype(str) != baseline_predictor
        ]
        .loc[:, ["experiment_name", "predictor"]]
        .drop_duplicates("experiment_name")
        .sort_values("experiment_name")
        .itertuples(index=False)
    )

    try:
        for exp_row in comparison_experiments:
            experiment_name = str(exp_row.experiment_name)
            predictor = str(exp_row.predictor)
            exp_df = cross_batch_summary[
                cross_batch_summary["experiment_name"].astype(str) == experiment_name
            ].copy()
            if exp_df.empty:
                continue

            # Pre-build compute savings pivot tables for this experiment.
            savings_tables: dict[tuple[str, str], pd.DataFrame] = {}
            if has_savings:
                for param_y, param_x in pairs:
                    agg_s = (
                        exp_df
                        .groupby([param_y, param_x], as_index=False, sort=True)["compute_savings_fraction"]
                        .mean()
                        .dropna(subset=["compute_savings_fraction"])
                    )
                    if not agg_s.empty:
                        savings_tables[(param_y, param_x)] = agg_s.pivot(
                            index=param_y, columns=param_x, values="compute_savings_fraction"
                        )

            has_final_pop = "final_population" in exp_df.columns
            final_pop_tables: dict[tuple[str, str], pd.DataFrame] = {}
            if has_final_pop:
                for param_y, param_x in pairs:
                    agg_p = (
                        exp_df
                        .groupby([param_y, param_x], as_index=False, sort=True)["final_population"]
                        .mean()
                        .dropna(subset=["final_population"])
                    )
                    if not agg_p.empty:
                        final_pop_tables[(param_y, param_x)] = agg_p.pivot(
                            index=param_y, columns=param_x, values="final_population"
                        )

            n_rows = len(pairs)
            n_cols = len(available_metrics)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes_grid = np.array([[axes]])
            elif n_rows == 1:
                axes_grid = np.array([np.atleast_1d(axes)])
            elif n_cols == 1:
                axes_grid = np.atleast_2d(np.atleast_1d(axes)).T
            else:
                axes_grid = np.atleast_2d(axes)

            for row_idx, (param_y, param_x) in enumerate(pairs):
                for col_idx, metric in enumerate(available_metrics):
                    axis = axes_grid[row_idx, col_idx]

                    baseline_table = baseline_tables.get((param_y, param_x, metric))
                    if baseline_table is None:
                        axis.axis("off")
                        axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no baseline)")
                        continue

                    agg = (
                        exp_df
                        .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                        .mean()
                        .dropna(subset=[metric])
                    )
                    if agg.empty:
                        axis.axis("off")
                        axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no data)")
                        continue

                    exp_table = agg.pivot(index=param_y, columns=param_x, values=metric)
                    delta_table = (exp_table - baseline_table).dropna(how="all").dropna(axis=1, how="all")
                    if delta_table.empty:
                        axis.axis("off")
                        axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no overlap)")
                        continue

                    delta_array = delta_table.to_numpy(dtype="float64")
                    max_abs = float(np.nanmax(np.abs(delta_array)))
                    if max_abs < 1e-9:
                        max_abs = 1e-6
                    cmap_name = _sensitivity_delta_cmap(metric)
                    render_heatmap(
                        axis=axis,
                        table=delta_table,
                        title=f"Δ {_sensitivity_metric_label(metric)}",
                        cmap_name=cmap_name,
                        vmin=-max_abs,
                        vmax=max_abs,
                        value_format="{:+.3f}",
                        annotate_values=True,
                    )
                    axis.set_xlabel(_sensitivity_param_label(param_x))
                    axis.set_ylabel(_sensitivity_param_label(param_y))

                    # Overlay compute savings in top-right corner of each cell.
                    savings_table = savings_tables.get((param_y, param_x))
                    if savings_table is not None:
                        _annotate_compute_savings_overlay(
                            axis,
                            savings_table,
                            row_labels=delta_table.index,
                            col_labels=delta_table.columns,
                            value_table=delta_table,
                        )

                    # Overlay final population count in bottom-left corner of each cell.
                    final_pop_table = final_pop_tables.get((param_y, param_x))
                    if final_pop_table is not None:
                        _annotate_final_population_overlay(
                            axis,
                            final_pop_table,
                            row_labels=delta_table.index,
                            col_labels=delta_table.columns,
                            value_table=delta_table,
                        )

            if show_title:
                layout_top = apply_figure_titles(
                    fig,
                    title=f"Δ vs {baseline_label} + Compute Savings — {experiment_name}",
                    subtitle=subtitle,
                    secondary_subtitle="top-right = compute savings · bottom-left = final population · green = better than baseline",
                )
                fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
            else:
                fig.tight_layout(pad=0.1, w_pad=1.5)
            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True
            out = output_dir / f"24_parameter_sensitivity_delta_compute_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote delta+compute sensitivity plot: %s", out)
    finally:
        if overview_pdf is not None:
            overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote delta+compute sensitivity overview PDF: %s", overview_path)

    return output_paths


# ---------------------------------------------------------------------------
# Plot 26: Combined absolute + delta sensitivity (no title/subtitle)
# ---------------------------------------------------------------------------

def render_sotf_combined_sensitivity(
    cross_batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: list[str] | None = None,
) -> list[Path]:
    """Render a combined figure with absolute (plot 20) and delta (plot 24) sensitivity rows.

    Top half = absolute metric heatmaps per parameter pair.
    Bottom half = delta-vs-baseline heatmaps per parameter pair.
    No title or subtitle.  One PNG per non-baseline experiment.

    Args:
        cross_batch_summary: Pooled dataset_summary DataFrame from all batches.
        output_dir: Directory to write PNG files.
        metrics: Column names to show; defaults to best_survivor_regret and oracle_top_k_recall.

    Returns:
        List of output paths written.
    """
    return []  # plot 26 disabled
    if cross_batch_summary.empty:
        return []

    if metrics is None:
        metrics = ["best_survivor_regret", "oracle_top_k_recall"]

    param_cols = ["selection_start_epoch", "selection_interval", "selection_survival_rate"]
    available_metrics = [m for m in metrics if m in cross_batch_summary.columns]
    missing_params = [c for c in param_cols if c not in cross_batch_summary.columns]
    if not available_metrics or missing_params:
        return []

    varying = [
        col for col in param_cols
        if cross_batch_summary[col].dropna().nunique() > 1
    ]
    if len(varying) < 2:
        return []

    resolved = resolve_baseline_and_comparison_predictors(cross_batch_summary["predictor"])
    if resolved is None:
        log.info("Combined sensitivity: no naive baseline found — skipping.")
        return []
    _, baseline_predictor = resolved

    baseline_df = cross_batch_summary[
        cross_batch_summary["predictor"].astype(str) == baseline_predictor
    ].copy()
    if baseline_df.empty:
        return []

    pairs = [(py, px) for py, px in _SENSITIVITY_PARAM_PAIRS if py in varying and px in varying]
    if not pairs:
        return []

    has_savings = "compute_savings_fraction" in cross_batch_summary.columns

    # Pre-aggregate baseline tables per (param_pair, metric).
    baseline_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for param_y, param_x in pairs:
        for metric in available_metrics:
            agg = (
                baseline_df
                .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                .mean()
                .dropna(subset=[metric])
            )
            if not agg.empty:
                baseline_tables[(param_y, param_x, metric)] = agg.pivot(
                    index=param_y, columns=param_x, values=metric
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    comparison_experiments = (
        cross_batch_summary[
            cross_batch_summary["predictor"].astype(str) != baseline_predictor
        ]
        .loc[:, ["experiment_name", "predictor"]]
        .drop_duplicates("experiment_name")
        .sort_values("experiment_name")
        .itertuples(index=False)
    )

    n_pairs = len(pairs)
    n_cols = len(available_metrics)
    total_rows = 2 * n_pairs  # top = absolute, bottom = delta

    for exp_row in comparison_experiments:
        experiment_name = str(exp_row.experiment_name)
        predictor = str(exp_row.predictor)
        exp_df = cross_batch_summary[
            cross_batch_summary["experiment_name"].astype(str) == experiment_name
        ].copy()
        if exp_df.empty:
            continue

        # Build savings and final-population pivot tables.
        savings_tables: dict[tuple[str, str], pd.DataFrame] = {}
        if has_savings:
            for param_y, param_x in pairs:
                agg_s = (
                    exp_df
                    .groupby([param_y, param_x], as_index=False, sort=True)["compute_savings_fraction"]
                    .mean()
                    .dropna(subset=["compute_savings_fraction"])
                )
                if not agg_s.empty:
                    savings_tables[(param_y, param_x)] = agg_s.pivot(
                        index=param_y, columns=param_x, values="compute_savings_fraction"
                    )

        has_final_pop = "final_population" in exp_df.columns
        final_pop_tables: dict[tuple[str, str], pd.DataFrame] = {}
        if has_final_pop:
            for param_y, param_x in pairs:
                agg_p = (
                    exp_df
                    .groupby([param_y, param_x], as_index=False, sort=True)["final_population"]
                    .mean()
                    .dropna(subset=["final_population"])
                )
                if not agg_p.empty:
                    final_pop_tables[(param_y, param_x)] = agg_p.pivot(
                        index=param_y, columns=param_x, values="final_population"
                    )

        fig, axes = plt.subplots(total_rows, n_cols, figsize=(5.5 * n_cols, 6.5 * total_rows))
        if total_rows == 1 and n_cols == 1:
            axes_grid = np.array([[axes]])
        elif total_rows == 1:
            axes_grid = np.array([np.atleast_1d(axes)])
        elif n_cols == 1:
            axes_grid = np.atleast_2d(np.atleast_1d(axes)).T
        else:
            axes_grid = np.atleast_2d(axes)

        # ── Top half: absolute metrics (plot-20 content) ──
        for row_idx, (param_y, param_x) in enumerate(pairs):
            for col_idx, metric in enumerate(available_metrics):
                axis = axes_grid[row_idx, col_idx]

                agg = (
                    exp_df
                    .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                    .mean()
                    .dropna(subset=[metric])
                )
                if agg.empty:
                    axis.axis("off")
                    axis.set_title(f"{_sensitivity_metric_label(metric)} (no data)")
                    continue

                table = agg.pivot(index=param_y, columns=param_x, values=metric)
                cmap_name, vmin, vmax = _sensitivity_metric_style(metric)
                render_heatmap(
                    axis=axis,
                    table=table,
                    title=_sensitivity_metric_label(metric),
                    cmap_name=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    value_format="{:.3f}",
                    annotate_values=True,
                )
                axis.set_xlabel(_sensitivity_param_label(param_x))
                axis.set_ylabel(_sensitivity_param_label(param_y))

                savings_table = savings_tables.get((param_y, param_x))
                if savings_table is not None:
                    _annotate_compute_savings_overlay(
                        axis, savings_table,
                        row_labels=table.index, col_labels=table.columns,
                        value_table=table,
                    )
                final_pop_table = final_pop_tables.get((param_y, param_x))
                if final_pop_table is not None:
                    _annotate_final_population_overlay(
                        axis, final_pop_table,
                        row_labels=table.index, col_labels=table.columns,
                        value_table=table,
                    )

        # ── Bottom half: delta vs baseline (plot-24 content) ──
        for row_idx, (param_y, param_x) in enumerate(pairs):
            axis_row = n_pairs + row_idx
            for col_idx, metric in enumerate(available_metrics):
                axis = axes_grid[axis_row, col_idx]

                baseline_table = baseline_tables.get((param_y, param_x, metric))
                if baseline_table is None:
                    axis.axis("off")
                    axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no baseline)")
                    continue

                agg = (
                    exp_df
                    .groupby([param_y, param_x], as_index=False, sort=True)[metric]
                    .mean()
                    .dropna(subset=[metric])
                )
                if agg.empty:
                    axis.axis("off")
                    axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no data)")
                    continue

                exp_table = agg.pivot(index=param_y, columns=param_x, values=metric)
                delta_table = (exp_table - baseline_table).dropna(how="all").dropna(axis=1, how="all")
                if delta_table.empty:
                    axis.axis("off")
                    axis.set_title(f"Δ {_sensitivity_metric_label(metric)} (no overlap)")
                    continue

                delta_array = delta_table.to_numpy(dtype="float64")
                max_abs = float(np.nanmax(np.abs(delta_array)))
                if max_abs < 1e-9:
                    max_abs = 1e-6
                cmap_name = _sensitivity_delta_cmap(metric)
                render_heatmap(
                    axis=axis,
                    table=delta_table,
                    title=f"Δ {_sensitivity_metric_label(metric)}",
                    cmap_name=cmap_name,
                    vmin=-max_abs,
                    vmax=max_abs,
                    value_format="{:+.3f}",
                    annotate_values=True,
                )
                axis.set_xlabel(_sensitivity_param_label(param_x))
                axis.set_ylabel(_sensitivity_param_label(param_y))

                savings_table = savings_tables.get((param_y, param_x))
                if savings_table is not None:
                    _annotate_compute_savings_overlay(
                        axis, savings_table,
                        row_labels=delta_table.index, col_labels=delta_table.columns,
                        value_table=delta_table,
                    )
                final_pop_table = final_pop_tables.get((param_y, param_x))
                if final_pop_table is not None:
                    _annotate_final_population_overlay(
                        axis, final_pop_table,
                        row_labels=delta_table.index, col_labels=delta_table.columns,
                        value_table=delta_table,
                    )

        fig.tight_layout()
        out = output_dir / f"26_combined_sensitivity_{sanitize_file_stem(experiment_name)}.png"
        _save_figure(fig, out)
        output_paths.append(out)
        log.info("Wrote combined sensitivity plot: %s", out)

    return output_paths


# ---------------------------------------------------------------------------
# Plot C (25): Efficiency ratio heatmap (recall × compute savings)
# ---------------------------------------------------------------------------

def render_sotf_efficiency_ratio(
    cross_batch_summary: pd.DataFrame,
    output_dir: Path,
    *,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    """Render efficiency-ratio heatmaps: recall × compute savings (plot 25).

    Defines a composite efficiency metric:

        efficiency = oracle_top_k_recall × compute_savings_fraction

    This rewards configurations that achieve high recall AND save compute.  Both
    factors are in [0, 1] so the product is also in [0, 1] and is easy to
    interpret.  A configuration that keeps many models (high recall, low savings)
    and one that aggressively prunes (low recall, high savings) will score lower
    than one that achieves a balanced trade-off.

    One figure per experiment.  Each figure has a grid of heatmaps:
      rows    = parameter pairs
      columns = [Absolute efficiency | Δ efficiency vs naive baseline]

    The delta column uses a diverging RdYlGn colormap: green means the predictor
    achieves a better quality-efficiency trade-off than naive, even accounting
    for any difference in how many models each configuration keeps.

    Args:
        cross_batch_summary: Pooled dataset_summary DataFrame from all batches.
        output_dir: Directory to write PNG (and mirrored PDF) files.
        subtitle: Subtitle line shown on every figure.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of output paths written.
    """
    required = {"oracle_top_k_recall", "compute_savings_fraction"}
    if cross_batch_summary.empty or not required.issubset(cross_batch_summary.columns):
        missing = required - set(cross_batch_summary.columns)
        log.info("Efficiency ratio: missing columns %s — skipping.", missing)
        return []

    param_cols = ["selection_start_epoch", "selection_interval", "selection_survival_rate"]
    missing_params = [c for c in param_cols if c not in cross_batch_summary.columns]
    if missing_params:
        log.warning("Efficiency ratio: missing param columns %s — skipping.", missing_params)
        return []

    varying = [
        col for col in param_cols
        if cross_batch_summary[col].dropna().nunique() > 1
    ]
    if len(varying) < 2:
        log.info("Efficiency ratio: fewer than 2 parameters vary — skipping.")
        return []

    pairs = [(py, px) for py, px in _SENSITIVITY_PARAM_PAIRS if py in varying and px in varying]
    if not pairs:
        return []

    # Build efficiency metric on a copy.
    df = cross_batch_summary.copy()
    df["efficiency_score"] = (
        pd.to_numeric(df["oracle_top_k_recall"], errors="coerce")
        * pd.to_numeric(df["compute_savings_fraction"], errors="coerce")
    )

    resolved = resolve_baseline_and_comparison_predictors(df["predictor"])
    if resolved is None:
        log.info("Efficiency ratio: no naive baseline found — showing absolute only.")
        baseline_predictor = None
        baseline_df = pd.DataFrame()
    else:
        _, baseline_predictor = resolved
        baseline_df = df[df["predictor"].astype(str) == baseline_predictor].copy()

    # Pre-build baseline efficiency pivot tables.
    baseline_tables: dict[tuple[str, str], pd.DataFrame] = {}
    if not baseline_df.empty:
        for param_y, param_x in pairs:
            agg = (
                baseline_df
                .groupby([param_y, param_x], as_index=False, sort=True)["efficiency_score"]
                .mean()
                .dropna(subset=["efficiency_score"])
            )
            if not agg.empty:
                baseline_tables[(param_y, param_x)] = agg.pivot(
                    index=param_y, columns=param_x, values="efficiency_score"
                )

    has_delta = bool(baseline_tables)
    n_cols = 2 if has_delta else 1

    non_baseline_rows = (
        df[df["predictor"].astype(str) != baseline_predictor]
        if baseline_predictor else df
    )
    experiments = (
        non_baseline_rows
        .loc[:, ["experiment_name", "predictor"]]
        .drop_duplicates("experiment_name")
        .sort_values("experiment_name")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "25_efficiency_ratio_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    try:
        for exp_row in experiments.itertuples(index=False):
            experiment_name = str(exp_row.experiment_name)
            predictor = str(exp_row.predictor)
            exp_df = df[df["experiment_name"].astype(str) == experiment_name].copy()
            if exp_df.empty:
                continue

            n_rows = len(pairs)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes_grid = np.array([[axes]])
            elif n_rows == 1:
                axes_grid = np.array([np.atleast_1d(axes)])
            elif n_cols == 1:
                axes_grid = np.atleast_2d(np.atleast_1d(axes)).T
            else:
                axes_grid = np.atleast_2d(axes)

            for row_idx, (param_y, param_x) in enumerate(pairs):
                agg = (
                    exp_df
                    .groupby([param_y, param_x], as_index=False, sort=True)["efficiency_score"]
                    .mean()
                    .dropna(subset=["efficiency_score"])
                )

                # --- Column 0: absolute efficiency ---
                abs_axis = axes_grid[row_idx, 0]
                if agg.empty:
                    abs_axis.axis("off")
                    abs_axis.set_title("Recall × Compute Savings (no data)")
                else:
                    table = agg.pivot(index=param_y, columns=param_x, values="efficiency_score")
                    render_heatmap(
                        axis=abs_axis,
                        table=table,
                        title="Recall × Compute Savings",
                        cmap_name="viridis",
                        vmin=0.0,
                        vmax=1.0,
                        value_format="{:.3f}",
                        annotate_values=True,
                    )
                    abs_axis.set_xlabel(_sensitivity_param_label(param_x))
                    abs_axis.set_ylabel(_sensitivity_param_label(param_y))

                # --- Column 1: delta vs baseline ---
                if not has_delta:
                    continue
                delta_axis = axes_grid[row_idx, 1]
                baseline_table = baseline_tables.get((param_y, param_x))
                if baseline_table is None or agg.empty:
                    delta_axis.axis("off")
                    delta_axis.set_title("Δ Recall × Compute Savings (no baseline)")
                    continue

                exp_table = agg.pivot(index=param_y, columns=param_x, values="efficiency_score")
                delta_table = (exp_table - baseline_table).dropna(how="all").dropna(axis=1, how="all")
                if delta_table.empty:
                    delta_axis.axis("off")
                    delta_axis.set_title("Δ Recall × Compute Savings (no overlap)")
                    continue

                delta_array = delta_table.to_numpy(dtype="float64")
                max_abs = float(np.nanmax(np.abs(delta_array)))
                if max_abs < 1e-9:
                    max_abs = 1e-6
                render_heatmap(
                    axis=delta_axis,
                    table=delta_table,
                    title=f"Δ (Recall × Savings) vs {predictor_display_name(baseline_predictor)}",
                    cmap_name="RdYlGn",
                    vmin=-max_abs,
                    vmax=max_abs,
                    value_format="{:+.3f}",
                    annotate_values=True,
                )
                delta_axis.set_xlabel(_sensitivity_param_label(param_x))
                delta_axis.set_ylabel(_sensitivity_param_label(param_y))

            layout_top = apply_figure_titles(
                fig,
                title=f"Efficiency Ratio (Recall × Compute Savings) — {experiment_name}",
                subtitle=subtitle,
                secondary_subtitle=secondary_subtitle,
            )
            fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True
            out = output_dir / f"25_efficiency_ratio_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote efficiency ratio plot: %s", out)
    finally:
        if overview_pdf is not None:
            overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote efficiency ratio overview PDF: %s", overview_path)

    return output_paths


def _load_ordered_run_details(latest_runs: pd.DataFrame) -> list[pd.DataFrame]:
    """Load round_details for each run in order, tagging each frame with run metadata."""
    sort_cols = [c for c in ("experiment_order", "dataset_number", "experiment_name", "predictor") if c in latest_runs.columns]
    ordered = latest_runs.sort_values(sort_cols, kind="stable").reset_index(drop=True) if sort_cols else latest_runs.reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for _, row in ordered.iterrows():
        path = Path(str(row["round_details_path"]))
        if not path.exists():
            log.warning("round_details not found: %s", path)
            continue
        df = pd.read_parquet(path)
        for col in ("experiment_name", "dataset_label", "predictor", "target_column", "dataset_number"):
            if col in row.index and col not in df.columns:
                df[col] = row[col]
        frames.append(df)
    return frames


def render_population_accuracy_comparison_grid(
    latest_runs: pd.DataFrame,
    output_path: Path,
    *,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    """Single figure: grouped boxplots per decision epoch, one colour per experiment.

    X-axis: decision epoch. At each epoch position all experiments are drawn side by
    side as narrow boxes so their distributions can be compared directly. Each
    experiment gets a unique colour from the tab20 palette. A legend maps colours to
    experiment names.

    Args:
        latest_runs (pd.DataFrame): One row per run, must contain `round_details_path`.
        output_path (Path): Destination path for the generated output.
        subtitle (str): Subtitle text for the output.
        secondary_subtitle (str): Optional secondary subtitle.
    """
    if latest_runs.empty or "round_details_path" not in latest_runs.columns:
        return

    ordered_frames = _load_ordered_run_details(latest_runs)
    if not ordered_frames:
        return

    all_details = pd.concat(ordered_frames, ignore_index=True)

    if "target_column" in all_details.columns:
        target_col = str(all_details["target_column"].mode().iloc[0])
    else:
        for candidate in ("test_acc", "train_acc"):
            if candidate in all_details.columns:
                target_col = candidate
                break
        else:
            return

    if target_col not in all_details.columns or "experiment_name" not in all_details.columns:
        return

    exp_order: list[str] = list(dict.fromkeys(all_details["experiment_name"].astype(str)))
    epochs = sorted(all_details["decision_epoch"].dropna().unique())
    if not epochs:
        return

    n_experiments = len(exp_order)
    n_epochs = len(epochs)

    cmap = plt.get_cmap("tab20")
    exp_colors = {name: cmap(i % 20 / 20) for i, name in enumerate(exp_order)}

    # Each epoch group is centred at its integer index.
    # Box width is 80 % of the group divided equally across experiments.
    group_fill = 0.80
    box_width = group_fill / n_experiments
    group_centers = list(range(n_epochs))

    # Figure width: at least 0.12 in per box, minimum 12 in.
    fig_width = max(12.0, n_epochs * max(1.5, n_experiments * 0.12))
    legend_cols = min(4, n_experiments)
    legend_rows = math.ceil(n_experiments / legend_cols)
    legend_height_in = 0.30 * legend_rows + 0.3
    fig, axis = plt.subplots(figsize=(fig_width, 5.0 + legend_height_in))

    for epoch_idx, (epoch, group_center) in enumerate(zip(epochs, group_centers)):
        epoch_df = all_details[all_details["decision_epoch"] == epoch]
        for exp_idx, exp_name in enumerate(exp_order):
            offset = (exp_idx - (n_experiments - 1) / 2.0) * box_width
            pos = group_center + offset
            values = epoch_df.loc[
                epoch_df["experiment_name"].astype(str) == exp_name, target_col
            ].dropna().to_numpy(dtype="float64")
            if len(values) == 0:
                continue
            bplot = axis.boxplot(
                [values],
                positions=[pos],
                widths=box_width * 0.9,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=0.6),
                capprops=dict(color="black", linewidth=0.6),
                flierprops=dict(marker=".", color="black", markersize=2, alpha=0.3),
                boxprops=dict(linewidth=0.6),
            )
            bplot["boxes"][0].set_facecolor(exp_colors[exp_name])

    axis.set_xticks(group_centers)
    axis.set_xticklabels([str(int(ep)) for ep in epochs])
    axis.set_xlabel("Epoch")
    axis.set_ylabel(target_col.replace("_", " "))
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlim(-0.55, n_epochs - 0.45)

    legend_handles = [
        Patch(facecolor=exp_colors[name], edgecolor="black", linewidth=0.5, label=name)
        for name in exp_order
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=legend_cols,
        frameon=False,
        fontsize=8,
    )

    layout_top = apply_figure_titles(
        fig,
        title="Experiment Comparison: Remaining Population Accuracy per Epoch",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Cross-dataset win-count plots
# ---------------------------------------------------------------------------

_CATEGORY_ORDER: list[str] = ["early", "mid", "end"]
_START_EPOCH_CATEGORY_LABELS: dict[str, str] = {
    "early": "Start",
    "start": "Start",
    "mid": "Mid",
    "middle": "Mid",
    "end": "End",
    "late": "End",
}
_START_EPOCH_CATEGORY_COLORS: dict[str, str] = {
    "early": "#457b9d",
    "mid": "#e9c46a",
    "end": "#2a9d8f",
}
_WIN_METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "best_survivor_regret": False,
    "oracle_top_k_recall": True,
    "compute_savings_fraction": True,
    "decision_accuracy_weighted": True,
    "false_kill_rate_weighted": False,
    "best_survivor_rank_percentile": True,
}
_WIN_METRIC_LABELS: dict[str, str] = {
    "best_survivor_regret": "Best Survivor Gap (↓ lower = better)",
    "oracle_top_k_recall": "Final Population Recall (↑ higher = better)",
    "compute_savings_fraction": "Compute Savings (↑)",
    "decision_accuracy_weighted": "Decision Accuracy (↑)",
    "false_kill_rate_weighted": "False Kill Rate (↓)",
    "best_survivor_rank_percentile": "Best Survivor Percentile (↑)",
}


def _normalise_start_epoch_category(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"early", "start"}:
        return "early"
    if text in {"mid", "middle"}:
        return "mid"
    if text in {"end", "late"}:
        return "end"
    return None


def _format_survival_rate(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _category_display_label(category: str) -> str:
    return _START_EPOCH_CATEGORY_LABELS.get(category, str(category).capitalize())


def _derive_start_epoch_categories(frame: pd.DataFrame, *, dataset_col: str) -> pd.Series:
    categories = pd.Series(index=frame.index, dtype="object")
    if "selection_start_epoch" not in frame.columns:
        return categories

    epochs = pd.to_numeric(frame["selection_start_epoch"], errors="coerce")
    for _, group in frame.assign(_numeric_start_epoch=epochs).groupby(dataset_col, sort=False):
        unique_epochs = sorted(group["_numeric_start_epoch"].dropna().unique().tolist())
        if not unique_epochs:
            continue
        if len(unique_epochs) == 1:
            mapping = {unique_epochs[0]: "early"}
        elif len(unique_epochs) == 2:
            mapping = {unique_epochs[0]: "early", unique_epochs[-1]: "end"}
        else:
            mapping = {
                epoch: ("early" if index == 0 else "end" if index == len(unique_epochs) - 1 else "mid")
                for index, epoch in enumerate(unique_epochs)
            }
        categories.loc[group.index] = group["_numeric_start_epoch"].map(mapping)
    return categories


def build_cross_dataset_best_survivor_regret_bars(
    pooled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[float]]:
    """Build cross-dataset best-survivor GAP values for grouped bar lanes."""
    required_columns = {"selection_survival_rate", "best_survivor_regret"}
    dataset_col = "dataset_name" if "dataset_name" in pooled_df.columns else "dataset_label"
    if pooled_df.empty or dataset_col not in pooled_df.columns or not required_columns.issubset(pooled_df.columns):
        return pd.DataFrame(), [], [], []

    select_cols = [
        dataset_col,
        "selection_survival_rate",
        "best_survivor_regret",
    ]
    for optional_col in (
        "dataset_number",
        "experiment_name",
        "experiment_order",
        "predictor",
        "selection_start_epoch",
        "start_epoch_category",
    ):
        if optional_col in pooled_df.columns:
            select_cols.append(optional_col)

    df = pooled_df.loc[:, select_cols].copy()
    df["_dataset_label"] = df[dataset_col].astype(str)
    df["_survival_rate"] = pd.to_numeric(df["selection_survival_rate"], errors="coerce")
    df["_best_survivor_regret"] = pd.to_numeric(df["best_survivor_regret"], errors="coerce")
    if "experiment_name" not in df.columns:
        df["experiment_name"] = "all_experiments"
    if "predictor" not in df.columns:
        df["predictor"] = ""
    df["experiment_name"] = df["experiment_name"].fillna("all_experiments").astype(str)
    df["predictor"] = df["predictor"].fillna("").astype(str)

    if "start_epoch_category" in df.columns:
        df["_start_epoch_category"] = df["start_epoch_category"].map(_normalise_start_epoch_category)
    else:
        df["_start_epoch_category"] = np.nan
    derived_categories = _derive_start_epoch_categories(df, dataset_col="_dataset_label")
    df["_start_epoch_category"] = df["_start_epoch_category"].fillna(derived_categories)

    df = df.dropna(subset=["_dataset_label", "_survival_rate", "_best_survivor_regret", "_start_epoch_category"])
    df = df[df["_start_epoch_category"].isin(_CATEGORY_ORDER)]
    if df.empty:
        return pd.DataFrame(), [], [], []

    dataset_order = sorted(df["_dataset_label"].dropna().astype(str).unique().tolist(), key=dataset_sort_key)
    categories = [category for category in _CATEGORY_ORDER if category in set(df["_start_epoch_category"].astype(str))]
    survival_rates = sorted(float(rate) for rate in df["_survival_rate"].dropna().unique().tolist())
    if not dataset_order or not categories or not survival_rates:
        return pd.DataFrame(), [], [], []

    group_keys = ["experiment_name", "predictor", "_dataset_label", "_survival_rate", "_start_epoch_category"]
    agg_spec: dict = {
        "mean_best_survivor_regret": ("_best_survivor_regret", "mean"),
        "run_count": ("_best_survivor_regret", "size"),
    }
    if "experiment_order" in df.columns:
        df["experiment_order"] = pd.to_numeric(df["experiment_order"], errors="coerce")
        agg_spec["experiment_order"] = ("experiment_order", "min")

    aggregated = (
        df.groupby(group_keys, as_index=False, sort=False)
        .agg(**agg_spec)
        .rename(
            columns={
                "_dataset_label": "dataset_label",
                "_survival_rate": "selection_survival_rate",
                "_start_epoch_category": "start_epoch_category",
            }
        )
    )
    if "experiment_order" not in aggregated.columns:
        experiment_order = (
            aggregated.loc[:, ["experiment_name"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(experiment_order=lambda frame: np.arange(len(frame), dtype="int64"))
        )
        aggregated = aggregated.merge(experiment_order, on="experiment_name", how="left")

    aggregated = aggregated.sort_values(
        ["experiment_order", "experiment_name", "predictor", "dataset_label", "selection_survival_rate", "start_epoch_category"],
        kind="stable",
    ).reset_index(drop=True)
    return aggregated, dataset_order, categories, survival_rates


def render_cross_dataset_best_survivor_regret_bars(
    pooled_df: pd.DataFrame,
    output_dir: Path,
    *,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    """Render cross-dataset grouped bar lanes for best-survivor GAP.

    One figure is generated per experiment. Each figure uses datasets on the
    x-axis, best-survivor GAP on the y-axis, grouped bars for start/mid/end,
    and one horizontal lane per survival rate.
    """
    aggregated, dataset_order, categories, survival_rates = build_cross_dataset_best_survivor_regret_bars(pooled_df)
    if aggregated.empty or not dataset_order or not categories or not survival_rates:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "32_cross_dataset_best_survivor_gap_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    experiments = (
        aggregated.loc[:, ["experiment_name", "predictor", "experiment_order"]]
        .drop_duplicates("experiment_name")
        .sort_values(["experiment_order", "experiment_name"], kind="stable")
    )
    x = np.arange(len(dataset_order))
    bar_group_width = 0.78
    bar_width = bar_group_width / max(1, len(categories))
    fig_width = max(11.0, 0.75 * len(dataset_order) + 3.0)
    fig_height = max(8.0, 2.0 * len(survival_rates) + 2.0)

    try:
        for exp_row in experiments.itertuples(index=False):
            experiment_name = str(exp_row.experiment_name)
            predictor = str(exp_row.predictor)
            exp_df = aggregated[aggregated["experiment_name"].astype(str) == experiment_name].copy()
            if exp_df.empty:
                continue

            fig, axes = plt.subplots(len(survival_rates), 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
            axes_list = list(np.atleast_1d(axes))
            finite_values = exp_df["mean_best_survivor_regret"].to_numpy(dtype="float64")
            finite_values = finite_values[np.isfinite(finite_values)]
            y_max = float(finite_values.max()) if finite_values.size else 1.0
            y_top = max(0.01, y_max * 1.15)

            for axis, survival_rate in zip(axes_list, survival_rates, strict=True):
                rate_df = exp_df[
                    np.isclose(exp_df["selection_survival_rate"].to_numpy(dtype="float64"), float(survival_rate))
                ].copy()
                for category_index, category in enumerate(categories):
                    category_values = (
                        rate_df[rate_df["start_epoch_category"] == category]
                        .set_index("dataset_label")
                        .reindex(dataset_order)["mean_best_survivor_regret"]
                        .to_numpy(dtype="float64")
                    )
                    offset = (category_index - (len(categories) - 1) / 2.0) * bar_width
                    axis.bar(
                        x + offset,
                        category_values,
                        width=bar_width * 0.92,
                        color=_START_EPOCH_CATEGORY_COLORS.get(category, "#6c757d"),
                        edgecolor="black",
                        linewidth=0.5,
                        label=_category_display_label(category),
                    )
                axis.set_ylabel(f"Survival rate {_format_survival_rate(survival_rate)}\nBest Survivor GAP")
                axis.set_ylim(0.0, y_top)
                axis.grid(axis="y", alpha=0.25)
                axis.margins(x=0.01)

            axes_list[-1].set_xticks(x, dataset_order, rotation=45, ha="right")
            axes_list[-1].set_xlabel("Dataset")
            handles = [
                Patch(
                    facecolor=_START_EPOCH_CATEGORY_COLORS.get(category, "#6c757d"),
                    edgecolor="black",
                    linewidth=0.5,
                    label=_category_display_label(category),
                )
                for category in categories
            ]
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)

            predictor_part = predictor_display_name(predictor) if predictor else ""
            exp_subtitle = predictor_part
            if subtitle:
                exp_subtitle = f"{exp_subtitle} | {subtitle}" if exp_subtitle else subtitle
            layout_top = apply_figure_titles(
                fig,
                title=f"Best Survivor GAP by Dataset - {experiment_name}",
                subtitle=exp_subtitle,
                secondary_subtitle=secondary_subtitle,
            )
            fig.tight_layout(rect=(0, 0.04, 1, layout_top), pad=0.1, h_pad=0.45)

            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True

            out = output_dir / f"32_cross_dataset_best_survivor_gap_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote cross-dataset best survivor GAP bars: %s", out)
    finally:
        if overview_pdf is not None:
            overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote cross-dataset best survivor GAP overview PDF: %s", overview_path)
    return output_paths


def _short_cross_dataset_experiment_label(value: object) -> str:
    label = str(value)
    replacements = {
        "weight_statistics_": "ws_",
        "latent_averaged_representations_": "avg_",
        "latent_tokens_representations_": "tok_",
        "latent_window_representations_": "win_",
        "_with_perf": "+perf",
        "_perf_only": "_perf",
        "_online_transformer": "_on_transformer",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def _build_cross_dataset_gap_dominance(
    pooled_df: pd.DataFrame,
    *,
    baseline_predictor: str = "naive",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = {"dataset_name", "experiment_name", "predictor", "selection_survival_rate", "best_survivor_regret"}
    if pooled_df.empty or not required.issubset(pooled_df.columns):
        return pd.DataFrame(), pd.DataFrame(), []

    df = pooled_df.copy()
    if "start_epoch_category" not in df.columns:
        df["start_epoch_category"] = _derive_start_epoch_categories(df, dataset_col="dataset_name")
    else:
        normalised_categories = df["start_epoch_category"].map(_normalise_start_epoch_category)
        df["start_epoch_category"] = normalised_categories.fillna(
            _derive_start_epoch_categories(df, dataset_col="dataset_name")
        )
    df["best_survivor_regret"] = pd.to_numeric(df["best_survivor_regret"], errors="coerce")
    df["selection_survival_rate"] = pd.to_numeric(df["selection_survival_rate"], errors="coerce")
    df = df.dropna(subset=["dataset_name", "experiment_name", "predictor", "start_epoch_category", "selection_survival_rate", "best_survivor_regret"])
    df = df[df["start_epoch_category"].isin(_CATEGORY_ORDER)]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    match_keys = ["dataset_name", "start_epoch_category", "selection_survival_rate"]
    baseline = (
        df[df["predictor"].astype(str) == baseline_predictor]
        .groupby(match_keys, as_index=False, sort=False)["best_survivor_regret"]
        .mean()
        .rename(columns={"best_survivor_regret": "baseline_best_survivor_regret"})
    )
    candidates = df[df["predictor"].astype(str) != baseline_predictor].copy()
    if baseline.empty or candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    merged = candidates.merge(baseline, on=match_keys, how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    merged["delta_gap_vs_baseline"] = merged["baseline_best_survivor_regret"] - merged["best_survivor_regret"]
    merged["wins_baseline"] = merged["delta_gap_vs_baseline"] > 1e-12
    merged["setting_label"] = (
        merged["experiment_name"].map(_short_cross_dataset_experiment_label)
        + " | "
        + merged["start_epoch_category"].map(_category_display_label).str.lower()
        + " | sr="
        + merged["selection_survival_rate"].map(_format_survival_rate)
    )

    summary = (
        merged.groupby(
            ["experiment_name", "predictor", "start_epoch_category", "selection_survival_rate", "setting_label"],
            as_index=False,
            sort=False,
        )
        .agg(
            wins=("wins_baseline", "sum"),
            datasets=("dataset_name", "nunique"),
            median_delta=("delta_gap_vs_baseline", "median"),
            worst_delta=("delta_gap_vs_baseline", "min"),
            mean_delta=("delta_gap_vs_baseline", "mean"),
            mean_gap=("best_survivor_regret", "mean"),
        )
    )
    summary["win_rate"] = summary["wins"] / summary["datasets"]
    summary["non_losing"] = summary["worst_delta"] >= -1e-12
    summary = summary.sort_values(
        ["wins", "non_losing", "worst_delta", "median_delta", "setting_label"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    dataset_order = sorted(merged["dataset_name"].dropna().astype(str).unique().tolist(), key=dataset_sort_key)
    return merged, summary, dataset_order


def render_cross_dataset_gap_dominance_matrix(
    pooled_df: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 28,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    merged, summary, dataset_order = _build_cross_dataset_gap_dominance(pooled_df)
    if merged.empty or summary.empty or not dataset_order:
        return []

    top_settings = summary.head(max(1, int(top_n)))["setting_label"].tolist()
    matrix = (
        merged[merged["setting_label"].isin(top_settings)]
        .pivot_table(index="setting_label", columns="dataset_name", values="delta_gap_vs_baseline", aggfunc="mean")
        .reindex(index=top_settings, columns=dataset_order)
    )
    if matrix.empty:
        return []

    values = np.abs(matrix.to_numpy(dtype="float64"))
    finite = values[np.isfinite(values)]
    scale = float(np.nanpercentile(finite, 92)) if finite.size else 0.02
    scale = min(max(scale, 0.01), 0.05)
    clipped_matrix = np.clip(matrix.to_numpy(dtype="float64"), -scale, scale)

    fig_width = max(12.0, 0.7 * len(dataset_order) + 6.0)
    fig_height = max(7.0, 0.35 * len(top_settings) + 2.5)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    image = axis.imshow(
        clipped_matrix,
        aspect="auto",
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
    )

    axis.set_xticks(np.arange(len(dataset_order)), dataset_order, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(top_settings)), top_settings, fontsize=8)
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Predictor / feature setting")
    summary_by_label = summary.set_index("setting_label")
    for row_idx, setting in enumerate(top_settings):
        row = summary_by_label.loc[setting]
        status = "all non-losing" if bool(row["non_losing"]) else f"worst {row['worst_delta']:+.3f}"
        axis.text(
            len(dataset_order) + 0.3,
            row_idx,
            f"{int(row['wins'])}/{int(row['datasets'])} wins | {status}",
            va="center",
            fontsize=8,
        )
    axis.set_xlim(-0.5, len(dataset_order) + 3.8)

    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Best Survivor GAP improvement vs naive")
    layout_top = apply_figure_titles(
        fig,
        title="Cross-Dataset GAP Dominance Matrix",
        subtitle=(
            "Cell = naive GAP - candidate GAP; green is lower GAP than naive. "
            "Rows sorted by win count and worst dataset."
            + (f" | {subtitle}" if subtitle else "")
        ),
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "33_cross_dataset_gap_dominance_matrix.png"
    _save_figure(fig, out)
    summary.to_csv(output_dir.parent / "cross_dataset_gap_dominance_summary.csv", index=False)
    log.info("Wrote cross-dataset GAP dominance matrix: %s", out)
    return [out]


def render_cross_dataset_gap_robustness_scorecard(
    pooled_df: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 34,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    _, summary, _ = _build_cross_dataset_gap_dominance(pooled_df)
    if summary.empty:
        return []

    plot_rows = summary.head(max(1, int(top_n))).copy().iloc[::-1]
    colors = np.where(
        plot_rows["wins"].to_numpy(dtype="float64") == plot_rows["datasets"].to_numpy(dtype="float64"),
        "#006d77",
        np.where(plot_rows["non_losing"].to_numpy(dtype=bool), "#e9c46a", "#d62828"),
    )
    fig_height = max(6.0, 0.28 * len(plot_rows) + 2.0)
    fig, axis = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(plot_rows))
    axis.barh(y, plot_rows["wins"], color=colors, edgecolor="black", linewidth=0.4)
    axis.set_yticks(y, plot_rows["setting_label"], fontsize=8)
    axis.set_xlabel("Datasets with strict improvement over naive")
    axis.set_xlim(0, max(float(plot_rows["datasets"].max()), 1.0) + 0.8)
    axis.grid(axis="x", alpha=0.25)
    for y_pos, row in zip(y, plot_rows.itertuples(index=False), strict=True):
        status = "all non-losing" if bool(row.non_losing) else f"worst {row.worst_delta:+.3f}"
        axis.text(
            float(row.wins) + 0.08,
            y_pos,
            f"{int(row.wins)}/{int(row.datasets)} | median {row.median_delta:+.3f} | {status}",
            va="center",
            fontsize=8,
        )
    axis.legend(
        handles=[
            Patch(facecolor="#006d77", edgecolor="black", label="strictly wins all datasets"),
            Patch(facecolor="#e9c46a", edgecolor="black", label="ties or wins all datasets"),
            Patch(facecolor="#d62828", edgecolor="black", label="loses at least one dataset"),
        ],
        frameon=False,
        loc="lower right",
    )
    layout_top = apply_figure_titles(
        fig,
        title="Cross-Dataset GAP Robustness Scorecard",
        subtitle="Compact view of whether one setting wins everywhere." + (f" | {subtitle}" if subtitle else ""),
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "34_cross_dataset_gap_robustness_scorecard.png"
    _save_figure(fig, out)
    log.info("Wrote cross-dataset GAP robustness scorecard: %s", out)
    return [out]


def render_cross_dataset_gap_parameter_map(
    pooled_df: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 18,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    _, summary, _ = _build_cross_dataset_gap_dominance(pooled_df)
    if summary.empty:
        return []

    experiments = (
        summary.groupby("experiment_name", as_index=False)
        .agg(
            best_wins=("wins", "max"),
            best_non_losing=("non_losing", "max"),
            best_worst=("worst_delta", "max"),
        )
        .sort_values(["best_wins", "best_non_losing", "best_worst", "experiment_name"], ascending=[False, False, False, True], kind="stable")
        .head(max(1, int(top_n)))["experiment_name"]
        .tolist()
    )
    categories = [cat for cat in _CATEGORY_ORDER if cat in set(summary["start_epoch_category"].astype(str))]
    survival_rates = sorted(float(rate) for rate in summary["selection_survival_rate"].dropna().unique().tolist())
    if not experiments or not categories or not survival_rates:
        return []

    col_keys = [(cat, rate) for cat in categories for rate in survival_rates]
    matrix = np.full((len(experiments), len(col_keys)), np.nan)
    non_losing = np.zeros_like(matrix, dtype=bool)
    for row_idx, experiment_name in enumerate(experiments):
        exp_rows = summary[summary["experiment_name"] == experiment_name]
        for col_idx, (category, survival_rate) in enumerate(col_keys):
            matches = exp_rows[
                (exp_rows["start_epoch_category"] == category)
                & np.isclose(exp_rows["selection_survival_rate"].to_numpy(dtype="float64"), float(survival_rate))
            ]
            if matches.empty:
                continue
            row = matches.iloc[0]
            matrix[row_idx, col_idx] = float(row["win_rate"])
            non_losing[row_idx, col_idx] = bool(row["non_losing"])

    fig_width = max(10.0, 0.7 * len(col_keys) + 3.0)
    fig_height = max(5.5, 0.38 * len(experiments) + 2.5)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
    for row_idx in range(len(experiments)):
        for col_idx in range(len(col_keys)):
            if np.isfinite(matrix[row_idx, col_idx]) and non_losing[row_idx, col_idx]:
                axis.text(col_idx, row_idx, "OK", ha="center", va="center", fontsize=6, color="#1d3557", fontweight="bold")
    for boundary in range(len(survival_rates), len(col_keys), len(survival_rates)):
        axis.axvline(boundary - 0.5, color="white", linewidth=2.2)
    axis.set_xticks(
        np.arange(len(col_keys)),
        [f"{_category_display_label(category).lower()}\n{_format_survival_rate(rate)}" for category, rate in col_keys],
        fontsize=8,
    )
    axis.set_yticks(np.arange(len(experiments)), [_short_cross_dataset_experiment_label(name) for name in experiments], fontsize=8)
    axis.set_xlabel("Start epoch category and survival rate")
    axis.set_ylabel("Predictor / feature combination")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Win fraction vs naive")
    layout_top = apply_figure_titles(
        fig,
        title="Cross-Dataset GAP Parameter Robustness Map",
        subtitle="Color = fraction of datasets strictly beaten vs naive; OK means no dataset worse than naive."
        + (f" | {subtitle}" if subtitle else ""),
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "35_cross_dataset_gap_parameter_robustness_map.png"
    _save_figure(fig, out)
    log.info("Wrote cross-dataset GAP parameter robustness map: %s", out)
    return [out]


def _cross_dataset_win_counts(
    pooled_df: pd.DataFrame,
    *,
    win_metric: str,
    baseline_predictor: str = "naive",
) -> pd.DataFrame:
    """Compute per-(experiment, survival_rate, start_epoch_category) win counts.

    For each matching (dataset_name, survival_rate, start_epoch_category) pair,
    determines whether the experiment outperforms the baseline on *win_metric*.

    Returns a DataFrame with columns:
        experiment_name, predictor, selection_survival_rate, start_epoch_category,
        wins, total_datasets
    """
    higher_better = _WIN_METRIC_HIGHER_IS_BETTER.get(win_metric, True)
    group_keys = ["dataset_name", "selection_survival_rate", "start_epoch_category"]
    merge_keys = group_keys

    baseline_df = pooled_df[pooled_df["predictor"].astype(str) == baseline_predictor].copy()
    nonbaseline_df = pooled_df[pooled_df["predictor"].astype(str) != baseline_predictor].copy()

    if baseline_df.empty or nonbaseline_df.empty:
        return pd.DataFrame()

    if win_metric not in baseline_df.columns or win_metric not in nonbaseline_df.columns:
        log.warning("win_metric %r not found in pooled_df columns — skipping win count.", win_metric)
        return pd.DataFrame()

    baseline_agg = (
        baseline_df.groupby(merge_keys, as_index=False)[win_metric]
        .mean()
        .rename(columns={win_metric: "_baseline_metric"})
    )

    merged = nonbaseline_df.merge(baseline_agg, on=merge_keys, how="inner")
    if merged.empty:
        return pd.DataFrame()

    exp_metric = pd.to_numeric(merged[win_metric], errors="coerce")
    base_metric = pd.to_numeric(merged["_baseline_metric"], errors="coerce")
    if higher_better:
        merged["_win"] = (exp_metric > base_metric).astype(int)
    else:
        merged["_win"] = (exp_metric < base_metric).astype(int)

    agg_keys = ["experiment_name", "predictor", "selection_survival_rate", "start_epoch_category"]
    result = (
        merged.groupby(agg_keys, as_index=False)
        .agg(wins=("_win", "sum"), total_datasets=("_win", "count"))
    )
    return result.sort_values(agg_keys).reset_index(drop=True)


def render_cross_dataset_win_heatmap(
    pooled_df: pd.DataFrame,
    output_dir: Path,
    *,
    win_metric: str = "best_survivor_regret",
    n_datasets: int = 0,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    """Render per-experiment win-count heatmaps across datasets.

    One figure per non-baseline experiment.  Each figure shows a 2-D heatmap:
      y-axis — survival rate
      x-axis — start-epoch category (early / mid / end)
      cell   — number of datasets where this (experiment, rate, category) setting
                outperformed the naive baseline on *win_metric*

    Cells are annotated with ``wins / total`` and coloured from white (0 wins) to
    dark green (all datasets win).

    Args:
        pooled_df: Concatenated dataset_summary rows from all datasets, with
            extra columns ``dataset_name`` and ``start_epoch_category``.
        output_dir: Directory for PNG (and mirrored PDF) output files.
        win_metric: Column name used to decide outperformance.
        n_datasets: Total number of distinct datasets (for figure title).
        subtitle: Subtitle shown on every figure.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of written output paths.
    """
    if pooled_df.empty or "start_epoch_category" not in pooled_df.columns:
        return []

    win_counts = _cross_dataset_win_counts(pooled_df, win_metric=win_metric)
    if win_counts.empty:
        return []

    categories = [c for c in _CATEGORY_ORDER if c in win_counts["start_epoch_category"].values]
    if not categories:
        return []

    survival_rates = sorted(win_counts["selection_survival_rate"].dropna().unique(), reverse=True)
    n_ds = n_datasets or int(win_counts["total_datasets"].max())

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    overview_path = _pdf_dir_for_output_dir(output_dir) / "30_cross_dataset_win_heatmap_overview.pdf"
    overview_pdf: PdfPages | None = None
    overview_written = False

    experiments = (
        win_counts[["experiment_name", "predictor"]]
        .drop_duplicates("experiment_name")
        .sort_values("experiment_name")
    )

    metric_label = _WIN_METRIC_LABELS.get(win_metric, win_metric.replace("_", " ").title())

    try:
        for exp_row in experiments.itertuples(index=False):
            experiment_name = str(exp_row.experiment_name)
            predictor = str(exp_row.predictor)
            exp_wins = win_counts[win_counts["experiment_name"] == experiment_name]

            matrix = np.full((len(survival_rates), len(categories)), np.nan)
            total_matrix = np.zeros_like(matrix)
            for i, sr in enumerate(survival_rates):
                for j, cat in enumerate(categories):
                    row = exp_wins[
                        np.isclose(exp_wins["selection_survival_rate"].to_numpy(dtype="float64"), float(sr))
                        & (exp_wins["start_epoch_category"] == cat)
                    ]
                    if not row.empty:
                        matrix[i, j] = float(row["wins"].iloc[0])
                        total_matrix[i, j] = float(row["total_datasets"].iloc[0])

            fig, axis = plt.subplots(figsize=(max(5.0, len(categories) * 2.0), max(4.0, len(survival_rates) * 1.2 + 1.5)))

            masked = np.ma.masked_invalid(matrix)
            im = axis.imshow(masked, aspect="auto", cmap="YlGn", vmin=0.0, vmax=float(n_ds))

            axis.set_xticks(np.arange(len(categories)), [c.capitalize() for c in categories])
            axis.set_yticks(np.arange(len(survival_rates)), [f"{sr:.0%}" for sr in survival_rates])
            axis.set_xlabel("Start-epoch category")
            axis.set_ylabel("Survival rate")

            for i in range(len(survival_rates)):
                for j in range(len(categories)):
                    wins_val = matrix[i, j]
                    tot_val = total_matrix[i, j]
                    if np.isnan(wins_val):
                        continue
                    label = f"{int(wins_val)}/{int(tot_val)}"
                    norm_val = wins_val / n_ds if n_ds > 0 else 0.0
                    text_color = "white" if norm_val > 0.55 else "black"
                    axis.text(j, i, label, ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")

            cbar = plt.colorbar(im, ax=axis, fraction=0.03, pad=0.04)
            cbar.set_label(f"# datasets won (out of {n_ds})", fontsize=9)
            cbar.set_ticks(np.arange(0, n_ds + 1))

            exp_subtitle = (
                f"{predictor_display_name(predictor)} vs Naive Baseline | metric: {metric_label}"
                + (f" | {subtitle}" if subtitle else "")
            )
            layout_top = apply_figure_titles(
                fig,
                title=f"Cross-Dataset Win Count — {experiment_name}",
                subtitle=exp_subtitle,
                secondary_subtitle=secondary_subtitle,
            )
            fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)

            if overview_pdf is None:
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_pdf = PdfPages(overview_path)
            overview_pdf.savefig(fig, bbox_inches="tight")
            overview_written = True

            out = output_dir / f"30_cross_dataset_wins_{sanitize_file_stem(experiment_name)}.png"
            _save_figure(fig, out)
            output_paths.append(out)
            log.info("Wrote cross-dataset win heatmap: %s", out)
    finally:
        if overview_pdf is not None:
            overview_pdf.close()

    if overview_written:
        output_paths.append(overview_path)
        log.info("Wrote cross-dataset win overview PDF: %s", overview_path)

    return output_paths


def render_cross_dataset_win_matrix(
    pooled_df: pd.DataFrame,
    output_dir: Path,
    *,
    win_metric: str = "best_survivor_regret",
    n_datasets: int = 0,
    subtitle: str = "",
    secondary_subtitle: str = "",
) -> list[Path]:
    """Render a single overview win-fraction matrix for all experiments.

    One figure, all non-baseline experiments at a glance:
      y-axis — experiment name (sorted by total wins descending)
      x-axis — (start-epoch category, survival rate) pairs, grouped by category
      colour  — win fraction = wins / total_datasets  (0 = red, 0.5 = yellow, 1 = green)
      cell    — annotated with ``wins/total``

    Group separators (vertical lines) are drawn between categories so the
    early / mid / end grouping remains legible.

    Args:
        pooled_df: Concatenated dataset_summary rows with ``dataset_name`` and
            ``start_epoch_category`` columns.
        output_dir: Directory for PNG (and mirrored PDF) output.
        win_metric: Column name used for outperformance comparison.
        n_datasets: Total number of distinct datasets (for figure title).
        subtitle: Subtitle line.
        secondary_subtitle: Optional second subtitle line.

    Returns:
        List of written output paths.
    """
    if pooled_df.empty or "start_epoch_category" not in pooled_df.columns:
        return []

    win_counts = _cross_dataset_win_counts(pooled_df, win_metric=win_metric)
    if win_counts.empty:
        return []

    categories = [c for c in _CATEGORY_ORDER if c in win_counts["start_epoch_category"].values]
    if not categories:
        return []

    survival_rates = sorted(win_counts["selection_survival_rate"].dropna().unique())
    n_ds = n_datasets or int(win_counts["total_datasets"].max())

    # Build column index: [(category, survival_rate), ...]
    col_keys: list[tuple[str, float]] = [(cat, sr) for cat in categories for sr in survival_rates]
    col_labels: list[str] = [f"{sr:.0%}" for _, sr in col_keys]

    # Compute total wins per experiment for sorting
    exp_total_wins = (
        win_counts.groupby("experiment_name")["wins"].sum().sort_values(ascending=False)
    )
    experiment_order = list(exp_total_wins.index)

    n_rows = len(experiment_order)
    n_cols = len(col_keys)
    if n_rows == 0 or n_cols == 0:
        return []

    win_matrix = np.full((n_rows, n_cols), np.nan)
    total_matrix = np.zeros_like(win_matrix)

    for i, exp_name in enumerate(experiment_order):
        exp_wins = win_counts[win_counts["experiment_name"] == exp_name]
        for j, (cat, sr) in enumerate(col_keys):
            row = exp_wins[
                np.isclose(exp_wins["selection_survival_rate"].to_numpy(dtype="float64"), float(sr))
                & (exp_wins["start_epoch_category"] == cat)
            ]
            if not row.empty:
                win_matrix[i, j] = float(row["wins"].iloc[0])
                total_matrix[i, j] = float(row["total_datasets"].iloc[0])

    fraction_matrix = np.where(total_matrix > 0, win_matrix / total_matrix, np.nan)

    fig_h = max(4.0, n_rows * 0.55 + 2.5)
    fig_w = max(8.0, n_cols * 0.85 + 3.5)
    fig, axis = plt.subplots(figsize=(fig_w, fig_h))

    masked_frac = np.ma.masked_invalid(fraction_matrix)
    im = axis.imshow(masked_frac, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            wins_val = win_matrix[i, j]
            tot_val = total_matrix[i, j]
            if np.isnan(wins_val):
                continue
            label = f"{int(wins_val)}/{int(tot_val)}"
            frac = fraction_matrix[i, j]
            text_color = "white" if (frac < 0.25 or frac > 0.75) else "black"
            axis.text(j, i, label, ha="center", va="center", fontsize=7, color=text_color)

    # Category group separators and group labels
    n_sr = len(survival_rates)
    for cat_idx, cat in enumerate(categories):
        group_start = cat_idx * n_sr
        group_end = group_start + n_sr - 1
        group_center = (group_start + group_end) / 2.0
        axis.text(group_center, -0.9, cat.capitalize(), ha="center", va="bottom", fontsize=9, fontweight="bold",
                  transform=axis.get_xaxis_transform())
        if cat_idx > 0:
            axis.axvline(x=group_start - 0.5, color="white", linewidth=2.5)

    axis.set_xticks(np.arange(n_cols), col_labels, fontsize=7, rotation=45, ha="right")
    axis.set_yticks(np.arange(n_rows), experiment_order, fontsize=8)
    axis.set_xlabel("Survival rate (grouped by start-epoch category)", labelpad=20)
    axis.set_ylabel("Experiment")

    cbar = plt.colorbar(im, ax=axis, fraction=0.02, pad=0.02)
    cbar.set_label(f"Win fraction (out of {n_ds} datasets)", fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])

    metric_label = _WIN_METRIC_LABELS.get(win_metric, win_metric.replace("_", " ").title())
    layout_top = apply_figure_titles(
        fig,
        title="Cross-Dataset Win-Rate Overview (vs Naive Baseline)",
        subtitle=f"metric: {metric_label}" + (f" | {subtitle}" if subtitle else ""),
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_top), pad=0.1, w_pad=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "31_cross_dataset_win_matrix.png"
    _save_figure(fig, out)
    log.info("Wrote cross-dataset win matrix: %s", out)
    return [out]
