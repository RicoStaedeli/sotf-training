from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
import numpy as np
import pandas as pd

from .analysis import filter_runs, load_completed_runs, normalize_predictor_name, resolve_results_root, select_latest_runs
from .plots import _save_figure
from .shared import build_experiment_subtitle, build_model_zoo_subtitle, predictor_display_name, sanitize_file_stem
from .tracking import resolve_default_evaluation_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


KEPT_COLOR = "#1f77b4"
KILLED_COLOR = "#d62828"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize kept and killed model counts for completed Survival Of The Fittest selection runs.",
    )
    parser.add_argument(
        "experiment_root",
        type=Path,
        help="Experiment root or results directory to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where plots will be written. "
            "Defaults to sotf-selection/outputs/evaluation/selection_counts/<experiment-name> when the source is under outputs/."
        ),
    )
    parser.add_argument(
        "--predictor",
        action="append",
        default=None,
        help="Predictor(s) to include, for example `forecast_model`, `xgboost`, or `naive`. Repeat to include multiple.",
    )
    parser.add_argument("--target-column", default=None, help="Filter to a specific target column, for example `test_acc`.")
    parser.add_argument("--target-mode", default=None, help="Filter to a specific sequence target mode, for example `horizon_value`.")
    parser.add_argument("--horizon", type=int, default=None, help="Filter to a specific forecast horizon.")
    parser.add_argument("--start-epoch", type=int, default=None, help="Filter to a specific selection start epoch.")
    parser.add_argument("--interval", type=int, default=None, help="Filter to a specific selection interval.")
    parser.add_argument("--survival-rate", type=float, default=None, help="Filter to a specific survival rate.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_root = args.experiment_root.resolve()
    results_root = resolve_results_root(experiment_root)
    completed_runs = load_completed_runs(results_root)
    if completed_runs.empty:
        raise SystemExit(f"No completed SOTF runs found under {experiment_root}")

    predictor_filters = [normalize_predictor_name(value) for value in args.predictor] if args.predictor else None
    filtered_runs = filter_runs(
        completed_runs,
        predictors=predictor_filters,
        target_column=args.target_column,
        target_mode=args.target_mode,
        horizon=args.horizon,
        start_epoch=args.start_epoch,
        interval=args.interval,
        survival_rate=args.survival_rate,
    )
    if filtered_runs.empty:
        raise SystemExit("No completed SOTF runs matched the requested metadata filters.")

    latest_runs = select_latest_runs(filtered_runs)
    count_frame = build_selection_count_frame(latest_runs)
    if count_frame.empty:
        raise SystemExit("No round summaries with selection counts were found for the selected runs.")

    output_stem_source = results_root.parent.name if results_root.name == "results" else results_root.name
    default_output_dir = resolve_default_evaluation_root(results_root) / "selection_counts" / sanitize_file_stem(output_stem_source)
    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    count_frame.to_csv(output_dir / "selection_counts.csv", index=False)

    subtitle = build_count_subtitle(latest_runs)
    secondary_subtitle = build_model_zoo_subtitle(latest_runs)
    png_dir = output_dir / "png"
    overview_path = png_dir / "selection_counts_overview_all_datasets.png"
    render_overview_plot(
        count_frame,
        overview_path,
        title="Kept and Killed Models Across Model Zoo Datasets",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    dataset_paths = render_dataset_plots(
        count_frame,
        output_dir=png_dir,
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )

    print(f"Wrote overview plot to {overview_path}")
    print(f"Wrote {len(dataset_paths)} dataset-specific plot(s) to {png_dir}")
    return 0


def build_selection_count_frame(latest_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in latest_runs.to_dict("records"):
        round_summary = load_round_summary(run)
        if round_summary.empty:
            continue
        rows.extend(selection_count_rows(run=run, round_summary=round_summary))

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "dataset_number",
            "dataset_label",
            "predictor",
            "target_column",
            "target_mode",
            "horizon",
            "selection_start_epoch",
            "selection_interval",
            "selection_survival_rate",
            "round_index",
        ],
        kind="stable",
    ).reset_index(drop=True)


def load_round_summary(run: dict[str, Any]) -> pd.DataFrame:
    run_dir = Path(str(run["run_dir"]))
    round_summary_path = run_dir / "round_summary.parquet"
    if round_summary_path.exists():
        return pd.read_parquet(round_summary_path)

    round_details_path = Path(str(run["round_details_path"]))
    if not round_details_path.exists():
        return pd.DataFrame()
    return summarize_round_details(pd.read_parquet(round_details_path))


def summarize_round_details(round_details: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"round_index", "decision_epoch", "population_before", "population_after"}
    if round_details.empty or not required_columns.issubset(round_details.columns):
        return pd.DataFrame()

    return (
        round_details.groupby("round_index", as_index=False, sort=True)
        .agg(
            decision_epoch=("decision_epoch", "first"),
            population_before=("population_before", "first"),
            population_after=("population_after", "first"),
        )
        .assign(kept_models=lambda frame: frame["population_after"])
    )


def selection_count_rows(*, run: dict[str, Any], round_summary: pd.DataFrame) -> list[dict[str, Any]]:
    summary = round_summary.copy()
    if "population_after" not in summary.columns and "kept_models" in summary.columns:
        summary["population_after"] = summary["kept_models"]
    if "kept_models" not in summary.columns and "population_after" in summary.columns:
        summary["kept_models"] = summary["population_after"]
    required_columns = {"round_index", "decision_epoch", "population_after"}
    if summary.empty or not required_columns.issubset(summary.columns):
        return []

    initial_population = int_or_default(run.get("initial_population"), default=0)
    if initial_population <= 0 and "population_before" in summary.columns:
        initial_population = int_or_default(pd.to_numeric(summary["population_before"], errors="coerce").max(), default=0)
    if initial_population <= 0:
        initial_population = int_or_default(pd.to_numeric(summary["population_after"], errors="coerce").max(), default=0)

    run_key = str(run["run_dir"])
    run_label = build_run_label(run)
    rows: list[dict[str, Any]] = [
        {
            "run_key": run_key,
            "run_label": run_label,
            "experiment_name": run["experiment_name"],
            "predictor": run["predictor"],
            "predictor_label": predictor_display_name(str(run["predictor"])),
            "dataset_number": run["dataset_number"],
            "dataset_label": run["dataset_label"],
            "dataset_slug": run["dataset_slug"],
            "target_column": run["target_column"],
            "target_mode": run["target_mode"],
            "horizon": run["horizon"],
            "selection_start_epoch": run["selection_start_epoch"],
            "selection_interval": run["selection_interval"],
            "selection_survival_rate": run["selection_survival_rate"],
            "round_index": 0,
            "decision_epoch": 0.0,
            "initial_population": initial_population,
            "population_before": initial_population,
            "kept_models": initial_population,
            "killed_models": 0,
            "killed_this_round": 0,
            "total_models": initial_population,
        }
    ]
    for summary_row in summary.sort_values("round_index").to_dict("records"):
        kept_models = int_or_default(summary_row.get("population_after"), default=0)
        kept_models = max(0, min(initial_population, kept_models))
        killed_models = max(0, initial_population - kept_models)
        population_before = int_or_default(summary_row.get("population_before"), default=initial_population)
        killed_this_round = max(0, population_before - kept_models)
        rows.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "experiment_name": run["experiment_name"],
                "predictor": run["predictor"],
                "predictor_label": predictor_display_name(str(run["predictor"])),
                "dataset_number": run["dataset_number"],
                "dataset_label": run["dataset_label"],
                "dataset_slug": run["dataset_slug"],
                "target_column": run["target_column"],
                "target_mode": run["target_mode"],
                "horizon": run["horizon"],
                "selection_start_epoch": run["selection_start_epoch"],
                "selection_interval": run["selection_interval"],
                "selection_survival_rate": run["selection_survival_rate"],
                "round_index": int_or_default(summary_row.get("round_index"), default=0),
                "decision_epoch": float_or_nan(summary_row.get("decision_epoch")),
                "initial_population": initial_population,
                "population_before": population_before,
                "kept_models": kept_models,
                "killed_models": killed_models,
                "killed_this_round": killed_this_round,
                "total_models": kept_models + killed_models,
            }
        )
    return rows


def render_overview_plot(
    count_frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    secondary_subtitle: str = "",
) -> None:
    run_order = run_order_frame(count_frame)
    column_count = min(4, max(1, len(run_order)))
    row_count = math.ceil(len(run_order) / column_count)
    fig_width = max(7.0, 4.5 * column_count)
    fig_height = max(4.8, 3.2 * row_count + 1.0)
    fig, axes = plt.subplots(row_count, column_count, figsize=(fig_width, fig_height), squeeze=False)
    flat_axes = list(axes.flat)

    for axis in flat_axes[len(run_order) :]:
        axis.axis("off")

    for axis, run_row in zip(flat_axes, run_order.to_dict("records"), strict=False):
        run_df = count_frame[count_frame["run_key"] == run_row["run_key"]]
        render_stacked_count_bars(
            axis,
            run_df,
            title=short_title(str(run_row["run_label"]), width=46),
            show_ylabel=True,
            show_xlabel=True,
        )

    add_shared_legend(fig)
    layout_top = apply_count_figure_titles(
        fig,
        title=title,
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )
    fig.tight_layout(rect=(0, 0.05, 1, layout_top))
    compact_top_margin(fig, axes=flat_axes, layout_top=layout_top)
    _save_figure(fig, output_path)


def render_dataset_plots(
    count_frame: pd.DataFrame,
    *,
    output_dir: Path,
    subtitle: str,
    secondary_subtitle: str = "",
) -> list[Path]:
    output_paths: list[Path] = []
    dataset_order = (
        count_frame.loc[:, ["dataset_number", "dataset_label"]]
        .drop_duplicates()
        .sort_values(["dataset_number", "dataset_label"], kind="stable")
    )
    for dataset_row in dataset_order.to_dict("records"):
        dataset_df = count_frame[
            (count_frame["dataset_number"] == dataset_row["dataset_number"])
            & (count_frame["dataset_label"] == dataset_row["dataset_label"])
        ].copy()
        if dataset_df.empty:
            continue

        run_order = run_order_frame(dataset_df)
        column_count = min(2, max(1, len(run_order)))
        row_count = math.ceil(len(run_order) / column_count)
        fig_width = 9.5 if len(run_order) == 1 else 6.0 * column_count
        fig_height = 5.0 if len(run_order) == 1 else 3.8 * row_count + 1.0
        fig, axes = plt.subplots(row_count, column_count, figsize=(fig_width, fig_height), squeeze=False)
        flat_axes = list(axes.flat)

        for axis in flat_axes[len(run_order) :]:
            axis.axis("off")

        for axis, run_row in zip(flat_axes, run_order.to_dict("records"), strict=False):
            run_df = dataset_df[dataset_df["run_key"] == run_row["run_key"]]
            render_stacked_count_bars(
                axis,
                run_df,
                title=run_panel_title(run_df),
                show_ylabel=True,
                show_xlabel=True,
            )

        add_shared_legend(fig)
        dataset_title = short_title(str(dataset_row["dataset_label"]), width=80)
        layout_top = apply_count_figure_titles(
            fig,
            title=f"Kept and Killed Models: {dataset_title}",
            subtitle=subtitle,
            secondary_subtitle=secondary_subtitle,
        )
        fig.tight_layout(rect=(0, 0.08, 1, layout_top))
        compact_top_margin(fig, axes=flat_axes, layout_top=layout_top)
        dataset_stem = sanitize_file_stem(f"{dataset_row['dataset_number']}_{dataset_row['dataset_label']}")
        output_path = output_dir / f"selection_counts_{dataset_stem}.png"
        _save_figure(fig, output_path)
        output_paths.append(output_path)
    return output_paths


def render_stacked_count_bars(
    axis,
    frame: pd.DataFrame,
    *,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    df = frame.sort_values("round_index", kind="stable").reset_index(drop=True)
    x = np.arange(len(df), dtype="float64")
    killed = df["killed_models"].to_numpy(dtype="float64")
    kept = df["kept_models"].to_numpy(dtype="float64")
    totals = df["total_models"].to_numpy(dtype="float64")
    axis.bar(x, killed, color=KILLED_COLOR, edgecolor="black", linewidth=0.5, label="Killed models")
    axis.bar(x, kept, bottom=killed, color=KEPT_COLOR, edgecolor="black", linewidth=0.5, label="Kept models")
    add_killed_count_labels(axis, x=x, killed=killed, totals=totals)
    add_initial_kept_count_labels(axis, x=x, killed=killed, kept=kept)
    axis.set_title(title, fontsize=10, pad=2)
    axis.set_xticks(x, [format_epoch(value) for value in df["decision_epoch"]])
    if len(df) > 8:
        axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    else:
        axis.tick_params(axis="x", labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    if show_xlabel:
        axis.set_xlabel("Decision epoch")
    if show_ylabel:
        axis.set_ylabel("Number of models")
    y_max = float(np.nanmax(totals)) if len(totals) else 1.0
    axis.set_ylim(0, max(1.0, y_max * 1.08))
    axis.grid(axis="y", alpha=0.25)


def add_killed_count_labels(axis, *, x: np.ndarray, killed: np.ndarray, totals: np.ndarray) -> None:
    max_total = float(np.nanmax(totals)) if len(totals) else 0.0
    if max_total <= 0:
        return

    small_label_height = max_total * 0.04
    for x_position, killed_count in zip(x, killed, strict=True):
        if not np.isfinite(killed_count) or killed_count <= 0:
            continue
        axis.text(
            x_position,
            killed_count / 2.0,
            format_count(killed_count),
            ha="center",
            va="center",
            color="white",
            fontsize=7 if killed_count < small_label_height else 8,
            fontweight="bold",
            clip_on=True,
        )


def add_initial_kept_count_labels(axis, *, x: np.ndarray, killed: np.ndarray, kept: np.ndarray) -> None:
    for x_position, killed_count, kept_count in zip(x[:2], killed[:2], kept[:2], strict=True):
        if not np.isfinite(kept_count) or kept_count <= 0:
            continue
        y_position = float(killed_count) + float(kept_count) / 2.0
        axis.text(
            x_position,
            y_position,
            format_count(kept_count),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            clip_on=True,
        )


def add_shared_legend(fig) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=KILLED_COLOR, ec="black", label="Killed models"),
        plt.Rectangle((0, 0), 1, 1, color=KEPT_COLOR, ec="black", label="Kept models"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)


def compact_top_margin(fig, *, axes: Sequence[Any], layout_top: float) -> None:
    visible_axes = [axis for axis in axes if axis.get_visible()]
    if not visible_axes:
        return

    figure_height = max(1.0, float(fig.get_figheight()))
    max_title_lines = max(max(1, axis.get_title().count("\n") + 1) for axis in visible_axes)
    axis_title_headroom = (10.0 / 72.0 / figure_height) * (1.25 * max_title_lines + 0.35)
    desired_top = max(0.68, layout_top - axis_title_headroom)
    fig.subplots_adjust(top=desired_top)


def apply_count_figure_titles(fig, *, title: str, subtitle: str, secondary_subtitle: str = "") -> float:
    title_line_count = max(1, str(title).count("\n") + 1)
    subtitle_line_count = 0 if not subtitle else max(1, str(subtitle).count("\n") + 1)
    secondary_line_count = 0 if not secondary_subtitle else max(1, str(secondary_subtitle).count("\n") + 1)
    figure_height = max(1.0, float(fig.get_figheight()))
    title_y = 0.99
    title_line_height = (16.0 / 72.0 / figure_height) * 1.15
    subtitle_gap = (9.0 / 72.0 / figure_height) * 0.65
    subtitle_line_height = (9.0 / 72.0 / figure_height) * 1.2
    secondary_gap = (8.0 / 72.0 / figure_height) * 0.55
    plot_gap = (9.0 / 72.0 / figure_height) * 0.8
    title_bottom = title_y - title_line_height * title_line_count

    fig.suptitle(title, fontsize=16, y=title_y)
    current_bottom = title_bottom
    if subtitle:
        subtitle_y = current_bottom - subtitle_gap
        fig.text(
            0.5,
            subtitle_y,
            subtitle,
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
            wrap=True,
        )
        current_bottom = subtitle_y - subtitle_line_height * subtitle_line_count
    if secondary_subtitle:
        secondary_y = current_bottom - secondary_gap
        fig.text(
            0.5,
            secondary_y,
            secondary_subtitle,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#555555",
            wrap=True,
        )
        current_bottom = secondary_y - subtitle_line_height * secondary_line_count

    layout_top = current_bottom - plot_gap
    return max(0.72, layout_top)


def run_order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[
            :,
            [
                "run_key",
                "run_label",
                "dataset_number",
                "dataset_label",
                "predictor",
                "target_column",
                "target_mode",
                "horizon",
                "selection_start_epoch",
                "selection_interval",
                "selection_survival_rate",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "dataset_number",
                "dataset_label",
                "predictor",
                "target_column",
                "target_mode",
                "horizon",
                "selection_start_epoch",
                "selection_interval",
                "selection_survival_rate",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_run_label(run: dict[str, Any]) -> str:
    dataset_label = str(run["dataset_label"])
    predictor_label = predictor_display_name(str(run["predictor"]))
    return (
        f"{dataset_label}\n"
        f"{predictor_label} | h={format_plain_value(run['horizon'])}, "
        f"start={format_plain_value(run['selection_start_epoch'])}, "
        f"int={format_plain_value(run['selection_interval'])}, "
        f"sr={format_plain_value(run['selection_survival_rate'])}"
    )


def run_panel_title(frame: pd.DataFrame) -> str:
    first = frame.iloc[0]
    return (
        f"{first['predictor_label']} | h={format_plain_value(first['horizon'])}, "
        f"start={format_plain_value(first['selection_start_epoch'])}, "
        f"int={format_plain_value(first['selection_interval'])}, "
        f"sr={format_plain_value(first['selection_survival_rate'])}"
    )


def build_count_subtitle(latest_runs: pd.DataFrame) -> str:
    return build_experiment_subtitle(latest_runs) or ""


def short_title(value: str, *, width: int) -> str:
    lines = []
    for line in str(value).splitlines():
        lines.extend(textwrap.wrap(line, width=width, max_lines=2, placeholder="...") or [""])
    return "\n".join(lines[:3])


def format_epoch(value: Any) -> str:
    numeric_value = float_or_nan(value)
    if np.isnan(numeric_value):
        return "n/a"
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"


def format_count(value: Any) -> str:
    numeric_value = float_or_nan(value)
    if np.isnan(numeric_value):
        return "n/a"
    return f"{int(round(numeric_value)):,}"


def format_plain_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "n/a"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def int_or_default(value: Any, *, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def float_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
