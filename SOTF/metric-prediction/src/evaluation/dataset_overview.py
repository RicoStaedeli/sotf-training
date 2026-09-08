from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

plt.rcParams["font.family"] = "Times New Roman"

from evaluation.forecasting import (
    R2_PLOT_LABEL,
    RMSE_PLOT_LABEL,
    annotation_text_color,
    clip_negative_r2_for_plot,
    compute_metric,
    dataset_number_from_name,
    heatmap_colormap,
    heatmap_value_limits,
    load_forecast_runs,
    ordered_modes,
    render_heatmap,
    resolve_results_dir,
    sanitize_file_name,
    select_latest_runs,
    with_target_metric,
)


_DEFAULT_FORECASTING_ROOT = Path("metric-prediction") / "outputs" / "forecasting"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-dataset overview: R²/RMSE heatmaps with experiment names as rows "
            "and forecasting modes as columns."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Root directory to scan for forecasting runs. "
            f"Defaults to {_DEFAULT_FORECASTING_ROOT} (all experiments)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to evaluate (e.g. ds14). Auto-detected when only one dataset is present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <results_dir>/evaluation.",
    )
    parser.add_argument(
        "--extra-results-root",
        dest="extra_results_roots",
        type=Path,
        action="append",
        default=None,
        help="Additional results directories to merge before filtering.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_root = args.results_dir if args.results_dir is not None else _DEFAULT_FORECASTING_ROOT
    requested_root = raw_root.resolve()
    results_dir = resolve_results_dir(requested_root)
    output_dir = (args.output_dir if args.output_dir is not None else requested_root / "evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_runs = load_forecast_runs(results_dir)
    for extra_root in args.extra_results_roots or []:
        extra_dir = resolve_results_dir(extra_root.resolve())
        extra_runs = load_forecast_runs(extra_dir)
        if not extra_runs.empty:
            completed_runs = pd.concat([completed_runs, extra_runs], ignore_index=True)
            print(f"Merged {len(extra_runs)} runs from {extra_dir}")

    if completed_runs.empty:
        raise SystemExit(f"No completed forecasting runs found under {results_dir}")

    latest_runs = select_latest_runs(completed_runs)
    dataset_name = _resolve_dataset(latest_runs, args.dataset)
    ds_runs = latest_runs[latest_runs["dataset_name"] == dataset_name].copy()
    if ds_runs.empty:
        raise SystemExit(f"No runs found for dataset '{dataset_name}'.")
    print(f"Generating dataset overview for '{dataset_name}' ({len(ds_runs)} experiments)")

    target_metrics = sorted(str(v) for v in ds_runs["target_metric"].dropna().unique())
    if not target_metrics:
        target_metrics = ["unknown"]

    multi_metric = len(target_metrics) > 1
    for target_metric in target_metrics:
        metric_runs = ds_runs[ds_runs["target_metric"] == target_metric].copy()
        if metric_runs.empty:
            metric_runs = ds_runs.copy()

        metric_output_dir = output_dir / sanitize_file_name(target_metric) if multi_metric else output_dir
        metric_output_dir.mkdir(parents=True, exist_ok=True)

        modes = ordered_modes(metric_runs["target_mode"].dropna().tolist())
        for plot_idx, mode in enumerate(modes, start=13):
            mode_runs = metric_runs[metric_runs["target_mode"] == mode].copy()
            if mode_runs.empty:
                continue
            r2_mode = build_experiment_mode_table(mode_runs, metric_name="r2")
            rmse_mode = build_experiment_mode_table_aligned_to(
                mode_runs, metric_name="rmse", row_order=r2_mode.index.tolist()
            )
            render_combined_overview(
                r2_table=r2_mode,
                rmse_table=rmse_mode,
                output_path=metric_output_dir / f"{plot_idx}_{mode}_r2_rmse_overview.png",
                title="Predictor Comparison",
                subtitle=f"{dataset_name} | {mode.replace('_', ' ')} | {target_metric.replace('_', ' ')} prediction",
            )

        epoch_plot_base = 13 + len(modes)
        for i, mode in enumerate(modes):
            mode_runs = metric_runs[metric_runs["target_mode"] == mode].copy()
            if mode_runs.empty:
                continue
            idx_r2 = epoch_plot_base + i * 2
            idx_rmse = epoch_plot_base + i * 2 + 1
            per_epoch_r2 = build_per_epoch_experiment_table(mode_runs, metric_name="r2")
            per_epoch_rmse = build_per_epoch_experiment_table(mode_runs, metric_name="rmse")
            render_heatmap(
                per_epoch_r2,
                output_path=metric_output_dir / f"{idx_r2}_r2_per_epoch_{mode}.png",
                title=with_target_metric(
                    f"Forecast {R2_PLOT_LABEL} per decision epoch — {dataset_name} — {mode}",
                    target_metric,
                ),
                x_label="Decision epoch",
                y_label="Experiment",
                colorbar_label=R2_PLOT_LABEL,
                metric_name="r2",
                clip_negative=True,
                annotate_values=False,
                compact=True,
                save_dpi=300,
                save_pdf=True,
            )
            render_heatmap(
                per_epoch_rmse,
                output_path=metric_output_dir / f"{idx_rmse}_rmse_per_epoch_{mode}.png",
                title=with_target_metric(
                    f"Forecast {RMSE_PLOT_LABEL} per decision epoch — {dataset_name} — {mode}",
                    target_metric,
                ),
                x_label="Decision epoch",
                y_label="Experiment",
                colorbar_label=RMSE_PLOT_LABEL,
                metric_name="rmse",
                reverse_scale=True,
                annotate_values=False,
                compact=True,
                save_dpi=300,
                save_pdf=True,
            )

    print(f"Wrote dataset overview to {output_dir}")
    return 0


def _resolve_dataset(runs_df: pd.DataFrame, dataset_filter: str | None) -> str:
    """Return the canonical dataset_name matching the filter, or auto-detect if None."""
    unique = [str(v) for v in runs_df["dataset_name"].dropna().unique()]
    if not unique:
        raise SystemExit("No dataset names found in runs.")

    if dataset_filter is None:
        if len(unique) == 1:
            return unique[0]
        raise SystemExit(f"Multiple datasets found: {unique}. Specify one with --dataset.")

    # Exact match
    if dataset_filter in unique:
        return dataset_filter

    # Match by numeric dataset number (handles "ds14" → 14 → "14_phase_...")
    filter_number = dataset_number_from_name(dataset_filter)
    if filter_number != float("inf"):
        matches = [name for name in unique if dataset_number_from_name(name) == filter_number]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(
                f"Dataset filter '{dataset_filter}' (number {int(filter_number)}) matches multiple datasets: {matches}."
            )

    # Prefix/substring match
    lower_filter = dataset_filter.lower()
    matches = [name for name in unique if lower_filter in name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"Dataset filter '{dataset_filter}' is ambiguous: {matches}.")

    raise SystemExit(
        f"No dataset matching '{dataset_filter}' found. Available: {unique}"
    )


def build_experiment_mode_table(runs_df: pd.DataFrame, *, metric_name: str) -> pd.DataFrame:
    """Pivot runs to (experiment × mode), sorted best-first by mean metric."""
    if runs_df.empty:
        return pd.DataFrame()

    mode_order = ordered_modes(runs_df["target_mode"].dropna().tolist())
    ascending = metric_name == "rmse"
    exp_order = (
        runs_df.groupby("experiment_name", sort=False)[metric_name]
        .mean()
        .sort_values(ascending=ascending)
        .index.tolist()
    )
    table = runs_df.pivot_table(
        index="experiment_name",
        columns="target_mode",
        values=metric_name,
        aggfunc="mean",
    )
    table = table.reindex(index=exp_order, columns=mode_order)
    table.index.name = "experiment"
    table.columns.name = "forecasting_mode"
    return table


def build_experiment_mode_table_aligned_to(
    runs_df: pd.DataFrame, *, metric_name: str, row_order: list[str]
) -> pd.DataFrame:
    """Same as build_experiment_mode_table but uses a fixed row_order."""
    if runs_df.empty:
        return pd.DataFrame()

    mode_order = ordered_modes(runs_df["target_mode"].dropna().tolist())
    table = runs_df.pivot_table(
        index="experiment_name",
        columns="target_mode",
        values=metric_name,
        aggfunc="mean",
    )
    all_rows = list(row_order) + [r for r in table.index if r not in row_order]
    table = table.reindex(index=all_rows, columns=mode_order)
    table.index.name = "experiment"
    table.columns.name = "forecasting_mode"
    return table


def render_combined_overview(
    *,
    r2_table: pd.DataFrame,
    rmse_table: pd.DataFrame,
    output_path: Path,
    title: str,
    subtitle: str = "",
    save_dpi: int = 200,
) -> None:
    """Render R² (left) and RMSE (right) heatmaps side by side in one figure."""
    all_experiments = list(
        dict.fromkeys(
            r2_table.index.tolist()
            + [e for e in rmse_table.index.tolist() if e not in r2_table.index]
        )
    )
    r2_aligned = r2_table.reindex(index=all_experiments)
    rmse_aligned = rmse_table.reindex(index=all_experiments)

    n_rows = len(all_experiments)
    n_r2_cols = r2_aligned.shape[1] if not r2_aligned.empty else 1
    n_rmse_cols = rmse_aligned.shape[1] if not rmse_aligned.empty else 1

    row_height = 0.48
    fig_height = max(4.5, row_height * n_rows + 2.8)
    col_width = 2.4
    fig_width = col_width * (n_r2_cols + n_rmse_cols) + 4.5

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    _render_metric_to_ax(
        axes[0],
        r2_aligned,
        metric_name="r2",
        subplot_title=R2_PLOT_LABEL,
        show_yticks=True,
        fig=fig,
    )
    _render_metric_to_ax(
        axes[1],
        rmse_aligned,
        metric_name="rmse",
        subplot_title=RMSE_PLOT_LABEL,
        show_yticks=False,
        reverse_scale=True,
        fig=fig,
    )

    title_y = 1.04 if subtitle else 1.01
    fig.suptitle(title, fontsize=14, y=title_y)
    if subtitle:
        fig.text(0.5, 1.005, subtitle, ha="center", va="bottom", fontsize=11, color="#555555", transform=fig.transFigure)
    fig.tight_layout()
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_metric_to_ax(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    metric_name: str,
    subplot_title: str,
    show_yticks: bool,
    fig: plt.Figure,
    reverse_scale: bool = False,
) -> None:
    """Render a metric heatmap onto an existing Axes."""
    if table.empty or not np.isfinite(table.to_numpy(dtype="float64")).any():
        ax.axis("off")
        ax.set_title(subplot_title)
        return

    plot_table = clip_negative_r2_for_plot(table) if metric_name == "r2" else table
    values = plot_table.to_numpy(dtype="float64")
    vmin, vmax = heatmap_value_limits(metric_name)
    cmap = heatmap_colormap(reverse_scale=reverse_scale)
    cmap.set_bad("#E6E6E6")

    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([])
    ax.set_xticks(np.arange(-0.5, table.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_yticks:
        ax.set_yticks(np.arange(table.shape[0]))
        ax.set_yticklabels([str(i) for i in table.index], fontsize=10)

    ax.set_title(subplot_title, fontsize=12)

    for row_i in range(table.shape[0]):
        for col_i in range(table.shape[1]):
            value = values[row_i, col_i]
            if not np.isfinite(value):
                continue
            rgba = image.cmap(image.norm(value))
            ax.text(
                col_i,
                row_i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=annotation_text_color(rgba),
                fontsize=10,
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(R2_PLOT_LABEL if metric_name == "r2" else RMSE_PLOT_LABEL)


def build_per_epoch_experiment_table(runs_df: pd.DataFrame, *, metric_name: str) -> pd.DataFrame:
    """For each experiment, compute metric at every prediction epoch from validation_predictions.parquet.

    Returns a DataFrame with experiment names as index and epochs as columns.
    """
    rows: list[dict] = []
    for run in runs_df.itertuples(index=False):
        predictions_path = getattr(run, "validation_predictions_path", None)
        if predictions_path is None:
            continue
        try:
            frame = pd.read_parquet(predictions_path)
        except Exception:
            continue
        if not {"epoch", "target", "prediction"}.issubset(frame.columns):
            continue
        for epoch, epoch_frame in frame.groupby("epoch", sort=True):
            rows.append(
                {
                    "experiment_name": run.experiment_name,
                    "epoch": int(epoch),
                    metric_name: compute_metric(
                        epoch_frame["target"], epoch_frame["prediction"], metric_name=metric_name
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    table = df.pivot_table(index="experiment_name", columns="epoch", values=metric_name, aggfunc="mean")
    table.index.name = "experiment"
    table.columns.name = "epoch"
    return table



if __name__ == "__main__":
    raise SystemExit(main())
