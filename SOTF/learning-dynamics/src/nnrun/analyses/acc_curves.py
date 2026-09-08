from __future__ import annotations

import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import AnalysisResult
from ..compute.matrices import build_metric_matrix_from_dfm
from ..zoo_info import zoo_plot_name


def _metric_display_name(metric: str) -> str:
    return metric.replace("_acc", " accuracy").replace("_loss", " loss").replace("_", " ")


def _natural_sort_key(value: Any) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def run_acc_curves(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    metric = config["metric"]
    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")
    epoch_range = tuple(config.get("epoch_range", [])) or None
    png_dpi = int(config.get("png_dpi", 100))
    boxplot_curves = bool(config.get("boxplot_curves", False))
    boxplot_overview_cfg = config.get("boxplot_overview", {}) or {}
    boxplot_overview_enabled = bool(boxplot_overview_cfg.get("enabled", False))
    boxplot_overview_single_cfg = boxplot_overview_cfg.get("single_page", {}) or {}
    boxplot_overview_paged_cfg = boxplot_overview_cfg.get("paged", {}) or {}
    boxplot_overview_show_title = bool(boxplot_overview_cfg.get("show_title", True))

    # Overlay plot settings (ALL models, full color, no avg)
    overlay_models = bool(config.get("overlay_models", True))
    overlay_models_max = int(config.get("overlay_models_max", 50))  # safeguard for PDF legibility
    overlay_models_legend = bool(config.get("overlay_models_legend", False))

    figures: List[Dict[str, plt.Figure]] = []
    tables: Dict[str, pd.DataFrame] = {}

    if "dataset_name" not in df.columns:
        raise ValueError("df must have dataset_name column for multi-dataset plotting.")

    available = set(df["dataset_name"].dropna().unique())
    config_order: List[str] = config.get("dataset_names", [])
    if config_order:
        datasets = [ds for ds in config_order if ds in available]
        extras = sorted(available - set(datasets), key=_natural_sort_key)
        datasets = datasets + extras
    else:
        datasets = sorted(available, key=_natural_sort_key)
    dataset_matrices: Dict[str, pd.DataFrame] = {}

    def add_nmodels_label(
        ax: plt.Axes,
        n_models: int,
        *,
        where: str = "top_right",
        fontsize: float = 10,
    ) -> None:
        if where == "top_right":
            xy = (0.99, 0.99)
            ha, va = "right", "top"
        elif where == "top_left":
            xy = (0.01, 0.99)
            ha, va = "left", "top"
        else:
            xy = (0.99, 0.01)
            ha, va = "right", "bottom"

        ax.text(
            *xy,
            f"n_models={n_models}",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9),
            zorder=10,
        )

    def get_dataset_matrix(ds: str) -> pd.DataFrame:
        mat = dataset_matrices.get(ds)
        if mat is not None:
            return mat

        dfd = df[df["dataset_name"] == ds]
        mat = build_metric_matrix_from_dfm(
            dfm=dfd,
            metric_key=metric,
            epoch_col=epoch_col,
            model_col=model_col,
            epoch_range=epoch_range,
        )
        if not mat.empty:
            mat = mat.reindex(sorted(mat.index, key=_natural_sort_key))
            mat = mat.sort_index(axis=1)
        dataset_matrices[ds] = mat
        return mat

    def style_boxplot(bp: dict[str, Any]) -> None:
        box_facecolor = (0.10, 0.65, 0.65, 0.22)
        edge_color = (0.10, 0.65, 0.65, 0.90)
        median_color = (0.05, 0.25, 0.25, 0.95)

        for patch in bp["boxes"]:
            patch.set_facecolor(box_facecolor)
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(1.2)

        for whisker in bp["whiskers"]:
            whisker.set_color(edge_color)
            whisker.set_linewidth(1.0)

        for cap in bp["caps"]:
            cap.set_color(edge_color)
            cap.set_linewidth(1.0)

        for median in bp["medians"]:
            median.set_color(median_color)
            median.set_linewidth(1.4)

    def plot_boxplot_on_ax(
        ax: plt.Axes,
        ds: str,
        *,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        title_fontsize: float = 12,
        tick_labelsize: float = 9,
        nmodels_fontsize: float = 10,
    ) -> None:
        mat = get_dataset_matrix(ds)

        if mat.empty:
            ax.set_title(f"{zoo_plot_name(ds)} | NO DATA", fontsize=title_fontsize)
            ax.set_xlabel("epoch" if show_xlabel else "")
            ax.set_ylabel(metric if show_ylabel else "")
            ax.grid(True)
            return

        epochs = mat.columns.to_list()
        data_by_epoch = [mat[e].dropna().to_numpy() for e in epochs]

        bp = ax.boxplot(
            data_by_epoch,
            positions=list(range(len(epochs))),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        style_boxplot(bp)

        ax.set_title(zoo_plot_name(ds), fontsize=title_fontsize)
        add_nmodels_label(ax, mat.shape[0], where="top_left", fontsize=nmodels_fontsize)
        ax.set_xlabel("Epoch" if show_xlabel else "", fontsize=tick_labelsize + 1)
        ax.set_ylabel(_metric_display_name(metric).title() if show_ylabel else "", fontsize=tick_labelsize + 1)

        n = len(epochs)
        if n > 150:
            tick_step = 50
        elif n > 50:
            tick_step = 10
        else:
            tick_step = 1
        tick_positions = list(range(0, n, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(epochs[i]) for i in tick_positions],
            rotation=45,
            ha="right",
            fontsize=tick_labelsize,
        )

        ax.tick_params(axis="y", labelsize=tick_labelsize)
        ax.grid(True, axis="y")

    def make_boxplot_for_dataset(ds: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=png_dpi)
        plot_boxplot_on_ax(ax, ds)
        fig.tight_layout()
        return fig

    def make_overlay_models_for_dataset(ds: str) -> plt.Figure:
        mat = get_dataset_matrix(ds)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=png_dpi)

        if mat.empty:
            ax.set_title(f"{metric} model curves | dataset={zoo_plot_name(ds)} | NO DATA")
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.grid(True)
            fig.tight_layout()
            return fig

        mat_plot = mat
        if mat.shape[0] > overlay_models_max:
            mat_plot = mat.sample(n=overlay_models_max, random_state=1)

        epochs = mat_plot.columns.to_numpy(dtype=float, copy=False)

        for idx, (model_name, row) in enumerate(mat_plot.iterrows(), start=1):
            y = row.to_numpy(dtype=float, copy=False)
            m = np.isfinite(epochs) & np.isfinite(y)
            if m.sum() >= 2:
                label = f"model {idx}" if overlay_models_legend else None
                ax.plot(epochs[m], y[m], linewidth=1.0, label=label)

        subtitle = f"Model Zoo: {zoo_plot_name(ds)} | Metric ({_metric_display_name(metric)}) | showing {mat_plot.shape[0]}"
        ax.set_title("Learning Trajectories in a model zoo", pad=30, fontsize=14)
        ax.text(0.5, 1.04, subtitle, ha="center", va="bottom", transform=ax.transAxes, fontsize=11, color="#555555")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(_metric_display_name(metric).title(), fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True)

        if overlay_models_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=1)

        fig.tight_layout()
        return fig

    def parse_figsize(cfg: Dict[str, Any], default_width: float, default_height: float) -> tuple[float, float]:
        figsize_raw = cfg.get("figsize")
        if isinstance(figsize_raw, (list, tuple)) and len(figsize_raw) == 2:
            return float(figsize_raw[0]), float(figsize_raw[1])
        return default_width, default_height

    def make_boxplot_overview_figure(
        page_datasets: List[str],
        *,
        nrows: int,
        ncols: int,
        figsize: tuple[float, float],
        title: str,
    ) -> plt.Figure:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=png_dpi, squeeze=False)
        axes_flat = np.array(axes).ravel()

        for idx, (ax, ds) in enumerate(zip(axes_flat, page_datasets)):
            row_idx = idx // ncols
            col_idx = idx % ncols
            plot_boxplot_on_ax(
                ax,
                ds,
                show_xlabel=row_idx == (nrows - 1),
                show_ylabel=col_idx == 0,
                title_fontsize=14,
                tick_labelsize=12,
                nmodels_fontsize=11,
            )

        for ax in axes_flat[len(page_datasets):]:
            ax.axis("off")

        if boxplot_overview_show_title:
            fig.suptitle(title, y=0.995, fontsize=22)
            fig.tight_layout(rect=(0, 0, 1.0, 0.99))
        else:
            fig.tight_layout()
        return fig

    def make_boxplot_overview_single_page() -> plt.Figure:
        ncols = max(1, int(boxplot_overview_single_cfg.get("ncols", 4)))
        nrows = max(1, int(np.ceil(len(datasets) / ncols)))
        figsize = parse_figsize(
            boxplot_overview_single_cfg,
            default_width=4.8 * ncols,
            default_height=3.6 * nrows,
        )
        return make_boxplot_overview_figure(
            datasets,
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            title=f"Distribution of {_metric_display_name(metric)} over epochs",
        )

    def make_boxplot_overview_paged() -> List[tuple[int, plt.Figure]]:
        nrows = max(1, int(boxplot_overview_paged_cfg.get("nrows", 2)))
        ncols = max(1, int(boxplot_overview_paged_cfg.get("ncols", 2)))
        plots_per_page = nrows * ncols
        total_pages = max(1, int(np.ceil(len(datasets) / plots_per_page)))
        figsize = parse_figsize(
            boxplot_overview_paged_cfg,
            default_width=5.0 * ncols,
            default_height=3.8 * nrows,
        )

        page_figures: List[tuple[int, plt.Figure]] = []
        for page_idx, start in enumerate(range(0, len(datasets), plots_per_page), start=1):
            page_datasets = datasets[start : start + plots_per_page]
            fig = make_boxplot_overview_figure(
                page_datasets,
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                title=f"Distribution of {_metric_display_name(metric)} over epochs ({page_idx}/{total_pages})",
            )
            page_figures.append((page_idx, fig))

        return page_figures

    for ds in datasets:
        if overlay_models:
            fig_ov = make_overlay_models_for_dataset(ds)
            figures.append({f"{ds}_acc_curves_overlay_models_{metric}": fig_ov})

        if boxplot_curves:
            fig_bp = make_boxplot_for_dataset(ds)
            figures.append({f"{ds}_acc_curves_boxplot_by_epoch_{metric}": fig_bp})

    if boxplot_curves and boxplot_overview_enabled and datasets:
        single_enabled = bool(boxplot_overview_single_cfg.get("enabled", True))
        paged_enabled = bool(boxplot_overview_paged_cfg.get("enabled", True))

        if single_enabled:
            ncols = max(1, int(boxplot_overview_single_cfg.get("ncols", 4)))
            fig_overview = make_boxplot_overview_single_page()
            figures.append({f"acc_curves_boxplot_overview_all_datasets_{metric}_{ncols}col": fig_overview})

        if paged_enabled:
            nrows = max(1, int(boxplot_overview_paged_cfg.get("nrows", 2)))
            ncols = max(1, int(boxplot_overview_paged_cfg.get("ncols", 2)))
            for page_idx, fig_page in make_boxplot_overview_paged():
                figures.append(
                    {
                        f"acc_curves_boxplot_overview_all_datasets_{metric}_{nrows}x{ncols}_page_{page_idx:02d}": (
                            fig_page
                        )
                    }
                )

    return AnalysisResult(tables=tables, figures=figures)
