from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .base import AnalysisResult
from ..compute.matrices import build_metric_matrix_from_dfm
from ..compute.ranks import pairwise_flip_table, rank_change_table, rank_per_epoch
from ..zoo_info import zoo_plot_name

log = logging.getLogger(__name__)


def _panel_params(cfg: Dict[str, Any]) -> Tuple[int, int, Tuple[float, float], int]:
    panel = cfg.get("panel", {}) or {}
    nrows = int(panel.get("nrows", 2))
    ncols = int(panel.get("ncols", 3))
    figsize_raw = panel.get("figsize", [12, 7])
    figsize = (float(figsize_raw[0]), float(figsize_raw[1]))
    png_dpi = cfg.get("png_dpi", 100)
    return nrows, ncols, figsize, png_dpi


def _drop_epoch0_to1_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Ignore the (0 -> 1) transition in plots (tables remain unchanged)."""
    if df is None or df.empty:
        return df
    if "from_epoch" in df.columns and "to_epoch" in df.columns:
        return df.loc[~((df["from_epoch"] == 0) & (df["to_epoch"] == 1))].copy()
    return df


def _plot_pairwise_panel(
        *,
        title: str,
        subtitle: str = "",
        metrics: List[str],
        results_by_metric: Dict[str, Optional[pd.DataFrame]],
        value_col: str,
        nrows: int,
        ncols: int,
        figsize: Tuple[float, float],
        dpi: int,
        show_title: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=True)
    axes = np.array(axes).ravel()

    for ax, metric_key in zip(axes, metrics):
        df = results_by_metric.get(metric_key)
        df = _drop_epoch0_to1_for_plot(df)  # <-- NEW
        if df is None or df.empty or value_col not in df.columns:
            ax.set_title(f"{metric_key} - no data", fontsize=18)
            ax.grid(True)
            continue

        x = df["to_epoch"].to_numpy()
        y = df[value_col].to_numpy()
        ax.plot(x, y)

        ax.set_title(f"{metric_key}", fontsize=18)
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("# pairwise flips", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    if show_title:
        fig.suptitle(title, y=1.06, fontsize=22)
        if subtitle:
            fig.text(0.5, 1.02, subtitle, ha="center", va="top", fontsize=16, color="#555555")
    fig.tight_layout()
    return fig


def _plot_pairwise_combined_panel(
        *,
        title: str,
        subtitle: str = "",
        metrics: List[str],
        results_by_metric: Dict[str, Optional[pd.DataFrame]],
        smoothening_threshold: Optional[float],
        nrows: int,
        ncols: int,
        figsize: Tuple[float, float],
        dpi: int,
        show_title: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=True)
    axes = np.array(axes).ravel()

    legend_handles: list = []
    legend_labels: list = []

    for ax, metric_key in zip(axes, metrics):
        df = results_by_metric.get(metric_key)
        df = _drop_epoch0_to1_for_plot(df)
        if df is None or df.empty or "pairwise_flips_without_smoothening" not in df.columns:
            ax.set_title(f"{metric_key} - no data", fontsize=18)
            ax.grid(True)
            continue

        x = df["to_epoch"].to_numpy()
        (line1,) = ax.plot(x, df["pairwise_flips_without_smoothening"].to_numpy(), label="without smoothening")

        if "pairwise_flips_with_smoothening" in df.columns:
            smooth_label = (
                f"with smoothening (threshold={smoothening_threshold:g})"
                if smoothening_threshold is not None
                else "with smoothening"
            )
            (line2,) = ax.plot(x, df["pairwise_flips_with_smoothening"].to_numpy(), label=smooth_label)
            if not legend_handles:
                legend_handles = [line1, line2]
                legend_labels = [line1.get_label(), line2.get_label()]
        elif not legend_handles:
            legend_handles = [line1]
            legend_labels = [line1.get_label()]

        ax.set_title(f"{metric_key}", fontsize=18)
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("# pairwise flips", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    if show_title:
        fig.suptitle(title, y=1.06, fontsize=22)
        if subtitle:
            fig.text(0.5, 1.02, subtitle, ha="center", va="top", fontsize=16, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_handles),
                   fontsize=14, frameon=True, bbox_to_anchor=(0.5, 0.0))
    return fig


def _parse_metrics(metrics_cfg: Any) -> Tuple[List[str], Dict[str, bool]]:
    """
      metrics:
        - name: train_loss
          higher_is_better: false
        - name: train_acc
          higher_is_better: true
    Returns (metrics, hib_map)
    """
    if not isinstance(metrics_cfg, list) or len(metrics_cfg) == 0:
        raise ValueError("Config must provide non-empty 'metrics' list.")

    if isinstance(metrics_cfg[0], dict):
        metrics: List[str] = []
        hib_map: Dict[str, bool] = {}
        for m in metrics_cfg:
            if "name" not in m:
                raise ValueError("Each metrics entry must have a 'name' field.")
            name = str(m["name"])
            metrics.append(name)
            hib_map[name] = bool(m.get("higher_is_better", True))
        return metrics, hib_map

    # fallback: allow simple list, default higher_is_better=True
    metrics = [str(x) for x in metrics_cfg]
    return metrics, {}

def _kendall_tau_table(rank_mat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Kendall's tau-b between consecutive epochs of rank_mat.
    Returns columns: from_epoch, to_epoch, kendall_tau, kendall_p_value.
    """
    epochs = list(rank_mat.columns)
    if len(epochs) < 2:
        return pd.DataFrame()

    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        r0 = rank_mat[e0].to_numpy(dtype=float)
        r1 = rank_mat[e1].to_numpy(dtype=float)

        mask = np.isfinite(r0) & np.isfinite(r1)
        r0 = r0[mask]
        r1 = r1[mask]

        if len(r0) < 2:
            rows.append({
                "from_epoch": e0,
                "to_epoch": e1,
                "kendall_tau": np.nan,
                "kendall_p_value": np.nan,
            })
            continue

        tau, p = kendalltau(r0, r1, variant="b")  # tau-b handles ties
        rows.append({
            "from_epoch": e0,
            "to_epoch": e1,
            "kendall_tau": float(tau),
            "kendall_p_value": float(p),
        })

    return pd.DataFrame(rows)

def _plot_kendall_to_final_panel(
    *,
    title: str,
    subtitle: str = "",
    metrics: List[str],
    results_by_metric: Dict[str, Optional[pd.DataFrame]],
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    dpi: int,
    show_title: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=True)
    axes = np.array(axes).ravel()

    for ax, metric_key in zip(axes, metrics):
        df = results_by_metric.get(metric_key)
        if df is None or df.empty or "kendall_tau_to_final" not in df.columns:
            ax.set_title(f"{metric_key} - no data", fontsize=18)
            ax.grid(True)
            continue

        ax.plot(df["epoch"].to_numpy(), df["kendall_tau_to_final"].to_numpy())

        ax.set_title(f"{metric_key}", fontsize=18)
        ax.set_xlabel("Epoch t", fontsize=16)
        ax.set_ylabel("Kendall τ(t, final)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    if show_title:
        fig.suptitle(title, y=1.06, fontsize=22)
        if subtitle:
            fig.text(0.5, 1.02, subtitle, ha="center", va="top", fontsize=16, color="#555555")
    fig.tight_layout()
    return fig

def _kendall_tau_to_final_table(rank_mat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Kendall's tau-b between each epoch and the final epoch.
    Returns columns: epoch, final_epoch, kendall_tau_to_final, kendall_p_value_to_final.
    """
    epochs = list(rank_mat.columns)
    if len(epochs) < 2:
        return pd.DataFrame()

    final_epoch = epochs[-1]
    r_final = rank_mat[final_epoch].to_numpy(dtype=float)

    rows = []
    for e in epochs[:-1]:  # exclude final itself
        r = rank_mat[e].to_numpy(dtype=float)

        mask = np.isfinite(r) & np.isfinite(r_final)
        r = r[mask]
        rf = r_final[mask]

        if len(r) < 2:
            rows.append({
                "epoch": e,
                "final_epoch": final_epoch,
                "kendall_tau_to_final": np.nan,
                "kendall_p_value_to_final": np.nan,
            })
            continue

        tau, p = kendalltau(r, rf, variant="b")
        rows.append({
            "epoch": e,
            "final_epoch": final_epoch,
            "kendall_tau_to_final": float(tau),
            "kendall_p_value_to_final": float(p),
        })

    return pd.DataFrame(rows)

def _plot_metric_panel(
    *,
    title: str,
    subtitle: str = "",
    metrics: List[str],
    results_by_metric: Dict[str, Optional[pd.DataFrame]],
    value_col: str,
    y_label: str,
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    dpi: int,
    y_lim: Tuple[float, float] | None = None,
    show_title: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=True)
    axes = np.array(axes).ravel()

    for ax, metric_key in zip(axes, metrics):
        df = results_by_metric.get(metric_key)
        df = _drop_epoch0_to1_for_plot(df)
        if df is None or df.empty or value_col not in df.columns:
            ax.set_title(f"{metric_key} - no data", fontsize=18)
            ax.grid(True)
            continue

        ax.plot(df["to_epoch"].to_numpy(), df[value_col].to_numpy())
        ax.set_title(f"{metric_key}", fontsize=18)
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.grid(True)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    if show_title:
        fig.suptitle(title, y=1.06, fontsize=22)
        if subtitle:
            fig.text(0.5, 1.02, subtitle, ha="center", va="top", fontsize=16, color="#555555")
    fig.tight_layout()
    return fig

def run_ranking_flips(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    """
    Multi-metric panel version of ranking + pairwise flips.

    Supports Option B config:
      metrics:
        - name: train_loss
          higher_is_better: false
        - name: train_acc
          higher_is_better: true
    """
    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")
    epoch_range = tuple(config.get("epoch_range", [])) or None

    metrics, hib_map = _parse_metrics(config.get("metrics"))

    smoothening_threshold = config.get("smoothening_threshold", config.get("tau", None))
    smoothening_threshold = None if smoothening_threshold is None else float(smoothening_threshold)

    nrows, ncols, figsize, dpi = _panel_params(config)
    show_title = bool(config.get("show_title", True))

    figures: List[Dict[str, plt.Figure]] = []
    tables: Dict[str, pd.DataFrame] = {}

    if "dataset_name" not in df.columns:
        raise ValueError("df must have dataset_name column for multi-dataset plotting.")

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())

    # --- per dataset panels ---
    for ds in datasets:
        dfd = df[df["dataset_name"] == ds]

        results_pairwise: Dict[str, Optional[pd.DataFrame]] = {}
        results_rankchg: Dict[str, Optional[pd.DataFrame]] = {}
        per_metric_switches: List[pd.DataFrame] = []
        results_kendall: Dict[str, Optional[pd.DataFrame]] = {}
        results_kendall_to_final: Dict[str, Optional[pd.DataFrame]] = {}

        for metric in metrics:
            mat = build_metric_matrix_from_dfm(
                dfd, metric_key=metric, epoch_col=epoch_col, model_col=model_col, epoch_range=epoch_range
            )

            n_models = mat.shape[0]
            if mat.shape[0] < 2 or mat.shape[1] < 2:
                results_pairwise[metric] = None
                results_rankchg[metric] = None
                continue

            # Pairwise flips table (contains both without/with smoothening columns)
            pf = pairwise_flip_table(
                mat,
                smoothening_threshold=float(smoothening_threshold) if smoothening_threshold is not None else 0.03,
            ).copy()
            if not pf.empty:
                pf["n_models"] = n_models
            results_pairwise[metric] = pf if not pf.empty else None

            # Rank switches with correct direction for each metric
            hib = bool(hib_map.get(metric, True))
            rmat = rank_per_epoch(mat, higher_is_better=hib, tie_method="average")
            rc = rank_change_table(
                mat,
                rank_mat=rmat,
                smoothening_threshold=smoothening_threshold,
            ).copy()
            if not rc.empty:
                rc["n_models"] = n_models
            results_rankchg[metric] = rc if not rc.empty else None

            # rank-change merged with flip table + avg_from/to/delta + metric + n_models
            if not rc.empty:
                avg_metric = mat.mean(axis=0)
                switch_df = rc.copy()

                if pf is not None and not pf.empty:
                    switch_df = switch_df.merge(pf, on=["from_epoch", "to_epoch"], how="left")

                switch_df["avg_from"] = switch_df["from_epoch"].map(avg_metric)
                switch_df["avg_to"] = switch_df["to_epoch"].map(avg_metric)
                switch_df["avg_delta"] = switch_df["avg_to"] - switch_df["avg_from"]
                switch_df["metric"] = metric
                switch_df["n_models"] = n_models
                per_metric_switches.append(switch_df)

            kt1 = _kendall_tau_table(rmat).copy()
            if not kt1.empty:
                kt1["n_models"] = n_models
            results_kendall[metric] = kt1 if not kt1.empty else None
            tables[f"{ds}_kendall_tau_step__{metric}"] = kt1

            ktF = _kendall_tau_to_final_table(rmat).copy()
            if not ktF.empty:
                ktF["n_models"] = n_models
            results_kendall_to_final[metric] = ktF if not ktF.empty else None
            tables[f"{ds}_kendall_tau_to_final__{metric}"] = ktF

        all_switches = (
            pd.concat(per_metric_switches, ignore_index=True)
            if per_metric_switches
            else pd.DataFrame()
        )
        tables[f"{ds}_all_switches_"] = all_switches

        # ---- FIGURE 1: Pairwise flips panel (without smoothening) ----
        fig = _plot_pairwise_panel(
            title="Ranking Flips in a model zoo",
            subtitle=f"Model Zoo: {zoo_plot_name(ds)} | Pairwise ranking flips · no smoothening",
            metrics=metrics,
            results_by_metric=results_pairwise,
            value_col="pairwise_flips_without_smoothening",
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            show_title=show_title,
        )
        fig_name = f"{ds}_pairwise_flips_without_smoothening"
        figures.append({fig_name: fig})

        # ---- FIGURE 2: Pairwise flips panel (with smoothening) ----
        if smoothening_threshold is not None:
            fig = _plot_pairwise_panel(
                title="Ranking Flips in a model zoo",
                subtitle=f"Model Zoo: {zoo_plot_name(ds)} | Pairwise ranking flips · smoothening threshold={smoothening_threshold:g}",
                metrics=metrics,
                results_by_metric=results_pairwise,
                value_col="pairwise_flips_with_smoothening",
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                dpi=dpi,
                show_title=show_title,
            )
            fig_name = f"{ds}_pairwise_flips_with_smoothening"
            figures.append({fig_name: fig})

        # ---- FIGURE 2b: Pairwise flips combined (without + with smoothening) ----
        if smoothening_threshold is not None:
            fig_combined = _plot_pairwise_combined_panel(
                title="Ranking Flips in a model zoo",
                subtitle=f"Model Zoo: {zoo_plot_name(ds)} | Pairwise ranking flips · combined",
                metrics=metrics,
                results_by_metric=results_pairwise,
                smoothening_threshold=smoothening_threshold,
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                dpi=dpi,
                show_title=show_title,
            )
            figures.append({f"{ds}_pairwise_flips_combined": fig_combined})

        # ---- FIGURE 3: Rank switches panel (without/with smoothening) ----
        fig3, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=True)
        axes = np.array(axes).ravel()

        for ax, metric_key in zip(axes, metrics):
            rc = results_rankchg.get(metric_key)
            rc = _drop_epoch0_to1_for_plot(rc)
            if rc is None or rc.empty:
                ax.set_title(f"{metric_key} (no data)", fontsize=18)
                ax.grid(True)
                continue

            x = rc["to_epoch"].to_numpy()
            N = rc["n_models"].iloc[0] if "n_models" in rc.columns and len(rc) > 0 else "?"

            ax.plot(x, rc["models_changed_rank"].to_numpy(), label="without smoothening")

            if "models_changed_rank_with_smoothening" in rc.columns:
                threshold_label = None
                if (
                    "smoothening_threshold_rank_change" in rc.columns
                    and not rc["smoothening_threshold_rank_change"].isna().all()
                ):
                    threshold_label = rc["smoothening_threshold_rank_change"].iloc[0]
                if threshold_label is not None:
                    ax.plot(
                        x,
                        rc["models_changed_rank_with_smoothening"].to_numpy(),
                        label=f"with smoothening (threshold={threshold_label:g})",
                    )
                else:
                    ax.plot(
                        x,
                        rc["models_changed_rank_with_smoothening"].to_numpy(),
                        label="with smoothening",
                    )

            ax.set_title(f"{metric_key}", fontsize=18)
            ax.set_xlabel("Epoch", fontsize=16)
            ax.set_ylabel("# models changed rank", fontsize=16)
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True)
            ax.legend(fontsize=14)

        for ax in axes[len(metrics):]:
            ax.axis("off")

        if show_title:
            fig3.suptitle("Rank Changes in a model zoo", y=1.06, fontsize=22)
            fig3.text(0.5, 1.02, f"Model Zoo: {zoo_plot_name(ds)} | Models changing rank per epoch", ha="center", va="top", fontsize=16, color="#555555")
        fig3.tight_layout()

        fig_name = f"{ds}_rank_switches"
        figures.append({fig_name: fig3})

        # ---- FIGURE 4: Kendall tau panel ----
        fig4 = _plot_metric_panel(
            title="Ranking Stability in a model zoo",
            subtitle=f"Model Zoo: {zoo_plot_name(ds)} | Kendall τ between consecutive epochs",
            metrics=metrics,
            results_by_metric=results_kendall,
            value_col="kendall_tau",
            y_label="Kendall τ",
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            y_lim=(-1.0, 1.0),
            show_title=show_title,
        )
        fig_name = f"{ds}_kendall_tau"
        figures.append({fig_name: fig4})

        fig5 = _plot_kendall_to_final_panel(
            title="Ranking Stability in a model zoo",
            subtitle=f"Model Zoo: {zoo_plot_name(ds)} | Kendall τ to final epoch",
            metrics=metrics,
            results_by_metric=results_kendall_to_final,
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            show_title=show_title,
        )
        figures.append({f"{ds}_kendall_tau_to_final": fig5})

    return AnalysisResult(tables=tables, figures=figures)
