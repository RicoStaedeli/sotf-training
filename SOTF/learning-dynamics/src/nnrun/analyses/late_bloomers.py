from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import AnalysisResult
from ..compute.matrices import build_metric_matrix_from_dfm
from ..compute.ranks import rank_per_epoch


def _resolve_epoch_mid(epoch_mid_cfg: Any, available_epochs: List[int]) -> int:
    """Resolve epoch_mid config value to an actual epoch number.

    A float in (0, 1] is treated as a fraction of training length:
      0.5 → the epoch at the 50th percentile of available epochs.
    Any other value (int or float > 1) is treated as an absolute epoch number,
    snapping down to the nearest available epoch if the exact epoch is missing.
    """
    if not available_epochs:
        return 0
    if isinstance(epoch_mid_cfg, float) and 0.0 < epoch_mid_cfg <= 1.0:
        idx = int(round(epoch_mid_cfg * (len(available_epochs) - 1)))
        idx = max(0, min(idx, len(available_epochs) - 1))
        return int(available_epochs[idx])
    target = int(epoch_mid_cfg)
    if target in available_epochs:
        return target
    earlier = [e for e in available_epochs if e <= target]
    return int(max(earlier)) if earlier else int(available_epochs[0])


def _parse_metrics(metrics_cfg: Any) -> Tuple[List[str], Dict[str, bool]]:
    """
    Calculates metrics from config with given value higher is better or not.
    Returns:
      metrics: list[str]
      hib_map: dict[str,bool]
    """
    if not isinstance(metrics_cfg, list) or len(metrics_cfg) == 0:
        raise ValueError("late_bloomers: config must provide non-empty 'metrics' list.")

    if isinstance(metrics_cfg[0], dict):
        metrics: List[str] = []
        hib_map: Dict[str, bool] = {}
        for m in metrics_cfg:
            if "name" not in m:
                raise ValueError("late_bloomers: each metrics entry must have a 'name' field.")
            name = str(m["name"])
            metrics.append(name)
            hib_map[name] = bool(m.get("higher_is_better", True))
        return metrics, hib_map

    # fallback: allow simple list of metric names (defaults to higher_is_better=True)
    metrics = [str(x) for x in metrics_cfg]
    return metrics, {}

def _df_to_table_figure(df: pd.DataFrame, title: str, *, max_rows: int = 40, dpi: int = 200):
    """
    Render a dataframe as a matplotlib table figure.
    """
    show = df.copy()
    if len(show) > max_rows:
        show = show.head(max_rows).copy()

    # convert to strings for nicer display
    show = show.fillna("").astype(str)

    fig, ax = plt.subplots(figsize=(12, 0.35 * (len(show) + 3)), dpi=dpi)
    ax.axis("off")
    ax.set_title(title, pad=12)

    tbl = ax.table(
        cellText=show.values,
        colLabels=show.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)

    fig.tight_layout()
    return fig

def run_late_bloomers(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    """
    Late bloomer:  bottom `bottom_frac_mid` at epoch_mid  → top `top_frac_end`    at epoch_end
    Early fader:   top   `top_frac_mid`    at epoch_mid  → bottom `bottom_frac_end` at epoch_end

    Both are computed in one pass and reported in a single combined summary table.
    Rank convention: 1 = best (acc: higher better; loss: lower better).
    """
    if "dataset_name" not in df.columns:
        raise ValueError("late_bloomers: df must contain 'dataset_name' column.")

    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")

    epoch_mid_cfg = config["epoch_mid"]
    epoch_mid_is_pct = isinstance(epoch_mid_cfg, float) and 0.0 < epoch_mid_cfg <= 1.0
    mid_label = f"{epoch_mid_cfg:.0%}" if epoch_mid_is_pct else str(int(epoch_mid_cfg))

    bottom_frac_mid = float(config.get("bottom_frac_mid", 0.30))
    top_frac_end    = float(config.get("top_frac_end",    0.30))
    top_frac_mid    = float(config.get("top_frac_mid",    top_frac_end))
    bottom_frac_end = float(config.get("bottom_frac_end", bottom_frac_mid))

    metrics, hib_map = _parse_metrics(config.get("metrics"))

    epoch_range_cfg = config.get("epoch_range", None)
    epoch_range = tuple(epoch_range_cfg) if epoch_range_cfg else None

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())

    lb_rows: List[Dict[str, Any]] = []
    ef_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        dfd = df[df["dataset_name"] == ds]

        for metric_key in metrics:
            mat = build_metric_matrix_from_dfm(
                dfd,
                metric_key=metric_key,
                epoch_col=epoch_col,
                model_col=model_col,
                epoch_range=epoch_range,
            )

            if mat.empty:
                summary_rows.append({
                    "dataset_name": ds,
                    "metric": metric_key,
                    "n_models": 0,
                    "late_bloomers": 0,
                    "early_faders": 0,
                    "epoch_mid": np.nan,
                    "epoch_end": np.nan,
                    "reason": "no data",
                })
                continue

            available_epochs = sorted(int(c) for c in mat.columns)
            epoch_mid = _resolve_epoch_mid(epoch_mid_cfg, available_epochs)
            epoch_end = int(max(available_epochs))

            n_models = int(mat.shape[0])
            if n_models == 0:
                summary_rows.append({
                    "dataset_name": ds,
                    "metric": metric_key,
                    "n_models": 0,
                    "late_bloomers": 0,
                    "early_faders": 0,
                    "epoch_mid": epoch_mid,
                    "epoch_end": epoch_end,
                    "reason": "no models",
                })
                continue

            hib = bool(hib_map.get(metric_key, True))
            rmat = rank_per_epoch(mat, higher_is_better=hib, tie_method="average")

            r_mid = rmat[epoch_mid]
            r_end = rmat[epoch_end]

            # late bloomers: worst at mid, best at end
            lb_bottom_rank = int(np.floor((1.0 - bottom_frac_mid) * n_models)) + 1
            lb_top_rank    = int(np.ceil(top_frac_end * n_models))
            lb_mask  = (r_mid >= lb_bottom_rank) & (r_end <= lb_top_rank)
            lb_models = r_mid.index[lb_mask].tolist()

            for m in lb_models:
                lb_rows.append({
                    "dataset_name": ds,
                    "model_name": m,
                    "metric": metric_key,
                    "rank_epoch_mid": float(r_mid.loc[m]),
                    "rank_epoch_end": float(r_end.loc[m]),
                    "value_epoch_mid": float(mat.loc[m, epoch_mid]),
                    "value_epoch_end": float(mat.loc[m, epoch_end]),
                    "epoch_mid": epoch_mid,
                    "epoch_end": epoch_end,
                })

            # early faders: best at mid, worst at end
            ef_top_rank    = int(np.ceil(top_frac_mid * n_models))
            ef_bottom_rank = int(np.floor((1.0 - bottom_frac_end) * n_models)) + 1
            ef_mask  = (r_mid <= ef_top_rank) & (r_end >= ef_bottom_rank)
            ef_models = r_mid.index[ef_mask].tolist()

            for m in ef_models:
                ef_rows.append({
                    "dataset_name": ds,
                    "model_name": m,
                    "metric": metric_key,
                    "rank_epoch_mid": float(r_mid.loc[m]),
                    "rank_epoch_end": float(r_end.loc[m]),
                    "value_epoch_mid": float(mat.loc[m, epoch_mid]),
                    "value_epoch_end": float(mat.loc[m, epoch_end]),
                    "epoch_mid": epoch_mid,
                    "epoch_end": epoch_end,
                })

            summary_rows.append({
                "dataset_name": ds,
                "metric": metric_key,
                "n_models": n_models,
                "late_bloomers": len(lb_models),
                "early_faders": len(ef_models),
                "bottom_frac_mid": bottom_frac_mid,
                "top_frac_end": top_frac_end,
                "top_frac_mid": top_frac_mid,
                "bottom_frac_end": bottom_frac_end,
                "epoch_mid": epoch_mid,
                "epoch_end": epoch_end,
                "reason": "ok",
            })

    summary = pd.DataFrame(summary_rows).sort_values(["dataset_name", "metric"], ascending=True)

    tables: Dict[str, pd.DataFrame] = {
        "late_bloomers_all": pd.DataFrame(lb_rows),
        "early_faders_all":  pd.DataFrame(ef_rows),
        "summary":           summary,
    }

    figures: List[Dict[str, Any]] = []

    if not summary.empty and bool(config.get("plot_summary_table", True)):
        default_cols = ["dataset_name", "metric", "n_models", "late_bloomers", "early_faders", "epoch_mid", "epoch_end"]
        cols = config.get("summary_columns", default_cols)
        cols = [c for c in cols if c in summary.columns]
        df_show = summary[cols].copy().sort_values(["dataset_name", "metric"], ascending=True)

        dpi = int(config.get("png_dpi", 200))
        title = (
            f"Late bloomers / Early faders | epoch_mid={mid_label} | "
            f"LB bottom={bottom_frac_mid:.2f}→top={top_frac_end:.2f}  "
            f"EF top={top_frac_mid:.2f}→bottom={bottom_frac_end:.2f}"
        )
        fig = _df_to_table_figure(df_show, title=title, max_rows=int(config.get("summary_max_rows", 50)), dpi=dpi)

        fig_name = f"late_bloomers_early_faders__summary_table__mid{mid_label}__last_epoch"
        figures.append({fig_name: fig})

    return AnalysisResult(tables=tables, figures=figures)


def run_early_faders(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    """
    Early fader definition:
      top `top_frac_mid` at epoch_mid, bottom `bottom_frac_end` at epoch_end (epoch_end = last epoch available)

    Rank convention: 1 = best (acc: higher better; loss: lower better).
    """
    if "dataset_name" not in df.columns:
        raise ValueError("early_faders: df must contain 'dataset_name' column.")

    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")

    epoch_mid_cfg = config["epoch_mid"]
    epoch_mid_is_pct = isinstance(epoch_mid_cfg, float) and 0.0 < epoch_mid_cfg <= 1.0
    mid_label = f"{epoch_mid_cfg:.0%}" if epoch_mid_is_pct else str(int(epoch_mid_cfg))

    top_frac_mid = float(config.get("top_frac_mid", 0.30))
    bottom_frac_end = float(config.get("bottom_frac_end", 0.20))

    metrics, hib_map = _parse_metrics(config.get("metrics"))

    epoch_range_cfg_val = config.get("epoch_range", None)
    epoch_range = tuple(epoch_range_cfg_val) if epoch_range_cfg_val else None

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        dfd = df[df["dataset_name"] == ds]

        for metric_key in metrics:
            mat = build_metric_matrix_from_dfm(
                dfd,
                metric_key=metric_key,
                epoch_col=epoch_col,
                model_col=model_col,
                epoch_range=epoch_range,
            )

            if mat.empty:
                summary_rows.append({
                    "dataset_name": ds,
                    "metric": metric_key,
                    "n_models": 0,
                    "early_faders": 0,
                    "epoch_mid": np.nan,
                    "epoch_end": np.nan,
                    "reason": "no data",
                })
                continue

            available_epochs = sorted(int(c) for c in mat.columns)
            epoch_mid = _resolve_epoch_mid(epoch_mid_cfg, available_epochs)
            epoch_end = int(max(available_epochs))

            n_models = int(mat.shape[0])
            if n_models == 0:
                summary_rows.append({
                    "dataset_name": ds,
                    "metric": metric_key,
                    "n_models": 0,
                    "early_faders": 0,
                    "epoch_mid": epoch_mid,
                    "epoch_end": epoch_end,
                    "reason": "no models",
                })
                continue

            hib = bool(hib_map.get(metric_key, True))
            rmat = rank_per_epoch(mat, higher_is_better=hib, tie_method="average")

            top_mid_rank = int(np.ceil(top_frac_mid * n_models))
            bottom_start_rank = int(np.floor((1.0 - bottom_frac_end) * n_models)) + 1

            r_mid = rmat[epoch_mid]
            r_end = rmat[epoch_end]

            mask = (r_mid <= top_mid_rank) & (r_end >= bottom_start_rank)
            models = r_mid.index[mask].tolist()
            count = len(models)

            for m in models:
                rows.append({
                    "dataset_name": ds,
                    "model_name": m,
                    "metric": metric_key,
                    "rank_epoch_mid": float(r_mid.loc[m]),
                    "rank_epoch_end": float(r_end.loc[m]),
                    "value_epoch_mid": float(mat.loc[m, epoch_mid]),
                    "value_epoch_end": float(mat.loc[m, epoch_end]),
                    "epoch_mid": epoch_mid,
                    "epoch_end": epoch_end,
                })

            summary_rows.append({
                "dataset_name": ds,
                "metric": metric_key,
                "n_models": n_models,
                "early_faders": int(count),
                "top_frac_mid": top_frac_mid,
                "bottom_frac_end": bottom_frac_end,
                "epoch_mid": epoch_mid,
                "epoch_end": epoch_end,
                "reason": "ok",
            })

    early_faders_all = pd.DataFrame(rows)
    early_faders_summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset_name", "metric"], ascending=True
    )

    tables: Dict[str, pd.DataFrame] = {
        "early_faders_all": early_faders_all,
        "early_faders_summary": early_faders_summary,
    }

    figures: List[Dict[str, Any]] = []

    if not early_faders_summary.empty and bool(config.get("plot_summary_table", True)):
        cols = config.get("summary_columns", ["dataset_name", "metric", "n_models", "early_faders", "epoch_mid", "epoch_end"])
        cols = [c for c in cols if c in early_faders_summary.columns]
        df_show = early_faders_summary[cols].copy()
        df_show = df_show.sort_values(["dataset_name", "early_faders"], ascending=[True, False])

        dpi = int(config.get("png_dpi", 200))
        title = f"Early faders summary | epoch_mid={mid_label} | top={top_frac_mid:.2f}, bottom={bottom_frac_end:.2f}"
        fig = _df_to_table_figure(df_show, title=title, max_rows=int(config.get("summary_max_rows", 50)), dpi=dpi)

        fig_name = f"early_faders__summary_table__mid{mid_label}__last_epoch"
        figures.append({fig_name: fig})

    return AnalysisResult(tables=tables, figures=figures)