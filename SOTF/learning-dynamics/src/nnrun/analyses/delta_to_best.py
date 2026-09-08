from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .base import AnalysisResult
from ..compute.matrices import build_metric_matrix_from_dfm
from ..compute.features import build_delta_features
from ..zoo_info import zoo_plot_name


def _fit_lr(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit y ~ slope*x + intercept using OLS.
    Returns slope, intercept, r2, se_slope, t_stat, p_value (two-sided test slope=0).
    Assumes x,y are finite and length >= 2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(len(x))

    # OLS via polyfit
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept

    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)

    # Standard error of slope:
    # se = sqrt( (SSE/(n-2)) / sum((x-xbar)^2) )
    df = n - 2
    sxx = float(np.sum((x - np.mean(x)) ** 2))

    if df <= 0 or sxx == 0.0:
        se_slope = np.nan
        t_stat = np.nan
        p_value = np.nan
    else:
        mse = ss_res / df
        se_slope = float(np.sqrt(mse / sxx))
        t_stat = float(slope / se_slope) if se_slope > 0 else np.nan
        # numerically stable two-sided p-value
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "se_slope": float(se_slope),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "df": int(df),
    }


def _short(s: str, maxlen: int = 28) -> str:
    s = str(s)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def _select_models_for_delta_plot(
    ranked_models: pd.DataFrame,
    *,
    selection_mode: str = "rank_bands",
    top_n: int = 12,
    best_n: int = 3,
    middle_n: int = 3,
    worst_n: int = 3,
) -> pd.DataFrame:
    """
    Select models to plot from a final-ranking table.

    `ranked_models` is expected to contain `final_rank`, already defined so that
    lower is better. The default `rank_bands` mode selects best, middle, and
    worst performers based on the final ranking.
    """
    if ranked_models.empty:
        return ranked_models.copy()

    ranked = ranked_models.sort_values("final_rank", ascending=True, na_position="last").reset_index(drop=True)
    selection_mode = str(selection_mode).lower()

    if selection_mode == "top_n":
        selected = ranked.head(top_n).copy()
        selected["selection_group"] = "top"
        return selected

    n_models = len(ranked)
    chosen_positions: List[int] = []
    chosen_groups: Dict[int, str] = {}

    def add_slice(start: int, count: int, group: str) -> None:
        start = max(0, start)
        stop = min(n_models, start + max(0, count))
        for pos in range(start, stop):
            if pos not in chosen_positions:
                chosen_positions.append(pos)
                chosen_groups[pos] = group

    add_slice(0, best_n, "best")

    middle_start = max(0, (n_models - max(0, middle_n)) // 2)
    add_slice(middle_start, middle_n, "middle")

    worst_start = max(0, n_models - max(0, worst_n))
    add_slice(worst_start, worst_n, "worst")

    if not chosen_positions:
        return ranked.head(0).copy()

    selected = ranked.iloc[chosen_positions].copy()
    selected["selection_group"] = [chosen_groups[pos] for pos in chosen_positions]
    return selected


def _rank_delta_models(lr_ds: pd.DataFrame) -> pd.DataFrame:
    ranked = lr_ds.copy()
    if ranked.empty:
        ranked["final_rank"] = pd.Series(dtype=int)
        return ranked

    ranked = ranked.sort_values(
        ["delta_final", "model_name"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["final_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _build_delta_plot_title(row: pd.Series, title_prefix: Optional[str] = None) -> str:
    p = row.get("p_value", np.nan)
    p_txt = f"p={p:.2g}" if np.isfinite(p) else "p=nan"

    final_delta = row.get("delta_final", np.nan)
    fd_txt = f"{final_delta:.2g}" if np.isfinite(final_delta) else "nan"

    final_rank = row.get("final_rank", np.nan)
    rk_txt = f"{int(final_rank)}" if np.isfinite(final_rank) else "nan"

    title_lines = []
    if title_prefix:
        title_lines.append(title_prefix)
    title_lines.append(f"Rank={rk_txt}  Δfinal={fd_txt}")
    title_lines.append(f"slope={row['slope']:.2g}, r2={row['r2']:.2f}, {p_txt}")
    return "\n".join(title_lines)


def _plot_delta_to_best_panel(
    ax: plt.Axes,
    curve_df: pd.DataFrame,
    row: pd.Series,
    *,
    use_log: bool,
    scatter_color: str,
    regression_color: str,
    show_regression: bool = True,
    title_prefix: Optional[str] = None,
) -> None:
    g = curve_df[["epoch", "delta_to_best"]].dropna()
    if g.empty:
        label = title_prefix or str(row.get("model_name", "model"))
        ax.set_title(f"{label}\n(no data)", fontsize=11, pad=2)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True)
        return

    x = g["epoch"].to_numpy(dtype=float)
    y = g["delta_to_best"].to_numpy(dtype=float)

    if use_log:
        ax.set_yscale("symlog", linthresh=1e-6)

    ax.scatter(x, y, s=10, color=scatter_color)

    slope = float(row["slope"]) if np.isfinite(row["slope"]) else np.nan
    intercept = float(row["intercept"]) if np.isfinite(row["intercept"]) else np.nan
    if show_regression and np.isfinite(slope) and np.isfinite(intercept) and x.size >= 2:
        xline = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        yline = slope * xline + intercept
        ax.plot(
            xline,
            yline,
            color=regression_color,
            linewidth=1.5,
            label="Linear regression",
        )
        ax.legend(loc="best", fontsize=11)

    ax.set_title(_build_delta_plot_title(row, title_prefix=title_prefix), fontsize=11, pad=2)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True)


def plot_end_vs_improvement(
    features: pd.DataFrame,
    model_col: str = "model_name",
    stability_col: Optional[str] = None,
) -> List[Dict[str, plt.Figure]]:
    figs: List[Dict[str, plt.Figure]] = []
    if features.empty:
        return figs

    if stability_col is None:
        stability_col = "end_mad_k" if "end_mad_k" in features.columns else "end_std_k"

    stab_label = "End Stability (MAD)" if "mad" in stability_col.lower() else "End Stability (Std)"

    for ds in sorted(features["dataset_name"].unique()):
        f = features[features["dataset_name"] == ds].copy()
        f = f[
            np.isfinite(f["end_mean_k"])
            & np.isfinite(f["improvement_k"])
            & np.isfinite(f[stability_col])
        ]
        if f.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
        sc = ax.scatter(
            f["end_mean_k"].to_numpy(),
            f["improvement_k"].to_numpy(),
            c=f[stability_col].to_numpy(),
            s=50,
            alpha=0.8,
        )
        ax.set_xlabel("Final Distance to Best (lower is better)", fontsize=16)
        ax.set_ylabel("Improvement over Training (higher is better)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(stab_label, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        fig.suptitle("End vs. Improvement", y=1.06, fontsize=22)
        fig.text(
            0.5, 1.02,
            f"Model Zoo: {zoo_plot_name(ds)} | stability encoded as color",
            ha="center", va="top", fontsize=16, color="#555555",
        )
        fig.tight_layout()

        figs.append({f"{ds}_end_vs_improvement": fig})

    return figs


def plot_end_vs_improvement_grouped(
    features: pd.DataFrame,
    model_col: str = "model_name",
    stability_col: Optional[str] = None,
    n_groups: int = 4,
) -> List[Dict[str, plt.Figure]]:
    """
    Divide models into n_groups performance bands by end_mean_k (lower = closer to best),
    then show violin distributions of improvement_k and stability within each band.
    Complements the scatter by highlighting whether top-performing models also improve
    more and stabilize earlier.
    """
    figs: List[Dict[str, plt.Figure]] = []
    if features.empty:
        return figs

    if stability_col is None:
        stability_col = "end_mad_k" if "end_mad_k" in features.columns else "end_std_k"

    stab_label = "End Stability (MAD)" if "mad" in stability_col.lower() else "End Stability (Std)"

    group_labels = {
        4: ["Q1\n(Best)", "Q2", "Q3", "Q4\n(Worst)"],
        3: ["Top", "Mid", "Bottom"],
        2: ["Top Half", "Bottom Half"],
    }
    labels = group_labels.get(n_groups, [f"G{i+1}" for i in range(n_groups)])

    _COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    for ds in sorted(features["dataset_name"].unique()):
        f = features[features["dataset_name"] == ds].copy()
        f = f[
            np.isfinite(f["end_mean_k"])
            & np.isfinite(f["improvement_k"])
            & np.isfinite(f[stability_col])
        ].copy()
        if f.empty:
            continue

        f["_perf_band"] = pd.qcut(
            f["end_mean_k"],
            q=n_groups,
            labels=labels[:n_groups],
            duplicates="drop",
        )
        actual_groups = [g for g in labels[:n_groups] if g in f["_perf_band"].values]
        n_actual = len(actual_groups)
        if n_actual == 0:
            continue

        colors = (_COLORS * ((n_actual // len(_COLORS)) + 1))[:n_actual]

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=160)

        for ax, col, ylabel, title_panel in [
            (axes[0], "improvement_k", "Improvement over Training\n(higher is better)", "Improvement"),
            (axes[1], stability_col, f"{stab_label}\n(lower is better)", "End Stability"),
        ]:
            group_data = [
                f.loc[f["_perf_band"] == g, col].dropna().to_numpy(dtype=float)
                for g in actual_groups
            ]
            positions = list(range(1, n_actual + 1))

            parts = ax.violinplot(
                group_data,
                positions=positions,
                showmedians=True,
                showextrema=True,
            )
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            for part_key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if part_key in parts:
                    parts[part_key].set_color("#333333")
                    parts[part_key].set_linewidth(1.4)

            # overlay individual model points
            for pos, gdata, color in zip(positions, group_data, colors):
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(gdata))
                ax.scatter(
                    np.full(len(gdata), pos) + jitter,
                    gdata,
                    s=18,
                    color=color,
                    alpha=0.6,
                    zorder=3,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(actual_groups, fontsize=14)
            ax.set_xlabel("Performance Band (by Final Distance to Best)", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.tick_params(axis="y", labelsize=13)
            ax.set_title(title_panel, fontsize=16, pad=6)
            ax.grid(True, axis="y", alpha=0.5)

        fig.suptitle("Performance Band Comparison", y=1.06, fontsize=22)
        fig.text(
            0.5, 1.02,
            f"Model Zoo: {zoo_plot_name(ds)} | models grouped by final distance to best",
            ha="center", va="top", fontsize=16, color="#555555",
        )
        fig.tight_layout()

        figs.append({f"{ds}_end_vs_improvement_grouped": fig})

    return figs


def _format_selection_group_label(group: str) -> str:
    group = str(group).lower()
    labels = {
        "best": "Best",
        "middle": "Middle",
        "worst": "Worst",
        "top": "Top",
    }
    return labels.get(group, _short(group, maxlen=12))


def add_z_outlier_score(features: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = features.copy()
    out["z_outlier_score"] = np.nan

    for ds, g in out.groupby("dataset_name", sort=False):
        gg = g.copy()
        M = gg[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask = np.all(np.isfinite(M), axis=1)
        if mask.sum() < 3:
            continue

        X = M[mask]
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        Z = (X - mu) / sd
        score = np.sqrt(np.nansum(Z**2, axis=1))  # Euclidean z-distance

        idx = gg.index.to_numpy()[mask]
        out.loc[idx, "z_outlier_score"] = score

    return out


def add_isoforest_flags(
    features: pd.DataFrame,
    cols: List[str],
    contamination: float = 0.1,
) -> pd.DataFrame:
    out = features.copy()
    out["iso_outlier"] = False
    out["iso_score"] = np.nan

    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return out  # sklearn not installed / available

    for ds, g in out.groupby("dataset_name", sort=False):
        gg = g.copy()
        X = gg[cols].apply(pd.to_numeric, errors="coerce")
        mask = np.all(np.isfinite(X.to_numpy(dtype=float)), axis=1)
        if mask.sum() < 10:
            continue

        Xv = X.to_numpy(dtype=float)[mask]
        clf = IsolationForest(random_state=0, contamination=contamination)
        clf.fit(Xv)

        pred = clf.predict(Xv)               # -1 outlier, +1 inlier
        score = clf.decision_function(Xv)    # higher = more normal

        idx = gg.index.to_numpy()[mask]
        out.loc[idx, "iso_outlier"] = (pred == -1)
        out.loc[idx, "iso_score"] = score

    return out

def plot_feature_distributions(
    features: pd.DataFrame,
    cols: List[str],
    bins: int = 30,
    max_cols: int = 3,   # grid width
) -> List[Dict[str, plt.Figure]]:
    """
    For each dataset, plot histograms of selected feature columns.
    Adds skewness in title (Fisher-Pearson sample skewness).
    """
    figs: List[Dict[str, plt.Figure]] = []
    if features.empty:
        return figs

    for ds in sorted(features["dataset_name"].unique()):
        f = features[features["dataset_name"] == ds].copy()

        # keep only finite values per col when plotting
        plot_cols = [c for c in cols if c in f.columns]
        if not plot_cols:
            continue

        nplots = len(plot_cols)
        ncols = int(min(max_cols, nplots))
        nrows = int(np.ceil(nplots / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), dpi=160)
        axes = np.array(axes).ravel()

        for ax, col in zip(axes, plot_cols):
            x = pd.to_numeric(f[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                ax.set_title(f"{col}\n(no finite data)")
                ax.axis("off")
                continue

            ax.hist(x, bins=bins)
            ax.grid(True)

            # sample skewness (no scipy needed)
            # skew = E[(X-mean)^3] / std^3
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
            if sd > 0:
                skew = float(np.mean((x - mu) ** 3) / (sd ** 3))
            else:
                skew = np.nan

            ax.set_title(f"{col}\nskew={skew:.2g}  n={x.size}")
            ax.set_xlabel(col)
            ax.set_ylabel("count")

        for ax in axes[nplots:]:
            ax.axis("off")

        fig.suptitle(f"{zoo_plot_name(ds)} | Feature Distribution", y=1.02)
        fig.tight_layout()
        figs.append({f"{ds}_feature_distributions": fig})

    return figs


def run_delta_to_best(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    """
    For each epoch, compute delta-to-best per model for ONE metric:

      - higher_is_better=True (accuracy):  delta = best(epoch) - value(model,epoch)
      - higher_is_better=False (loss):     delta = value(model,epoch) - best(epoch)

    Output table columns:
      dataset_name, epoch, model_name, value, best_value, delta_to_best
    """
    if "dataset_name" not in df.columns:
        raise ValueError("delta_to_best: df must contain dataset_name column.")

    metric = config["metric"]
    higher_is_better = bool(config.get("higher_is_better", True))

    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")
    epoch_range = tuple(config.get("epoch_range", [])) or None

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())
    rows: List[pd.DataFrame] = []

    for ds in datasets:
        dfd = df[df["dataset_name"] == ds]

        mat = build_metric_matrix_from_dfm(
            dfm=dfd,
            metric_key=metric,
            epoch_col=epoch_col,
            model_col=model_col,
            epoch_range=epoch_range,
        )
        if mat.empty:
            continue

        # best per epoch
        best = mat.max(axis=0) if higher_is_better else mat.min(axis=0)

        # delta-to-best (always "how far behind best", so >=0 when values are finite)
        delta = (best - mat) if higher_is_better else (mat - best)

        # long-form
        long_val = mat.stack().rename("value").reset_index()
        long_val.columns = [model_col, "epoch", "value"]

        long_delta = delta.stack().rename("delta_to_best").reset_index()
        long_delta.columns = [model_col, "epoch", "delta_to_best"]

        long_best = best.rename("best_value").reset_index()
        long_best.columns = ["epoch", "best_value"]

        out = (
            long_val.merge(long_delta, on=[model_col, "epoch"], how="left")
                    .merge(long_best, on="epoch", how="left")
        )
        out.insert(0, "dataset_name", ds)
        rows.append(out)

    delta_to_best_all = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["dataset_name", "epoch", model_col, "value", "best_value", "delta_to_best"])
    )

    # enforce numeric types
    for c in ["epoch", "value", "best_value", "delta_to_best"]:
        if c in delta_to_best_all.columns:
            delta_to_best_all[c] = pd.to_numeric(delta_to_best_all[c], errors="coerce")

    # ---------------- linear regression summary table ----------------
    rows_lr: List[Dict[str, Any]] = []

    if not delta_to_best_all.empty:
        delta_to_best_all["epoch"] = pd.to_numeric(delta_to_best_all["epoch"], errors="coerce")
        delta_to_best_all["delta_to_best"] = pd.to_numeric(delta_to_best_all["delta_to_best"], errors="coerce")

        for (ds, model), g in delta_to_best_all.groupby(["dataset_name", model_col], sort=False):
            gg = g[["epoch", "delta_to_best"]].dropna()
            gg = gg[np.isfinite(gg["epoch"]) & np.isfinite(gg["delta_to_best"])]

            n = int(len(gg))
            if n < 2:
                rows_lr.append({
                    "dataset_name": ds,
                    "model_name": model,
                    "n_points": n,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "epoch_min": np.nan,
                    "epoch_max": np.nan,
                    "delta_mean": np.nan,
                    "delta_final": np.nan,
                    "se_slope": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "df": np.nan,
                })
                continue

            x = gg["epoch"].to_numpy(dtype=float)
            y = gg["delta_to_best"].to_numpy(dtype=float)

            fit = _fit_lr(x, y)
            i_end = int(np.argmax(x))  # index of max epoch
            delta_final = float(y[i_end])

            rows_lr.append({
                "dataset_name": ds,
                "model_name": model,
                "n_points": n,
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "epoch_min": float(np.min(x)),
                "epoch_max": float(np.max(x)),
                "delta_mean": float(np.mean(y)),
                "delta_final": delta_final,
                "se_slope": fit["se_slope"],
                "t_stat": fit["t_stat"],
                "p_value": fit["p_value"],
                "df": fit["df"],
            })

    delta_to_best_lr = pd.DataFrame(rows_lr)

    # ---------------- feature table + outliers ----------------
    features = build_delta_features(
        delta_to_best_all=delta_to_best_all,
        model_col=model_col,
        k=int(config.get("window_k", 5)),
        use_mad=bool(config.get("use_mad", True)),
    )

    if not features.empty:
        stability_col = "end_mad_k" if "end_mad_k" in features.columns else "end_std_k"
        feature_cols = ["end_mean_k", "improvement_k", "auc_mean", stability_col]

        features = add_z_outlier_score(features, cols=feature_cols)
        features = add_isoforest_flags(
            features,
            cols=feature_cols,
            contamination=float(config.get("isoforest_contamination", 0.1)),
        )

    # ---------------- tables ----------------
    tables = {
        "delta_to_best_all": delta_to_best_all,
        "delta_to_best_lr": delta_to_best_lr,
        "delta_to_best_features": features,
    }

    # ---------------- figures ----------------
    plot_cfg = config.get("plot", {}) or {}
    plot_enabled = bool(plot_cfg.get("enabled", False))
    figures: List[Dict[str, plt.Figure]] = []

    # (A) your existing per-model fitted-line grid
    if plot_enabled and (not delta_to_best_all.empty) and (not delta_to_best_lr.empty):
        top_n = int(plot_cfg.get("top_n", 12))
        selection_mode = str(plot_cfg.get("selection_mode", "rank_bands")).lower()
        best_n = int(plot_cfg.get("best_n", 3))
        middle_n = int(plot_cfg.get("middle_n", 3))
        worst_n = int(plot_cfg.get("worst_n", 3))
        ncols = int(plot_cfg.get("ncols", 4))
        figsize_raw = plot_cfg.get("figsize", [12, 9])
        figsize = (float(figsize_raw[0]), float(figsize_raw[1]))
        best_model_plot_enabled = bool(plot_cfg.get("best_model_overview_enabled", True))
        best_model_figsize_raw = plot_cfg.get("overview_figsize", [7.5, 5.5])
        best_model_figsize = (float(best_model_figsize_raw[0]), float(best_model_figsize_raw[1]))
        dpi = int(plot_cfg.get("dpi", 200))
        use_log = bool(plot_cfg.get("use_log", False))

        scatter_color = "#4C78A8"
        regression_color = "#E45756"

        for ds in sorted(delta_to_best_all["dataset_name"].unique().tolist()):
            lr_ds = delta_to_best_lr[delta_to_best_lr["dataset_name"] == ds].copy()
            if lr_ds.empty:
                continue

            # --- ranking always based on delta_final (lower is better) ---
            lr_ds = _rank_delta_models(lr_ds)

            lr_top = _select_models_for_delta_plot(
                lr_ds,
                selection_mode=selection_mode,
                top_n=top_n,
                best_n=best_n,
                middle_n=middle_n,
                worst_n=worst_n,
            )
            models = lr_top["model_name"].tolist()
            if not models:
                continue

            if selection_mode == "top_n":
                nrows = int(np.ceil(len(models) / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize, dpi=dpi, sharex=True, sharey=True
                )
                axes = np.array(axes).ravel()
                grouped_axes: List[tuple[pd.DataFrame, List[plt.Axes]]] = [(lr_top, axes.tolist())]
            else:
                grouped_axes = []
                group_specs = []
                for group, group_df in lr_top.groupby("selection_group", sort=False):
                    group_models = group_df["model_name"].tolist()
                    if not group_models:
                        continue
                    rank_min = int(group_df["final_rank"].min())
                    rank_max = int(group_df["final_rank"].max())
                    rank_span = f"Rank {rank_min}" if rank_min == rank_max else f"Rank {rank_min} - {rank_max}"
                    row_lengths = [
                        len(group_models[i : i + ncols]) for i in range(0, len(group_models), ncols)
                    ]
                    group_specs.append(
                        {
                            "group_df": group_df,
                            "row_title": f"{_format_selection_group_label(group)} {rank_span}",
                            "row_lengths": row_lengths,
                            "rows_needed": len(row_lengths),
                        }
                    )

                height_ratios: List[float] = []
                for spec in group_specs:
                    height_ratios.append(0.04)
                    height_ratios.extend([1.0] * spec["rows_needed"])

                fig = plt.figure(figsize=figsize, dpi=dpi)
                gs = fig.add_gridspec(nrows=len(height_ratios), ncols=ncols, height_ratios=height_ratios)
                axes_list: List[plt.Axes] = []
                shared_x_ax: Optional[plt.Axes] = None
                gs_row = 0

                for spec in group_specs:
                    header_ax = fig.add_subplot(gs[gs_row, :])
                    header_ax.axis("off")
                    header_ax.text(
                        0.5,
                        0.5,
                        spec["row_title"],
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color="#333333",
                    )
                    gs_row += 1

                    group_axes: List[plt.Axes] = []
                    for row_len in spec["row_lengths"]:
                        row_shared_y_ax: Optional[plt.Axes] = None
                        row_gs = gs[gs_row, :].subgridspec(1, row_len)
                        for col in range(row_len):
                            share_kwargs: Dict[str, plt.Axes] = {}
                            if shared_x_ax is not None:
                                share_kwargs["sharex"] = shared_x_ax
                            if row_shared_y_ax is not None:
                                share_kwargs["sharey"] = row_shared_y_ax
                            ax = fig.add_subplot(row_gs[0, col], **share_kwargs)
                            if shared_x_ax is None:
                                shared_x_ax = ax
                            if row_shared_y_ax is None:
                                row_shared_y_ax = ax
                            else:
                                ax.tick_params(axis="y", labelleft=False)
                            group_axes.append(ax)
                            axes_list.append(ax)
                        gs_row += 1

                    grouped_axes.append((spec["group_df"], group_axes))

                axes = np.array(axes_list, dtype=object)

            data_ds = delta_to_best_all[delta_to_best_all["dataset_name"] == ds]

            for group_df, group_axes in grouped_axes:
                group_models = group_df["model_name"].tolist()
                for ax, model in zip(group_axes, group_models):
                    row = group_df.loc[group_df["model_name"] == model].iloc[0]
                    curve_df = data_ds[data_ds["model_name"] == model]
                    _plot_delta_to_best_panel(
                        ax,
                        curve_df,
                        row,
                        use_log=use_log,
                        scatter_color=scatter_color,
                        regression_color=regression_color,
                    )

                for ax in group_axes[len(group_models):]:
                    ax.axis("off")

            fig.suptitle("Delta to Best Model", y=1.06, fontsize=22)
            fig.text(0.5, 1.02, f"Model Zoo: {zoo_plot_name(ds)}", ha="center", va="top", fontsize=16, color="#555555")
            fig.supxlabel("Epoch", fontsize=16)
            fig.supylabel("Delta to best ranked model", fontsize=16)
            fig.tight_layout()

            if selection_mode == "top_n":
                fig_name = f"{ds}_delta_to_best_fit_{metric}_top{top_n}"
            else:
                fig_name = (
                    f"{ds}_delta_to_best_fit_{metric}_best{best_n}_middle{middle_n}_worst{worst_n}"
                )
            figures.append({fig_name: fig})

            if best_model_plot_enabled and not lr_ds.empty:
                best_row = lr_ds.iloc[0]
                best_model = best_row["model_name"]
                best_curve_df = data_ds[data_ds["model_name"] == best_model]
                for show_regression, suffix, title_suffix in [
                    (True, "with_regression", "with regression"),
                    (False, "without_regression", "without regression"),
                ]:
                    fig_best, ax_best = plt.subplots(figsize=best_model_figsize, dpi=dpi)
                    _plot_delta_to_best_panel(
                        ax_best,
                        best_curve_df,
                        best_row,
                        use_log=use_log,
                        scatter_color=scatter_color,
                        regression_color=regression_color,
                        show_regression=show_regression,
                        title_prefix=_short(best_model, maxlen=28),
                    )
                    fig_best.suptitle(f"Delta to Best Model", y=1.06, fontsize=22)
                    fig_best.text(0.5, 1.02, f"Model Zoo: {zoo_plot_name(ds)} | Best model {title_suffix}", ha="center", va="top", fontsize=16, color="#555555")
                    fig_best.supxlabel("Epoch", fontsize=16)
                    fig_best.supylabel("Delta to best ranked model", fontsize=16)
                    fig_best.tight_layout()
                    figures.append({f"{ds}_delta_to_best_fit_{metric}_best_model_{suffix}": fig_best})

    # (B) “end vs improvement” scatter + grouped violin alternative
    if plot_enabled and (not features.empty):
        figures.extend(plot_end_vs_improvement(features, model_col=model_col))
        figures.extend(plot_end_vs_improvement_grouped(features, model_col=model_col))


    if plot_enabled and (not features.empty):
        stability_col = "end_mad_k" if "end_mad_k" in features.columns else "end_std_k"
        dist_cols = [
            "end_mean_k",
            "improvement_k",
            "auc_mean",
            stability_col,
            "z_outlier_score",
            "iso_score",
        ]
        figures.extend(plot_feature_distributions(features, cols=dist_cols, bins=30))

    return AnalysisResult(tables=tables, figures=figures)
