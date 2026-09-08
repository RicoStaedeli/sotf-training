from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

def extract_curve_features(
    mat: pd.DataFrame,
    t95: float = 0.95,
    early_epochs: int = 10,
    late_epochs: int = 10,
) -> pd.DataFrame:
    """
    Compute per-model curve features from mat (index=model, columns=epochs).
    Works for both accuracy and loss (it's just numbers).
    """
    mat = mat.sort_index(axis=1)
    epochs = mat.columns.to_numpy()
    E = len(epochs)
    if E < 2:
        return pd.DataFrame(index=mat.index)

    X = mat.to_numpy(dtype=float)

    final_val = X[:, -1]
    peak_val = np.max(X, axis=1)
    peak_epoch_idx = np.argmax(X, axis=1)
    peak_epoch = epochs[peak_epoch_idx]
    peak_drop = peak_val - final_val

    auc = np.trapezoid(X, x=epochs, axis=1)
    auc_norm = auc / (epochs[-1] - epochs[0] + 1e-12)

    dX = np.diff(X, axis=1)
    mean_delta = dX.mean(axis=1)
    std_delta = dX.std(axis=1)
    std_val = X.std(axis=1)

    kE = min(early_epochs, E - 1)
    kL = min(late_epochs, E - 1)

    early_slope = (X[:, kE] - X[:, 0]) / (epochs[kE] - epochs[0] + 1e-12)
    late_slope = (X[:, -1] - X[:, -(kL + 1)]) / (epochs[-1] - epochs[-(kL + 1)] + 1e-12)

    # time-to-95% of final (first epoch where value >= 0.95*final)
    thresh = t95 * final_val
    t95_epoch = np.full(X.shape[0], np.nan, dtype=float)
    for i in range(X.shape[0]):
        hit = np.where(X[i] >= thresh[i])[0]
        if len(hit) > 0:
            t95_epoch[i] = float(epochs[hit[0]])

    avg_regret = np.mean(final_val[:, None] - X, axis=1)

    feat_df = pd.DataFrame({
        "model_name": mat.index,
        "final_val": final_val,
        "auc_norm": auc_norm,
        "peak_val": peak_val,
        "peak_epoch": peak_epoch.astype(float),
        "peak_drop": peak_drop,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "std_val": std_val,
        "early_slope": early_slope,
        "late_slope": late_slope,
        "t95_epoch": t95_epoch,
        "avg_regret": avg_regret,
    }, index=mat.index)

    return feat_df

# Robust dispersion (Median Absolute Deviation)
def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def _auc_mean(x: np.ndarray, y: np.ndarray) -> float:
    """
    AUC normalized by x-range (so it becomes 'average delta' over time),
    using trapezoidal integration. Works even if epochs not evenly spaced.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    xr = float(x[-1] - x[0])
    if xr == 0.0:
        return float(np.mean(y))
    area = float(np.trapezoid(y, x))
    return area / xr  # average y over x-range

def build_delta_features(
    delta_to_best_all: pd.DataFrame,
    model_col: str = "model_name",
    k: int = 5,
    use_mad: bool = True,
) -> pd.DataFrame:
    """
    Returns one row per (dataset_name, model) with:
      start_mean_k, end_mean_k, improvement_k,
      end_std_k (or end_mad_k),
      auc_mean,
      theil_sen_slope, spearman_rho (+ spearman_p)
    """
    rows = []
    if delta_to_best_all.empty:
        return pd.DataFrame()

    # ensure numeric
    d = delta_to_best_all.copy()
    d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
    d["delta_to_best"] = pd.to_numeric(d["delta_to_best"], errors="coerce")
    d = d[np.isfinite(d["epoch"]) & np.isfinite(d["delta_to_best"])]

    for (ds, model), g in d.groupby(["dataset_name", model_col], sort=False):
        gg = g.sort_values("epoch")
        x = gg["epoch"].to_numpy(dtype=float)
        y = gg["delta_to_best"].to_numpy(dtype=float)
        n = int(len(y))

        if n == 0:
            continue

        kk = int(min(k, n))
        start = y[:kk]
        end = y[-kk:]

        start_mean_k = float(np.mean(start))
        end_mean_k = float(np.mean(end))
        improvement_k = float(start_mean_k - end_mean_k)  # positive = improved

        if use_mad:
            end_disp = _mad(end)              # robust stability
            end_disp_name = "end_mad_k"
        else:
            end_disp = float(np.std(end, ddof=1)) if kk >= 2 else np.nan
            end_disp_name = "end_std_k"

        auc_mean = _auc_mean(x, y)

        # Trend: Theil–Sen (robust slope)
        # Returns (slope, intercept, lo_slope, hi_slope)
        if n >= 2:
            ts = stats.theilslopes(y, x)
            theil_sen_slope = float(ts[0])
        else:
            theil_sen_slope = np.nan

        # Monotonic trend: Spearman correlation
        if n >= 2:
            rho, rho_p = stats.spearmanr(x, y)
            spearman_rho = float(rho) if np.isfinite(rho) else np.nan
            spearman_p = float(rho_p) if np.isfinite(rho_p) else np.nan
        else:
            spearman_rho = np.nan
            spearman_p = np.nan

        rows.append({
            "dataset_name": ds,
            model_col: model,
            "n_points": n,
            "k": kk,
            "start_mean_k": start_mean_k,
            "end_mean_k": end_mean_k,
            "improvement_k": improvement_k,
            end_disp_name: end_disp,
            "auc_mean": float(auc_mean),
            "theil_sen_slope": theil_sen_slope,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
        })

    return pd.DataFrame(rows)