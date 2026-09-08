from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr

from .base import AnalysisResult
from .weight_statistics import (
    ensure_weight_metric_cache,
    load_or_build_partitioned_weight_metric_manifest,
    load_or_compute_df_long,
)
from ..compute.matrices import build_metric_matrix_from_dfm
from ..compute.ranks import rank_per_epoch
from ..zoo_info import zoo_plot_name

log = logging.getLogger(__name__)


def _safe_name_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "empty"


def _to_int_epoch(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _as_list_or_all(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        if v.lower() == "all":
            return None
        return [v]
    if isinstance(v, Sequence):
        vals = [str(x) for x in v if x is not None]
        if len(vals) == 1 and vals[0].lower() == "all":
            return None
        return vals
    return [str(v)]


def _canonical_model_key(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
    )


def _trial_token(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"([A-Za-z0-9]{5}_\d{5})", expand=False)


def _safe_corr(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) < 3:
        return np.nan
    if np.allclose(xv, xv[0], atol=1e-12, rtol=0.0):
        return np.nan
    if np.allclose(yv, yv[0], atol=1e-12, rtol=0.0):
        return np.nan
    try:
        if method == "pearson":
            return float(pearsonr(xv, yv).statistic)
        if method == "spearman":
            return float(spearmanr(xv, yv).statistic)
    except Exception:
        return np.nan
    return np.nan


def _partial_corr_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    zv = np.asarray(z, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv)
    xv = xv[mask]
    yv = yv[mask]
    zv = zv[mask]
    if len(xv) < 4:
        return np.nan

    rxy = _safe_corr(xv, yv, method="pearson")
    rxz = _safe_corr(xv, zv, method="pearson")
    ryz = _safe_corr(yv, zv, method="pearson")
    if not (np.isfinite(rxy) and np.isfinite(rxz) and np.isfinite(ryz)):
        return np.nan

    denom = np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    return float((rxy - rxz * ryz) / denom)


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) < 2:
        return np.nan

    y01 = (y > 0.5).astype(int)
    n_pos = int(np.sum(y01 == 1))
    n_neg = int(np.sum(y01 == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = rankdata(s, method="average")
    rank_sum_pos = float(np.sum(ranks[y01 == 1]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u_stat / float(n_pos * n_neg)
    return float(auc)


def _binary_auc_components(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) < 2:
        return {"scores": np.array([], dtype=float), "labels": np.array([], dtype=int)}

    y01 = (y > 0.5).astype(int)
    n_pos = int(np.sum(y01 == 1))
    n_neg = int(np.sum(y01 == 0))
    if n_pos == 0 or n_neg == 0:
        return {"scores": s, "labels": y01, "n_pos": n_pos, "n_neg": n_neg}

    ranks = rankdata(s, method="average")
    rank_sum_pos = float(np.sum(ranks[y01 == 1]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    auc_raw = u_stat / float(n_pos * n_neg)
    return {
        "scores": s,
        "labels": y01,
        "ranks": ranks,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "rank_sum_pos": rank_sum_pos,
        "u_stat": float(u_stat),
        "auc_raw": float(auc_raw),
        "auc_best": float(max(auc_raw, 1.0 - auc_raw)),
        "auc_best_direction": 1.0 if auc_raw >= 0.5 else -1.0,
    }


def _binary_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = (y[mask] > 0.5).astype(int)
    s = s[mask]
    if len(y) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    distinct_last = np.where(np.r_[np.diff(s_sorted) != 0, True])[0]

    tpr = np.r_[0.0, tp[distinct_last] / float(n_pos), 1.0]
    fpr = np.r_[0.0, fp[distinct_last] / float(n_neg), 1.0]
    return fpr.astype(float), tpr.astype(float)


def _top_mask(n: int, q: float) -> np.ndarray:
    k = int(max(1, np.ceil(float(q) * float(n))))
    mask = np.zeros(n, dtype=bool)
    mask[:k] = True
    return mask


def _binary_quantile_lift(scores: np.ndarray, y_true: np.ndarray, q: float) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(float)
    mask = np.isfinite(s) & np.isfinite(y)
    s = s[mask]
    y = y[mask]
    n = int(len(s))
    if n == 0:
        return {
            "top_n": 0,
            "base_rate": np.nan,
            "top_rate": np.nan,
            "top_rate_lift": np.nan,
        }

    order = np.argsort(-s)
    y_sorted = y[order]
    top = _top_mask(n, q)
    base_rate = float(np.mean(y_sorted))
    top_rate = float(np.mean(y_sorted[top])) if np.any(top) else np.nan
    lift = np.nan
    if np.isfinite(base_rate) and base_rate > 0.0 and np.isfinite(top_rate):
        lift = float(top_rate / base_rate)
    return {
        "top_n": int(np.sum(top)),
        "base_rate": base_rate,
        "top_rate": top_rate,
        "top_rate_lift": lift,
    }


def _continuous_quantile_gain(scores: np.ndarray, target: np.ndarray, q: float) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    t = np.asarray(target, dtype=float)
    mask = np.isfinite(s) & np.isfinite(t)
    s = s[mask]
    t = t[mask]
    n = int(len(s))
    if n == 0:
        return {
            "top_n": 0,
            "base_mean": np.nan,
            "top_mean": np.nan,
            "top_minus_base": np.nan,
        }

    order = np.argsort(-s)
    t_sorted = t[order]
    top = _top_mask(n, q)
    base_mean = float(np.mean(t_sorted))
    top_mean = float(np.mean(t_sorted[top])) if np.any(top) else np.nan
    gain = float(top_mean - base_mean) if np.isfinite(top_mean) and np.isfinite(base_mean) else np.nan
    return {
        "top_n": int(np.sum(top)),
        "base_mean": base_mean,
        "top_mean": top_mean,
        "top_minus_base": gain,
    }


def _build_partition_combo_sources(
    manifest: pd.DataFrame,
    *,
    metric_families: Optional[List[str]],
    metrics: Optional[List[str]],
    scopes: Optional[List[str]],
    include_model_row: bool,
) -> List[Tuple[Tuple[str, str, str], Dict[str, Any]]]:
    if manifest is None or manifest.empty:
        return []

    manifest_sel = manifest.copy()
    if metric_families:
        manifest_sel = manifest_sel[manifest_sel["metric_family"].isin(metric_families)]
    if metrics:
        manifest_sel = manifest_sel[manifest_sel["metric"].isin(metrics)]
    if scopes:
        manifest_sel = manifest_sel[manifest_sel["scope"].isin(scopes)]

    combo_sources: List[Tuple[Tuple[str, str, str], Dict[str, Any]]] = []
    for row in manifest_sel.itertuples(index=False):
        combo_key = (str(row.metric_family), str(row.metric), str(row.scope))
        source_path: Optional[str] = None
        source_kind = "layers"

        if include_model_row and bool(row.has_model_rows):
            source_path = row.model_path
            source_kind = "model"
        elif bool(row.has_layer_rows):
            source_path = row.layers_path
            source_kind = "layers"

        if source_path is None:
            continue

        combo_sources.append(
            (
                combo_key,
                {
                    "path": str(source_path),
                    "source_kind": source_kind,
                },
            )
        )

    return combo_sources


def _prepare_weight_metrics_frame(dfw: pd.DataFrame, *, drop_epoch_zero: bool) -> pd.DataFrame:
    if dfw.empty:
        return dfw.copy()
    out = dfw.copy()
    out["epoch"] = _to_int_epoch(out["epoch"])
    out = out.dropna(subset=["model_id", "epoch", "value"]).copy()
    out["epoch"] = out["epoch"].astype(int)
    out["model_key"] = _canonical_model_key(out["model_id"])
    out["trial_token"] = _trial_token(out["model_id"])
    if drop_epoch_zero:
        out = out[out["epoch"] != 0].copy()
    return out


def _prepare_weight_series_from_group(group_df: pd.DataFrame, *, include_model_row: bool) -> Tuple[pd.DataFrame, str]:
    if group_df.empty:
        return pd.DataFrame(), "none"

    use_model_row = bool(
        include_model_row
        and ("layer_name" in group_df.columns)
        and (group_df["layer_name"] == "__model__").any()
    )
    if use_model_row:
        out = group_df[group_df["layer_name"] == "__model__"].copy()
        value_source = "model_row"
    else:
        out = group_df.groupby(["model_id", "epoch"], as_index=False, sort=False)["value"].mean()
        value_source = "layer_mean"

    if out.empty:
        return pd.DataFrame(), value_source

    out = out.rename(columns={"value": "weight_value"})
    keep_cols = [c for c in out.columns if c in {"model_id", "epoch", "weight_value", "model_key", "trial_token"}]
    out = out.loc[:, keep_cols].copy()
    if "model_key" not in out.columns:
        out["model_key"] = _canonical_model_key(out["model_id"])
    if "trial_token" not in out.columns:
        out["trial_token"] = _trial_token(out["model_id"])
    return out, value_source


def _resolve_dataset_cache_mode(
    *,
    ds_name: str,
    root_dir: Path,
    dstype: str,
    dataset_split: str,
    epoch_lst: Sequence[int],
    cache_dir: Path,
    force_recompute: bool,
    skip_missing_cache: bool,
    requested_use_partitioned_cache: bool,
    max_full_cache_size_mb: float,
) -> Tuple[bool, Optional[Path]]:
    cache_path = ensure_weight_metric_cache(
        ds_name=ds_name,
        root_dir=root_dir,
        dstype=dstype,
        dataset_split=dataset_split,
        epoch_lst=epoch_lst,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        skip_missing_cache=skip_missing_cache,
        preload_properties=True,
        validate_checkpoints=False,
        show_progress=True,
    )
    if requested_use_partitioned_cache:
        return True, cache_path
    if max_full_cache_size_mb <= 0.0 or cache_path is None or not cache_path.exists():
        return False, cache_path

    cache_size_mb = float(cache_path.stat().st_size) / float(1024 ** 2)
    if cache_size_mb <= max_full_cache_size_mb:
        return False, cache_path

    log.warning(
        "weight_early_signal: dataset '%s' cached weight table is %.1f MiB at %s; "
        "overriding use_partitioned_cache=false to avoid a high-memory full-table load.",
        ds_name,
        cache_size_mb,
        cache_path,
    )
    return True, cache_path


def _aggregate_functional_on_key(target_df: pd.DataFrame, *, key_col: str) -> pd.DataFrame:
    if target_df.empty or key_col not in target_df.columns:
        return pd.DataFrame()
    d = target_df.dropna(subset=[key_col]).copy()
    if d.empty:
        return pd.DataFrame()
    agg = (
        d.groupby(key_col, as_index=False, sort=False)
        .agg(
            functional_value_pred=("functional_value_pred", "mean"),
            future_delta=("future_delta", "mean"),
            late_bloomer=("late_bloomer", "max"),
            rank_pred=("rank_pred", "mean"),
            rank_final=("rank_final", "mean"),
        )
    )
    agg["late_bloomer"] = agg["late_bloomer"].astype(float) > 0.5
    return agg


def _aggregate_weight_on_key(weight_df: pd.DataFrame, *, key_col: str) -> pd.DataFrame:
    if weight_df.empty or key_col not in weight_df.columns:
        return pd.DataFrame()
    d = weight_df.dropna(subset=[key_col]).copy()
    if d.empty:
        return pd.DataFrame()
    return d.groupby(key_col, as_index=False, sort=False).agg(weight_value=("weight_value", "mean"))


def _align_weight_and_targets(weight_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    wt = _aggregate_weight_on_key(weight_df, key_col="trial_token")
    ft = _aggregate_functional_on_key(target_df, key_col="trial_token")
    aligned_trial = wt.merge(ft, on="trial_token", how="inner") if (not wt.empty and not ft.empty) else pd.DataFrame()

    wk = _aggregate_weight_on_key(weight_df, key_col="model_key")
    fk = _aggregate_functional_on_key(target_df, key_col="model_key")
    aligned_key = wk.merge(fk, on="model_key", how="inner") if (not wk.empty and not fk.empty) else pd.DataFrame()

    if len(aligned_trial) >= len(aligned_key):
        return aligned_trial, "trial_token"
    return aligned_key, "model_key"


def _normalize_metric_matrix_epochs(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.empty:
        return mat
    epoch_vals = pd.to_numeric(pd.Index(mat.columns), errors="coerce")
    keep = np.isfinite(epoch_vals)
    if not np.any(keep):
        return pd.DataFrame(index=mat.index)

    cols_keep = np.array(mat.columns)[keep]
    cols_int = np.round(epoch_vals[keep]).astype(int)

    out = mat.loc[:, cols_keep].copy()
    out.columns = cols_int
    out = out.T.groupby(level=0, sort=True).mean().T
    return out.sort_index(axis=1)


def _select_prediction_epochs(
    *,
    available_epochs: List[int],
    epoch_mid: int,
    config_prediction_epochs: Optional[Sequence[int]],
    prediction_stride: int,
    max_prediction_epochs: int,
) -> List[int]:
    if not available_epochs:
        return []
    available_set = set(available_epochs)

    if config_prediction_epochs:
        selected = sorted({int(e) for e in config_prediction_epochs if int(e) in available_set and int(e) <= int(epoch_mid)})
    else:
        selected = [int(e) for e in available_epochs if int(e) <= int(epoch_mid)]

    if not selected:
        selected = [int(available_epochs[0])]

    stride = max(1, int(prediction_stride))
    selected = selected[::stride]
    if max_prediction_epochs > 0 and len(selected) > max_prediction_epochs:
        idx = np.linspace(0, len(selected) - 1, num=max_prediction_epochs)
        selected = [selected[int(round(i))] for i in idx]
        selected = sorted(set(selected))
    return selected


def _build_functional_targets(
    dfd: pd.DataFrame,
    *,
    functional_metric: str,
    higher_is_better: bool,
    epoch_col: str,
    model_col: str,
    epoch_range: Optional[Tuple[int, int]],
    epoch_mid_cfg: Optional[int],
    bottom_frac_mid: float,
    top_frac_end: float,
    config_prediction_epochs: Optional[Sequence[int]],
    prediction_stride: int,
    max_prediction_epochs: int,
    drop_epoch_zero: bool,
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[str, Any]]:
    mat = build_metric_matrix_from_dfm(
        dfd,
        metric_key=functional_metric,
        epoch_col=epoch_col,
        model_col=model_col,
        epoch_range=epoch_range,
    )
    mat = _normalize_metric_matrix_epochs(mat)
    if drop_epoch_zero and (0 in mat.columns):
        mat = mat.drop(columns=[0]).copy()

    if mat.empty or mat.shape[0] < 2 or mat.shape[1] < 2:
        return pd.DataFrame(), {}, {"reason": "functional_matrix_too_small"}

    available_epochs = sorted([int(c) for c in mat.columns])
    if not available_epochs:
        return pd.DataFrame(), {}, {"reason": "no_epochs"}

    if epoch_mid_cfg is not None:
        epoch_mid = int(epoch_mid_cfg)
        if epoch_mid not in set(available_epochs):
            earlier = [e for e in available_epochs if e <= epoch_mid]
            epoch_mid = max(earlier) if earlier else available_epochs[len(available_epochs) // 2]
    else:
        epoch_mid = int(available_epochs[len(available_epochs) // 2])

    epoch_end = int(max(available_epochs))

    rmat = rank_per_epoch(mat, higher_is_better=higher_is_better, tie_method="average")
    n_models = int(mat.shape[0])
    bottom_start_rank = int(np.floor((1.0 - float(bottom_frac_mid)) * n_models)) + 1
    top_end_rank = int(np.ceil(float(top_frac_end) * n_models))

    late_bloomer = (rmat[epoch_mid] >= bottom_start_rank) & (rmat[epoch_end] <= top_end_rank)
    final_vals = mat[epoch_end]
    direction = 1.0 if bool(higher_is_better) else -1.0

    prediction_epochs = _select_prediction_epochs(
        available_epochs=available_epochs,
        epoch_mid=epoch_mid,
        config_prediction_epochs=config_prediction_epochs,
        prediction_stride=prediction_stride,
        max_prediction_epochs=max_prediction_epochs,
    )

    target_by_epoch: Dict[int, pd.DataFrame] = {}
    for e in prediction_epochs:
        if e not in mat.columns:
            continue
        pred_vals = mat[e]
        fut_delta = direction * (final_vals - pred_vals)
        tdf = pd.DataFrame(
            {
                "model_name_raw": mat.index.astype(str),
                "functional_value_pred": pd.to_numeric(pred_vals, errors="coerce"),
                "future_delta": pd.to_numeric(fut_delta, errors="coerce"),
                "late_bloomer": late_bloomer.astype(bool),
                "rank_pred": pd.to_numeric(rmat[e], errors="coerce"),
                "rank_final": pd.to_numeric(rmat[epoch_end], errors="coerce"),
            }
        )
        tdf["model_key"] = _canonical_model_key(tdf["model_name_raw"])
        tdf["trial_token"] = _trial_token(tdf["model_name_raw"])
        target_by_epoch[int(e)] = tdf

    meta = {
        "reason": "ok",
        "n_models": n_models,
        "epoch_mid": int(epoch_mid),
        "epoch_end": int(epoch_end),
        "prediction_epochs": prediction_epochs,
        "late_bloomers": int(np.sum(late_bloomer.to_numpy(dtype=bool))),
    }
    return mat, target_by_epoch, meta


def _evaluate_late_bloomer_signal(
    aligned: pd.DataFrame,
    *,
    top_quantile: float,
) -> Dict[str, Any]:
    if aligned.empty:
        return {
            "n_aligned_models": 0,
            "n_late_bloomers": 0,
        }

    x = pd.to_numeric(aligned["weight_value"], errors="coerce").to_numpy(dtype=float)
    yb = aligned["late_bloomer"].astype(bool).to_numpy(dtype=bool)
    y = yb.astype(float)
    z = pd.to_numeric(aligned["functional_value_pred"], errors="coerce").to_numpy(dtype=float)

    n = int(len(y))
    n_pos = int(np.sum(yb))
    n_neg = int(n - n_pos)

    out: Dict[str, Any] = {
        "n_aligned_models": n,
        "n_late_bloomers": n_pos,
        "late_bloomer_rate": float(np.mean(y)) if n > 0 else np.nan,
        "pearson_weight_vs_late_bloomer": _safe_corr(x, y, method="pearson"),
        "spearman_weight_vs_late_bloomer": _safe_corr(x, y, method="spearman"),
        "partial_pearson_weight_vs_late_bloomer_given_early_metric": _partial_corr_pearson(x, y, z),
        "mannwhitney_u_p_value": np.nan,
        "auc_raw": np.nan,
        "auc_best": np.nan,
        "auc_best_direction": np.nan,
        "top_n": 0,
        "base_rate": np.nan,
        "top_rate": np.nan,
        "top_rate_lift": np.nan,
    }

    if n_pos == 0 or n_neg == 0:
        return out

    try:
        x_pos = x[yb]
        x_neg = x[~yb]
        mask_pos = np.isfinite(x_pos)
        mask_neg = np.isfinite(x_neg)
        if np.any(mask_pos) and np.any(mask_neg):
            out["mannwhitney_u_p_value"] = float(
                mannwhitneyu(x_pos[mask_pos], x_neg[mask_neg], alternative="two-sided").pvalue
            )
    except Exception:
        out["mannwhitney_u_p_value"] = np.nan

    auc_raw = _binary_auc(y, x)
    out["auc_raw"] = auc_raw
    if np.isfinite(auc_raw):
        direction = 1.0 if auc_raw >= 0.5 else -1.0
        out["auc_best_direction"] = direction
        out["auc_best"] = float(max(auc_raw, 1.0 - auc_raw))
        lift = _binary_quantile_lift(x * direction, y, top_quantile)
        out.update(lift)

    return out


def _evaluate_future_delta_signal(
    aligned: pd.DataFrame,
    *,
    top_quantile: float,
) -> Dict[str, Any]:
    if aligned.empty:
        return {
            "n_aligned_models": 0,
        }

    x = pd.to_numeric(aligned["weight_value"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(aligned["future_delta"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(aligned["functional_value_pred"], errors="coerce").to_numpy(dtype=float)

    out: Dict[str, Any] = {
        "n_aligned_models": int(len(aligned)),
        "pearson_weight_vs_future_delta": _safe_corr(x, y, method="pearson"),
        "spearman_weight_vs_future_delta": _safe_corr(x, y, method="spearman"),
        "partial_pearson_weight_vs_future_delta_given_early_metric": _partial_corr_pearson(x, y, z),
        "top_n": 0,
        "base_mean": np.nan,
        "top_mean": np.nan,
        "top_minus_base": np.nan,
        "top_direction": np.nan,
    }

    plus = _continuous_quantile_gain(x, y, top_quantile)
    minus = _continuous_quantile_gain(-x, y, top_quantile)
    plus_gain = plus.get("top_minus_base", np.nan)
    minus_gain = minus.get("top_minus_base", np.nan)

    if np.isfinite(plus_gain) and (not np.isfinite(minus_gain) or plus_gain >= minus_gain):
        out["top_direction"] = 1.0
        out.update(plus)
    elif np.isfinite(minus_gain):
        out["top_direction"] = -1.0
        out.update(minus)

    return out


def _combo_label(metric_family: str, metric: str, scope: str) -> str:
    return f"{metric_family}/{metric}/{scope}"


def _plot_top_metric_traces(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    value_col: str,
    title: str,
    ylabel: str,
    top_k: int,
    dpi: int,
    use_abs: bool = False,
    zero_line: bool = False,
) -> Optional[plt.Figure]:
    d = df[df["dataset_name"] == dataset_name].copy()
    d = d[np.isfinite(pd.to_numeric(d[value_col], errors="coerce"))].copy()
    if d.empty:
        return None

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    metric_score = d[value_col].abs() if use_abs else d[value_col]
    d["_metric_score"] = metric_score

    top_metrics = (
        d.groupby("combo_label", as_index=False)["_metric_score"]
        .max()
        .sort_values("_metric_score", ascending=False)
        .head(max(1, int(top_k)))
    )
    keep = set(top_metrics["combo_label"].tolist())
    p = d[d["combo_label"].isin(keep)].copy()
    if p.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    for combo, sub in p.groupby("combo_label", sort=False):
        sub = sub.sort_values("prediction_epoch", kind="stable")
        ax.plot(
            sub["prediction_epoch"].to_numpy(dtype=float),
            sub[value_col].to_numpy(dtype=float),
            marker="o",
            label=combo,
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _select_top_combo_labels(
    df: pd.DataFrame,
    *,
    score_col: str,
    top_k: int,
    use_abs: bool = False,
) -> List[str]:
    if df.empty or score_col not in df.columns:
        return []

    d = df.copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d[np.isfinite(d[score_col])].copy()
    if d.empty:
        return []

    d["_score"] = d[score_col].abs() if use_abs else d[score_col]
    top_metrics = (
        d.groupby("combo_label", as_index=False)["_score"]
        .max()
        .sort_values("_score", ascending=False)
        .head(max(1, int(top_k)))
    )
    return top_metrics["combo_label"].astype(str).tolist()


def _plot_metric_epoch_heatmap(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    value_col: str,
    title: str,
    top_k: int,
    dpi: int,
    use_abs: bool = False,
    center_zero: bool = False,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    d = df[df["dataset_name"] == dataset_name].copy()
    if d.empty or value_col not in d.columns:
        log.info(
            "Skipping heatmap for dataset '%s' and value '%s': rows=%d column_present=%s.",
            dataset_name,
            value_col,
            len(d),
            value_col in d.columns,
        )
        return None

    keep = _select_top_combo_labels(d, score_col=value_col, top_k=top_k, use_abs=use_abs)
    if not keep:
        finite_count = int(np.isfinite(pd.to_numeric(d[value_col], errors="coerce")).sum())
        log.info(
            "Skipping heatmap for dataset '%s' and value '%s': no top combos with finite scores "
            "(rows=%d finite_scores=%d top_k=%d use_abs=%s).",
            dataset_name,
            value_col,
            len(d),
            finite_count,
            int(top_k),
            bool(use_abs),
        )
        return None

    d = d[d["combo_label"].isin(keep)].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["prediction_epoch"] = pd.to_numeric(d["prediction_epoch"], errors="coerce")
    d = d[np.isfinite(d[value_col]) & np.isfinite(d["prediction_epoch"])].copy()
    if d.empty:
        log.info(
            "Skipping heatmap for dataset '%s' and value '%s': selected combos have no finite value/epoch pairs.",
            dataset_name,
            value_col,
        )
        return None

    p = d.pivot_table(
        index="combo_label",
        columns="prediction_epoch",
        values=value_col,
        aggfunc="mean",
        observed=True,
    )
    p = p.reindex(keep)
    p = p.sort_index(axis=1)
    if p.empty:
        log.info(
            "Skipping heatmap for dataset '%s' and value '%s': pivot table is empty after aggregation.",
            dataset_name,
            value_col,
        )
        return None

    mat = p.to_numpy(dtype=float)
    fig_h = max(4.0, 0.45 * len(p.index) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=dpi)
    imshow_kwargs: Dict[str, Any] = {"aspect": "auto", "origin": "lower", "cmap": cmap}
    if vmin is not None:
        imshow_kwargs["vmin"] = float(vmin)
    if vmax is not None:
        imshow_kwargs["vmax"] = float(vmax)
    if center_zero and "vmin" not in imshow_kwargs and "vmax" not in imshow_kwargs:
        max_abs = np.nanmax(np.abs(mat)) if mat.size else np.nan
        if np.isfinite(max_abs) and max_abs > 0.0:
            imshow_kwargs["vmin"] = -float(max_abs)
            imshow_kwargs["vmax"] = float(max_abs)
    im = ax.imshow(mat, **imshow_kwargs)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight metric")
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_xticklabels([str(int(x)) for x in p.columns.to_numpy(dtype=float)])
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels(p.index.tolist(), fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def _plot_auc_detail(
    aligned: pd.DataFrame,
    *,
    dataset_name: str,
    functional_metric: str,
    combo_label: str,
    prediction_epoch: int,
    epoch_mid: int,
    epoch_end: int,
    auc_raw: float,
    auc_best: float,
    auc_best_direction: float,
    dpi: int,
) -> Optional[plt.Figure]:
    if aligned.empty:
        return None

    x = pd.to_numeric(aligned["weight_value"], errors="coerce").to_numpy(dtype=float)
    y = aligned["late_bloomer"].astype(bool).to_numpy(dtype=bool).astype(float)
    parts = _binary_auc_components(y, x)
    scores = np.asarray(parts.get("scores", np.array([], dtype=float)), dtype=float)
    labels = np.asarray(parts.get("labels", np.array([], dtype=int)), dtype=int)
    ranks = np.asarray(parts.get("ranks", np.array([], dtype=float)), dtype=float)
    n_pos = int(parts.get("n_pos", 0))
    n_neg = int(parts.get("n_neg", 0))
    if len(scores) == 0 or n_pos == 0 or n_neg == 0 or len(ranks) != len(scores):
        return None

    rank_sum_pos = float(parts.get("rank_sum_pos", np.nan))
    u_stat = float(parts.get("u_stat", np.nan))
    direction_label = "higher weight => late bloomer" if float(auc_best_direction) >= 0.0 else "lower weight => late bloomer"

    fpr_raw, tpr_raw = _binary_roc_curve(labels, scores)
    fpr_best, tpr_best = _binary_roc_curve(labels, scores * float(auc_best_direction))

    fig, (ax_rank, ax_roc) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    neg_mask = labels == 0
    pos_mask = labels == 1
    ax_rank.scatter(
        ranks[neg_mask],
        scores[neg_mask],
        c="#4C78A8",
        alpha=0.80,
        s=36,
        label=f"Non-late bloomer (n={n_neg})",
    )
    ax_rank.scatter(
        ranks[pos_mask],
        scores[pos_mask],
        c="#E45756",
        alpha=0.90,
        s=42,
        label=f"Late bloomer (n={n_pos})",
    )
    ax_rank.set_title("Rank view used by AUC")
    ax_rank.set_xlabel("Rank of weight score (ties = average)")
    ax_rank.set_ylabel("Weight statistic value")
    ax_rank.grid(True, alpha=0.25)
    ax_rank.legend(fontsize=8, loc="best")
    ax_rank.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"R+ = {rank_sum_pos:.1f}",
                f"U = {u_stat:.1f}",
                f"AUC_raw = {auc_raw:.3f}",
                f"AUC_best = {auc_best:.3f}",
                direction_label,
            ]
        ),
        transform=ax_rank.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    if len(fpr_raw) > 0:
        ax_roc.plot(fpr_raw, tpr_raw, color="#4C78A8", lw=2, label=f"Raw score (AUC={auc_raw:.3f})")
    if len(fpr_best) > 0:
        ax_roc.plot(
            fpr_best,
            tpr_best,
            color="#54A24B",
            lw=2,
            linestyle="--",
            label=f"Best direction (AUC={auc_best:.3f})",
        )
    ax_roc.plot([0.0, 1.0], [0.0, 1.0], color="0.5", lw=1.25, linestyle=":")
    ax_roc.set_title("ROC curve")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.grid(True, alpha=0.25)
    ax_roc.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"{zoo_plot_name(dataset_name)} | functional={functional_metric} | {combo_label} | prediction_epoch={int(prediction_epoch)} | "
        f"late bloomers from epoch_mid={int(epoch_mid)} to epoch_end={int(epoch_end)}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _plot_mannwhitney_detail(
    aligned: pd.DataFrame,
    *,
    dataset_name: str,
    functional_metric: str,
    combo_label: str,
    prediction_epoch: int,
    epoch_mid: int,
    epoch_end: int,
    mannwhitney_p_value: float,
    auc_raw: float,
    dpi: int,
) -> Optional[plt.Figure]:
    if aligned.empty:
        return None

    scores = pd.to_numeric(aligned["weight_value"], errors="coerce").to_numpy(dtype=float)
    labels = aligned["late_bloomer"].astype(bool).to_numpy(dtype=bool)
    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) < 2:
        return None

    bloom_scores = scores[labels]
    non_scores = scores[~labels]
    if len(bloom_scores) == 0 or len(non_scores) == 0:
        return None

    parts = _binary_auc_components(labels.astype(float), scores)
    u_stat = float(parts.get("u_stat", np.nan))

    def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals = np.sort(np.asarray(values, dtype=float))
        n = len(vals)
        if n == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        y = np.arange(1, n + 1, dtype=float) / float(n)
        return vals, y

    bloom_x, bloom_y = _ecdf(bloom_scores)
    non_x, non_y = _ecdf(non_scores)

    fig, (ax_dist, ax_ecdf) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    positions = [0, 1]
    bp = ax_dist.boxplot(
        [non_scores, bloom_scores],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    colors = ["#4C78A8", "#E45756"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(0)
    jitter_non = rng.normal(loc=positions[0], scale=0.04, size=len(non_scores))
    jitter_bloom = rng.normal(loc=positions[1], scale=0.04, size=len(bloom_scores))
    ax_dist.scatter(jitter_non, non_scores, c=colors[0], alpha=0.7, s=26)
    ax_dist.scatter(jitter_bloom, bloom_scores, c=colors[1], alpha=0.8, s=28)
    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(
        [
            f"Non-late bloomer\n(n={len(non_scores)})",
            f"Late bloomer\n(n={len(bloom_scores)})",
        ]
    )
    ax_dist.set_ylabel("Weight statistic value")
    ax_dist.set_title("Group distributions")
    ax_dist.grid(True, axis="y", alpha=0.25)
    ax_dist.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"MW p = {mannwhitney_p_value:.3g}" if np.isfinite(mannwhitney_p_value) else "MW p = nan",
                f"U = {u_stat:.1f}" if np.isfinite(u_stat) else "U = nan",
                f"AUC_raw = {auc_raw:.3f}" if np.isfinite(auc_raw) else "AUC_raw = nan",
                f"median non = {np.median(non_scores):.3g}",
                f"median bloom = {np.median(bloom_scores):.3g}",
            ]
        ),
        transform=ax_dist.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    ax_ecdf.step(non_x, non_y, where="post", color=colors[0], lw=2, label="Non-late bloomer")
    ax_ecdf.step(bloom_x, bloom_y, where="post", color=colors[1], lw=2, label="Late bloomer")
    ax_ecdf.set_title("Empirical CDF")
    ax_ecdf.set_xlabel("Weight statistic value")
    ax_ecdf.set_ylabel("Cumulative fraction")
    ax_ecdf.set_ylim(0.0, 1.0)
    ax_ecdf.grid(True, alpha=0.25)
    ax_ecdf.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{zoo_plot_name(dataset_name)} | functional={functional_metric} | {combo_label} | prediction_epoch={int(prediction_epoch)} | "
        f"late bloomers from epoch_mid={int(epoch_mid)} to epoch_end={int(epoch_end)}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def run_weight_early_signal(
    df: pd.DataFrame,
    config: Dict[str, Any],
    registry: Dict[str, Any],
) -> AnalysisResult:
    if "dataset_name" not in df.columns:
        raise ValueError("weight_early_signal: df must contain dataset_name column.")

    functional_metric = str(config.get("functional_metric", config.get("rank_metric", "test_acc")))
    functional_metric_key = _safe_name_component(functional_metric)
    functional_metric_title = str(functional_metric)
    functional_higher_is_better = bool(config.get("higher_is_better", True))
    epoch_col = str(config.get("epoch_col", "training_iteration"))
    model_col = str(config.get("model_col", "model_name"))
    epoch_range = tuple(config.get("epoch_range", [])) or None
    drop_epoch_zero = bool(config.get("drop_epoch_zero", True))

    epoch_mid_cfg = config.get("epoch_mid")
    epoch_mid_cfg = int(epoch_mid_cfg) if epoch_mid_cfg is not None else None
    bottom_frac_mid = float(config.get("bottom_frac_mid", 0.30))
    top_frac_end = float(config.get("top_frac_end", 0.20))
    prediction_epochs_cfg = config.get("prediction_epochs", None)
    prediction_stride = int(config.get("prediction_stride", 1))
    max_prediction_epochs = int(config.get("max_prediction_epochs", 12))

    cache_dir = Path(config.get("cache_dir", "learning-dynamics/outputs/cache/weight_metrics"))
    dataset_split = str(config.get("dataset_split", "full"))
    force_recompute = bool(config.get("force_recompute", False))
    skip_missing_cache = bool(config.get("skip_missing_cache", True))
    use_partitioned_cache = bool(config.get("use_partitioned_cache", True))
    partition_batch_size = int(config.get("partition_batch_size", 250_000))
    max_full_cache_size_mb = float(config.get("max_full_cache_size_mb", 256.0))

    metric_families = _as_list_or_all(config.get("metric_family", "all"))
    metrics = _as_list_or_all(config.get("metric", "all"))
    scopes = _as_list_or_all(config.get("scope", "all"))
    include_model_row = bool(config.get("include_model_row", True))

    top_quantile = float(config.get("top_quantile", 0.20))
    top_quantile = float(np.clip(top_quantile, 1e-6, 1.0))

    plot_cfg = config.get("plot", {}) or {}
    plot_enabled = bool(plot_cfg.get("enabled", False))
    plot_top_k_metrics = int(plot_cfg.get("top_k_metrics", 10))
    plot_dpi = int(plot_cfg.get("dpi", config.get("png_dpi", 200)))
    plot_include_traces = bool(plot_cfg.get("include_traces", True))
    plot_include_heatmaps = bool(plot_cfg.get("include_heatmaps", True))
    plot_include_auc_detail = bool(plot_cfg.get("include_auc_detail", False))
    plot_auc_detail_top_k = max(1, int(plot_cfg.get("auc_detail_top_k", 1)))
    plot_include_mannwhitney_detail = bool(plot_cfg.get("include_mannwhitney_detail", False))
    plot_mannwhitney_detail_top_k = max(1, int(plot_cfg.get("mannwhitney_detail_top_k", 1)))

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())
    late_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    auc_detail_candidates: Dict[str, List[Dict[str, Any]]] = {}
    mannwhitney_candidates: Dict[str, List[Dict[str, Any]]] = {}

    for ds in datasets:
        dfd = df[df["dataset_name"] == ds].copy()

        functional_mat, targets_by_epoch, target_meta = _build_functional_targets(
            dfd,
            functional_metric=functional_metric,
            higher_is_better=functional_higher_is_better,
            epoch_col=epoch_col,
            model_col=model_col,
            epoch_range=epoch_range,
            epoch_mid_cfg=epoch_mid_cfg,
            bottom_frac_mid=bottom_frac_mid,
            top_frac_end=top_frac_end,
            config_prediction_epochs=prediction_epochs_cfg,
            prediction_stride=prediction_stride,
            max_prediction_epochs=max_prediction_epochs,
            drop_epoch_zero=drop_epoch_zero,
        )

        summary_row = {
            "dataset_name": ds,
            "functional_metric": functional_metric,
            "higher_is_better": functional_higher_is_better,
            "reason": str(target_meta.get("reason", "unknown")),
            "n_models": int(target_meta.get("n_models", 0)),
            "epoch_mid": target_meta.get("epoch_mid", np.nan),
            "epoch_end": target_meta.get("epoch_end", np.nan),
            "n_prediction_epochs": len(target_meta.get("prediction_epochs", [])),
            "prediction_epochs": ",".join(str(x) for x in target_meta.get("prediction_epochs", [])),
            "late_bloomers": int(target_meta.get("late_bloomers", 0)),
            "bottom_frac_mid": bottom_frac_mid,
            "top_frac_end": top_frac_end,
        }
        target_rows.append(summary_row)

        if functional_mat.empty or not targets_by_epoch:
            log.warning("weight_early_signal: skipping dataset '%s' (%s).", ds, summary_row["reason"])
            continue

        spec = registry.get(ds)
        if spec is None:
            log.warning("weight_early_signal: dataset '%s' is missing in registry. Skipping.", ds)
            continue

        epoch_lst = sorted([int(c) for c in functional_mat.columns])
        dataset_use_partitioned_cache, weight_cache_path = _resolve_dataset_cache_mode(
            ds_name=ds,
            root_dir=Path(spec.root_dir),
            dstype=spec.dstype,
            dataset_split=dataset_split,
            epoch_lst=epoch_lst,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            skip_missing_cache=skip_missing_cache,
            requested_use_partitioned_cache=use_partitioned_cache,
            max_full_cache_size_mb=max_full_cache_size_mb,
        )
        parquet_columns = [
            "metric_family",
            "metric",
            "scope",
            "layer_name",
            "model_id",
            "epoch",
            "value",
        ]
        parquet_filters: List[Tuple[str, str, Any]] = []
        if metric_families:
            parquet_filters.append(("metric_family", "in", metric_families))
        if metrics:
            parquet_filters.append(("metric", "in", metrics))
        if scopes:
            parquet_filters.append(("scope", "in", scopes))

        combo_sources: List[Tuple[Tuple[str, str, str], Dict[str, Any]]] = []
        if dataset_use_partitioned_cache:
            manifest = load_or_build_partitioned_weight_metric_manifest(
                ds_name=ds,
                dataset_split=dataset_split,
                cache_dir=cache_dir,
                source_path=weight_cache_path,
                rebuild=force_recompute,
                batch_size=partition_batch_size,
            )

            if manifest is None:
                weight_cache_path = ensure_weight_metric_cache(
                    ds_name=ds,
                    root_dir=Path(spec.root_dir),
                    dstype=spec.dstype,
                    dataset_split=dataset_split,
                    epoch_lst=epoch_lst,
                    cache_dir=cache_dir,
                    force_recompute=force_recompute,
                    skip_missing_cache=skip_missing_cache,
                    preload_properties=True,
                    validate_checkpoints=False,
                    show_progress=True,
                )
                manifest = load_or_build_partitioned_weight_metric_manifest(
                    ds_name=ds,
                    dataset_split=dataset_split,
                    cache_dir=cache_dir,
                    source_path=weight_cache_path,
                    rebuild=force_recompute,
                    batch_size=partition_batch_size,
                )

            combo_sources = _build_partition_combo_sources(
                manifest if manifest is not None else pd.DataFrame(),
                metric_families=metric_families,
                metrics=metrics,
                scopes=scopes,
                include_model_row=include_model_row,
            )
            if combo_sources:
                log.info(
                    "weight_early_signal: dataset '%s' will process %d metric combinations from partitioned cache.",
                    ds,
                    len(combo_sources),
                )

        if combo_sources:
            combo_iter = combo_sources
        else:
            dfw = load_or_compute_df_long(
                ds_name=ds,
                root_dir=Path(spec.root_dir),
                dstype=spec.dstype,
                dataset_split=dataset_split,
                epoch_lst=epoch_lst,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
                skip_missing_cache=skip_missing_cache,
                preload_properties=True,
                validate_checkpoints=False,
                show_progress=True,
                parquet_columns=parquet_columns,
                parquet_filters=parquet_filters or None,
            )
            if dfw is None or dfw.empty:
                log.warning("weight_early_signal: no weight-stat table for dataset '%s'.", ds)
                continue

            dfw = _prepare_weight_metrics_frame(dfw, drop_epoch_zero=drop_epoch_zero)
            if dfw.empty:
                continue

            dfw_sel = dfw
            if metric_families:
                dfw_sel = dfw_sel[dfw_sel["metric_family"].isin(metric_families)]
            if metrics:
                dfw_sel = dfw_sel[dfw_sel["metric"].isin(metrics)]
            if scopes:
                dfw_sel = dfw_sel[dfw_sel["scope"].isin(scopes)]
            if dfw_sel.empty:
                log.warning("weight_early_signal: no matching weight metrics for dataset '%s'.", ds)
                continue

            combo_iter = (
                (tuple(key), {"dataframe": group_df, "source_kind": "full_cache"})
                for key, group_df in dfw_sel.groupby(["metric_family", "metric", "scope"], sort=True)
            )

        for (metric_family, metric, scope), combo_source in combo_iter:
            if "path" in combo_source:
                gdf = pd.read_parquet(
                    combo_source["path"],
                    columns=["layer_name", "model_id", "epoch", "value"],
                )
            else:
                gdf = combo_source["dataframe"]

            gdf = _prepare_weight_metrics_frame(gdf, drop_epoch_zero=drop_epoch_zero)
            if gdf.empty:
                continue

            weight_series, value_source = _prepare_weight_series_from_group(
                gdf,
                include_model_row=include_model_row,
            )
            if weight_series.empty:
                continue

            combo = _combo_label(str(metric_family), str(metric), str(scope))
            for pred_epoch, target_df in targets_by_epoch.items():
                w_epoch = weight_series[weight_series["epoch"] == int(pred_epoch)].copy()
                if w_epoch.empty:
                    continue

                aligned, align_method = _align_weight_and_targets(w_epoch, target_df)
                if aligned.empty:
                    continue

                meta = {
                    "dataset_name": ds,
                    "functional_metric": functional_metric,
                    "prediction_epoch": int(pred_epoch),
                    "epoch_mid": int(target_meta["epoch_mid"]),
                    "epoch_end": int(target_meta["epoch_end"]),
                    "metric_family": str(metric_family),
                    "metric": str(metric),
                    "scope": str(scope),
                    "combo_label": combo,
                    "weight_value_source": value_source,
                    "alignment_key": align_method,
                }

                late_eval = _evaluate_late_bloomer_signal(aligned, top_quantile=top_quantile)
                delta_eval = _evaluate_future_delta_signal(aligned, top_quantile=top_quantile)

                late_rows.append({**meta, **late_eval})
                delta_rows.append({**meta, **delta_eval})

                auc_best_val = pd.to_numeric(late_eval.get("auc_best"), errors="coerce")
                if plot_include_auc_detail and np.isfinite(auc_best_val):
                    auc_detail_candidates.setdefault(ds, []).append(
                        {
                            **meta,
                            "auc_raw": float(late_eval.get("auc_raw", np.nan)),
                            "auc_best": float(auc_best_val),
                            "auc_best_direction": float(late_eval.get("auc_best_direction", np.nan)),
                            "aligned": aligned.loc[:, ["weight_value", "late_bloomer"]].copy(),
                        }
                    )
                mw_p_val = pd.to_numeric(late_eval.get("mannwhitney_u_p_value"), errors="coerce")
                if plot_include_mannwhitney_detail and np.isfinite(mw_p_val):
                    mannwhitney_candidates.setdefault(ds, []).append(
                        {
                            **meta,
                            "mannwhitney_u_p_value": float(mw_p_val),
                            "auc_raw": float(late_eval.get("auc_raw", np.nan)),
                            "auc_best": float(late_eval.get("auc_best", np.nan)),
                            "aligned": aligned.loc[:, ["weight_value", "late_bloomer"]].copy(),
                        }
                    )

    late_df = pd.DataFrame(late_rows)
    if not late_df.empty:
        late_df = late_df.sort_values(
            ["dataset_name", "prediction_epoch", "auc_best"],
            ascending=[True, True, False],
            kind="stable",
        ).reset_index(drop=True)

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(
            ["dataset_name", "prediction_epoch", "spearman_weight_vs_future_delta"],
            ascending=[True, True, False],
            kind="stable",
        ).reset_index(drop=True)

    target_df = pd.DataFrame(target_rows).sort_values("dataset_name", kind="stable").reset_index(drop=True)

    figures: List[Dict[str, plt.Figure]] = []
    if plot_enabled and plot_include_traces and not late_df.empty:
        for ds in sorted(late_df["dataset_name"].unique().tolist()):
            fig = _plot_top_metric_traces(
                late_df,
                dataset_name=ds,
                value_col="auc_best",
                title=(
                    f"{zoo_plot_name(ds)} | functional={functional_metric_title} | "
                    "Early weight metrics vs late bloomer (AUC best direction)"
                ),
                ylabel="AUC (best direction)",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=False,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_late_bloomer_auc": fig})

            fig = _plot_top_metric_traces(
                late_df,
                dataset_name=ds,
                value_col="pearson_weight_vs_late_bloomer",
                title=(
                    f"{zoo_plot_name(ds)} | functional={functional_metric_title} | "
                    "Early weight metrics vs late bloomer (Pearson)"
                ),
                ylabel="Pearson(weight, late_bloomer)",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                zero_line=True,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_late_bloomer_pearson": fig})

            fig = _plot_top_metric_traces(
                late_df,
                dataset_name=ds,
                value_col="spearman_weight_vs_late_bloomer",
                title=(
                    f"{zoo_plot_name(ds)} | functional={functional_metric_title} | "
                    "Early weight metrics vs late bloomer (Spearman)"
                ),
                ylabel="Spearman(weight, late_bloomer)",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                zero_line=True,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_late_bloomer_spearman": fig})

    if plot_enabled and plot_include_traces and not delta_df.empty:
        for ds in sorted(delta_df["dataset_name"].unique().tolist()):
            fig = _plot_top_metric_traces(
                delta_df,
                dataset_name=ds,
                value_col="pearson_weight_vs_future_delta",
                title=(
                    f"{zoo_plot_name(ds)} | functional={functional_metric_title} | "
                    "Early weight metrics vs future delta (Pearson)"
                ),
                ylabel="Pearson(weight, future_delta)",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                zero_line=True,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_future_delta_pearson": fig})

            fig = _plot_top_metric_traces(
                delta_df,
                dataset_name=ds,
                value_col="spearman_weight_vs_future_delta",
                title=(
                    f"{zoo_plot_name(ds)} | functional={functional_metric_title} | "
                    "Early weight metrics vs future delta (Spearman)"
                ),
                ylabel="Spearman(weight, future_delta)",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                zero_line=True,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_future_delta_spearman": fig})

    if plot_enabled and plot_include_heatmaps and not late_df.empty:
        for ds in sorted(late_df["dataset_name"].unique().tolist()):
            late_ds = late_df[late_df["dataset_name"] == ds]
            finite_auc_best = int(np.isfinite(pd.to_numeric(late_ds["auc_best"], errors="coerce")).sum())
            if finite_auc_best == 0:
                log.info(
                    "Dataset '%s' has no finite late-bloomer AUC values, so the late-bloomer heatmap will be skipped "
                    "(rows=%d combos=%d prediction_epochs=%d). This usually means each aligned sample set had only one "
                    "class or too few aligned models.",
                    ds,
                    len(late_ds),
                    int(late_ds["combo_label"].nunique()),
                    int(pd.to_numeric(late_ds["prediction_epoch"], errors="coerce").nunique()),
                )
            fig = _plot_metric_epoch_heatmap(
                late_df,
                dataset_name=ds,
                value_col="auc_best",
                title=f"{zoo_plot_name(ds)} | functional={functional_metric_title} | Late bloomer signal heatmap",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=False,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_late_bloomer_heatmap": fig})

            fig = _plot_metric_epoch_heatmap(
                late_df,
                dataset_name=ds,
                value_col="pearson_weight_vs_late_bloomer",
                title=f"{zoo_plot_name(ds)} | functional={functional_metric_title} | Late bloomer Pearson correlation heatmap",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                center_zero=True,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_late_bloomer_pearson_heatmap": fig})

            fig = _plot_metric_epoch_heatmap(
                late_df,
                dataset_name=ds,
                value_col="spearman_weight_vs_late_bloomer",
                title=f"{zoo_plot_name(ds)} | functional={functional_metric_title} | Late bloomer Spearman correlation heatmap",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                center_zero=True,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_late_bloomer_spearman_heatmap": fig})

    if plot_enabled and plot_include_heatmaps and not delta_df.empty:
        for ds in sorted(delta_df["dataset_name"].unique().tolist()):
            fig = _plot_metric_epoch_heatmap(
                delta_df,
                dataset_name=ds,
                value_col="pearson_weight_vs_future_delta",
                title=   f"Correlation of weight statistic vs. future improvement | {functional_metric_title} | Top {plot_top_k_metrics}", # f"{ds} | functional={functional_metric_title} | Future delta Pearson correlation heatmap",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                center_zero=True,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_future_delta_pearson_heatmap": fig})

            fig = _plot_metric_epoch_heatmap(
                delta_df,
                dataset_name=ds,
                value_col="spearman_weight_vs_future_delta",
                title= f"Correlation of weight statistic vs. future improvement | {functional_metric_title} | Top {plot_top_k_metrics}", # f"{ds} | functional={functional_metric_title} | Future delta Spearman correlation heatmap",
                top_k=plot_top_k_metrics,
                dpi=plot_dpi,
                use_abs=True,
                center_zero=True,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            if fig is not None:
                figures.append({f"{ds}_weight_early_signal_{functional_metric_key}_future_delta_heatmap": fig})

    if plot_enabled and plot_include_auc_detail:
        for ds, candidates in sorted(auc_detail_candidates.items()):
            if not candidates:
                continue
            top_candidates = sorted(
                candidates,
                key=lambda item: (
                    pd.to_numeric(item.get("auc_best"), errors="coerce"),
                    -int(item.get("prediction_epoch", 0)),
                ),
                reverse=True,
            )[:plot_auc_detail_top_k]
            for idx, candidate in enumerate(top_candidates, start=1):
                fig = _plot_auc_detail(
                    candidate["aligned"],
                    dataset_name=ds,
                    functional_metric=functional_metric_title,
                    combo_label=str(candidate["combo_label"]),
                    prediction_epoch=int(candidate["prediction_epoch"]),
                    epoch_mid=int(candidate["epoch_mid"]),
                    epoch_end=int(candidate["epoch_end"]),
                    auc_raw=float(candidate["auc_raw"]),
                    auc_best=float(candidate["auc_best"]),
                    auc_best_direction=float(candidate["auc_best_direction"]),
                    dpi=plot_dpi,
                )
                if fig is not None:
                    key = f"{ds}_weight_early_signal_{functional_metric_key}_auc_detail_{idx}"
                    figures.append({key: fig})

    if plot_enabled and plot_include_mannwhitney_detail:
        for ds, candidates in sorted(mannwhitney_candidates.items()):
            if not candidates:
                continue
            top_candidates = sorted(
                candidates,
                key=lambda item: (
                    np.inf if not np.isfinite(pd.to_numeric(item.get("mannwhitney_u_p_value"), errors="coerce")) else float(pd.to_numeric(item.get("mannwhitney_u_p_value"), errors="coerce")),
                    -np.inf if not np.isfinite(pd.to_numeric(item.get("auc_best"), errors="coerce")) else -float(pd.to_numeric(item.get("auc_best"), errors="coerce")),
                ),
            )[:plot_mannwhitney_detail_top_k]
            for idx, candidate in enumerate(top_candidates, start=1):
                fig = _plot_mannwhitney_detail(
                    candidate["aligned"],
                    dataset_name=ds,
                    functional_metric=functional_metric_title,
                    combo_label=str(candidate["combo_label"]),
                    prediction_epoch=int(candidate["prediction_epoch"]),
                    epoch_mid=int(candidate["epoch_mid"]),
                    epoch_end=int(candidate["epoch_end"]),
                    mannwhitney_p_value=float(candidate["mannwhitney_u_p_value"]),
                    auc_raw=float(candidate["auc_raw"]),
                    dpi=plot_dpi,
                )
                if fig is not None:
                    key = f"{ds}_weight_early_signal_{functional_metric_key}_mannwhitney_detail_{idx}"
                    figures.append({key: fig})

    tables = {
        f"weight_early_signal_targets__{functional_metric_key}": target_df,
        f"weight_early_signal_late_bloomer_summary__{functional_metric_key}": late_df,
        f"weight_early_signal_delta_summary__{functional_metric_key}": delta_df,
    }

    return AnalysisResult(tables=tables, figures=figures)
