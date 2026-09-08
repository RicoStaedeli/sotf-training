from __future__ import annotations

import logging
from pathlib import Path
import re
from time import perf_counter
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, kendalltau, pearsonr, spearmanr, ttest_rel, wilcoxon
from tqdm import tqdm

from .base import AnalysisResult
from .result_store import make_result_stores
from .ranking_flips import run_ranking_flips
from .weight_statistics import load_or_build_partitioned_weight_metric_manifest, load_or_compute_df_long
from ..zoo_info import zoo_plot_name
from ..compute.matrices import build_metric_matrix_from_dfm
from ..compute.ranks import count_pairwise_flips, rank_change_table, rank_per_epoch

log = logging.getLogger(__name__)

# Smaller blocks reduce peak memory; larger blocks reduce Python overhead.
_PAIRWISE_BLOCK_SIZE = 2048


def _log_df_size(label: str, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        log.debug("%s: <none>", label)
        return

    rows, cols = df.shape
    mem_mib = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    log.debug("%s: rows=%d cols=%d memory=%.2f MiB", label, rows, cols, mem_mib)


def _log_matrix_size(label: str, mat: pd.DataFrame) -> None:
    rows, cols = mat.shape
    mem_mib = float(mat.memory_usage(deep=True).sum()) / (1024 * 1024)
    log.debug("%s: rows=%d cols=%d memory=%.2f MiB", label, rows, cols, mem_mib)


def _build_weight_cache_filters(
    *,
    metric_families: Optional[List[str]],
    metrics: Optional[List[str]],
    scopes: Optional[List[str]],
    model_rows_only: bool,
) -> List[tuple[str, str, Any]]:
    filters: List[tuple[str, str, Any]] = []
    if model_rows_only:
        filters.append(("layer_name", "==", "__model__"))
    if metric_families:
        filters.append(("metric_family", "in", metric_families))
    if metrics:
        filters.append(("metric", "in", metrics))
    if scopes:
        filters.append(("scope", "in", scopes))
    return filters


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


def _to_int_epoch(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _infer_available_epochs(
    df: pd.DataFrame,
    *,
    epoch_col: str,
    epoch_range: Optional[Tuple[int, int]] = None,
) -> List[int]:
    if epoch_col not in df.columns:
        return []

    epochs = _to_int_epoch(df[epoch_col]).dropna().astype(int)
    if epoch_range is not None:
        lo, hi = epoch_range
        epochs = epochs[(epochs >= int(lo)) & (epochs <= int(hi))]

    return sorted(epochs.unique().tolist())


def _is_effectively_constant(x: np.ndarray, atol: float = 1e-12) -> bool:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return True
    return bool(np.allclose(vals, vals[0], rtol=0.0, atol=atol))


def _safe_key_part(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


def _combo_id(metric_family: str, metric: str, scope: str) -> str:
    return f"{_safe_key_part(metric_family)}__{_safe_key_part(metric)}__{_safe_key_part(scope)}"


def _combo_label(metric_family: str, metric: str, scope: str) -> str:
    return f"{metric_family}/{metric}/{scope}"


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


def _prepare_rank_long(
    dfd: pd.DataFrame,
    *,
    rank_metric: str,
    higher_is_better: bool,
    epoch_col: str,
    model_col: str,
    epoch_range: Optional[Tuple[int, int]],
    drop_epoch_zero: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mat = build_metric_matrix_from_dfm(
        dfd,
        metric_key=rank_metric,
        epoch_col=epoch_col,
        model_col=model_col,
        epoch_range=epoch_range,
    )
    if mat.empty or mat.shape[0] < 2 or mat.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    rmat = rank_per_epoch(mat, higher_is_better=higher_is_better, tie_method="average")
    rank_long = (
        rmat.stack()
        .rename("rank")
        .reset_index()
        .rename(columns={model_col: "model_name_raw", epoch_col: "epoch"})
    )
    rank_long["epoch"] = _to_int_epoch(rank_long["epoch"])
    rank_long["model_key"] = _canonical_model_key(rank_long["model_name_raw"])
    rank_long["trial_token"] = _trial_token(rank_long["model_name_raw"])
    rank_long = rank_long.dropna(subset=["epoch"]).copy()
    rank_long["epoch"] = rank_long["epoch"].astype(int)

    if drop_epoch_zero:
        rank_long = rank_long[rank_long["epoch"] != 0].copy()

    return mat, rank_long


def _preprocess_weight_metrics_frame(
    dfw: pd.DataFrame,
    *,
    drop_epoch_zero: bool,
) -> pd.DataFrame:
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


def _discover_metric_groups(
    dfw: pd.DataFrame,
    *,
    metric_families: Optional[List[str]],
    metrics: Optional[List[str]],
    scopes: Optional[List[str]],
) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    d = dfw
    if metric_families:
        d = d[d["metric_family"].isin(metric_families)]
    if metrics:
        d = d[d["metric"].isin(metrics)]
    if scopes:
        d = d[d["scope"].isin(scopes)]
    if d.empty:
        return {}

    groups: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for key, sub in d.groupby(["metric_family", "metric", "scope"], sort=True):
        groups[tuple(key)] = sub

    return groups


def _prepare_weight_long_from_group(
    *,
    group_df: pd.DataFrame,
    include_model_row: bool,
) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame()

    out: pd.DataFrame
    if include_model_row and "layer_name" in group_df.columns and (group_df["layer_name"] == "__model__").any():
        out = group_df[group_df["layer_name"] == "__model__"].copy()
    else:
        out = group_df.groupby(["model_id", "epoch"], as_index=False, sort=False)["value"].mean()

    if out.empty:
        return out

    if "weight_value" not in out.columns:
        out = out.rename(columns={"value": "weight_value"})

    keep_cols = [c for c in out.columns if c in {"model_id", "epoch", "weight_value", "model_key", "trial_token"}]
    out = out.loc[:, keep_cols].copy()

    if "model_key" not in out.columns:
        out["model_key"] = _canonical_model_key(out["model_id"])
    if "trial_token" not in out.columns:
        out["trial_token"] = _trial_token(out["model_id"])

    return out


def _build_weight_trajectory_matrix(weight_long: pd.DataFrame) -> pd.DataFrame:
    if weight_long.empty:
        return pd.DataFrame()

    # Duplicates can still happen after aggregation in some paths, so keep pivot_table.
    mat = weight_long.pivot_table(
        index="model_id",
        columns="epoch",
        values="weight_value",
        aggfunc="mean",
    )
    if mat.empty:
        return mat
    return mat.sort_index().sort_index(axis=1)


def _iter_upper_triangle_blocks(n: int, block_size: int = _PAIRWISE_BLOCK_SIZE) -> Iterator[Tuple[int, int, int, int, bool]]:
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            yield i0, i1, j0, j1, (i0 == j0)


def _count_pairwise_flips_blocked(
    values_t0: np.ndarray,
    values_t1: np.ndarray,
    *,
    tau: Optional[float] = None,
    block_size: int = _PAIRWISE_BLOCK_SIZE,
) -> int:
    a0 = np.asarray(values_t0, dtype=float)
    a1 = np.asarray(values_t1, dtype=float)

    mask = np.isfinite(a0) & np.isfinite(a1)
    a0 = a0[mask]
    a1 = a1[mask]

    n = len(a0)
    if n < 2:
        return 0

    flips_total = 0

    for i0, i1, j0, j1, is_diag in _iter_upper_triangle_blocks(n, block_size=block_size):
        d0 = a0[i0:i1, None] - a0[None, j0:j1]
        d1 = a1[i0:i1, None] - a1[None, j0:j1]

        flipped = (np.sign(d0) * np.sign(d1)) < 0

        if tau is not None:
            flipped &= (np.abs(d0) >= tau) & (np.abs(d1) >= tau)

        if is_diag:
            flipped = np.triu(flipped, k=1)

        flips_total += int(np.count_nonzero(flipped))

    return flips_total


def _weight_pairwise_flip_table(
    weight_mat: pd.DataFrame,
    *,
    tau: Optional[float] = None,
) -> pd.DataFrame:
    if weight_mat.empty or weight_mat.shape[0] < 2 or weight_mat.shape[1] < 2:
        return pd.DataFrame()

    epochs = list(weight_mat.columns)
    rows: List[Dict[str, Any]] = []

    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        a0 = weight_mat[e0].to_numpy(dtype=float, copy=False)
        a1 = weight_mat[e1].to_numpy(dtype=float, copy=False)
        mask = np.isfinite(a0) & np.isfinite(a1)

        n_models = int(mask.sum())
        n_pairs = int(n_models * (n_models - 1) // 2)

        if n_models < 2:
            flips = 0
        else:
            if tau is None:
                try:
                    flips = int(count_pairwise_flips(a0[mask], a1[mask], tau=None))
                except Exception:
                    flips = _count_pairwise_flips_blocked(a0[mask], a1[mask], tau=None)
            else:
                flips = _count_pairwise_flips_blocked(a0[mask], a1[mask], tau=tau)

        rows.append(
            {
                "from_epoch": int(e0),
                "to_epoch": int(e1),
                "n_models": n_models,
                "n_model_pairs": n_pairs,
                "pairwise_flips": int(flips),
                "pairwise_flip_fraction": float(flips / n_pairs) if n_pairs > 0 else np.nan,
                "tau_pairwise_flip": float(tau) if tau is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _filter_epoch_zero_from_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.empty:
        return mat
    keep_cols = [c for c in mat.columns if pd.notna(c) and int(c) != 0]
    return mat.loc[:, keep_cols]


def _normalize_matrix_model_index(
    mat: pd.DataFrame,
    *,
    use_trial_token: bool = False,
) -> pd.DataFrame:
    if mat.empty:
        return mat

    idx = pd.Index(mat.index.astype(str))
    idx_series = idx.to_series(index=idx)

    if use_trial_token:
        norm_idx = idx_series.str.extract(r"([A-Za-z0-9]{5}_\d{5})", expand=False)
    else:
        norm_idx = idx_series.str.strip().str.replace("\\", "/", regex=False).str.split("/").str[-1]

    out = mat.copy()
    out.index = norm_idx.to_numpy()
    out = out[~pd.isna(out.index)].copy()
    if out.empty:
        return out
    return out.groupby(level=0, sort=True).mean().sort_index()


def _align_model_matrices_precomputed(
    weight_mat: pd.DataFrame,
    *,
    functional_mat_norm_model: pd.DataFrame,
    functional_mat_norm_trial: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w_model = _normalize_matrix_model_index(weight_mat, use_trial_token=False)
    w_trial = _normalize_matrix_model_index(weight_mat, use_trial_token=True)

    candidates = [
        (w_model, functional_mat_norm_model),
        (w_trial, functional_mat_norm_trial),
    ]

    best_weight = pd.DataFrame()
    best_functional = pd.DataFrame()
    best_overlap = -1

    for w_mat, f_mat in candidates:
        common_models = sorted(set(w_mat.index).intersection(f_mat.index))
        common_epochs = sorted(set(w_mat.columns).intersection(f_mat.columns))
        overlap = min(len(common_models), len(common_epochs))
        if overlap > best_overlap:
            best_overlap = overlap
            best_weight = w_mat
            best_functional = f_mat

    return best_weight, best_functional


def _pairwise_flip_pair_ids(
    values_t0: np.ndarray,
    values_t1: np.ndarray,
    *,
    tau: Optional[float] = None,
    block_size: int = _PAIRWISE_BLOCK_SIZE,
) -> set[int]:
    a0 = np.asarray(values_t0, dtype=float)
    a1 = np.asarray(values_t1, dtype=float)

    mask = np.isfinite(a0) & np.isfinite(a1)
    a0 = a0[mask]
    a1 = a1[mask]

    n = len(a0)
    if n < 2:
        return set()

    out: set[int] = set()

    for i0, i1, j0, j1, is_diag in _iter_upper_triangle_blocks(n, block_size=block_size):
        d0 = a0[i0:i1, None] - a0[None, j0:j1]
        d1 = a1[i0:i1, None] - a1[None, j0:j1]

        flipped = (np.sign(d0) * np.sign(d1)) < 0

        if tau is not None:
            flipped &= (np.abs(d0) >= tau) & (np.abs(d1) >= tau)

        if is_diag:
            flipped = np.triu(flipped, k=1)

        ii, jj = np.where(flipped)
        if len(ii) == 0:
            continue

        for bi, bj in zip(ii.tolist(), jj.tolist()):
            gi = i0 + bi
            gj = j0 + bj
            a, b = (gi, gj) if gi < gj else (gj, gi)
            out.add(a * n + b)

    return out


def _pairwise_flip_overlap_table(
    weight_mat: pd.DataFrame,
    functional_mat: pd.DataFrame,
    *,
    weight_tau: Optional[float] = None,
    functional_tau: Optional[float] = None,
) -> pd.DataFrame:
    if weight_mat.empty or functional_mat.empty:
        return pd.DataFrame()

    common_models = sorted(set(weight_mat.index).intersection(functional_mat.index))
    common_epochs = sorted(set(weight_mat.columns).intersection(functional_mat.columns))
    if len(common_models) < 2 or len(common_epochs) < 2:
        return pd.DataFrame()

    wm = weight_mat.loc[common_models, common_epochs]
    fm = functional_mat.loc[common_models, common_epochs]

    rows: List[Dict[str, Any]] = []

    for e0, e1 in zip(common_epochs[:-1], common_epochs[1:]):
        w_pairs = _pairwise_flip_pair_ids(
            wm[e0].to_numpy(dtype=float, copy=False),
            wm[e1].to_numpy(dtype=float, copy=False),
            tau=weight_tau,
        )
        f_pairs = _pairwise_flip_pair_ids(
            fm[e0].to_numpy(dtype=float, copy=False),
            fm[e1].to_numpy(dtype=float, copy=False),
            tau=functional_tau,
        )
        overlap = w_pairs & f_pairs
        union = w_pairs | f_pairs

        rows.append(
            {
                "from_epoch": int(e0),
                "to_epoch": int(e1),
                "n_weight_flip_pairs": int(len(w_pairs)),
                "n_functional_flip_pairs": int(len(f_pairs)),
                "n_overlap_flip_pairs": int(len(overlap)),
                "overlap_jaccard": float(len(overlap) / len(union)) if len(union) > 0 else np.nan,
                "overlap_precision_weight_vs_functional": float(len(overlap) / len(w_pairs)) if len(w_pairs) > 0 else np.nan,
                "overlap_recall_weight_vs_functional": float(len(overlap) / len(f_pairs)) if len(f_pairs) > 0 else np.nan,
                "same_pair_overlap_pct_of_weight": float(100.0 * len(overlap) / len(w_pairs)) if len(w_pairs) > 0 else np.nan,
                "same_pair_overlap_pct_of_functional": float(100.0 * len(overlap) / len(f_pairs)) if len(f_pairs) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _load_or_compute_functional_switches(
    *,
    df: pd.DataFrame,
    ds: str,
    functional_metric: str,
    functional_higher_is_better: bool,
    epoch_col: str,
    model_col: str,
    epoch_range: Optional[Tuple[int, int]],
    tables_dir: Path,
) -> pd.DataFrame:
    tables_dir.mkdir(parents=True, exist_ok=True)
    metric_key = _safe_key_part(functional_metric)
    hib_key = "hib1" if functional_higher_is_better else "hib0"
    table_path = tables_dir / f"{ds}_all_switches__{metric_key}__{hib_key}.parquet"
    legacy_table_path = tables_dir / f"{ds}_all_switches_.parquet"

    if table_path.exists():
        return pd.read_parquet(table_path)

    existing = pd.DataFrame()
    if legacy_table_path.exists():
        existing = pd.read_parquet(legacy_table_path)
        if not existing.empty and "metric" in existing.columns:
            metric_existing = existing[existing["metric"].astype(str) == functional_metric].copy()
            if not metric_existing.empty:
                metric_existing.to_parquet(table_path, index=False)
                return metric_existing

    dfd = df[df["dataset_name"] == ds].copy()
    res = run_ranking_flips(
        dfd,
        {
            "epoch_col": epoch_col,
            "model_col": model_col,
            "epoch_range": list(epoch_range) if epoch_range is not None else [],
            "metrics": [
                {
                    "name": functional_metric,
                    "higher_is_better": functional_higher_is_better,
                }
            ],
            "tau": None,
            "panel": {
                "nrows": 1,
                "ncols": 1,
                "figsize": [8, 4],
            },
            "png_dpi": 100,
        },
        {},
    )

    generated = res.tables.get(f"{ds}_all_switches_", pd.DataFrame()).copy()
    for item in res.figures:
        if isinstance(item, dict):
            _, fig = next(iter(item.items()))
            plt.close(fig)

    if generated.empty:
        return existing

    generated.to_parquet(table_path, index=False)
    return generated


def _prepare_functional_flip_table(
    switches: pd.DataFrame,
    *,
    functional_metric: str,
) -> pd.DataFrame:
    if switches.empty or "metric" not in switches.columns:
        return pd.DataFrame()

    out = switches[switches["metric"].astype(str) == functional_metric].copy()
    if out.empty:
        return out

    if "pairwise_flips_without_smoothening" in out.columns:
        flip_col = "pairwise_flips_without_smoothening"
    elif "pairwise_flips_without_tau" in out.columns:
        flip_col = "pairwise_flips_without_tau"
    else:
        flip_col = "pairwise_flips"
    out["functional_pairwise_flips"] = pd.to_numeric(out[flip_col], errors="coerce")

    if "n_models" in out.columns:
        n_models = pd.to_numeric(out["n_models"], errors="coerce")
        n_pairs = n_models * (n_models - 1) / 2.0
        out["functional_pairwise_flip_fraction"] = out["functional_pairwise_flips"] / n_pairs.replace(0, np.nan)
    else:
        out["functional_pairwise_flip_fraction"] = np.nan

    return out.sort_values("to_epoch")


def _flip_correspondence_summary(
    weight_flip_df: pd.DataFrame,
    functional_flip_df: pd.DataFrame,
) -> Dict[str, Any]:
    aligned = weight_flip_df.merge(
        functional_flip_df[
            [
                "to_epoch",
                "functional_pairwise_flips",
                "functional_pairwise_flip_fraction",
            ]
        ],
        on="to_epoch",
        how="inner",
    )
    if aligned.empty:
        return {
            "n_aligned_epochs": 0,
            "pearson_pairwise_flips": np.nan,
            "spearman_pairwise_flips": np.nan,
            "pearson_pairwise_flip_fraction": np.nan,
            "spearman_pairwise_flip_fraction": np.nan,
        }

    def _corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 3 or _is_effectively_constant(x) or _is_effectively_constant(y):
            return np.nan, np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            try:
                pr, _ = pearsonr(x, y)
            except Exception:
                pr = np.nan
            try:
                sr, _ = spearmanr(x, y)
            except Exception:
                sr = np.nan
        return float(pr) if np.isfinite(pr) else np.nan, float(sr) if np.isfinite(sr) else np.nan

    w_count = aligned["pairwise_flips"].to_numpy(dtype=float)
    f_count = aligned["functional_pairwise_flips"].to_numpy(dtype=float)
    w_frac = aligned["pairwise_flip_fraction"].to_numpy(dtype=float)
    f_frac = aligned["functional_pairwise_flip_fraction"].to_numpy(dtype=float)

    pr_count, sr_count = _corr(w_count, f_count)
    pr_frac, sr_frac = _corr(w_frac, f_frac)

    return {
        "n_aligned_epochs": int(len(aligned)),
        "pearson_pairwise_flips": pr_count,
        "spearman_pairwise_flips": sr_count,
        "pearson_pairwise_flip_fraction": pr_frac,
        "spearman_pairwise_flip_fraction": sr_frac,
    }


def _epoch_association_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for epoch, sub in joined.groupby("epoch", dropna=False):
        x = sub["weight_value"].to_numpy(dtype=float)
        y = sub["rank"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = int(len(x))
        if n < 3:
            rows.append(
                {
                    "epoch": int(epoch),
                    "n_models": n,
                    "spearman_rho": np.nan,
                    "spearman_p_value": np.nan,
                    "kendall_tau": np.nan,
                    "kendall_p_value": np.nan,
                }
            )
            continue

        if _is_effectively_constant(x) or _is_effectively_constant(y):
            rho, rho_p = np.nan, np.nan
            tau, tau_p = np.nan, np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                rho, rho_p = spearmanr(x, y)
                tau, tau_p = kendalltau(x, y, variant="b")

        rows.append(
            {
                "epoch": int(epoch),
                "n_models": n,
                "spearman_rho": float(rho),
                "spearman_p_value": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p_value": float(tau_p),
            }
        )

    return pd.DataFrame(rows).sort_values("epoch")


def _rank_change_signal(
    mat: pd.DataFrame,
    *,
    higher_is_better: bool,
) -> pd.DataFrame:
    rmat = rank_per_epoch(mat, higher_is_better=higher_is_better, tie_method="average")
    rc = rank_change_table(mat, rank_mat=rmat, tau=None).copy()
    if rc.empty:
        return rc
    rc["n_models"] = mat.shape[0]
    rc["rank_change_fraction"] = rc["models_changed_rank"] / max(mat.shape[0], 1)
    return rc


def _detect_crossover_events(
    rc: pd.DataFrame,
    *,
    quantile: float,
    min_changed_rank: int,
    min_separation: int,
) -> pd.DataFrame:
    if rc.empty:
        return rc

    q = float(np.clip(quantile, 0.0, 1.0))
    threshold = float(np.nanquantile(rc["models_changed_rank"], q))
    threshold = max(threshold, float(min_changed_rank))

    cand = rc[rc["models_changed_rank"] >= threshold].sort_values("to_epoch").copy()
    if cand.empty:
        return cand

    selected: List[pd.Series] = []
    last_epoch: Optional[int] = None
    for _, row in cand.iterrows():
        e = int(row["to_epoch"])
        if last_epoch is None or (e - last_epoch) >= int(min_separation):
            selected.append(row)
            last_epoch = e

    if not selected:
        return pd.DataFrame(columns=cand.columns)

    out = pd.DataFrame(selected).reset_index(drop=True)
    out["crossover_threshold_models_changed"] = threshold
    return out


def _event_alignment_table(
    weight_epoch: pd.DataFrame,
    events: pd.DataFrame,
    *,
    align_window: int,
) -> pd.DataFrame:
    if weight_epoch.empty or events.empty:
        return pd.DataFrame()

    w = int(max(1, align_window))
    signal = weight_epoch.set_index("epoch")["weight_mean"]
    rows: List[Dict[str, Any]] = []

    for _, ev in events.iterrows():
        e = int(ev["to_epoch"])
        for rel in range(-w, w + 1):
            abs_e = e + rel
            if abs_e in signal.index:
                rows.append(
                    {
                        "event_epoch": e,
                        "relative_epoch": rel,
                        "epoch": abs_e,
                        "weight_mean": float(signal.loc[abs_e]),
                    }
                )

    return pd.DataFrame(rows)


def _event_shift_tests(
    weight_epoch: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pre_window: int,
    post_window: int,
) -> pd.DataFrame:
    if weight_epoch.empty or events.empty:
        return pd.DataFrame()

    signal = weight_epoch.set_index("epoch")["weight_mean"]
    pre_w = int(max(1, pre_window))
    post_w = int(max(1, post_window))

    event_rows: List[Dict[str, Any]] = []
    pre_vals: List[float] = []
    at_vals: List[float] = []
    post_vals: List[float] = []

    for _, ev in events.iterrows():
        e = int(ev["to_epoch"])
        pre_idx = [x for x in range(e - pre_w, e) if x in signal.index]
        post_idx = [x for x in range(e + 1, e + 1 + post_w) if x in signal.index]
        if (e not in signal.index) or (len(pre_idx) == 0) or (len(post_idx) == 0):
            continue

        pre = float(signal.loc[pre_idx].mean())
        at = float(signal.loc[e])
        post = float(signal.loc[post_idx].mean())
        pre_vals.append(pre)
        at_vals.append(at)
        post_vals.append(post)

        event_rows.append(
            {
                "event_epoch": e,
                "pre_mean": pre,
                "at_value": at,
                "post_mean": post,
                "at_minus_pre": at - pre,
                "post_minus_at": post - at,
                "post_minus_pre": post - pre,
            }
        )

    per_event = pd.DataFrame(event_rows)
    if per_event.empty:
        return per_event

    def _paired(name: str, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        finite_mask = np.isfinite(a) & np.isfinite(b)
        a_finite = a[finite_mask]
        b_finite = b[finite_mask]
        diff = b_finite - a_finite
        n_events = int(len(diff))
        mean_diff = float(diff.mean()) if n_events > 0 else np.nan

        if n_events < 2:
            return {
                "comparison": name,
                "n_events": n_events,
                "mean_diff": mean_diff,
                "ttest_p_value": np.nan,
                "wilcoxon_p_value": np.nan,
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _, p_t = ttest_rel(a_finite, b_finite, nan_policy="omit")

        if _is_effectively_constant(diff):
            return {
                "comparison": name,
                "n_events": n_events,
                "mean_diff": mean_diff,
                "ttest_p_value": float(p_t) if np.isfinite(p_t) else np.nan,
                "wilcoxon_p_value": np.nan,
            }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _, p_w = wilcoxon(diff)
                p_w = float(p_w)
        except Exception:
            p_w = np.nan

        return {
            "comparison": name,
            "n_events": n_events,
            "mean_diff": mean_diff,
            "ttest_p_value": float(p_t),
            "wilcoxon_p_value": p_w,
        }

    summary = pd.DataFrame(
        [
            _paired("at_vs_pre", np.array(pre_vals), np.array(at_vals)),
            _paired("post_vs_at", np.array(at_vals), np.array(post_vals)),
            _paired("post_vs_pre", np.array(pre_vals), np.array(post_vals)),
        ]
    )
    summary["stat"] = "summary"
    per_event["stat"] = "per_event"
    return pd.concat([summary, per_event], ignore_index=True, sort=False)


def _lagged_correlation_table(
    weight_epoch: pd.DataFrame,
    rank_change: pd.DataFrame,
    *,
    max_lag: int,
) -> pd.DataFrame:
    if weight_epoch.empty or rank_change.empty:
        return pd.DataFrame()

    w = weight_epoch.set_index("epoch")["weight_mean"].sort_index()
    r = rank_change.set_index("to_epoch")["rank_change_fraction"].sort_index()
    base = pd.DataFrame({"weight_mean": w, "rank_change_fraction": r})

    rows: List[Dict[str, Any]] = []
    for lag in range(-int(max_lag), int(max_lag) + 1):
        y = base["rank_change_fraction"].shift(-lag)
        x = base["weight_mean"]
        aligned = pd.concat([x, y], axis=1).dropna()
        n = int(len(aligned))

        if n < 3:
            rows.append(
                {
                    "lag": lag,
                    "n_points": n,
                    "pearson_r": np.nan,
                    "pearson_p_value": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p_value": np.nan,
                }
            )
            continue

        x_vals = aligned["weight_mean"].to_numpy(dtype=float)
        y_vals = aligned["rank_change_fraction"].to_numpy(dtype=float)

        if _is_effectively_constant(x_vals) or _is_effectively_constant(y_vals):
            rows.append(
                {
                    "lag": lag,
                    "n_points": n,
                    "pearson_r": np.nan,
                    "pearson_p_value": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p_value": np.nan,
                }
            )
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            try:
                pr, pp = pearsonr(x_vals, y_vals)
            except Exception:
                pr, pp = np.nan, np.nan

            try:
                sr, sp = spearmanr(x_vals, y_vals)
            except Exception:
                sr, sp = np.nan, np.nan

        rows.append(
            {
                "lag": lag,
                "n_points": n,
                "pearson_r": float(pr) if np.isfinite(pr) else np.nan,
                "pearson_p_value": float(pp) if np.isfinite(pp) else np.nan,
                "spearman_rho": float(sr) if np.isfinite(sr) else np.nan,
                "spearman_p_value": float(sp) if np.isfinite(sp) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    out["is_best_lag_by_abs_spearman"] = False

    valid = out["abs_spearman_rho"].dropna()
    if not valid.empty:
        best_idx = valid.idxmax()
        out.loc[best_idx, "is_best_lag_by_abs_spearman"] = True

    return out.drop(columns=["abs_spearman_rho"])


def _plot_epoch_association(ds: str, metric_label: str, assoc: pd.DataFrame, dpi: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.plot(assoc["epoch"], assoc["spearman_rho"], label="Spearman rho")
    ax.plot(assoc["epoch"], assoc["kendall_tau"], label="Kendall tau", linestyle="--")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(f"{zoo_plot_name(ds)} | Epoch-wise association: {metric_label} vs rank")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_event_alignment(ds: str, metric_label: str, align_df: pd.DataFrame, dpi: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    for _, sub in align_df.groupby("event_epoch"):
        ax.plot(sub["relative_epoch"], sub["weight_mean"], color="gray", alpha=0.25, linewidth=1)
    mean_curve = align_df.groupby("relative_epoch", as_index=False)["weight_mean"].mean()
    ax.plot(mean_curve["relative_epoch"], mean_curve["weight_mean"], color="tab:blue", linewidth=2, label="mean")
    ax.axvline(0, color="black", linewidth=1, alpha=0.7, linestyle="--")
    ax.set_title(f"{zoo_plot_name(ds)} | Event-aligned {metric_label} around cross-over epochs")
    ax.set_xlabel("Relative epoch (0 = cross-over)")
    ax.set_ylabel("Mean weight statistic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_lagged_corr(ds: str, metric_label: str, lag_df: pd.DataFrame, dpi: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.plot(lag_df["lag"], lag_df["spearman_rho"], label="Spearman rho")
    ax.plot(lag_df["lag"], lag_df["pearson_r"], label="Pearson r", linestyle="--")
    best = lag_df[lag_df["is_best_lag_by_abs_spearman"] == True]  # noqa: E712
    if not best.empty:
        b = best.iloc[0]
        ax.scatter([b["lag"]], [b["spearman_rho"]], color="red", zorder=3, label=f"best lag={int(b['lag'])}")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(f"{zoo_plot_name(ds)} | Lagged relationship: {metric_label} vs rank-instability")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_metric_score_bars(
    ds: str,
    summary: pd.DataFrame,
    *,
    score_col: str,
    title_suffix: str,
    top_k: int,
    dpi: int,
) -> plt.Figure:
    if summary.empty or score_col not in summary.columns or "metric_label" not in summary.columns:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
        ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix} (no data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    s = summary.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(int(max(1, top_k)))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    if s.empty:
        ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix} (no data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    x = np.arange(len(s))
    ax.bar(x, s[score_col].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(s["metric_label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel(score_col)
    ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_metric_heatmap(
    ds: str,
    summary: pd.DataFrame,
    *,
    value_col: str,
    title_suffix: str,
    dpi: int,
) -> plt.Figure:
    if summary.empty or value_col not in summary.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix} (no data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    p = summary.pivot_table(index="metric_family", columns="metric", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    if p.empty:
        ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix} (no data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    mat = p.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_xticklabels(p.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels(p.index.tolist())
    ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix}")
    ax.set_xlabel("metric")
    ax.set_ylabel("metric_family")
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def _plot_metric_flip_heatmap(
    ds: str,
    flip_long: pd.DataFrame,
    *,
    value_col: str,
    title_suffix: str,
    dpi: int,
) -> plt.Figure:
    if flip_long.empty or value_col not in flip_long.columns:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
        ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix} (no data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    ordered_metrics = (
        flip_long[["metric_family", "metric", "scope", "metric_label"]]
        .drop_duplicates()
        .sort_values(["metric_family", "metric", "scope"])["metric_label"]
        .tolist()
    )
    p = (
        flip_long.pivot_table(index="metric_label", columns="to_epoch", values=value_col, aggfunc="mean")
        .reindex(ordered_metrics)
        .sort_index(axis=1)
    )

    fig_height = max(8, 0.35 * max(len(p.index), 1))
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=dpi)
    if p.empty:
        ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix} (no data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    im = ax.imshow(p.to_numpy(dtype=float), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_xticklabels(p.columns.tolist())
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels(p.index.tolist(), fontsize=8)
    ax.set_title(f"{zoo_plot_name(ds)} | {title_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def _plot_metric_flip_counts(
    ds: str,
    metric_label: str,
    flip_df: pd.DataFrame,
    *,
    functional_metric: Optional[str] = None,
    functional_flip_df: Optional[pd.DataFrame] = None,
    overlap_df: Optional[pd.DataFrame] = None,
    dpi: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.plot(flip_df["to_epoch"], flip_df["pairwise_flips"], marker="o", label="pairwise flips")
    if functional_flip_df is not None and not functional_flip_df.empty:
        ax.plot(
            functional_flip_df["to_epoch"],
            functional_flip_df["functional_pairwise_flips"],
            marker="o",
            linestyle="--",
            label=f"functional flips ({functional_metric})" if functional_metric else "functional flips",
        )
    if overlap_df is not None and not overlap_df.empty:
        ax.plot(
            overlap_df["to_epoch"],
            overlap_df["n_overlap_flip_pairs"],
            marker="o",
            linestyle=":",
            label="same-pair overlap",
        )
    metric_suffix = f" vs {functional_metric}" if functional_metric else ""
    ax.set_title(f"{zoo_plot_name(ds)} | Pairwise cross-overs per epoch: {metric_label}{metric_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pairwise flip count")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    if overlap_df is not None and not overlap_df.empty:
        ax2.plot(
            overlap_df["to_epoch"],
            overlap_df["overlap_jaccard"],
            color="tab:red",
            linewidth=1.5,
            alpha=0.8,
            label="overlap jaccard",
        )
    ax2.set_ylabel("Pair overlap")
    ax2.set_ylim(0.0, 1.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    return fig


def _plot_metric_overlap_percent(
    ds: str,
    metric_label: str,
    overlap_df: pd.DataFrame,
    *,
    functional_metric: str,
    dpi: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.plot(
        overlap_df["to_epoch"],
        overlap_df["same_pair_overlap_pct_of_functional"],
        marker="o",
        label=f"% of {functional_metric} flip pairs matched",
    )
    ax.plot(
        overlap_df["to_epoch"],
        overlap_df["same_pair_overlap_pct_of_weight"],
        marker="o",
        linestyle="--",
        label="% of weight flip pairs matched",
    )
    ax.set_title(f"{zoo_plot_name(ds)} | Same-pair overlap percentage: {metric_label} vs {functional_metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overlap percentage")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def run_weight_rank_correspondence(
    df: pd.DataFrame,
    config: Dict[str, Any],
    registry: Dict[str, Any],
) -> AnalysisResult:
    epoch_col = str(config.get("epoch_col", "training_iteration"))
    model_col = str(config.get("model_col", "model_name"))
    epoch_range = tuple(config.get("epoch_range", [])) or None
    drop_epoch_zero = bool(config.get("drop_epoch_zero", True))
    png_dpi = int(config.get("png_dpi", 200))

    functional_metric = str(config.get("functional_metric", config.get("rank_metric", "test_acc")))
    functional_metric_key = _safe_key_part(functional_metric)
    functional_higher_is_better = bool(config.get("functional_higher_is_better", config.get("higher_is_better", True)))
    metric_families = _as_list_or_all(config.get("metric_family", "all"))
    metrics = _as_list_or_all(config.get("metric", "all"))
    scopes = _as_list_or_all(config.get("scope", "all"))
    include_model_row = bool(config.get("include_model_row", True))
    dataset_split = str(config.get("dataset_split", "full"))
    cache_dir = Path(config.get("cache_dir", "./cache/weight_metrics"))
    force_recompute = bool(config.get("force_recompute", False))
    skip_missing_cache = bool(config.get("skip_missing_cache", False))
    ranking_flips_tables_dir = Path(
        config.get("ranking_flips_tables_dir", "learning-dynamics/outputs/reports/tables/ranking_flips")
    )
    top_k_metrics = int(config.get("top_k_metrics", 12))
    pairwise_flip_tau = config.get("pairwise_flip_tau", None)
    pairwise_flip_tau = float(pairwise_flip_tau) if pairwise_flip_tau is not None else None
    show_progress = bool(config.get("show_progress", True))
    model_rows_only = bool(config.get("model_rows_only", False))
    use_partitioned_cache = bool(config.get("use_partitioned_cache", True))
    partition_batch_size = int(config.get("partition_batch_size", 250_000))

    parquet_columns = sorted(
        {
            "metric_family",
            "metric",
            "scope",
            "layer_name",
            "model_id",
            "epoch",
            "value",
        }
    )
    parquet_filters = _build_weight_cache_filters(
        metric_families=metric_families,
        metrics=metrics,
        scopes=scopes,
        model_rows_only=model_rows_only,
    )

    log.info(
        "run_weight_rank_correspondence config: functional_metric=%s dataset_split=%s drop_epoch_zero=%s "
        "force_recompute=%s skip_missing_cache=%s include_model_row=%s model_rows_only=%s "
        "use_partitioned_cache=%s pairwise_flip_tau=%s",
        functional_metric,
        dataset_split,
        drop_epoch_zero,
        force_recompute,
        skip_missing_cache,
        include_model_row,
        model_rows_only,
        use_partitioned_cache,
        pairwise_flip_tau,
    )
    log.info(
        "Weight cache parquet read config: columns=%s filters=%s",
        parquet_columns,
        parquet_filters if parquet_filters else None,
    )

    tables, figures = make_result_stores(config, default_png_dpi=png_dpi)

    if "dataset_name" not in df.columns:
        raise ValueError("df must have 'dataset_name' column.")

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())
    dataset_iterator = tqdm(datasets, desc="Weight-rank correspondence", unit="dataset") if show_progress else datasets

    for ds in dataset_iterator:
        dataset_start = perf_counter()
        if show_progress:
            dataset_iterator.set_description(f"Weight-rank correspondence [{ds}]")

        try:
            if ds not in registry:
                log.warning("Dataset '%s' is in dataframe but missing in registry. Skipping.", ds)
                continue

            log.info("Starting weight-rank correspondence for dataset '%s'.", ds)

            functional_switches = _load_or_compute_functional_switches(
                df=df,
                ds=ds,
                functional_metric=functional_metric,
                functional_higher_is_better=functional_higher_is_better,
                epoch_col=epoch_col,
                model_col=model_col,
                epoch_range=epoch_range,
                tables_dir=ranking_flips_tables_dir,
            )
            _log_df_size(f"{ds} functional switches", functional_switches)
            functional_flip_df = _prepare_functional_flip_table(
                functional_switches,
                functional_metric=functional_metric,
            )
            if functional_flip_df.empty:
                log.warning(
                    "No functional flip table for dataset '%s' and metric '%s'. Skipping.",
                    ds,
                    functional_metric,
                )
                continue
            _log_df_size(f"{ds} functional flip table", functional_flip_df)
            tables[f"{ds}_functional_pairwise_flips__{functional_metric_key}"] = functional_flip_df

            dfd = df[df["dataset_name"] == ds].copy()
            _log_df_size(f"{ds} source dataset", dfd)
            epoch_lst = _infer_available_epochs(
                dfd,
                epoch_col=epoch_col,
                epoch_range=None,
            )
            if not epoch_lst:
                log.warning("Could not infer available epochs for dataset '%s'. Skipping.", ds)
                continue
            log.info(
                "Dataset '%s' inferred %d epochs (min=%d, max=%d).",
                ds,
                len(epoch_lst),
                min(epoch_lst),
                max(epoch_lst),
            )

            functional_mat = build_metric_matrix_from_dfm(
                dfd,
                metric_key=functional_metric,
                epoch_col=epoch_col,
                model_col=model_col,
                epoch_range=epoch_range,
            )
            functional_mat = _filter_epoch_zero_from_matrix(functional_mat) if drop_epoch_zero else functional_mat
            if functional_mat.empty or functional_mat.shape[0] < 2 or functional_mat.shape[1] < 2:
                log.warning(
                    "No functional metric matrix for dataset '%s' and metric '%s'. Skipping.",
                    ds,
                    functional_metric,
                )
                continue
            _log_matrix_size(f"{ds} functional matrix", functional_mat)

            functional_mat_norm_model = _normalize_matrix_model_index(functional_mat, use_trial_token=False)
            functional_mat_norm_trial = _normalize_matrix_model_index(functional_mat, use_trial_token=True)

            spec = registry[ds]

            combo_sources: List[Tuple[Tuple[str, str, str], Dict[str, Any]]] = []
            if use_partitioned_cache:
                if force_recompute:
                    # Refresh the source full cache before rebuilding partitions.
                    _ = load_or_compute_df_long(
                        ds_name=ds,
                        root_dir=Path(spec.root_dir),
                        dstype=spec.dstype,
                        dataset_split=dataset_split,
                        epoch_lst=epoch_lst,
                        cache_dir=cache_dir,
                        force_recompute=True,
                        skip_missing_cache=skip_missing_cache,
                        preload_properties=True,
                        validate_checkpoints=False,
                        show_progress=show_progress,
                    )

                manifest = load_or_build_partitioned_weight_metric_manifest(
                    ds_name=ds,
                    dataset_split=dataset_split,
                    cache_dir=cache_dir,
                    rebuild=force_recompute,
                    batch_size=partition_batch_size,
                )
                if manifest is not None and not manifest.empty:
                    _log_df_size(f"{ds} partition manifest", manifest)

                    manifest_sel = manifest.copy()
                    if metric_families:
                        manifest_sel = manifest_sel[manifest_sel["metric_family"].isin(metric_families)]
                    if metrics:
                        manifest_sel = manifest_sel[manifest_sel["metric"].isin(metrics)]
                    if scopes:
                        manifest_sel = manifest_sel[manifest_sel["scope"].isin(scopes)]
                    if model_rows_only and include_model_row:
                        manifest_sel = manifest_sel[manifest_sel["has_model_rows"]].copy()

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

                    log.info(
                        "Dataset '%s' will process %d metric combinations from partitioned cache.",
                        ds,
                        len(combo_sources),
                    )

            if not combo_sources:
                dfw_all = load_or_compute_df_long(
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
                    show_progress=show_progress,
                    parquet_columns=parquet_columns,
                    parquet_filters=parquet_filters or None,
                )
                if dfw_all is None:
                    continue
                _log_df_size(f"{ds} cached weight table", dfw_all)

                dfw_all = _preprocess_weight_metrics_frame(
                    dfw_all,
                    drop_epoch_zero=drop_epoch_zero,
                )
                _log_df_size(f"{ds} preprocessed weight table", dfw_all)

                combo_groups = _discover_metric_groups(
                    dfw_all,
                    metric_families=metric_families,
                    metrics=metrics,
                    scopes=scopes,
                )
                if not combo_groups:
                    log.warning("No matching weight metric combinations for dataset '%s'.", ds)
                    continue

                combo_sources = [
                    (key, {"dataframe": group_df, "source_kind": "full_cache"})
                    for key, group_df in combo_groups.items()
                ]
                log.info(
                    "Dataset '%s' will process %d metric combinations from the full cache fallback.",
                    ds,
                    len(combo_sources),
                )

            summary_rows: List[Dict[str, Any]] = []
            metric_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
            flip_long_rows: List[pd.DataFrame] = []
            overlap_long_rows: List[pd.DataFrame] = []

            for combo_idx, ((metric_family, metric, scope), combo_source) in enumerate(combo_sources, start=1):
                metric_id = _combo_id(metric_family, metric, scope)
                metric_label = _combo_label(metric_family, metric, scope)
                log.info(
                    "Dataset '%s' processing combo %d/%d: %s",
                    ds,
                    combo_idx,
                    len(combo_sources),
                    metric_label,
                )

                if "path" in combo_source:
                    group_df = pd.read_parquet(combo_source["path"])
                    group_df = _preprocess_weight_metrics_frame(
                        group_df,
                        drop_epoch_zero=drop_epoch_zero,
                    )
                    log.info(
                        "Dataset '%s' loaded combo '%s' from partitioned cache (%s rows).",
                        ds,
                        metric_label,
                        len(group_df),
                    )
                else:
                    group_df = combo_source["dataframe"]

                _log_df_size(f"{ds} combo group [{metric_label}]", group_df)

                weight_long = _prepare_weight_long_from_group(
                    group_df=group_df,
                    include_model_row=include_model_row,
                )
                if weight_long.empty:
                    log.info("Dataset '%s' combo '%s' produced empty weight_long. Skipping.", ds, metric_label)
                    continue
                _log_df_size(f"{ds} weight_long [{metric_label}]", weight_long)

                weight_mat = _build_weight_trajectory_matrix(weight_long)
                if weight_mat.empty or weight_mat.shape[0] < 2 or weight_mat.shape[1] < 2:
                    log.info("Dataset '%s' combo '%s' produced empty/small weight matrix. Skipping.", ds, metric_label)
                    continue
                _log_matrix_size(f"{ds} weight matrix [{metric_label}]", weight_mat)

                flip_df = _weight_pairwise_flip_table(weight_mat, tau=pairwise_flip_tau)
                if flip_df.empty:
                    log.info("Dataset '%s' combo '%s' produced empty flip table. Skipping.", ds, metric_label)
                    continue
                _log_df_size(f"{ds} weight flip table [{metric_label}]", flip_df)

                weight_trajectory_table = weight_mat.reset_index().rename(columns={"model_id": "model_id"})
                tables[f"{ds}_weight_trajectory_matrix__{functional_metric_key}__{metric_id}"] = weight_trajectory_table
                tables[f"{ds}_weight_pairwise_flips__{functional_metric_key}__{metric_id}"] = flip_df

                aligned_weight_mat, aligned_functional_mat = _align_model_matrices_precomputed(
                    weight_mat,
                    functional_mat_norm_model=functional_mat_norm_model,
                    functional_mat_norm_trial=functional_mat_norm_trial,
                )
                overlap_df = _pairwise_flip_overlap_table(
                    aligned_weight_mat,
                    aligned_functional_mat,
                    weight_tau=pairwise_flip_tau,
                    functional_tau=None,
                )
                if not overlap_df.empty:
                    tables[f"{ds}_weight_functional_flip_overlap__{functional_metric_key}__{metric_id}"] = overlap_df
                _log_df_size(f"{ds} overlap table [{metric_label}]", overlap_df)

                flip_df_annotated = flip_df.copy()
                flip_df_annotated["metric_family"] = metric_family
                flip_df_annotated["metric"] = metric
                flip_df_annotated["scope"] = scope
                flip_df_annotated["metric_id"] = metric_id
                flip_df_annotated["metric_label"] = metric_label
                flip_df_annotated["functional_metric"] = functional_metric
                flip_long_rows.append(flip_df_annotated)

                corr_summary = _flip_correspondence_summary(flip_df, functional_flip_df)

                overlap_jaccard_mean = float(overlap_df["overlap_jaccard"].mean()) if not overlap_df.empty else np.nan
                overlap_jaccard_max = float(overlap_df["overlap_jaccard"].max()) if not overlap_df.empty else np.nan
                overlap_count_mean = float(overlap_df["n_overlap_flip_pairs"].mean()) if not overlap_df.empty else np.nan
                overlap_count_max = float(overlap_df["n_overlap_flip_pairs"].max()) if not overlap_df.empty else np.nan
                overlap_pct_functional_mean = (
                    float(overlap_df["same_pair_overlap_pct_of_functional"].mean()) if not overlap_df.empty else np.nan
                )
                overlap_pct_functional_max = (
                    float(overlap_df["same_pair_overlap_pct_of_functional"].max()) if not overlap_df.empty else np.nan
                )
                overlap_pct_weight_mean = (
                    float(overlap_df["same_pair_overlap_pct_of_weight"].mean()) if not overlap_df.empty else np.nan
                )
                overlap_pct_weight_max = (
                    float(overlap_df["same_pair_overlap_pct_of_weight"].max()) if not overlap_df.empty else np.nan
                )

                if not overlap_df.empty:
                    overlap_df_annotated = overlap_df.copy()
                    overlap_df_annotated["metric_family"] = metric_family
                    overlap_df_annotated["metric"] = metric
                    overlap_df_annotated["scope"] = scope
                    overlap_df_annotated["metric_id"] = metric_id
                    overlap_df_annotated["metric_label"] = metric_label
                    overlap_df_annotated["functional_metric"] = functional_metric
                    overlap_long_rows.append(overlap_df_annotated)

                mean_flips = float(flip_df["pairwise_flips"].mean()) if not flip_df.empty else np.nan
                max_flips = float(flip_df["pairwise_flips"].max()) if not flip_df.empty else np.nan
                mean_flip_fraction = float(flip_df["pairwise_flip_fraction"].mean()) if not flip_df.empty else np.nan
                max_flip_fraction = float(flip_df["pairwise_flip_fraction"].max()) if not flip_df.empty else np.nan
                best_epoch = int(flip_df.loc[flip_df["pairwise_flips"].idxmax(), "to_epoch"]) if not flip_df.empty else np.nan

                summary_rows.append(
                    {
                        "functional_metric": functional_metric,
                        "metric_family": metric_family,
                        "metric": metric,
                        "scope": scope,
                        "metric_id": metric_id,
                        "metric_label": metric_label,
                        "n_models": int(weight_mat.shape[0]),
                        "n_epoch_pairs": int(len(flip_df)),
                        "mean_pairwise_flips": mean_flips,
                        "max_pairwise_flips": max_flips,
                        "mean_pairwise_flip_fraction": mean_flip_fraction,
                        "max_pairwise_flip_fraction": max_flip_fraction,
                        "peak_flip_epoch": best_epoch,
                        "mean_overlap_flip_pairs": overlap_count_mean,
                        "max_overlap_flip_pairs": overlap_count_max,
                        "mean_overlap_jaccard": overlap_jaccard_mean,
                        "max_overlap_jaccard": overlap_jaccard_max,
                        "mean_same_pair_overlap_pct_of_functional": overlap_pct_functional_mean,
                        "max_same_pair_overlap_pct_of_functional": overlap_pct_functional_max,
                        "mean_same_pair_overlap_pct_of_weight": overlap_pct_weight_mean,
                        "max_same_pair_overlap_pct_of_weight": overlap_pct_weight_max,
                        **corr_summary,
                    }
                )

                metric_cache[metric_id] = {
                    "metric_label": metric_label,
                    "flip_df": flip_df,
                    "overlap_df": overlap_df,
                }

            summary = pd.DataFrame(summary_rows)
            if not summary.empty and "mean_overlap_jaccard" in summary.columns:
                summary = summary.sort_values("mean_overlap_jaccard", ascending=False, na_position="last")
            tables[f"{ds}_metric_flip_summary__{functional_metric_key}"] = summary
            _log_df_size(f"{ds} metric flip summary", summary)

            flip_long = pd.concat(flip_long_rows, ignore_index=True) if flip_long_rows else pd.DataFrame()
            if not flip_long.empty:
                tables[f"{ds}_metric_pairwise_flips_long__{functional_metric_key}"] = flip_long
            _log_df_size(f"{ds} metric pairwise flips long", flip_long)

            overlap_long = pd.concat(overlap_long_rows, ignore_index=True) if overlap_long_rows else pd.DataFrame()
            if not overlap_long.empty:
                tables[f"{ds}_metric_pairwise_flip_overlap_long__{functional_metric_key}"] = overlap_long
            _log_df_size(f"{ds} metric overlap long", overlap_long)
            log.info(
                "Dataset '%s' finished aggregation: summary_rows=%d cached_metric_plots=%d",
                ds,
                len(summary_rows),
                len(metric_cache),
            )

            figures.append(
                {
                    f"{ds}_metric_scan_top_mean_pairwise_flips__{functional_metric_key}": _plot_metric_score_bars(
                        ds,
                        summary,
                        score_col="mean_overlap_jaccard",
                        title_suffix=f"Top weight statistics by mean same-pair Jaccard with {functional_metric}",
                        top_k=top_k_metrics,
                        dpi=png_dpi,
                    )
                }
            )
            figures.append(
                {
                    f"{ds}_metric_scan_top_mean_pairwise_flip_fraction__{functional_metric_key}": _plot_metric_score_bars(
                        ds,
                        summary,
                        score_col="mean_same_pair_overlap_pct_of_weight",
                        title_suffix=f"Top weight statistics by mean % of weight pairs matched vs {functional_metric}",
                        top_k=top_k_metrics,
                        dpi=png_dpi,
                    )
                }
            )
            figures.append(
                {
                    f"{ds}_metric_scan_heatmap_mean_pairwise_flip_fraction__{functional_metric_key}": _plot_metric_heatmap(
                        ds,
                        summary,
                        value_col="mean_same_pair_overlap_pct_of_functional",
                        title_suffix=f"Mean % of {functional_metric} pairs matched heatmap",
                        dpi=png_dpi,
                    )
                }
            )
            figures.append(
                {
                    f"{ds}_metric_pairwise_flip_count_heatmap__{functional_metric_key}": _plot_metric_flip_heatmap(
                        ds,
                        overlap_long,
                        value_col="same_pair_overlap_pct_of_functional",
                        title_suffix=f"% of {functional_metric} pairs matched per epoch per metric",
                        dpi=png_dpi,
                    )
                }
            )

            for metric_id, cached in metric_cache.items():
                metric_label = cached.get("metric_label", metric_id)
                flip_df = cached.get("flip_df", pd.DataFrame())
                overlap_df = cached.get("overlap_df", pd.DataFrame())

                if not flip_df.empty:
                    figures.append(
                        {
                            f"{ds}_weight_pairwise_flips__{functional_metric_key}__{metric_id}": _plot_metric_flip_counts(
                                ds,
                                metric_label,
                                flip_df,
                                functional_metric=functional_metric,
                                functional_flip_df=functional_flip_df,
                                overlap_df=overlap_df,
                                dpi=png_dpi,
                            )
                        }
                    )
                if overlap_df is not None and not overlap_df.empty:
                    figures.append(
                        {
                            f"{ds}_weight_functional_overlap_pct__{functional_metric_key}__{metric_id}": _plot_metric_overlap_percent(
                                ds,
                                metric_label,
                                overlap_df,
                                functional_metric=functional_metric,
                                dpi=png_dpi,
                            )
                        }
                    )
        finally:
            elapsed = perf_counter() - dataset_start
            log.info("Finished weight-rank correspondence for dataset '%s' in %.2fs", ds, elapsed)
            if show_progress:
                dataset_iterator.set_postfix_str(f"last={elapsed:.1f}s")

    return AnalysisResult(tables=tables, figures=figures)
