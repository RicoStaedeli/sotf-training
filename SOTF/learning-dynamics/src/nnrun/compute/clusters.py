from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging

log = logging.getLogger(__name__)

DEFAULT_USE_COLS = [
    "final_val", "auc_norm", "peak_drop",
    "std_delta", "early_slope", "late_slope",
    "t95_epoch", "avg_regret",
]


def _prepare_features(feat_df: pd.DataFrame, use_cols=None) -> pd.DataFrame:
    if use_cols is None:
        use_cols = DEFAULT_USE_COLS

    missing = [c for c in use_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    feat_used = feat_df[use_cols].copy()

    nan_rows_mask = feat_used.isna().any(axis=1)
    if nan_rows_mask.any():
        nan_cols = feat_used.columns[feat_used.isna().any()].tolist()
        example_idx = feat_used.index[nan_rows_mask].tolist()[:10]
        n_drop = int(nan_rows_mask.sum())

        log.warning(
            "NaN values found in feature matrix.\n"
            f"NaN columns: {nan_cols}\n"
            f"Example rows (index/model): {example_idx}\n"
            f"Dropping {n_drop} model(s) with NaNs"
        )

        feat_used = feat_used.loc[~nan_rows_mask].copy()

    log.info("Prepared features: n_models=%d, n_features=%d", len(feat_used), feat_used.shape[1])
    return feat_used

def choose_k_by_silhouette(
    feat_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    use_cols=None,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Returns a table with silhouette scores for each k in [k_min, k_max].
    """
    feat_used = _prepare_features(feat_df, use_cols=use_cols)
    n = len(feat_used)

    k_min_eff = max(2, k_min)
    k_max_eff = min(k_max, n - 1)

    if k_max_eff < k_min_eff:
        log.warning(
            "Silhouette selection not possible: n_models=%d requires 2 <= k <= n-1, got range [%d,%d].",
            n, k_min, k_max
        )
        return pd.DataFrame([{"k": np.nan, "silhouette": np.nan}])

    log.info("Choosing k by silhouette in range [%d, %d] (n_models=%d)", k_min_eff, k_max_eff, n)

    scaler = StandardScaler()
    Z = scaler.fit_transform(feat_used.to_numpy())

    rows = []
    for k in range(k_min_eff, k_max_eff + 1):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = kmeans.fit_predict(Z)

        sil = np.nan
        try:
            sil = float(silhouette_score(Z, labels))
        except Exception as e:
            log.warning("Silhouette failed for k=%d: %s", k, e)
            sil = np.nan

        rows.append({"k": k, "silhouette": sil})

    tbl = pd.DataFrame(rows).sort_values("k")
    best = tbl.dropna(subset=["silhouette"])
    if not best.empty:
        best_k = int(best.loc[best["silhouette"].idxmax(), "k"])
        best_s = float(best["silhouette"].max())
        log.info("Silhouette best: k=%d, silhouette=%.4f", best_k, best_s)

    return tbl


def cluster_models_by_features(
    feat_df: pd.DataFrame,
    k: int = 5,
    use_cols=None,
    random_state: int = 0,
    k_min: int = 2,
    k_max: int = 10,
    *,
    return_all_models: bool = True,  # optional convenience
):
    """
    Cluster models based on feature dataframe.

    If k == 0, chooses k by maximizing silhouette score on [k_min, k_max].
    Returns: labels, feat_used, scaler, kmeans, silhouette, chosen_k
    """
    feat_used = _prepare_features(feat_df, use_cols=use_cols)
    n = len(feat_used)

    if n < 2:
        log.warning("Not enough models to cluster: n_models=%d. Returning single cluster.", n)
        labels_used = pd.Series([0] * n, index=feat_used.index, name="cluster")
        if return_all_models:
            labels_all = pd.Series(np.nan, index=feat_df.index, name="cluster")
            labels_all.loc[labels_used.index] = labels_used
            return labels_all, feat_used, None, None, np.nan, 1
        return labels_used, feat_used, None, None, np.nan, 1

    if k == 0:
        sil_table = choose_k_by_silhouette(
            feat_df, k_min=k_min, k_max=k_max, use_cols=use_cols, random_state=random_state
        )
        sil_table_valid = sil_table.dropna(subset=["silhouette"])
        if sil_table_valid.empty:
            chosen_k = 2 if n >= 3 else 1
            log.warning("No valid silhouette scores; fallback chosen_k=%d", chosen_k)
        else:
            chosen_k = int(sil_table_valid.loc[sil_table_valid["silhouette"].idxmax(), "k"])
        log.info("Auto-selected k=%d", chosen_k)
    else:
        chosen_k = int(k)
        log.info("Using provided k=%d", chosen_k)

    chosen_k = max(1, min(chosen_k, n))

    scaler = StandardScaler()
    Z = scaler.fit_transform(feat_used.to_numpy())

    log.info("Clustering with KMeans: n_models=%d, chosen_k=%d", n, chosen_k)
    kmeans = KMeans(n_clusters=chosen_k, n_init=20, random_state=random_state)
    y = kmeans.fit_predict(Z)

    sil = np.nan
    if 2 <= chosen_k <= (n - 1):
        try:
            sil = float(silhouette_score(Z, y))
        except Exception as e:
            log.warning("Final silhouette failed (k=%d): %s", chosen_k, e)
            sil = np.nan

    counts = pd.Series(y).value_counts().sort_index()
    log.info("Cluster sizes: %s", counts.to_dict())
    if not np.isnan(sil):
        log.info("Final silhouette: %.4f", sil)

    # index must match y length: use feat_used.index (after dropping NaNs)
    labels_used = pd.Series(y, index=feat_used.index, name="cluster")

    # Optional: return a labels series aligned to the original input (NaN for dropped models)
    if return_all_models:
        labels_all = pd.Series(np.nan, index=feat_df.index, name="cluster")
        labels_all.loc[labels_used.index] = labels_used
        return labels_all, feat_used, scaler, kmeans, sil, chosen_k

    return labels_used, feat_used, scaler, kmeans, sil, chosen_k