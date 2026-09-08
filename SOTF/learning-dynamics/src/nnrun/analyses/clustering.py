from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import logging

from .base import AnalysisResult
from ..compute.matrices import build_metric_matrix_from_dfm
from ..compute.features import extract_curve_features
from ..compute.clusters import cluster_models_by_features
from ..viz.curves import plot_cluster_median_curves
from ..zoo_info import zoo_plot_name

log = logging.getLogger(__name__)

def run_clustering(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    metric = config["metric"]
    png_dpi = config.get("png_dpi", 100)
    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")
    epoch_range = tuple(config.get("epoch_range", [])) or None

    k = int(config.get("k", 5))
    tag = config.get("tag", "")

    figures: List[Dict[str, plt.Figure]] = []
    tables: Dict[str, pd.DataFrame] = {}

    datasets = sorted(df["dataset_name"].dropna().unique().tolist())
    for ds in datasets:
        log.info(f"Cluster analyzing dataset: {ds}")
        dfd = df[df["dataset_name"] == ds]
        mat = build_metric_matrix_from_dfm(
            dfd, metric_key=metric, epoch_col=epoch_col, model_col=model_col, epoch_range=epoch_range
        )
        if mat.shape[1] < 2 or mat.shape[0] < 2:
            continue

        feat = extract_curve_features(mat)
        labels, feat_used, scaler, kmeans, sil, k = cluster_models_by_features(feat, k=k, random_state=42)

        # Final table: curve features + assigned cluster
        final_table = feat.copy()
        final_table["cluster"] = labels

        tables[f"{ds}_cluster_curve_features_{metric}_{tag}"] = final_table
        tables[f"{ds}_cluster_features_used_{metric}_{tag}"] = feat_used

        fig = plot_cluster_median_curves(
            mat=mat,
            labels=labels,
            ylabel=metric,
            title=f"{zoo_plot_name(ds)} | Cluster median curves with IQR",
            figsize=(12, 7),
            dpi=png_dpi,
            show_median=True,
            show_iqr=True,
        )
        fig_name = f"{ds}_clusters_{metric}_{tag}"
        figures.append({fig_name: fig})

    return AnalysisResult(tables=tables, figures=figures)