from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cluster_median_curves(
    mat: pd.DataFrame,
    labels: pd.Series,
    ylabel: str,
    title: str,
    figsize=(10, 6),
    dpi=180,
    show_median=True,
    show_iqr=True,
    show_mean=False,
    show_std=False,
    min_models_per_cluster=1,
):
    mat = mat.sort_index(axis=1)
    epochs = mat.columns.to_numpy()

    cluster_sizes = labels.value_counts().sort_values(ascending=False)
    cluster_order = cluster_sizes.index.tolist()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for c in cluster_order:
        members = labels.index[labels == c]
        if len(members) < min_models_per_cluster:
            continue

        M = mat.loc[members].to_numpy(dtype=float)
        med = np.median(M, axis=0)
        mean = np.mean(M, axis=0)
        std = np.std(M, axis=0)

        if show_median:
            ax.plot(epochs, med, linewidth=2, label=f"cluster {c} median (n={len(members)})")

        if show_mean:
            ax.plot(epochs, mean, linestyle="--", linewidth=1.8, label=f"cluster {c} mean (n={len(members)})")

        if show_iqr:
            q25 = np.quantile(M, 0.25, axis=0)
            q75 = np.quantile(M, 0.75, axis=0)
            ax.fill_between(epochs, q25, q75, alpha=0.15)

        if show_std:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.10)

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig