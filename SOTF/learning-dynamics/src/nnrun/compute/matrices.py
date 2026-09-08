from __future__ import annotations

import numpy as np
import pandas as pd



def build_metric_matrix_from_dfm(
    dfm: pd.DataFrame,
    metric_key: str,
    epoch_col: str = "training_iteration",
    model_col: str = "model_name",
    epoch_range: tuple[int, int] | None = None,
    sentinels: tuple[float, ...] = (-999.0, -999),
) -> pd.DataFrame:
    """
    Build model x epoch matrix from dfm (long format).
    Returns DataFrame index=model, columns=epochs, values=metric.
    """
    df = dfm[[model_col, epoch_col, metric_key]].copy()

    df[epoch_col] = pd.to_numeric(df[epoch_col], errors="coerce")
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
    df[metric_key] = df[metric_key].replace(list(sentinels), np.nan)
    df = df.dropna(subset=[model_col, epoch_col, metric_key])

    if epoch_range is not None:
        e_min, e_max = epoch_range
        df = df[(df[epoch_col] >= e_min) & (df[epoch_col] <= e_max)]

    mat = df.pivot_table(
        index=model_col,
        columns=epoch_col,
        values=metric_key,
        aggfunc="max",
    )

    return mat.sort_index(axis=1)