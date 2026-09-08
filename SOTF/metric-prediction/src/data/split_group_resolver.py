from __future__ import annotations

import pandas as pd


def resolve_split_groups(metadata: pd.DataFrame) -> tuple[str, pd.Series] | None:
    """Resolve split groups.

    Args:
        metadata (pd.DataFrame): Metadata frame to process.

    Returns:
        tuple[str, pd.Series] | None: Result of the function.
    """
    for column in ("hyperparameter_key", "model_id"):
        if column not in metadata.columns:
            continue

        groups = _normalize_group_series(metadata[column])
        if column == "hyperparameter_key" and "model_id" in metadata.columns:
            groups = groups.fillna(_normalize_group_series(metadata["model_id"]))

        if groups.notna().all() and groups.nunique() >= 2:
            return column, groups.astype(str)

    return None


def _normalize_group_series(values: pd.Series) -> pd.Series:
    """Normalize group series.

    Args:
        values (pd.Series): Values to process.

    Returns:
        pd.Series: Computed series.
    """
    normalized = values.astype("string")
    normalized = normalized.mask(normalized.isin(["", "nan", "None", "<NA>"]), pd.NA)
    return normalized
