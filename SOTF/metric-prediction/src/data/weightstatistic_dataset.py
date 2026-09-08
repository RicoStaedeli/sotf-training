from __future__ import annotations

from dataclasses import dataclass, field
import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class WeightStatisticDataset:
    """Structured output of the feature-preparation pipeline.

    Attributes:
        features: Wide feature matrix indexed by ``sample_idx``.
        target: Target values aligned to ``features``.
        metadata: Additional sample-level metadata aligned to ``features``.
        feature_metadata: Additional feature-level metadata aligned to ``features.columns``.
    """

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame
    feature_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)


def _drop_sparse_and_constant_features(
    feature_table: pd.DataFrame,
    feature_cfg: DictConfig,
) -> pd.DataFrame:
    """Drop sparse and constant features.

    Args:
        feature_table (pd.DataFrame): pd.DataFrame value used by the function.
        feature_cfg (DictConfig): Configuration for feature selection.

    Returns:
        pd.DataFrame: Computed data frame.
    """

    filtered = feature_table

    max_missing_fraction = float(feature_cfg.max_missing_fraction)
    if max_missing_fraction < 1.0:
        keep_sparse = filtered.isna().mean() <= max_missing_fraction
        filtered = filtered.loc[:, keep_sparse]

    if bool(feature_cfg.drop_constant_features):
        keep_non_constant = filtered.nunique(dropna=True) > 1
        filtered = filtered.loc[:, keep_non_constant]

    return filtered
