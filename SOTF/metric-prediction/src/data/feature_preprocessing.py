from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(slots=True)
class FeaturePreprocessor:
    strategy: str = "median"
    fill_values: pd.Series = field(init=False)
    columns: list[str] = field(init=False)

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit.

        Args:
            frame (pd.DataFrame): Data frame to process.

        Returns:
            "FeaturePreprocessor": Result of the function.
        """
        self.columns = list(frame.columns)
        strategy = self.strategy.lower()
        log.info("Fitting preprocessor on %d samples and %d features", len(frame), len(frame.columns))
        sanitized = _replace_infinite_values(frame)

        if strategy == "median":
            fill_values = sanitized.median(numeric_only=True)
        elif strategy == "zero":
            fill_values = pd.Series(0.0, index=sanitized.columns, dtype="float64")
        elif strategy == "none":
            fill_values = pd.Series(dtype="float64")
        else:
            raise ValueError(f"Unsupported missing value strategy '{self.strategy}'.")

        self.fill_values = fill_values.reindex(sanitized.columns).fillna(0.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform.

        Args:
            frame (pd.DataFrame): Data frame to process.

        Returns:
            pd.DataFrame: Computed data frame.
        """
        reordered = frame.reindex(columns=self.columns)
        sanitized = _replace_infinite_values(reordered)
        if self.strategy.lower() == "none":
            if sanitized.isna().any().any():
                raise ValueError("Missing or non-finite values remain but features.missing_value_fill=none.")
            return sanitized
        missing_values = int(sanitized.isna().sum().sum())
        if missing_values > 0:
            log.info("Imputing %d missing feature values using strategy '%s'", missing_values, self.strategy)
        return sanitized.fillna(self.fill_values)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit transform.

        Args:
            frame (pd.DataFrame): Data frame to process.

        Returns:
            pd.DataFrame: Computed data frame.
        """
        return self.fit(frame).transform(frame)


def _replace_infinite_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values.

    Args:
        frame (pd.DataFrame): Data frame to process.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    infinite_mask = np.isinf(frame.to_numpy(dtype="float64", copy=False))
    infinite_values = int(infinite_mask.sum())
    if infinite_values == 0:
        return frame

    log.info("Replacing %d infinite feature values with NaN before preprocessing", infinite_values)
    return frame.replace([np.inf, -np.inf], np.nan)
