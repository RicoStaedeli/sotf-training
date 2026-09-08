from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StandardScalerState:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(slots=True)
class TargetScalerState:
    mean: float
    scale: float


def fit_standard_scaler(values: np.ndarray, *, enabled: bool) -> StandardScalerState:
    """Fit standard scaler.

    Args:
        values (np.ndarray): Values to process.
        enabled (bool): bool value used by the function.

    Returns:
        StandardScalerState: Result of the function.
    """
    if not enabled:
        return StandardScalerState(
            mean=np.zeros(values.shape[1], dtype=np.float32),
            scale=np.ones(values.shape[1], dtype=np.float32),
        )

    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    return StandardScalerState(mean=mean, scale=scale)


def apply_standard_scaler(values: np.ndarray, scaler: StandardScalerState) -> np.ndarray:
    """Apply standard scaler.

    Args:
        values (np.ndarray): Values to process.
        scaler (StandardScalerState): StandardScalerState value used by the function.

    Returns:
        np.ndarray: Result of the function.
    """
    return ((values - scaler.mean) / scaler.scale).astype(np.float32, copy=False)


def fit_target_scaler(values: np.ndarray, *, enabled: bool) -> TargetScalerState:
    """Fit target scaler.

    Args:
        values (np.ndarray): Values to process.
        enabled (bool): bool value used by the function.

    Returns:
        TargetScalerState: Result of the function.
    """
    if not enabled:
        return TargetScalerState(mean=0.0, scale=1.0)

    mean = float(values.mean(dtype=np.float64))
    scale = float(values.std(dtype=np.float64))
    if scale < 1.0e-6:
        scale = 1.0
    return TargetScalerState(mean=mean, scale=scale)


def apply_target_scaler(values: np.ndarray, scaler: TargetScalerState) -> np.ndarray:
    """Apply target scaler.

    Args:
        values (np.ndarray): Values to process.
        scaler (TargetScalerState): TargetScalerState value used by the function.

    Returns:
        np.ndarray: Result of the function.
    """
    return ((values - scaler.mean) / scaler.scale).astype(np.float32, copy=False)


def invert_target_scaler(values: np.ndarray, scaler: TargetScalerState) -> np.ndarray:
    """Invert target scaler.

    Args:
        values (np.ndarray): Values to process.
        scaler (TargetScalerState): TargetScalerState value used by the function.

    Returns:
        np.ndarray: Result of the function.
    """
    return (values * scaler.scale + scaler.mean).astype(np.float32, copy=False)
