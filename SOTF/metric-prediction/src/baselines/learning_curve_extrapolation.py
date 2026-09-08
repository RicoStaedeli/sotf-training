from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Callable

import numpy as np
import pandas as pd


CurveFunction = Callable[..., np.ndarray]


@dataclass(frozen=True, slots=True)
class CurveFamily:
    name: str
    function: CurveFunction
    initial_params: Callable[[np.ndarray, np.ndarray, tuple[float, float]], list[float]]
    bounds: Callable[[np.ndarray, np.ndarray, tuple[float, float]], tuple[list[float], list[float]]]
    min_points: int


@dataclass(frozen=True, slots=True)
class ExtrapolationResult:
    prediction: float
    method: str
    n_curve_predictions: int


class LearningCurveExtrapolator:
    """Parametric learning-curve extrapolation baseline.

    The predictor fits a small ensemble of monotone and weakly constrained curve
    families to the observed prefix of one model's learning curve and returns the
    median extrapolated value. This follows the baseline spirit of parametric
    learning-curve extrapolation while keeping the implementation deterministic
    and robust for small checkpoint histories.
    """

    def __init__(
        self,
        *,
        target_bounds: tuple[float, float],
        min_points: int = 3,
        max_fit_evaluations: int = 5000,
    ) -> None:
        lower, upper = target_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid target_bounds={target_bounds!r}.")
        self.target_bounds = (float(lower), float(upper))
        self.min_points = int(min_points)
        self.max_fit_evaluations = int(max_fit_evaluations)
        self.curve_families = default_curve_families()

    def predict_one(
        self,
        history_epochs: np.ndarray,
        history_values: np.ndarray,
        *,
        target_epoch: float,
        fallback_value: float,
    ) -> ExtrapolationResult:
        x_raw, y = _clean_history(history_epochs, history_values)
        fallback = self._clip_prediction(float(fallback_value))
        if len(y) < self.min_points:
            return ExtrapolationResult(fallback, "fallback_too_few_points", 0)

        first_epoch = float(x_raw.min())
        x = x_raw - first_epoch + 1.0
        x_target = float(target_epoch) - first_epoch + 1.0
        if not math.isfinite(x_target) or x_target <= 0.0:
            return ExtrapolationResult(fallback, "fallback_invalid_target_epoch", 0)

        predictions: list[float] = []
        methods: list[str] = []
        for family in self.curve_families:
            if len(y) < max(self.min_points, family.min_points):
                continue
            pred = self._fit_family(family, x, y, x_target)
            if pred is None:
                continue
            predictions.append(pred)
            methods.append(family.name)

        if not predictions:
            return ExtrapolationResult(fallback, "fallback_fit_failed", 0)

        prediction = self._clip_prediction(float(np.median(predictions)))
        return ExtrapolationResult(prediction, "+".join(methods), len(predictions))

    def _fit_family(
        self,
        family: CurveFamily,
        x: np.ndarray,
        y: np.ndarray,
        x_target: float,
    ) -> float | None:
        try:
            from scipy.optimize import curve_fit
        except Exception as exc:  # pragma: no cover - exercised only without SciPy.
            raise RuntimeError(
                "LearningCurveExtrapolator requires scipy.optimize.curve_fit. "
                "Install scipy or use the fallback-only baseline."
            ) from exc

        try:
            p0 = family.initial_params(x, y, self.target_bounds)
            bounds = family.bounds(x, y, self.target_bounds)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                params, _ = curve_fit(
                    family.function,
                    x,
                    y,
                    p0=p0,
                    bounds=bounds,
                    maxfev=self.max_fit_evaluations,
                )
            pred = float(family.function(np.asarray([x_target], dtype="float64"), *params)[0])
        except Exception:
            return None

        if not math.isfinite(pred):
            return None
        return self._clip_prediction(pred)

    def _clip_prediction(self, value: float) -> float:
        lower, upper = self.target_bounds
        return float(np.clip(value, lower, upper))


def predict_forecast_targets(
    *,
    full_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    target_column: str,
    target_mode: str,
    horizon: int,
    target_bounds: tuple[float, float] | None = None,
    min_points: int = 3,
) -> pd.DataFrame:
    """Predict validation targets from observed learning-curve prefixes."""
    if target_column not in full_metadata.columns:
        raise ValueError(f"full_metadata is missing target column '{target_column}'.")
    required_columns = {"model_id", "epoch", "current_target", "future_target"}
    missing = sorted(required_columns.difference(validation_metadata.columns))
    if missing:
        raise ValueError("validation_metadata is missing required columns: " + ", ".join(missing))

    bounds = target_bounds or infer_target_bounds(
        pd.to_numeric(full_metadata[target_column], errors="coerce").to_numpy(dtype="float64"),
        target_column=target_column,
    )
    extrapolator = LearningCurveExtrapolator(target_bounds=bounds, min_points=min_points)
    curve_frame = _curve_frame(full_metadata, target_column=target_column)
    curves_by_model = {
        str(model_id): model_curve
        for model_id, model_curve in curve_frame.groupby("model_id", sort=False)
    }

    rows: list[dict[str, object]] = []
    for sample_idx, row in validation_metadata.iterrows():
        model_id = str(row["model_id"])
        current_epoch = float(row["epoch"])
        target_epoch = _target_epoch(row, target_mode=target_mode, horizon=horizon)
        model_curve = curves_by_model.get(model_id, curve_frame.iloc[0:0])
        observed = model_curve[model_curve["epoch"] <= current_epoch]
        result = extrapolator.predict_one(
            observed["epoch"].to_numpy(dtype="float64"),
            observed[target_column].to_numpy(dtype="float64"),
            target_epoch=target_epoch,
            fallback_value=float(row["current_target"]),
        )
        rows.append(
            {
                "sample_idx": sample_idx,
                "prediction": result.prediction,
                "baseline_method": result.method,
                "n_curve_predictions": result.n_curve_predictions,
                "target_epoch": target_epoch,
            }
        )

    prediction_frame = pd.DataFrame(rows).set_index("sample_idx")
    prediction_frame.index.name = "sample_idx"
    return prediction_frame


def infer_target_bounds(values: np.ndarray, *, target_column: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot infer target bounds from an empty/non-finite target array.")

    observed_min = float(np.min(finite))
    observed_max = float(np.max(finite))
    span = max(observed_max - observed_min, 1e-6)
    lower = observed_min - 0.1 * span
    upper = observed_max + 0.1 * span

    normalized_name = target_column.lower()
    if "acc" in normalized_name:
        lower = max(0.0, lower)
        if observed_max <= 1.5:
            upper = min(1.0, upper)
        elif observed_max <= 105.0:
            upper = min(100.0, upper)
    elif "loss" in normalized_name:
        lower = max(0.0, lower)

    if lower >= upper:
        upper = lower + 1e-6
    return float(lower), float(upper)


def default_curve_families() -> tuple[CurveFamily, ...]:
    return (
        CurveFamily("linear", _linear, _linear_initial, _linear_bounds, 2),
        CurveFamily("log", _logarithmic, _log_initial, _log_bounds, 2),
        CurveFamily("exponential", _exponential_saturation, _saturation_initial, _saturation_bounds, 3),
        CurveFamily("power", _power_saturation, _saturation_initial, _saturation_bounds, 3),
    )


def _curve_frame(full_metadata: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    required = {"model_id", "epoch", target_column}
    missing = sorted(required.difference(full_metadata.columns))
    if missing:
        raise ValueError("full_metadata is missing required columns: " + ", ".join(missing))
    frame = full_metadata.loc[:, ["model_id", "epoch", target_column]].copy()
    frame["model_id"] = frame["model_id"].astype(str)
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=["model_id", "epoch", target_column])
    return frame.sort_values(["model_id", "epoch"], kind="stable")


def _target_epoch(row: pd.Series, *, target_mode: str, horizon: int) -> float:
    mode = str(target_mode).lower()
    if mode == "final_value":
        return float(row["max_epoch"])
    if mode == "horizon_value":
        return float(row["epoch"]) + float(horizon)
    raise ValueError(f"Unsupported target_mode={target_mode!r}.")


def _clean_history(epochs: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(epochs, dtype="float64")
    y = np.asarray(values, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return x, y
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_epochs, unique_indices = np.unique(x, return_index=True)
    return unique_epochs.astype("float64"), y[unique_indices].astype("float64")


def _linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def _logarithmic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log(np.maximum(x, 1e-9)) + b


def _exponential_saturation(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c - a * np.exp(-b * x)


def _power_saturation(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c - a * np.power(np.maximum(x, 1e-9), -b)


def _linear_initial(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> list[float]:
    slope = (float(y[-1]) - float(y[0])) / max(float(x[-1]) - float(x[0]), 1.0)
    return [slope, float(y[0])]


def _linear_bounds(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> tuple[list[float], list[float]]:
    lower, upper = bounds
    span = max(upper - lower, 1e-6)
    max_slope = span / max(float(np.ptp(x)), 1.0) * 5.0
    return [-max_slope, lower - span], [max_slope, upper + span]


def _log_initial(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> list[float]:
    log_span = max(float(np.log(max(x[-1], 1.0)) - np.log(max(x[0], 1.0))), 1.0)
    slope = (float(y[-1]) - float(y[0])) / log_span
    return [slope, float(y[0])]


def _log_bounds(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> tuple[list[float], list[float]]:
    lower, upper = bounds
    span = max(upper - lower, 1e-6)
    return [-5.0 * span, lower - span], [5.0 * span, upper + span]


def _saturation_initial(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> list[float]:
    lower, upper = bounds
    asymptote = min(upper, max(float(y[-1]), float(np.max(y))) + 0.25 * max(upper - lower, 1e-6))
    amplitude = max(asymptote - float(y[0]), 1e-6)
    return [amplitude, 0.1, asymptote]


def _saturation_bounds(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> tuple[list[float], list[float]]:
    lower, upper = bounds
    span = max(upper - lower, 1e-6)
    return [0.0, 1e-6, lower], [5.0 * span, 10.0, upper]
