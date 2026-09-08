from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd

from data.feature_preprocessing import FeaturePreprocessor
from forecasting.dataset import WINDOW_SEQUENCE_COLUMN
from models.factory import create_regressor
from utils.artifacts import load_model_bundle
from forecasting.dataset import ForecastDataset

log = logging.getLogger(__name__)

_METRIC_FEATURE_PREFIX = "observed_metric__"
_warned_missing_observed_cols: set[frozenset] = set()
_DISALLOWED_OBSERVED_METRIC_COLUMNS = frozenset(
    {
        "future_target",
        "actual_future_target",
        "predicted_future_target",
        "target",
        "prediction_horizon",
        "final_target",
        "final_epoch",
    }
)


class PopulationPredictor(Protocol):
    name: str

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        """Predict the function.

        Args:
            features (pd.DataFrame): Feature frame used for prediction.
            metadata (pd.DataFrame): Metadata frame aligned with the feature rows.

        Returns:
            np.ndarray: Computed array.
        """
        ...


@dataclass(slots=True)
class LoadedForecastModelBundle:
    model: Any
    preprocessor: Any | None
    feature_columns: list[str]
    config: dict[str, Any]
    path: Path


@dataclass(slots=True)
class ForecastModelPredictor:
    bundle: LoadedForecastModelBundle
    name: str = "forecast_model"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        """Predict the function.

        Args:
            features (pd.DataFrame): Feature frame used for prediction.
            metadata (pd.DataFrame): Metadata frame aligned with the feature rows.

        Returns:
            np.ndarray: Computed array.
        """
        del metadata
        missing_columns = [column for column in self.bundle.feature_columns if column not in features.columns]
        if missing_columns:
            missing_obs = [c for c in missing_columns if c.startswith(_METRIC_FEATURE_PREFIX)]
            missing_other = [c for c in missing_columns if not c.startswith(_METRIC_FEATURE_PREFIX)]
            if missing_other:
                preview = ", ".join(missing_other[:10])
                raise ValueError(
                    "Simulation features are missing columns required by the saved forecast model: "
                    f"{preview}"
                )
            key = frozenset(missing_obs)
            if key not in _warned_missing_observed_cols:
                _warned_missing_observed_cols.add(key)
                log.warning(
                    "Forecast model bundle requires %d observed-metric feature column(s) not present "
                    "in the simulation dataset; filling with NaN: %s",
                    len(missing_obs),
                    ", ".join(missing_obs),
                )
            features = features.copy()
            for col in missing_obs:
                features[col] = np.nan

        raw_features = _select_bundle_features(
            features=features,
            feature_columns=self.bundle.feature_columns,
        )
        if self.bundle.preprocessor is None:
            predictions = self.bundle.model.predict(raw_features)
        else:
            transformed = self.bundle.preprocessor.transform(raw_features)
            predictions = self.bundle.model.predict(transformed)
        return np.asarray(predictions, dtype="float64")

    def predict_for_round(
        self,
        *,
        dataset: "ForecastDataset",
        candidate: pd.DataFrame,
        observed_until_by_model: Mapping[str, float],
    ) -> "tuple[np.ndarray, dict[str, Any]]":
        """Predict with dynamically-computed observed-metric lag/delta features.

        When the bundle was trained with ``observed_metric__*__lag_*`` /
        ``__delta_*`` columns (the ``_with_perf`` variants), those features are
        not pre-built in the simulation dataset.  This method computes them
        on-the-fly from the model histories recorded in ``dataset.metadata``
        and ``dataset.target``, then delegates to :meth:`predict`.
        """
        obs_required = [c for c in self.bundle.feature_columns if c.startswith(_METRIC_FEATURE_PREFIX)]
        if not obs_required:
            return self.predict(features=dataset.features.loc[candidate.index], metadata=candidate), {}

        augmented = _augment_features_with_lag_cols(
            dataset=dataset,
            candidate=candidate,
            observed_until_by_model=observed_until_by_model,
            required_obs_cols=obs_required,
        )
        return self.predict(features=augmented, metadata=candidate), {}


@dataclass(slots=True)
class NaivePredictor:
    strategy: str = "current_target"
    name: str = "naive"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        """Predict the function.

        Args:
            features (pd.DataFrame): Feature frame used for prediction.
            metadata (pd.DataFrame): Metadata frame aligned with the feature rows.

        Returns:
            np.ndarray: Computed array.
        """
        del features
        strategy = self.strategy.lower()
        if strategy == "current_target":
            if "current_target" not in metadata.columns:
                raise ValueError("Naive predictor requires metadata column 'current_target'.")
            return metadata["current_target"].to_numpy(dtype="float64")

        raise ValueError(f"Unsupported naive predictor strategy '{self.strategy}'.")


@dataclass(slots=True)
class OnlineXGBoostPredictor:
    model_params: dict[str, Any]
    missing_value_fill: str = "median"
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    include_observed_metrics: bool = True
    observed_metric_columns: tuple[str, ...] = (
        "current_target",
        "test_acc",
        "train_acc",
        "test_loss",
        "train_loss",
    )
    name: str = "xgboost"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        """Predict the function.

        Args:
            features (pd.DataFrame): Feature frame used for prediction.
            metadata (pd.DataFrame): Metadata frame aligned with the feature rows.

        Returns:
            np.ndarray: Computed array.
        """
        del features, metadata
        raise RuntimeError(
            "OnlineXGBoostPredictor requires per-round training context. "
            "Use predict_for_round() from the simulation loop."
        )

    def predict_for_round(
        self,
        *,
        dataset: ForecastDataset,
        candidate: pd.DataFrame,
        observed_until_by_model: Mapping[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict for round.

        Args:
            dataset (ForecastDataset): Dataset to process.
            candidate (pd.DataFrame): pd.DataFrame value used by the function.
            observed_until_by_model (Mapping[str, float]): Mapping[str, float] value used by the function.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Result of the function.
        """
        feature_matrix, metric_columns = self._build_design_matrix(dataset)
        train_mask = _observed_training_mask(dataset.metadata, observed_until_by_model)
        X_train = feature_matrix.loc[train_mask].astype("float64")
        y_train = dataset.target.loc[train_mask].astype("float64")
        train_models = int(dataset.metadata.loc[train_mask, "model_id"].astype(str).nunique())
        details = {
            "fit_strategy": "xgboost",
            "training_rows": int(len(X_train)),
            "training_models": train_models,
            "feature_count": int(feature_matrix.shape[1]),
            "metric_feature_count": int(len(metric_columns)),
            "metric_feature_columns": ",".join(_metric_feature_name(column) for column in metric_columns),
        }

        if len(X_train) < int(self.min_train_rows) or train_models < int(self.min_train_models):
            details["fit_strategy"] = f"fallback_{self.fallback_strategy}"
            predictions = _fallback_predictions(candidate, strategy=self.fallback_strategy)
            return predictions, details

        preprocessor = FeaturePreprocessor(strategy=self.missing_value_fill)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_candidate = preprocessor.transform(feature_matrix.loc[candidate.index].astype("float64"))
        model = create_regressor({"name": "xgboost", "params": dict(self.model_params)})
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_candidate)
        return np.asarray(predictions, dtype="float64"), details

    def required_metadata_columns(self) -> list[str]:
        """Resolve metadata columns.

        Returns:
            list[str]: List of string values.
        """
        return _required_metadata_columns(
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )

    def used_feature_columns(self, dataset: "ForecastDataset") -> list[str]:
        """Resolve feature columns.

        Args:
            dataset ("ForecastDataset"): Dataset to process.

        Returns:
            list[str]: List of string values.
        """
        return _used_feature_columns(
            dataset=dataset,
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )

    def _build_design_matrix(self, dataset: "ForecastDataset") -> tuple[pd.DataFrame, list[str]]:
        """Build design matrix.

        Args:
            dataset ("ForecastDataset"): Dataset to process.

        Returns:
            tuple[pd.DataFrame, list[str]]: Computed design matrix.
        """
        return _build_design_matrix(
            dataset=dataset,
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )


@dataclass(slots=True)
class OnlineLSTMPredictor:
    model_params: dict[str, Any]
    missing_value_fill: str = "median"
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    include_observed_metrics: bool = True
    observed_metric_columns: tuple[str, ...] = (
        "current_target",
        "test_acc",
        "train_acc",
        "test_loss",
        "train_loss",
    )
    name: str = "online_lstm"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        """Predict the function.

        Args:
            features (pd.DataFrame): Feature frame used for prediction.
            metadata (pd.DataFrame): Metadata frame aligned with the feature rows.

        Returns:
            np.ndarray: Computed array.
        """
        del features, metadata
        raise RuntimeError(
            "OnlineLSTMPredictor requires per-round training context. "
            "Use predict_for_round() from the simulation loop."
        )

    def predict_for_round(
        self,
        *,
        dataset: ForecastDataset,
        candidate: pd.DataFrame,
        observed_until_by_model: Mapping[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict for round.

        Args:
            dataset ("ForecastDataset"): Dataset to process.
            candidate (pd.DataFrame): pd.DataFrame value used by the function.
            observed_until_by_model (Mapping[str, float]): Mapping[str, float] value used by the function.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Result of the function.
        """
        if WINDOW_SEQUENCE_COLUMN not in dataset.features.columns:
            raise ValueError(
                "OnlineLSTMPredictor requires a latent-window forecast dataset with feature column "
                f"'{WINDOW_SEQUENCE_COLUMN}'."
            )
        feature_matrix, metric_columns = _build_design_matrix(
            dataset=dataset,
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )
        feature_matrix.insert(0, WINDOW_SEQUENCE_COLUMN, dataset.features.loc[feature_matrix.index, WINDOW_SEQUENCE_COLUMN])

        train_mask = _observed_training_mask(dataset.metadata, observed_until_by_model)
        X_train = feature_matrix.loc[train_mask].copy()
        y_train = dataset.target.loc[train_mask].astype("float64")
        train_models = int(dataset.metadata.loc[train_mask, "model_id"].astype(str).nunique())
        details = {
            "fit_strategy": "latent_window_lstm",
            "training_rows": int(len(X_train)),
            "training_models": train_models,
            "feature_count": int(feature_matrix.shape[1]),
            "metric_feature_count": int(len(metric_columns)),
            "metric_feature_columns": ",".join(_metric_feature_name(column) for column in metric_columns),
        }

        if len(X_train) < int(self.min_train_rows) or train_models < int(self.min_train_models):
            details["fit_strategy"] = f"fallback_{self.fallback_strategy}"
            predictions = _fallback_predictions(candidate, strategy=self.fallback_strategy)
            return predictions, details

        X_candidate = feature_matrix.loc[candidate.index].copy()
        X_train_processed, X_candidate_processed = _preprocess_sequence_scalar_features(
            X_train,
            X_candidate,
            missing_value_fill=self.missing_value_fill,
        )
        model = create_regressor({"name": "latent_window_lstm", "params": dict(self.model_params)})
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_candidate_processed)
        return np.asarray(predictions, dtype="float64"), details

    def required_metadata_columns(self) -> list[str]:
        """Resolve metadata columns.

        Returns:
            list[str]: List of string values.
        """
        return _required_metadata_columns(
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )

    def used_feature_columns(self, dataset: ForecastDataset) -> list[str]:
        """Resolve feature columns.

        Args:
            dataset (ForecastDataset): Dataset to process.

        Returns:
            list[str]: List of string values.
        """
        return _used_feature_columns(
            dataset=dataset,
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )


@dataclass(slots=True)
class OnlineHierarchicalTransformerPredictor:
    """Online hierarchical transformer predictor.

    Structurally identical to :class:`OnlineLSTMPredictor` but delegates to
    ``latent_hierarchical_transformer`` instead of ``latent_window_lstm``.
    Supports both flat mode (``local_window_size=1``) and hierarchical mode
    (``local_window_size>1``) via ``model_params``.
    """

    model_params: dict[str, Any]
    missing_value_fill: str = "median"
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    include_observed_metrics: bool = True
    observed_metric_columns: tuple[str, ...] = (
        "current_target",
        "test_acc",
        "train_acc",
        "test_loss",
        "train_loss",
    )
    name: str = "online_hierarchical_transformer"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        """Predict the function.

        Args:
            features (pd.DataFrame): Feature frame used for prediction.
            metadata (pd.DataFrame): Metadata frame aligned with the feature rows.

        Returns:
            np.ndarray: Computed array.
        """
        del features, metadata
        raise RuntimeError(
            "OnlineHierarchicalTransformerPredictor requires per-round training context. "
            "Use predict_for_round() from the simulation loop."
        )

    def predict_for_round(
        self,
        *,
        dataset: ForecastDataset,
        candidate: pd.DataFrame,
        observed_until_by_model: Mapping[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict for round.

        Args:
            dataset (ForecastDataset): Dataset to process.
            candidate (pd.DataFrame): Candidate rows (survivors of the current round).
            observed_until_by_model (Mapping[str, float]): Latest observed epoch per model.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Predictions and diagnostic details.
        """
        if WINDOW_SEQUENCE_COLUMN not in dataset.features.columns:
            raise ValueError(
                "OnlineHierarchicalTransformerPredictor requires a latent-window forecast dataset "
                f"with feature column '{WINDOW_SEQUENCE_COLUMN}'."
            )
        feature_matrix, metric_columns = _build_design_matrix(
            dataset=dataset,
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )
        feature_matrix.insert(0, WINDOW_SEQUENCE_COLUMN, dataset.features.loc[feature_matrix.index, WINDOW_SEQUENCE_COLUMN])

        train_mask = _observed_training_mask(dataset.metadata, observed_until_by_model)
        X_train = feature_matrix.loc[train_mask].copy()
        y_train = dataset.target.loc[train_mask].astype("float64")
        train_models = int(dataset.metadata.loc[train_mask, "model_id"].astype(str).nunique())
        details = {
            "fit_strategy": "latent_hierarchical_transformer",
            "training_rows": int(len(X_train)),
            "training_models": train_models,
            "feature_count": int(feature_matrix.shape[1]),
            "metric_feature_count": int(len(metric_columns)),
            "metric_feature_columns": ",".join(_metric_feature_name(c) for c in metric_columns),
        }

        if len(X_train) < int(self.min_train_rows) or train_models < int(self.min_train_models):
            details["fit_strategy"] = f"fallback_{self.fallback_strategy}"
            predictions = _fallback_predictions(candidate, strategy=self.fallback_strategy)
            return predictions, details

        X_candidate = feature_matrix.loc[candidate.index].copy()
        X_train_processed, X_candidate_processed = _preprocess_sequence_scalar_features(
            X_train,
            X_candidate,
            missing_value_fill=self.missing_value_fill,
        )
        model = create_regressor(
            {"name": "latent_hierarchical_transformer", "params": dict(self.model_params)}
        )
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_candidate_processed)
        return np.asarray(predictions, dtype="float64"), details

    def required_metadata_columns(self) -> list[str]:
        """Resolve metadata columns.

        Returns:
            list[str]: List of string values.
        """
        return _required_metadata_columns(
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )

    def used_feature_columns(self, dataset: ForecastDataset) -> list[str]:
        """Resolve feature columns.

        Args:
            dataset (ForecastDataset): Dataset to process.

        Returns:
            list[str]: List of string values.
        """
        return _used_feature_columns(
            dataset=dataset,
            include_observed_metrics=bool(self.include_observed_metrics),
            observed_metric_columns=self.observed_metric_columns,
        )


@dataclass(slots=True)
class OnlineLCExtrapolationPredictor:
    """Online learning-curve extrapolation predictor.

    At each decision round, fits a small ensemble of parametric curve families
    to each candidate model's observable learning-curve prefix and extrapolates
    to the target epoch. No cross-model training is required — each model is
    predicted independently.
    """

    min_points: int = 3
    fallback_strategy: str = "current_target"
    name: str = "online_lc_extrapolation"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        del features, metadata
        raise RuntimeError(
            "OnlineLCExtrapolationPredictor requires per-round context. Use predict_for_round()."
        )

    def predict_for_round(
        self,
        *,
        dataset: "ForecastDataset",
        candidate: pd.DataFrame,
        observed_until_by_model: Mapping[str, float],
    ) -> "tuple[np.ndarray, dict[str, Any]]":
        from baselines.learning_curve_extrapolation import LearningCurveExtrapolator, infer_target_bounds

        meta = dataset.metadata[["model_id", "epoch", "current_target"]].copy()
        meta["model_id"] = meta["model_id"].astype(str)
        meta["epoch"] = pd.to_numeric(meta["epoch"], errors="coerce")
        meta["current_target"] = pd.to_numeric(meta["current_target"], errors="coerce")

        all_known = pd.concat(
            [
                pd.to_numeric(dataset.target, errors="coerce"),
                meta["current_target"],
            ]
        ).dropna().to_numpy(dtype="float64")
        finite_vals = all_known[np.isfinite(all_known)]
        target_column_hint = str(getattr(dataset.target, "name", "")) or "current_target"
        bounds = infer_target_bounds(finite_vals, target_column=target_column_hint)

        extrapolator = LearningCurveExtrapolator(target_bounds=bounds, min_points=self.min_points)

        predictions: list[float] = []
        n_extrapolated = 0
        for _, row in candidate.iterrows():
            model_id = str(row["model_id"])
            current_epoch = float(row["epoch"])
            obs_until = observed_until_by_model.get(model_id, current_epoch)

            hist = meta.loc[
                (meta["model_id"] == model_id) & (meta["epoch"] <= obs_until + 1e-9)
            ].sort_values("epoch").dropna(subset=["epoch", "current_target"])

            history_epochs = hist["epoch"].to_numpy(dtype="float64")
            history_values = hist["current_target"].to_numpy(dtype="float64")

            if "prediction_horizon" in row.index:
                target_epoch = current_epoch + float(row["prediction_horizon"])
            else:
                target_epoch = current_epoch + 5.0

            fallback_val = float(row["current_target"]) if "current_target" in row.index else np.nan
            if not np.isfinite(fallback_val):
                fallback_val = float(history_values[-1]) if history_values.size > 0 else float(bounds[0])

            result = extrapolator.predict_one(
                history_epochs,
                history_values,
                target_epoch=target_epoch,
                fallback_value=fallback_val,
            )
            predictions.append(result.prediction)
            if result.n_curve_predictions > 0:
                n_extrapolated += 1

        details: dict[str, Any] = {
            "fit_strategy": "lc_extrapolation",
            "n_extrapolated": n_extrapolated,
            "n_fallback": len(predictions) - n_extrapolated,
        }
        return np.array(predictions, dtype="float64"), details


@dataclass(slots=True)
class OnlineLCNetPredictor:
    """Online LCNet-style learning-curve predictor.

    At each decision round, builds fixed-length learning-curve prefix features
    for all candidate models, trains a Bayesian neural network on the observable
    history with available labels, and predicts for the current candidates.
    Follows the LCNet baseline spirit (cross-model BNN) in a causal online setting.
    """

    model_params: dict[str, Any]
    max_history_points: int = 8
    min_train_rows: int = 10
    min_train_models: int = 2
    fallback_strategy: str = "current_target"
    name: str = "online_lcnet"

    def predict(self, *, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        del features, metadata
        raise RuntimeError(
            "OnlineLCNetPredictor requires per-round training context. Use predict_for_round()."
        )

    def predict_for_round(
        self,
        *,
        dataset: "ForecastDataset",
        candidate: pd.DataFrame,
        observed_until_by_model: Mapping[str, float],
    ) -> "tuple[np.ndarray, dict[str, Any]]":
        from baselines.lcnet import LCNetRegressor, build_lcnet_features
        from baselines.learning_curve_extrapolation import infer_target_bounds

        train_mask = _observed_training_mask(dataset.metadata, observed_until_by_model)
        train_models = int(dataset.metadata.loc[train_mask, "model_id"].astype(str).nunique())
        details: dict[str, Any] = {
            "fit_strategy": "online_lcnet",
            "training_rows": int(train_mask.sum()),
            "training_models": train_models,
        }

        if train_mask.sum() < int(self.min_train_rows) or train_models < int(self.min_train_models):
            details["fit_strategy"] = f"fallback_{self.fallback_strategy}"
            return _fallback_predictions(candidate, strategy=self.fallback_strategy), details

        all_known = pd.concat(
            [pd.to_numeric(dataset.target, errors="coerce"), pd.to_numeric(dataset.metadata["current_target"], errors="coerce")]
        ).dropna().to_numpy(dtype="float64")
        finite_vals = all_known[np.isfinite(all_known)]
        target_column_hint = str(getattr(dataset.target, "name", "")) or "current_target"
        bounds = infer_target_bounds(finite_vals, target_column=target_column_hint)

        full_meta = self._build_lcnet_metadata(dataset, observed_until_by_model)
        train_meta = self._build_lcnet_metadata(dataset, observed_until_by_model, mask=train_mask)
        candidate_meta = self._build_lcnet_metadata(dataset, observed_until_by_model, index=candidate.index)

        # Use target_mode="final_value" with max_epoch set to epoch+prediction_horizon per row,
        # so _target_epoch correctly returns the intended prediction target epoch for both modes.
        X_train = build_lcnet_features(
            full_metadata=full_meta,
            forecast_metadata=train_meta,
            target_column="current_target",
            target_mode="final_value",
            horizon=5,
            max_history_points=int(self.max_history_points),
        )
        y_train = dataset.target.loc[train_mask].astype("float64")
        X_candidate = build_lcnet_features(
            full_metadata=full_meta,
            forecast_metadata=candidate_meta,
            target_column="current_target",
            target_mode="final_value",
            horizon=5,
            max_history_points=int(self.max_history_points),
        )

        model = LCNetRegressor(**dict(self.model_params))
        model.fit(X_train, y_train)
        pred_dist = model.predict_distribution(X_candidate)
        predictions = np.clip(pred_dist.mean, bounds[0], bounds[1])
        details["feature_count"] = int(X_train.shape[1])
        return np.asarray(predictions, dtype="float64"), details

    def _build_lcnet_metadata(
        self,
        dataset: "ForecastDataset",
        observed_until_by_model: Mapping[str, float],
        *,
        mask: "pd.Series | None" = None,
        index=None,
    ) -> pd.DataFrame:
        meta = dataset.metadata.copy()
        meta["model_id"] = meta["model_id"].astype(str)
        meta["epoch"] = pd.to_numeric(meta["epoch"], errors="coerce")
        meta["current_target"] = pd.to_numeric(meta["current_target"], errors="coerce")

        # Set max_epoch = epoch + prediction_horizon so that target_mode="final_value" resolves
        # to the correct target epoch for both horizon_value and final_value SOTF configs.
        if "prediction_horizon" in meta.columns:
            ph = pd.to_numeric(meta["prediction_horizon"], errors="coerce").fillna(5.0)
        else:
            ph = pd.Series(5.0, index=meta.index)
        meta["max_epoch"] = meta["epoch"].fillna(0.0) + ph

        obs_counts: dict[str, int] = {}
        for model_id, obs_until in observed_until_by_model.items():
            obs_counts[model_id] = int(
                ((meta["model_id"] == model_id) & (meta["epoch"] <= obs_until + 1e-9)).sum()
            )
        meta["history_length"] = meta["model_id"].map(obs_counts).fillna(0).astype(float)
        global_max_epoch = float(meta["max_epoch"].dropna().max()) or 1.0
        meta["progress_fraction"] = meta["epoch"].fillna(0.0) / max(global_max_epoch, 1e-9)

        if mask is not None:
            return meta.loc[mask].copy()
        if index is not None:
            return meta.loc[index].copy()
        return meta


def load_forecast_model_bundle(
    model_bundle_path: str | Path,
    *,
    used_features_path: str | Path | None = None,
) -> LoadedForecastModelBundle:
    """Load forecast model bundle.

    Args:
        model_bundle_path (str | Path): Path to the model bundle.
        used_features_path (str | Path | None): Path to the used features.

    Returns:
        LoadedForecastModelBundle: Computed forecast model bundle.
    """
    bundle_path = Path(model_bundle_path).expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Forecast model bundle does not exist: {bundle_path}")

    try:
        payload = load_model_bundle(bundle_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading the forecast model bundle failed because the current Python environment "
            f"is missing module '{exc.name}'."
        ) from exc

    missing_keys = {"model"}.difference(payload)
    if missing_keys:
        raise ValueError(f"Forecast model bundle is missing required keys: {sorted(missing_keys)}")

    feature_columns = payload.get("feature_columns")
    if feature_columns is None:
        if used_features_path is None:
            raise ValueError(
                "Forecast model bundle is missing 'feature_columns'. Provide predictor.used_features_path."
            )
        feature_columns = _load_used_features(used_features_path)

    if not isinstance(feature_columns, list) or not all(isinstance(column, str) for column in feature_columns):
        raise ValueError("Forecast model bundle feature columns must be a list of strings.")

    preprocessor = payload.get("preprocessor")
    if preprocessor is not None:
        if not hasattr(preprocessor, "transform"):
            raise ValueError("Forecast model bundle preprocessor is missing a transform method.")

        preprocessor_columns = getattr(preprocessor, "columns", None)
        if preprocessor_columns is not None and list(preprocessor_columns) != list(feature_columns):
            raise ValueError(
                "Feature mismatch between the saved forecast model preprocessor columns and feature_columns."
            )

    bundle_config = payload.get("config", {})
    if bundle_config is None:
        bundle_config = {}
    if not isinstance(bundle_config, dict):
        raise ValueError("Forecast model bundle config must be a dict when present.")
    bundle_config = dict(bundle_config)
    sequence_config = payload.get("sequence_config")
    if "sequence" not in bundle_config and isinstance(sequence_config, dict):
        bundle_config["sequence"] = dict(sequence_config)

    return LoadedForecastModelBundle(
        model=payload["model"],
        preprocessor=preprocessor,
        feature_columns=list(feature_columns),
        config=bundle_config,
        path=bundle_path,
    )


def _load_used_features(path: str | Path) -> list[str]:
    """Load used features.

    Args:
        path (str | Path): Path to the input resource.

    Returns:
        list[str]: List of string values.
    """
    used_features_path = Path(path).expanduser().resolve()
    if not used_features_path.exists():
        raise FileNotFoundError(f"Used-features JSON does not exist: {used_features_path}")

    payload = json.loads(used_features_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {used_features_path} to contain a JSON object.")

    features = payload.get("features")
    if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
        raise ValueError(f"{used_features_path} is missing a valid string list under 'features'.")
    return list(features)


def _select_bundle_features(
    *,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Select bundle feature columns while preserving non-numeric sequence payloads."""
    selected = features.loc[:, feature_columns].copy()
    numeric_columns = [column for column in selected.columns if column != WINDOW_SEQUENCE_COLUMN]
    if numeric_columns:
        selected.loc[:, numeric_columns] = selected.loc[:, numeric_columns].astype("float64")
    return selected


def _fallback_predictions(metadata: pd.DataFrame, *, strategy: str) -> np.ndarray:
    """Fallback predictions.

    Args:
        metadata (pd.DataFrame): Metadata frame aligned with the feature rows.
        strategy (str): str value used by the function.

    Returns:
        np.ndarray: Computed array.
    """
    strategy_name = str(strategy).lower()
    if strategy_name == "current_target":
        if "current_target" not in metadata.columns:
            raise ValueError("Fallback strategy 'current_target' requires metadata column 'current_target'.")
        return metadata["current_target"].to_numpy(dtype="float64")

    raise ValueError(f"Unsupported fallback strategy '{strategy}'.")


def _available_metric_columns(
    *,
    metadata: pd.DataFrame | None,
    include_observed_metrics: bool,
    observed_metric_columns: Iterable[str],
) -> list[str]:
    """Available metric columns.

    Args:
        metadata (pd.DataFrame | None): Metadata frame aligned with the feature rows.
        include_observed_metrics (bool): bool value used by the function.
        observed_metric_columns (Iterable[str]): Iterable[str] value used by the function.

    Returns:
        list[str]: List of string values.
    """
    if not include_observed_metrics:
        return []

    normalized_columns = normalize_observed_metric_columns(observed_metric_columns)
    if metadata is None:
        return normalized_columns
    return [column for column in normalized_columns if column in metadata.columns]


def _required_metadata_columns(
    *,
    include_observed_metrics: bool,
    observed_metric_columns: Iterable[str],
) -> list[str]:
    """Resolve required metadata columns for online predictors."""
    if not include_observed_metrics:
        return []

    metric_columns = _available_metric_columns(
        metadata=None,
        include_observed_metrics=include_observed_metrics,
        observed_metric_columns=observed_metric_columns,
    )
    return [column for column in metric_columns if column != "current_target"]


def _used_feature_columns(
    *,
    dataset: ForecastDataset,
    include_observed_metrics: bool,
    observed_metric_columns: Iterable[str],
) -> list[str]:
    """Resolve feature columns consumed by an online predictor."""
    metric_columns = _available_metric_columns(
        metadata=dataset.metadata,
        include_observed_metrics=include_observed_metrics,
        observed_metric_columns=observed_metric_columns,
    )
    return list(dataset.features.columns) + [_metric_feature_name(column) for column in metric_columns]


def _build_design_matrix(
    *,
    dataset: ForecastDataset,
    include_observed_metrics: bool,
    observed_metric_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the per-round design matrix, optionally augmenting observed metrics."""
    feature_matrix = dataset.features
    if WINDOW_SEQUENCE_COLUMN in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=[WINDOW_SEQUENCE_COLUMN])
    metric_columns = _available_metric_columns(
        metadata=dataset.metadata,
        include_observed_metrics=include_observed_metrics,
        observed_metric_columns=observed_metric_columns,
    )
    if not metric_columns:
        return feature_matrix, []

    metric_frame = pd.DataFrame(index=feature_matrix.index)
    for column in metric_columns:
        numeric_values = pd.to_numeric(dataset.metadata[column], errors="coerce").astype("float64")
        metric_frame[_metric_feature_name(column)] = numeric_values.to_numpy(dtype="float64")

    return pd.concat([feature_matrix, metric_frame], axis=1), metric_columns


def _preprocess_sequence_scalar_features(
    X_train: pd.DataFrame,
    X_candidate: pd.DataFrame,
    *,
    missing_value_fill: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute only scalar side-channel features while preserving raw window sequences."""
    scalar_columns = [column for column in X_train.columns if column != WINDOW_SEQUENCE_COLUMN]
    if not scalar_columns:
        return X_train.copy(), X_candidate.copy()

    preprocessor = FeaturePreprocessor(strategy=missing_value_fill)
    train_scalar = preprocessor.fit_transform(X_train.loc[:, scalar_columns].astype("float64"))
    candidate_scalar = preprocessor.transform(X_candidate.loc[:, scalar_columns].astype("float64"))

    train_processed = X_train.copy()
    train_processed.loc[:, scalar_columns] = train_scalar
    candidate_processed = X_candidate.copy()
    candidate_processed.loc[:, scalar_columns] = candidate_scalar
    return train_processed, candidate_processed


def _parse_lag_specs(columns: list[str]) -> dict[str, tuple[str, str, int]]:
    """Parse ``observed_metric__<base>__lag_<k>`` / ``__delta_<k>`` column names.

    Returns a mapping from column name to ``(base_metric, kind, k)`` where
    ``kind`` is ``"lag"`` or ``"delta"`` and ``k`` is the integer step.
    Columns that don't match the expected pattern are silently skipped.
    """
    prefix = _METRIC_FEATURE_PREFIX
    specs: dict[str, tuple[str, str, int]] = {}
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        for kind in ("lag", "delta"):
            marker = f"__{kind}_"
            idx = suffix.rfind(marker)
            if idx == -1:
                continue
            base = suffix[:idx]
            try:
                k = int(suffix[idx + len(marker):])
            except ValueError:
                continue
            specs[col] = (base, kind, k)
            break
    return specs


def _augment_features_with_lag_cols(
    *,
    dataset: "ForecastDataset",
    candidate: pd.DataFrame,
    observed_until_by_model: Mapping[str, float],
    required_obs_cols: list[str],
) -> pd.DataFrame:
    """Augment the candidate feature slice with observed-metric lag/delta columns.

    For each column in *required_obs_cols* (e.g. ``observed_metric__train_acc__lag_2``),
    compute the value from the observable checkpoint history for each candidate model,
    mirroring the lag computation used at training time (positional shift within each
    model's sorted epoch sequence, restricted to epochs ≤ ``observed_until_by_model``).
    Delta features are computed as ``lag_0 – lag_k``.
    """
    specs = _parse_lag_specs(required_obs_cols)
    if not specs:
        return dataset.features.loc[candidate.index].copy()

    base_metrics = {base for base, _, _ in specs.values()}
    max_lag = max(k for _, _, k in specs.values())

    # Build observable history frame.
    # Use full_metadata (all epochs including pre-valid rows) when available so that
    # lag-k lookups at early decision rounds include checkpoints that predate the
    # ForecastDataset's valid-row window — matching how training lags are computed.
    obs_source = getattr(dataset, "full_metadata", None)
    if obs_source is None:
        obs_source = dataset.metadata
    meta = obs_source[["model_id", "epoch"]].copy()
    meta["model_id"] = meta["model_id"].astype(str)
    meta["epoch"] = pd.to_numeric(meta["epoch"], errors="coerce")

    target_col_name = dataset.target.name if hasattr(dataset.target, "name") else None
    for base in base_metrics:
        if base == "current_target":
            meta["current_target"] = dataset.target.reindex(meta.index).to_numpy(dtype="float64")
        elif base == target_col_name:
            meta[base] = dataset.target.reindex(meta.index).to_numpy(dtype="float64")
        elif base in obs_source.columns:
            meta[base] = pd.to_numeric(obs_source[base], errors="coerce").to_numpy(dtype="float64")

    obs_until = meta["model_id"].map(observed_until_by_model)
    obs_mask = meta["epoch"].notna() & (meta["epoch"] <= obs_until.fillna(float("-inf")) + 1e-9)
    hist = meta.loc[obs_mask].copy().sort_values(["model_id", "epoch"])

    # Lag index from end: 0 = most recent checkpoint, 1 = one back, etc.
    group_count = hist.groupby("model_id", sort=False)["epoch"].transform("count").astype(int)
    cumulative = hist.groupby("model_id", sort=False).cumcount().astype(int)
    hist["__lag"] = group_count - 1 - cumulative

    # Pre-build a lookup dict per (base, lag_k)
    lag_maps: dict[tuple[str, int], dict[str, float]] = {}
    for base in base_metrics:
        if base not in hist.columns:
            continue
        for k in range(max_lag + 1):
            rows = hist.loc[hist["__lag"] == k, ["model_id", base]]
            lag_maps[(base, k)] = rows.groupby("model_id", sort=False)[base].last().to_dict()

    candidate_model_ids = candidate["model_id"].astype(str)
    result = dataset.features.loc[candidate.index].copy()

    for col, (base, kind, k) in specs.items():
        if kind == "lag":
            lk = lag_maps.get((base, k), {})
            result[col] = candidate_model_ids.map(lk).to_numpy(dtype="float64")
        else:
            l0 = lag_maps.get((base, 0), {})
            lk = lag_maps.get((base, k), {})
            lag0_vals = candidate_model_ids.map(l0).to_numpy(dtype="float64")
            lagk_vals = candidate_model_ids.map(lk).to_numpy(dtype="float64")
            result[col] = lag0_vals - lagk_vals

    return result


def normalize_observed_metric_columns(columns: Iterable[str]) -> list[str]:
    """Normalize observed metric columns.

    Args:
        columns (Iterable[str]): Column names to process.

    Returns:
        list[str]: List of string values.
    """
    normalized: list[str] = []
    disallowed_columns: list[str] = []
    for column in columns:
        text = str(column).strip()
        if not text or text in normalized:
            continue
        normalized_lower = text.lower()
        is_disallowed = (
            normalized_lower in _DISALLOWED_OBSERVED_METRIC_COLUMNS
            or "future_target" in normalized_lower
            or normalized_lower.startswith("future_")
            or normalized_lower.startswith("actual_future_")
            or normalized_lower.startswith("predicted_future_")
        )
        if is_disallowed:
            disallowed_columns.append(text)
            continue
        normalized.append(text)

    if disallowed_columns:
        raise ValueError(
            "predictor.observed_metric_columns contains leakage-prone columns that are not allowed: "
            f"{sorted(disallowed_columns)}"
        )

    return normalized


def _metric_feature_name(metric_column: str) -> str:
    """Metric feature name.

    Args:
        metric_column (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    return f"{_METRIC_FEATURE_PREFIX}{metric_column}"


def _observed_training_mask(
    metadata: pd.DataFrame,
    observed_until_by_model: Mapping[str, float],
) -> pd.Series:
    """Observed training mask.

    Args:
        metadata (pd.DataFrame): Metadata frame aligned with the feature rows.
        observed_until_by_model (Mapping[str, float]): Mapping[str, float] value used by the function.

    Returns:
        pd.Series: Computed series.
    """
    if "prediction_horizon" not in metadata.columns:
        raise ValueError("Forecast dataset metadata is missing 'prediction_horizon', required for online training.")

    model_ids = metadata["model_id"].astype(str)
    observed_until = model_ids.map(observed_until_by_model)
    epoch_values = pd.to_numeric(metadata["epoch"], errors="coerce")
    horizon_values = pd.to_numeric(metadata["prediction_horizon"], errors="coerce")
    label_available_epoch = epoch_values + horizon_values

    return (
        observed_until.notna()
        & epoch_values.notna()
        & horizon_values.notna()
        & (label_available_epoch <= observed_until + 1e-9)
    ).astype(bool)
