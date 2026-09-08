from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.weightstatistic_dataset import WeightStatisticDataset

from .config import SequenceFeatureConfig


@dataclass(slots=True)
class ForecastDataset:
    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame
    full_metadata: pd.DataFrame | None = None


@dataclass(slots=True)
class ForecastTargetDataset:
    target: pd.Series
    metadata: pd.DataFrame


@dataclass(slots=True)
class LatentWindowForecastDataset:
    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


WINDOW_SEQUENCE_COLUMN = "window_sequence"


def build_forecast_target_dataset(
    prepared_dataset: WeightStatisticDataset,
    sequence_cfg: SequenceFeatureConfig,
) -> ForecastTargetDataset:
    """Build forecast target dataset.

    Args:
        prepared_dataset (WeightStatisticDataset): WeightStatisticDataset value used by the function.
        sequence_cfg (SequenceFeatureConfig): SequenceFeatureConfig value used by the function.

    Returns:
        ForecastTargetDataset: Result of the function.
    """
    frame, metadata_columns = _prepare_forecast_frame(prepared_dataset)
    future_target = _future_target(
        frame,
        frame.groupby("model_id", sort=False)["current_target"],
        sequence_cfg,
    )
    valid_mask = _valid_rows(frame, future_target, sequence_cfg)
    return _finalize_forecast_targets(
        frame,
        future_target,
        valid_mask=valid_mask,
        metadata_columns=metadata_columns,
        sequence_cfg=sequence_cfg,
    )


def build_forecast_dataset(
    prepared_dataset: WeightStatisticDataset,
    sequence_cfg: SequenceFeatureConfig,
) -> ForecastDataset:
    """Build forecast dataset.

    Args:
        prepared_dataset (WeightStatisticDataset): WeightStatisticDataset value used by the function.
        sequence_cfg (SequenceFeatureConfig): SequenceFeatureConfig value used by the function.

    Returns:
        ForecastDataset: Result of the function.
    """
    frame, metadata_columns = _prepare_forecast_frame(prepared_dataset)
    feature_frame = prepared_dataset.features.loc[frame["sample_idx"]].copy()
    feature_frame.index = frame.index
    feature_frame = feature_frame.astype("float64")

    group_keys = frame["model_id"]
    grouped_target = frame.groupby("model_id", sort=False)["current_target"]
    future_target = _future_target(frame, grouped_target, sequence_cfg)

    feature_blocks: list[pd.DataFrame] = []
    feature_blocks.append(feature_frame.add_suffix("__lag_0"))
    grouped_features = feature_frame.groupby(group_keys, sort=False)

    for lag in range(1, int(sequence_cfg.max_lag) + 1):
        lagged = grouped_features.shift(lag)
        lagged.columns = [f"{column}__lag_{lag}" for column in lagged.columns]
        feature_blocks.append(lagged)

        if bool(sequence_cfg.add_delta_features):
            delta = feature_frame - grouped_features.shift(lag)
            delta.columns = [f"{column}__delta_{lag}" for column in delta.columns]
            feature_blocks.append(delta)

    time_features = pd.DataFrame(index=frame.index)
    if bool(sequence_cfg.include_time_features):
        time_features = pd.DataFrame(
            {
                "epoch": frame["epoch"].to_numpy(dtype="float64"),
                "history_length": frame["history_length"].to_numpy(dtype="float64"),
                "max_epoch": frame["max_epoch"].to_numpy(dtype="float64"),
                "epochs_remaining": frame["epochs_remaining"].to_numpy(dtype="float64"),
                "epoch_gap_prev": frame["epoch_gap_prev"].fillna(0.0).to_numpy(dtype="float64"),
                "progress_fraction": frame["progress_fraction"].to_numpy(dtype="float64"),
            },
            index=frame.index,
        )
    if bool(sequence_cfg.include_current_target):
        time_features["current_target"] = frame["current_target"].to_numpy(dtype="float64")
    if not time_features.empty:
        feature_blocks.append(time_features)

    obs_metric_blocks = _build_observed_metric_feature_blocks(
        frame=frame,
        metadata=prepared_dataset.metadata,
        observed_metric_columns=list(sequence_cfg.observed_metric_columns),
        group_keys=group_keys,
        max_lag=int(sequence_cfg.max_lag),
        add_delta_features=bool(sequence_cfg.add_delta_features),
    )
    feature_blocks.extend(obs_metric_blocks)

    design_matrix = pd.concat(feature_blocks, axis=1)
    valid_mask = _valid_rows(frame, future_target, sequence_cfg)
    targets = _finalize_forecast_targets(
        frame,
        future_target,
        valid_mask=valid_mask,
        metadata_columns=metadata_columns,
        sequence_cfg=sequence_cfg,
    )

    out_features = design_matrix.loc[valid_mask].copy()
    out_features.index = targets.metadata.index
    out_features.index.name = "sample_idx"

    if out_features.empty:
        raise ValueError("No forecasting samples remain after applying the sequence configuration.")

    return ForecastDataset(
        features=out_features,
        target=targets.target,
        metadata=targets.metadata,
        full_metadata=prepared_dataset.metadata.copy(),
    )


def build_latent_window_forecast_dataset(
    prepared_dataset: WeightStatisticDataset,
    sequence_cfg: SequenceFeatureConfig,
) -> LatentWindowForecastDataset:
    """Build a checkpoint-level dataset whose features are variable-length window sequences."""
    if WINDOW_SEQUENCE_COLUMN not in prepared_dataset.features.columns:
        raise ValueError(
            "Latent window forecasting requires a prepared feature column named "
            f"'{WINDOW_SEQUENCE_COLUMN}'."
        )

    frame, metadata_columns = _prepare_forecast_frame(prepared_dataset)
    grouped_target = frame.groupby("model_id", sort=False)["current_target"]
    future_target = _future_target(frame, grouped_target, sequence_cfg)
    valid_mask = _valid_rows(frame, future_target, sequence_cfg)
    targets = _finalize_forecast_targets(
        frame,
        future_target,
        valid_mask=valid_mask,
        metadata_columns=metadata_columns,
        sequence_cfg=sequence_cfg,
    )

    sample_index = targets.metadata.index
    sequence_series = prepared_dataset.features.loc[sample_index, WINDOW_SEQUENCE_COLUMN]
    out_features = pd.DataFrame({WINDOW_SEQUENCE_COLUMN: sequence_series.to_numpy()}, index=sample_index)

    valid_frame = frame.loc[valid_mask].copy()
    valid_frame.index = sample_index
    scalar_features = _latent_window_scalar_features(valid_frame, sequence_cfg)
    if not scalar_features.empty:
        out_features = pd.concat([out_features, scalar_features], axis=1)

    out_features.index.name = "sample_idx"
    if out_features.empty:
        raise ValueError("No latent-window forecasting samples remain after applying the sequence configuration.")
    return LatentWindowForecastDataset(features=out_features, target=targets.target, metadata=targets.metadata)


def _latent_window_scalar_features(
    frame: pd.DataFrame,
    sequence_cfg: SequenceFeatureConfig,
) -> pd.DataFrame:
    """Build optional scalar features that accompany each window sequence."""
    scalar_features = pd.DataFrame(index=frame.index)
    if bool(sequence_cfg.include_time_features):
        scalar_features = pd.DataFrame(
            {
                "epoch": frame["epoch"].to_numpy(dtype="float64"),
                "history_length": frame["history_length"].to_numpy(dtype="float64"),
                "max_epoch": frame["max_epoch"].to_numpy(dtype="float64"),
                "epochs_remaining": frame["epochs_remaining"].to_numpy(dtype="float64"),
                "epoch_gap_prev": frame["epoch_gap_prev"].fillna(0.0).to_numpy(dtype="float64"),
                "progress_fraction": frame["progress_fraction"].to_numpy(dtype="float64"),
            },
            index=frame.index,
        )
    if bool(sequence_cfg.include_current_target):
        scalar_features["current_target"] = frame["current_target"].to_numpy(dtype="float64")
    for col in sequence_cfg.observed_metric_columns:
        if col == "current_target":
            scalar_features["observed_metric__current_target__lag_0"] = frame["current_target"].to_numpy(dtype="float64")
        elif col in frame.columns:
            scalar_features[f"observed_metric__{col}__lag_0"] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64")
    return scalar_features


def _build_observed_metric_feature_blocks(
    *,
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    observed_metric_columns: list[str],
    group_keys: pd.Series,
    max_lag: int,
    add_delta_features: bool,
) -> list[pd.DataFrame]:
    """Build lag/delta feature blocks for observed performance metrics.

    Each requested column produces `observed_metric__<col>__lag_0 … lag_N` features,
    plus `__delta_1 … delta_N` when ``add_delta_features`` is True.
    ``current_target`` is resolved from ``frame``; all other columns are looked up
    from ``metadata`` by ``sample_idx`` and must be present there.
    """
    if not observed_metric_columns:
        return []

    available = []
    skipped = []
    for col in observed_metric_columns:
        if col == "current_target":
            available.append(col)
        elif col in metadata.columns:
            available.append(col)
        else:
            skipped.append(col)

    if skipped:
        import logging
        logging.getLogger(__name__).warning(
            "sequence.observed_metric_columns: skipping %s — not found in dataset metadata.", skipped
        )

    if not available:
        return []

    obs_frame = pd.DataFrame(index=frame.index)
    for col in available:
        if col == "current_target":
            obs_frame[col] = frame["current_target"].to_numpy(dtype="float64")
        else:
            raw = metadata.loc[frame["sample_idx"], col].to_numpy()
            obs_frame[col] = pd.to_numeric(raw, errors="coerce").astype("float64")

    grouped_obs = obs_frame.groupby(group_keys, sort=False)
    blocks: list[pd.DataFrame] = []

    lag0 = obs_frame.rename(columns={col: f"observed_metric__{col}__lag_0" for col in obs_frame.columns})
    blocks.append(lag0)

    for lag in range(1, max_lag + 1):
        lagged = grouped_obs.shift(lag)
        lagged.columns = pd.Index([f"observed_metric__{col}__lag_{lag}" for col in obs_frame.columns])
        blocks.append(lagged)

        if add_delta_features:
            delta = obs_frame - grouped_obs.shift(lag)
            delta.columns = pd.Index([f"observed_metric__{col}__delta_{lag}" for col in obs_frame.columns])
            blocks.append(delta)

    return blocks


def _prepare_forecast_frame(
    prepared_dataset: WeightStatisticDataset,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare forecast frame.

    Args:
        prepared_dataset (WeightStatisticDataset): WeightStatisticDataset value used by the function.

    Returns:
        tuple[pd.DataFrame, list[str]]: Result of the function.
    """
    metadata = prepared_dataset.metadata.copy()
    required_metadata_columns = {"model_id", "epoch"}
    missing_metadata = sorted(required_metadata_columns.difference(metadata.columns))
    if missing_metadata:
        raise ValueError(
            "Prepared dataset metadata is missing required forecasting columns: "
            + ", ".join(missing_metadata)
        )

    frame = metadata.copy()
    frame["model_id"] = frame["model_id"].astype(str)
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["current_target"] = prepared_dataset.target.astype("float64")
    frame["sample_idx"] = prepared_dataset.features.index
    frame = frame.dropna(subset=["model_id", "epoch", "current_target"]).copy()
    frame = frame.reset_index(drop=True)
    frame = frame.sort_values(["model_id", "epoch", "sample_idx"], kind="stable")

    duplicated = frame.duplicated(subset=["model_id", "epoch"], keep=False)
    if duplicated.any():
        preview = frame.loc[duplicated, ["model_id", "epoch"]].drop_duplicates().head(10)
        raise ValueError(
            "Forecasting requires one checkpoint row per (model_id, epoch). Duplicates found: "
            + preview.to_dict(orient="records").__repr__()
        )

    grouped_epoch = frame.groupby("model_id", sort=False)["epoch"]
    frame["history_length"] = grouped_epoch.cumcount() + 1
    frame["max_epoch"] = grouped_epoch.transform("max")
    frame["epochs_remaining"] = frame["max_epoch"] - frame["epoch"]
    frame["epoch_gap_prev"] = grouped_epoch.diff()
    epoch_values = frame["epoch"].to_numpy(dtype="float64")
    max_epoch_values = frame["max_epoch"].to_numpy(dtype="float64")
    frame["progress_fraction"] = np.divide(
        epoch_values,
        max_epoch_values,
        out=np.zeros_like(epoch_values, dtype="float64"),
        where=max_epoch_values > 0.0,
    )
    return frame, [column for column in metadata.columns if column in frame.columns]


def _finalize_forecast_targets(
    frame: pd.DataFrame,
    future_target: pd.Series,
    *,
    valid_mask: pd.Series,
    metadata_columns: list[str],
    sequence_cfg: SequenceFeatureConfig,
) -> ForecastTargetDataset:
    """Finalize forecast targets.

    Args:
        frame (pd.DataFrame): Data frame to process.
        future_target (pd.Series): pd.Series value used by the function.
        valid_mask (pd.Series): pd.Series value used by the function.
        metadata_columns (list[str]): list[str] value used by the function.
        sequence_cfg (SequenceFeatureConfig): SequenceFeatureConfig value used by the function.

    Returns:
        ForecastTargetDataset: Result of the function.
    """
    out_target = future_target.loc[valid_mask].astype("float64").rename("target")
    out_metadata = frame.loc[valid_mask, metadata_columns].copy()
    out_metadata["current_target"] = frame.loc[valid_mask, "current_target"].to_numpy(dtype="float64")
    out_metadata["future_target"] = out_target.to_numpy(dtype="float64")
    out_metadata["history_length"] = frame.loc[valid_mask, "history_length"].to_numpy(dtype="int64")
    out_metadata["max_epoch"] = frame.loc[valid_mask, "max_epoch"].to_numpy(dtype="float64")
    out_metadata["epochs_remaining"] = frame.loc[valid_mask, "epochs_remaining"].to_numpy(dtype="float64")
    out_metadata["epoch_gap_prev"] = frame.loc[valid_mask, "epoch_gap_prev"].fillna(0.0).to_numpy(dtype="float64")
    out_metadata["progress_fraction"] = frame.loc[valid_mask, "progress_fraction"].to_numpy(dtype="float64")

    if sequence_cfg.target_mode == "horizon_value":
        out_metadata["prediction_horizon"] = int(sequence_cfg.horizon)
    else:
        out_metadata["prediction_horizon"] = frame.loc[valid_mask, "epochs_remaining"].to_numpy(dtype="float64")

    sample_index = frame.loc[valid_mask, "sample_idx"].to_numpy()
    out_target.index = sample_index
    out_metadata.index = sample_index
    out_target.index.name = "sample_idx"
    out_metadata.index.name = "sample_idx"

    if out_metadata.empty:
        raise ValueError("No forecasting samples remain after applying the sequence configuration.")

    return ForecastTargetDataset(target=out_target, metadata=out_metadata)


def _future_target(
    frame: pd.DataFrame,
    grouped_target,
    sequence_cfg: SequenceFeatureConfig,
) -> pd.Series:
    """Future target.

    Args:
        frame (pd.DataFrame): Data frame to process.
        grouped_target (Any): Value used by the function.
        sequence_cfg (SequenceFeatureConfig): SequenceFeatureConfig value used by the function.

    Returns:
        pd.Series: Computed series.
    """
    mode = str(sequence_cfg.target_mode).lower()
    if mode == "final_value":
        return grouped_target.transform("last").astype("float64")
    if mode == "horizon_value":
        horizon = int(sequence_cfg.horizon)
        if horizon <= 0:
            raise ValueError("sequence.horizon must be positive when target_mode='horizon_value'.")
        # horizon is in epoch units; look up target at (model_id, current_epoch + horizon)
        # rather than shifting by checkpoint steps, so sparse datasets (every-N-th epoch)
        # and dense datasets interpret the same horizon value identically.
        key_to_target: dict[tuple[str, int], float] = dict(
            zip(
                zip(frame["model_id"], frame["epoch"].round().astype(int)),
                frame["current_target"],
            )
        )
        future_epochs = frame["epoch"].round().astype(int) + horizon
        return pd.Series(
            [key_to_target.get((mid, int(ep)), float("nan")) for mid, ep in zip(frame["model_id"], future_epochs)],
            index=frame.index,
            dtype="float64",
        )

    raise ValueError(
        "Unsupported sequence.target_mode. Expected one of: "
        "'final_value', 'horizon_value'."
    )


def _valid_rows(
    frame: pd.DataFrame,
    future_target: pd.Series,
    sequence_cfg: SequenceFeatureConfig,
) -> pd.Series:
    """Valid rows.

    Args:
        frame (pd.DataFrame): Data frame to process.
        future_target (pd.Series): pd.Series value used by the function.
        sequence_cfg (SequenceFeatureConfig): SequenceFeatureConfig value used by the function.

    Returns:
        pd.Series: Computed series.
    """
    valid_mask = future_target.notna()
    valid_mask &= frame["history_length"] >= int(sequence_cfg.min_history)

    if not bool(sequence_cfg.include_terminal_samples):
        valid_mask &= frame["epochs_remaining"] > 0

    return valid_mask.astype(bool)
