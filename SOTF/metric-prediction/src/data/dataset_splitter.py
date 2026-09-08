from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from data.weightstatistic_dataset import WeightStatisticDataset
from data.split_group_resolver import resolve_split_groups
from forecasting.dataset import ForecastDataset

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame


def split_row_indices(
    metadata: pd.DataFrame,
    *,
    validation_size: float,
    random_seed: int,
    shuffle: bool | None = None,
    split_csv_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices.

    Args:
        metadata (pd.DataFrame): Metadata frame to process.
        validation_size (float): float value used by the function.
        random_seed (int): Random seed used by the function.
        shuffle (bool | None): bool | None value used by the function.
        split_csv_path (str | Path | None): str | Path | None value used by the function.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result of the function.
    """
    if split_csv_path:
        return _split_row_indices_from_csv(metadata, split_csv_path=split_csv_path)

    resolved_groups = resolve_split_groups(metadata)
    if resolved_groups is None:
        raise ValueError(
            "Dataset metadata must contain either 'hyperparameter_key' or 'model_id' for grouped splitting."
        )
    group_column, groups = resolved_groups

    unique_groups = pd.Index(groups.astype(str).unique())
    if len(unique_groups) < 2:
        raise ValueError(
            f"Grouped forecasting split requires at least two distinct {group_column} values."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(validation_size),
        random_state=int(random_seed),
    )
    placeholder = np.zeros((len(metadata), 1), dtype=np.uint8)
    train_idx, val_idx = next(splitter.split(placeholder, groups=groups))
    log.info(
        "Dataset split complete: train=%d validation=%d split_group=%s train_groups=%d validation_groups=%d "
        "shuffle_config_ignored=%s",
        len(train_idx),
        len(val_idx),
        group_column,
        int(groups.iloc[train_idx].astype(str).nunique()),
        int(groups.iloc[val_idx].astype(str).nunique()),
        shuffle,
    )
    return train_idx, val_idx


def split_dataset(
    dataset: WeightStatisticDataset | ForecastDataset,
    *,
    validation_size: float,
    random_seed: int,
    shuffle: bool | None = None,
    split_csv_path: str | Path | None = None,
) -> DatasetSplit:
    """Split dataset.

    Args:
        dataset (WeightStatisticDataset | ForecastDataset): Dataset to process.
        validation_size (float): float value used by the function.
        random_seed (int): Random seed used by the function.
        shuffle (bool | None): bool | None value used by the function.
        split_csv_path (str | Path | None): str | Path | None value used by the function.

    Returns:
        DatasetSplit: Computed dataset split.
    """
    train_idx, val_idx = split_row_indices(
        dataset.metadata,
        validation_size=validation_size,
        random_seed=random_seed,
        shuffle=shuffle,
        split_csv_path=split_csv_path,
    )

    return DatasetSplit(
        X_train=dataset.features.iloc[train_idx].copy(),
        X_val=dataset.features.iloc[val_idx].copy(),
        y_train=dataset.target.iloc[train_idx].copy(),
        y_val=dataset.target.iloc[val_idx].copy(),
        metadata_train=dataset.metadata.iloc[train_idx].copy(),
        metadata_val=dataset.metadata.iloc[val_idx].copy(),
    )


def _split_row_indices_from_csv(
    metadata: pd.DataFrame,
    *,
    split_csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices from csv.

    Args:
        metadata (pd.DataFrame): Metadata frame to process.
        split_csv_path (str | Path): str | Path value used by the function.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result of the function.
    """
    if "model_id" not in metadata.columns:
        raise ValueError("Configured data.split_csv_path requires metadata to contain 'model_id'.")

    path = Path(split_csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured data.split_csv_path does not exist: {path}")

    exact_split_by_model_id, basename_split_by_model_id = _load_split_csv(path)
    model_ids = metadata["model_id"].astype(str)
    assigned_splits = model_ids.map(exact_split_by_model_id)
    missing_mask = assigned_splits.isna()
    if missing_mask.any():
        basename_matches = model_ids.loc[missing_mask].map(
            lambda model_id: basename_split_by_model_id.get(Path(model_id).name)
        )
        assigned_splits.loc[missing_mask] = basename_matches

    train_mask = assigned_splits == "train"
    val_mask = assigned_splits == "val"
    unmatched_count = int(assigned_splits.isna().sum())
    test_count = int((assigned_splits == "test").sum())
    ignored_count = int((~(train_mask | val_mask) & assigned_splits.notna()).sum())

    if not train_mask.any() or not val_mask.any():
        split_counts = assigned_splits.fillna("__unmatched__").value_counts(dropna=False).to_dict()
        raise ValueError(
            "CSV split did not produce both train and validation rows. "
            f"split_csv_path={path}, counts={split_counts}"
        )

    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    log.info(
        "Dataset split loaded from CSV: train=%d validation=%d train_models=%d validation_models=%d "
        "test_rows_excluded=%d unmatched_rows_excluded=%d other_rows_excluded=%d path=%s",
        len(train_idx),
        len(val_idx),
        int(model_ids.iloc[train_idx].nunique()),
        int(model_ids.iloc[val_idx].nunique()),
        test_count,
        unmatched_count,
        ignored_count - test_count,
        path,
    )
    return train_idx, val_idx


def _load_split_csv(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load split csv.

    Args:
        path (Path): Path to process.

    Returns:
        tuple[dict[str, str], dict[str, str]]: Result of the function.
    """
    exact_split_by_model_id: dict[str, str] = {}
    basename_split_by_model_id: dict[str, str] = {}
    basename_conflicts: set[str] = set()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required_columns = {"model_id", "split_name"}
        missing_columns = sorted(required_columns.difference(fieldnames))
        if missing_columns:
            raise ValueError(
                f"Split CSV is missing required columns {missing_columns}: {path}"
            )

        for row in reader:
            model_id = str(row["model_id"]).strip()
            split_name = _normalize_split_name(row["split_name"])
            if not model_id:
                continue

            previous_split = exact_split_by_model_id.get(model_id)
            if previous_split is not None and previous_split != split_name:
                raise ValueError(
                    f"Model ID appears with multiple split names in {path}: "
                    f"{model_id!r} ({previous_split}, {split_name})"
                )
            exact_split_by_model_id[model_id] = split_name

            basename = Path(model_id).name
            previous_basename_split = basename_split_by_model_id.get(basename)
            if previous_basename_split is not None and previous_basename_split != split_name:
                basename_conflicts.add(basename)
            else:
                basename_split_by_model_id[basename] = split_name

    for basename in basename_conflicts:
        basename_split_by_model_id.pop(basename, None)

    if not exact_split_by_model_id:
        raise ValueError(f"Split CSV contains no model IDs: {path}")
    return exact_split_by_model_id, basename_split_by_model_id


def _normalize_split_name(raw_split_name: str) -> str:
    """Normalize split name.

    Args:
        raw_split_name (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    split_name = str(raw_split_name).strip().lower()
    if split_name in {"train", "training"}:
        return "train"
    if split_name in {"val", "valid", "validation", "valt"}:
        return "val"
    if split_name == "test":
        return "test"
    raise ValueError(
        "Unsupported split_name in split CSV. Expected train, val, validation, or test; "
        f"received {raw_split_name!r}."
    )
