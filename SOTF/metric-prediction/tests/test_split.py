from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from data.dataset_splitter import split_dataset
from data.weightstatistic_dataset import WeightStatisticDataset


def _prepared_dataset() -> WeightStatisticDataset:
    features = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        },
        index=pd.Index([100, 101, 102, 200, 201, 202], name="sample_idx"),
    )
    target = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], index=features.index, name="target")
    metadata = pd.DataFrame(
        {
            "model_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "path": ["a", "b", "c", "d", "e", "f"],
            "epoch": [50, 50, 50, 50, 50, 50],
            "hyperparameter_key": ["hp_a", "hp_a", "hp_b", "hp_b", "hp_c", "hp_c"],
        },
        index=features.index,
    )
    feature_metadata = pd.DataFrame(index=features.columns)
    feature_metadata.index.name = "feature"
    return WeightStatisticDataset(
        features=features,
        target=target,
        metadata=metadata,
        feature_metadata=feature_metadata,
    )


def test_split_dataset_groups_by_hyperparameter_key() -> None:
    features = pd.DataFrame(
        {
            "feature_a": [1.0, 1.1, 2.0, 2.1, 10.0, 10.1, 11.0, 11.1],
        },
        index=pd.Index([100, 101, 200, 201, 300, 301, 400, 401], name="sample_idx"),
    )
    target = pd.Series([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0], index=features.index, name="target")
    metadata = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"],
            "path": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "epoch": [10, 20, 10, 20, 10, 20, 10, 20],
            "hyperparameter_key": ["hp_a", "hp_a", "hp_a", "hp_a", "hp_b", "hp_b", "hp_b", "hp_b"],
        },
        index=features.index,
    )
    dataset = WeightStatisticDataset(
        features=features,
        target=target,
        metadata=metadata,
        feature_metadata=pd.DataFrame(index=features.columns),
    )
    train_cfg = OmegaConf.create(
        {
            "validation_size": 0.5,
            "random_seed": 42,
            "shuffle": True,
        }
    )

    split = split_dataset(
        dataset,
        validation_size=float(train_cfg.validation_size),
        random_seed=int(train_cfg.random_seed),
        shuffle=bool(train_cfg.shuffle),
    )

    train_models = set(split.metadata_train["model_id"].astype(str))
    val_models = set(split.metadata_val["model_id"].astype(str))
    train_keys = set(split.metadata_train["hyperparameter_key"].astype(str))
    val_keys = set(split.metadata_val["hyperparameter_key"].astype(str))
    assert train_models
    assert val_models
    assert train_keys
    assert val_keys
    assert train_models.isdisjoint(val_models)
    assert train_keys.isdisjoint(val_keys)


def test_split_dataset_requires_grouping_columns() -> None:
    dataset = _prepared_dataset()
    dataset = WeightStatisticDataset(
        features=dataset.features,
        target=dataset.target,
        metadata=dataset.metadata.drop(columns=["hyperparameter_key", "model_id"]),
        feature_metadata=dataset.feature_metadata,
    )
    train_cfg = OmegaConf.create(
        {
            "validation_size": 1 / 3,
            "random_seed": 42,
            "shuffle": True,
        }
    )

    with pytest.raises(ValueError, match="hyperparameter_key|model_id"):
        split_dataset(
            dataset,
            validation_size=float(train_cfg.validation_size),
            random_seed=int(train_cfg.random_seed),
            shuffle=bool(train_cfg.shuffle),
        )
