from __future__ import annotations

import unittest

import pandas as pd
from omegaconf import OmegaConf

from data.weightstatistic_dataset_loader import _select_feature_columns


def _feature_cfg(
    *,
    metric_families: list[str] | None = None,
    metrics: list[str] | None = None,
    scopes: list[str] | None = None,
    layer_types: list[str] | None = None,
):
    return OmegaConf.create(
        {
            "include_model_metrics": True,
            "include_layer_metrics": True,
            "metric_families": metric_families or [],
            "metrics": metrics or [],
            "scopes": scopes or [],
            "layer_types": layer_types or [],
            "include_feature_names": [],
            "exclude_feature_names": [],
            "include_feature_patterns": [],
            "exclude_feature_patterns": [],
        }
    )


def _feature_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": "distance__l2__weight_only____model__",
                "row_kind": "model",
                "layer_name": "__model__",
                "layer_safe_name": "__model__",
                "layer_type": "model",
                "metric_family": "distance",
                "metric": "l2",
                "scope": "weight_only",
            },
            {
                "feature": "distribution__p25__weight_only__layers__encoder_0",
                "row_kind": "layers",
                "layer_name": "encoder.0",
                "layer_safe_name": "encoder_0",
                "layer_type": "linear",
                "metric_family": "distribution",
                "metric": "p25",
                "scope": "weight_only",
            },
            {
                "feature": "distribution__p25__grad_only__layers__encoder_1",
                "row_kind": "layers",
                "layer_name": "encoder.1",
                "layer_safe_name": "encoder_1",
                "layer_type": "conv2d",
                "metric_family": "distribution",
                "metric": "p25",
                "scope": "grad_only",
            },
        ]
    ).set_index("feature")


class SelectFeatureColumnsMetadataTests(unittest.TestCase):
    def test_metric_family_filter_keeps_matching_features(self) -> None:
        selected = _select_feature_columns(
            list(_feature_metadata().index),
            feature_cfg=_feature_cfg(metric_families=["distance"]),
            feature_metadata=_feature_metadata(),
        )

        self.assertEqual(selected, ["distance__l2__weight_only____model__"])

    def test_metric_scope_and_layer_type_filters_compose(self) -> None:
        selected = _select_feature_columns(
            list(_feature_metadata().index),
            feature_cfg=_feature_cfg(
                metrics=["p25"],
                scopes=["weight_only"],
                layer_types=["linear"],
            ),
            feature_metadata=_feature_metadata(),
        )

        self.assertEqual(selected, ["distribution__p25__weight_only__layers__encoder_0"])

    def test_metadata_filtering_requires_sidecar_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prepared feature metadata is required when filtering by"):
            _select_feature_columns(
                ["distance__l2__weight_only____model__"],
                feature_cfg=_feature_cfg(metric_families=["distance"]),
                feature_metadata=None,
            )

    def test_metadata_filtering_errors_when_no_features_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "No prepared-dataset feature columns match the configured metadata filters"):
            _select_feature_columns(
                list(_feature_metadata().index),
                feature_cfg=_feature_cfg(layer_types=["attention"]),
                feature_metadata=_feature_metadata(),
            )


if __name__ == "__main__":
    unittest.main()
