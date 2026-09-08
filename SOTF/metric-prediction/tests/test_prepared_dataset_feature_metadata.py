from __future__ import annotations

import unittest

import pandas as pd

from data.weightstatistic_dataset_loader import _align_feature_metadata
from scripts.data_preparation.generate_prepared_dataset import _finalize_feature_metadata


class PreparedDatasetFeatureMetadataTests(unittest.TestCase):
    def test_finalize_feature_metadata_filters_and_orders_selected_features(self) -> None:
        metadata_blocks = [
            pd.DataFrame(
                [
                    {
                        "feature": "distribution__p25__weight_only____model__",
                        "row_kind": "model",
                        "layer_name": "__model__",
                        "layer_safe_name": "__model__",
                        "layer_type": "model",
                        "metric_family": "distribution",
                        "metric": "p25",
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
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "feature": "distribution__p25__weight_only__layers__encoder_0",
                        "row_kind": "layers",
                        "layer_name": "encoder.0",
                        "layer_safe_name": "encoder_0",
                        "layer_type": "linear",
                        "metric_family": "distribution",
                        "metric": "p25",
                        "scope": "weight_only",
                    }
                ]
            ),
        ]

        feature_metadata = _finalize_feature_metadata(
            feature_metadata_blocks=metadata_blocks,
            feature_columns=pd.Index(
                [
                    "distribution__p25__weight_only__layers__encoder_0",
                    "distribution__p25__weight_only____model__",
                ]
            ),
        )

        self.assertEqual(
            feature_metadata.index.tolist(),
            [
                "distribution__p25__weight_only__layers__encoder_0",
                "distribution__p25__weight_only____model__",
            ],
        )
        self.assertEqual(feature_metadata.loc["distribution__p25__weight_only__layers__encoder_0", "layer_type"], "linear")
        self.assertEqual(feature_metadata.loc["distribution__p25__weight_only____model__", "row_kind"], "model")

    def test_finalize_feature_metadata_errors_on_conflicting_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting feature metadata were produced"):
            _finalize_feature_metadata(
                feature_metadata_blocks=[
                    pd.DataFrame(
                        [
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
                                "feature": "distribution__p25__weight_only__layers__encoder_0",
                                "row_kind": "layers",
                                "layer_name": "encoder.0",
                                "layer_safe_name": "encoder_0",
                                "layer_type": "conv2d",
                                "metric_family": "distribution",
                                "metric": "p25",
                                "scope": "weight_only",
                            },
                        ]
                    )
                ],
                feature_columns=pd.Index(["distribution__p25__weight_only__layers__encoder_0"]),
            )

    def test_align_feature_metadata_returns_empty_frame_when_sidecar_is_missing(self) -> None:
        aligned = _align_feature_metadata(
            feature_metadata=None,
            feature_columns=pd.Index(["distribution__p25__weight_only____model__"]),
        )

        self.assertEqual(aligned.index.tolist(), ["distribution__p25__weight_only____model__"])
        self.assertTrue(aligned.empty)

    def test_align_feature_metadata_errors_on_missing_feature_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prepared feature metadata is missing rows for selected features"):
            _align_feature_metadata(
                feature_metadata=pd.DataFrame(
                    [
                        {
                            "feature": "distribution__p25__weight_only____model__",
                            "row_kind": "model",
                        }
                    ]
                ).set_index("feature"),
                feature_columns=pd.Index(
                    [
                        "distribution__p25__weight_only____model__",
                        "distribution__p25__weight_only__layers__encoder_0",
                    ]
                ),
            )


if __name__ == "__main__":
    unittest.main()
