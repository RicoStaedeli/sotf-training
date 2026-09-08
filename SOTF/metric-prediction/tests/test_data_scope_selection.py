from __future__ import annotations

import unittest

import pandas as pd
from omegaconf import OmegaConf

from data.weightstatistic_dataset_loader import _select_feature_columns


def _feature_cfg(*, include_model_metrics: bool, include_layer_metrics: bool):
    return OmegaConf.create(
        {
            "include_model_metrics": include_model_metrics,
            "include_layer_metrics": include_layer_metrics,
            "metric_families": [],
            "metrics": [],
            "scopes": [],
            "layer_types": [],
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
                "feature": "distribution__percentile_25__weight_only____model__",
                "row_kind": "model",
            },
            {
                "feature": "distribution__percentile_25__weight_only__layers__module_list_0",
                "row_kind": "layers",
            },
        ]
    ).set_index("feature")


class SelectFeatureColumnsScopeTests(unittest.TestCase):
    def test_model_only_selection_keeps_only_model_features(self) -> None:
        cfg = _feature_cfg(include_model_metrics=True, include_layer_metrics=False)

        selected = _select_feature_columns(
            [
                "distribution__percentile_25__weight_only____model__",
                "distribution__percentile_25__weight_only__layers__module_list_0",
            ],
            feature_cfg=cfg,
            feature_metadata=_feature_metadata(),
        )

        self.assertEqual(selected, ["distribution__percentile_25__weight_only____model__"])

    def test_layer_only_selection_keeps_only_layer_features(self) -> None:
        cfg = _feature_cfg(include_model_metrics=False, include_layer_metrics=True)

        selected = _select_feature_columns(
            [
                "distribution__percentile_25__weight_only____model__",
                "distribution__percentile_25__weight_only__layers__module_list_0",
            ],
            feature_cfg=cfg,
            feature_metadata=_feature_metadata(),
        )

        self.assertEqual(selected, ["distribution__percentile_25__weight_only__layers__module_list_0"])

    def test_scope_selection_errors_when_requested_scope_is_absent(self) -> None:
        cfg = _feature_cfg(include_model_metrics=False, include_layer_metrics=True)

        with self.assertRaisesRegex(ValueError, "No prepared-dataset feature columns match the requested layer-level scope"):
            _select_feature_columns(
                ["distribution__percentile_25__weight_only____model__"],
                feature_cfg=cfg,
                feature_metadata=_feature_metadata(),
            )


if __name__ == "__main__":
    unittest.main()
