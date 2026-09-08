from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from data.weightstatistic_dataset_loader import _load_feature_metadata_from_pt_payload


def _data_cfg(*, parquet_path: str | None = None):
    return OmegaConf.create(
        {
            "parquet_path": parquet_path,
            "pt_train_path": "train.pt",
            "pt_val_path": "val.pt",
            "target_column": "test_acc",
            "metadata_columns": ["model_id", "path", "epoch", "hyperparameter_key"],
            "epoch_filter": [],
            "target_model_min_threshold": None,
            "split_csv_path": None,
        }
    )


class PtFeatureMetadataLoadingTests(unittest.TestCase):
    def test_embedded_feature_metadata_is_used_without_sidecar_lookup(self) -> None:
        payload = {
            "feature_metadata": {
                "feature": ["feat_a", "feat_b"],
                "row_kind": ["model", "layers"],
                "metric_family": ["distribution", "distribution"],
            }
        }
        feature_metadata = _load_feature_metadata_from_pt_payload(
            payload,
            data_cfg=_data_cfg(),
            pt_path=Path("dummy.pt"),
            prefix="[train] ",
        )

        self.assertEqual(list(feature_metadata.index), ["feat_a", "feat_b"])
        self.assertEqual(feature_metadata.loc["feat_a", "row_kind"], "model")
        self.assertEqual(feature_metadata.loc["feat_b", "metric_family"], "distribution")

    def test_configured_parquet_path_is_used_before_pt_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            prepared_path = tmp_path / "prepared__prepared_dataset.parquet"
            prepared_path.touch()
            sidecar_path = tmp_path / "prepared__prepared_dataset__feature_metadata.parquet"
            pd.DataFrame(
                {
                    "feature": ["feat_cfg"],
                    "row_kind": ["model"],
                    "metric_family": ["distribution"],
                }
            ).to_parquet(sidecar_path, index=False)

            payload = {
                "source_prepared_dataset_path": "/definitely/missing/prepared.parquet",
            }
            feature_metadata = _load_feature_metadata_from_pt_payload(
                payload,
                data_cfg=_data_cfg(parquet_path=str(prepared_path)),
                pt_path=tmp_path / "weights.pt",
                prefix="[train] ",
            )

            self.assertEqual(list(feature_metadata.index), ["feat_cfg"])
            self.assertEqual(feature_metadata.loc["feat_cfg", "row_kind"], "model")


if __name__ == "__main__":
    unittest.main()
