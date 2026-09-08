from __future__ import annotations

import unittest

from hydra import compose, initialize_config_module

from utils.config_utils import register_omegaconf_resolvers
from forecasting.config import forecast_config_from_omegaconf


class ForecastHydraConfigTests(unittest.TestCase):
    def test_default_forecast_config_composes(self) -> None:
        register_omegaconf_resolvers()
        with initialize_config_module(config_module="metric_prediction.config.configs", version_base=None):
            cfg = compose(config_name="config", overrides=["mode=forecasting"])

        self.assertEqual(cfg.model.name, "xgboost")
        self.assertEqual(cfg.sequence.target_mode, "final_value")
        self.assertTrue(cfg.sequence.include_time_features)
        self.assertEqual(cfg.features.metric_families, [])
        self.assertEqual(cfg.logging.filename, "metric_prediction.log")
        self.assertFalse(cfg.train.deterministic)
        self.assertFalse(cfg.train.disable_tf32)

    def test_override_metric_family_and_horizon(self) -> None:
        register_omegaconf_resolvers()
        with initialize_config_module(config_module="metric_prediction.config.configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "mode=forecasting",
                    "data.parquet_path=/tmp/example.parquet",
                    "data.epoch_filter=[3,5]",
                    "data.target_model_min_threshold=0.1",
                    "features.metric_families=[distribution]",
                    "sequence.target_mode=horizon_value",
                    "sequence.horizon=5",
                    "sequence.include_time_features=false",
                ],
            )

        forecast_cfg = forecast_config_from_omegaconf(cfg)
        self.assertEqual(forecast_cfg.data.parquet_path, "/tmp/example.parquet")
        self.assertEqual(forecast_cfg.data.epoch_filter, [3, 5])
        self.assertEqual(forecast_cfg.data.target_model_min_threshold, 0.1)
        self.assertEqual(forecast_cfg.features.metric_families, ["distribution"])
        self.assertEqual(forecast_cfg.sequence.target_mode, "horizon_value")
        self.assertEqual(forecast_cfg.sequence.horizon, 5)
        self.assertFalse(forecast_cfg.sequence.include_time_features)

    def test_reproducibility_flags_compose_into_forecast_config(self) -> None:
        register_omegaconf_resolvers()
        with initialize_config_module(config_module="metric_prediction.config.configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "mode=forecasting",
                    "train.deterministic=true",
                    "train.disable_tf32=true",
                ],
            )

        forecast_cfg = forecast_config_from_omegaconf(cfg)
        self.assertTrue(forecast_cfg.train.deterministic)
        self.assertTrue(forecast_cfg.train.disable_tf32)

    def test_mlp_model_override_composes(self) -> None:
        register_omegaconf_resolvers()
        with initialize_config_module(config_module="metric_prediction.config.configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "mode=forecasting",
                    "model=mlp",
                    "model.params.hidden_layer_sizes=[64,32]",
                    "model.params.max_iter=50",
                ],
            )

        forecast_cfg = forecast_config_from_omegaconf(cfg)
        self.assertEqual(forecast_cfg.model.name, "mlp_weight_stats")
        self.assertEqual(forecast_cfg.model.params["hidden_layer_sizes"], [64, 32])
        self.assertEqual(forecast_cfg.model.params["max_iter"], 50)


if __name__ == "__main__":
    unittest.main()
