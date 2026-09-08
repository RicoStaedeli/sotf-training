from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from utils.config_utils import register_omegaconf_resolvers
from pipeline.batch import parse_batch_config
from utils.settings import simulation_config_from_omegaconf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "config"
EXPERIMENT_PLANS = sorted(
    path.relative_to(CONFIG_DIR).with_suffix("").as_posix()
    for path in (CONFIG_DIR / "experiments").glob("*.yaml")
)


class SotfHydraConfigTests(unittest.TestCase):
    def test_default_config_composes(self) -> None:
        """Verify that default config composes.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name="config")

        self.assertEqual(cfg.predictor.name, "naive")
        self.assertEqual(cfg.sequence.target_mode, "horizon_value")
        self.assertEqual(cfg.selection.interval, 10)
        self.assertEqual(cfg.logging.filename, "sotf_selection.log")

    def test_experiment_config_composes(self) -> None:
        """Verify that a batch experiment plan config composes.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name="experiments/ds12_test_horizon_5_sr_05_se_10_i_5")

        plan = parse_batch_config(cfg)
        self.assertEqual(plan.batch_name, "ds12_test_horizon_5_sr_05_se_10_i_5")
        self.assertEqual(len(plan.runs), 13)
        self.assertEqual(plan.runs[0].experiment_name, "naive_baseline")
        self.assertIn("online_lstm_latent_window_representations", [run.experiment_name for run in plan.runs])

    def test_all_experiment_plan_configs_compose(self) -> None:
        """Verify that all experiment plan configs compose.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            for plan_name in EXPERIMENT_PLANS:
                cfg = compose(config_name=plan_name)
                plan = parse_batch_config(cfg)

                self.assertTrue(plan.batch_name)
                self.assertTrue(plan.runs)

    def test_forecast_model_override_parses(self) -> None:
        """Verify that forecast model override parses.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "predictor=forecast_model",
                    "predictor.model_bundle_path=/tmp/model_bundle.pkl",
                    "data.parquet_path=/tmp/example.parquet",
                    "sequence.horizon=7",
                    "selection.interval=14",
                    "features.metric_families=[distribution]",
                ],
            )

        simulation_cfg = simulation_config_from_omegaconf(cfg, run_dir="/tmp/sotf-test")
        self.assertEqual(simulation_cfg.predictor.name, "forecast_model")
        self.assertEqual(simulation_cfg.predictor.model_bundle_path, "/tmp/model_bundle.pkl")
        self.assertEqual(simulation_cfg.data.parquet_path, "/tmp/example.parquet")
        self.assertEqual(simulation_cfg.sequence.horizon, 7)
        self.assertEqual(simulation_cfg.selection.interval, 14)
        self.assertEqual(simulation_cfg.features.metric_families, ["distribution"])

    def test_prepared_dataset_pt_path_override_parses(self) -> None:
        """Verify that prepared-dataset PT override parses.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "data.pt_path=/tmp/weight_statistics_test.pt",
                    "sequence.horizon=5",
                ],
            )

        simulation_cfg = simulation_config_from_omegaconf(cfg, run_dir="/tmp/sotf-test")
        self.assertEqual(simulation_cfg.data.source, "prepared_dataset")
        self.assertIsNone(simulation_cfg.data.parquet_path)
        self.assertEqual(simulation_cfg.data.pt_path, "/tmp/weight_statistics_test.pt")

    def test_latent_forecast_data_override_parses(self) -> None:
        """Verify that latent forecast data override parses.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "predictor=forecast_model",
                    "predictor.model_bundle_path=/tmp/model_bundle.pkl",
                    "data.source=latent_forecast",
                    "data.parquet_path=/tmp/targets.parquet",
                    "data.pt_path=/tmp/latent.pt",
                    "data.target_parquet_path=/tmp/targets.parquet",
                    "sequence.horizon=5",
                ],
            )

        simulation_cfg = simulation_config_from_omegaconf(cfg, run_dir="/tmp/sotf-test")
        self.assertEqual(simulation_cfg.data.source, "latent_forecast")
        self.assertEqual(simulation_cfg.data.parquet_path, "/tmp/targets.parquet")
        self.assertEqual(simulation_cfg.data.pt_path, "/tmp/latent.pt")
        self.assertEqual(simulation_cfg.data.target_parquet_path, "/tmp/targets.parquet")

    def test_xgboost_override_parses(self) -> None:
        """Verify that xgboost override parses.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "predictor=xgboost",
                    "predictor.min_train_rows=12",
                    "predictor.min_train_models=3",
                    "predictor.include_observed_metrics=true",
                    "predictor.observed_metric_columns=[current_target,test_acc,test_loss]",
                    "predictor.params.n_estimators=25",
                    "predictor.params.random_state=7",
                    "data.parquet_path=/tmp/example.parquet",
                    "sequence.horizon=5",
                ],
            )

        simulation_cfg = simulation_config_from_omegaconf(cfg, run_dir="/tmp/sotf-test")
        self.assertEqual(simulation_cfg.predictor.name, "xgboost")
        self.assertEqual(simulation_cfg.predictor.min_train_rows, 12)
        self.assertEqual(simulation_cfg.predictor.min_train_models, 3)
        self.assertTrue(simulation_cfg.predictor.include_observed_metrics)
        self.assertEqual(
            simulation_cfg.predictor.observed_metric_columns,
            ["current_target", "test_acc", "test_loss"],
        )
        self.assertEqual(simulation_cfg.predictor.params["n_estimators"], 25)
        self.assertEqual(simulation_cfg.predictor.params["random_state"], 7)
        self.assertEqual(simulation_cfg.sequence.horizon, 5)

    def test_online_lstm_override_parses(self) -> None:
        """Verify that online_lstm override parses.

        Returns:
            None: None.
        """
        register_omegaconf_resolvers()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "predictor=online_lstm",
                    "predictor.min_train_rows=12",
                    "predictor.min_train_models=3",
                    "predictor.include_observed_metrics=true",
                    "predictor.observed_metric_columns=[current_target,test_acc,test_loss]",
                    "predictor.params.hidden_size=96",
                    "predictor.params.max_epochs=25",
                    "data.source=latent_forecast",
                    "data.parquet_path=/tmp/targets.parquet",
                    "data.pt_path=/tmp/latent_windows.pt",
                    "sequence.horizon=5",
                ],
            )

        simulation_cfg = simulation_config_from_omegaconf(cfg, run_dir="/tmp/sotf-test")
        self.assertEqual(simulation_cfg.predictor.name, "online_lstm")
        self.assertEqual(simulation_cfg.predictor.min_train_rows, 12)
        self.assertEqual(simulation_cfg.predictor.min_train_models, 3)
        self.assertTrue(simulation_cfg.predictor.include_observed_metrics)
        self.assertEqual(
            simulation_cfg.predictor.observed_metric_columns,
            ["current_target", "test_acc", "test_loss"],
        )
        self.assertEqual(simulation_cfg.predictor.params["hidden_size"], 96)
        self.assertEqual(simulation_cfg.predictor.params["max_epochs"], 25)
        self.assertEqual(simulation_cfg.predictor.params["head_hidden_layer_sizes"], [32])
        self.assertEqual(simulation_cfg.data.source, "latent_forecast")
        self.assertEqual(simulation_cfg.data.pt_path, "/tmp/latent_windows.pt")
        self.assertEqual(simulation_cfg.sequence.horizon, 5)


if __name__ == "__main__":
    unittest.main()
