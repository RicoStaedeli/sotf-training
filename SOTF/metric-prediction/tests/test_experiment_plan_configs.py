from __future__ import annotations

from hydra import compose, initialize_config_module

from runner.orchestrator import _parse_plan
from utils.config_utils import register_omegaconf_resolvers


EXPERIMENT_PLANS = [
    "experiments/forecasting/exp_1",
    "experiments/forecasting/exp_2",
    "experiments/forecasting/exp_3",
    "experiments/forecasting/exp_4",
    "experiments/forecasting/exp_5",
    "experiments/forecasting/exp_6",
    "experiments/forecasting/exp_7",
    "experiments/forecasting/exp_8",
    "experiments/forecasting/exp_9",
    "experiments/forecasting/exp_10",
    "experiments/forecasting/exp_latent",
    "experiments/prediction/exp_1a",
    "experiments/prediction/exp_1b",
    "experiments/prediction/exp_2a",
    "experiments/prediction/exp_2b",
    "experiments/prediction/exp_3a",
    "experiments/prediction/exp_3b",
    "experiments/prediction/exp_4a",
    "experiments/prediction/exp_4b",
    "experiments/prediction/exp_5",
    "experiments/prediction/exp_6",
    "experiments/prediction/exp_7",
    "experiments/prediction/exp_test",
    "experiments/prediction/test_cfg",
]


def test_all_experiment_plan_configs_compose() -> None:
    register_omegaconf_resolvers()
    with initialize_config_module(config_module="metric_prediction.config.configs", version_base=None):
        for plan_name in EXPERIMENT_PLANS:
            cfg = compose(config_name=plan_name)
            plan = _parse_plan(cfg)

            assert plan.batch_name
            assert plan.runs
