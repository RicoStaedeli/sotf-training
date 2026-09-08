from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.config_utils import register_omegaconf_resolvers

from pipeline.batch import parse_batch_config, run_batch_config
from simulation.runner import run_simulation
from utils.settings import simulation_config_from_omegaconf

register_omegaconf_resolvers()


@hydra.main(version_base=None, config_path="pkg://config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the function.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        None: None.
    """
    if _is_experiment_plan(cfg):
        run_batch_config(parse_batch_config(cfg), config_source_path=None)
        return

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    simulation_cfg = simulation_config_from_omegaconf(cfg, run_dir=str(output_dir))
    result = run_simulation(simulation_cfg)
    summary = result["summary"]

    print("Simulation summary:")
    print(f"  predictor: {summary['predictor']}")
    print(f"  rounds_executed: {summary['rounds_executed']}")
    print(f"  initial_population: {summary['initial_population']}")
    print(f"  final_population: {summary['final_population']}")
    print(f"  retained_oracle_best: {summary['retained_oracle_best']}")
    if summary["oracle_top_k_recall"] is not None:
        print(f"  oracle_top_k_recall: {summary['oracle_top_k_recall']:.6f}")
    if summary["mean_survivor_rank_gap"] is not None:
        print(f"  mean_survivor_rank_gap: {summary['mean_survivor_rank_gap']:.6f}")
    if summary["best_survivor_model_id"] is not None:
        print(f"  best_survivor_model_id: {summary['best_survivor_model_id']}")
        print(f"  best_survivor_final_target: {summary['best_survivor_final_target']:.6f}")
    print(f"Artifacts: {result['artifacts_dir']}")


def _is_experiment_plan(cfg: DictConfig) -> bool:
    """Detect Hydra experiment-plan configs in either global or config-group package form.

    Args:
        cfg (DictConfig): Configuration object used by the function.

    Returns:
        bool: Boolean result.
    """
    if "runs" in cfg:
        return True
    experiments_cfg = cfg.get("experiments")
    return isinstance(experiments_cfg, DictConfig) and "runs" in experiments_cfg


if __name__ == "__main__":
    main()
