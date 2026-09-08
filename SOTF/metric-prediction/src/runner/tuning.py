from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping

# Ray 2.53 auto-propagates `uv run` into worker runtime environments. In this
# project that leads to mixed Ray installations during worker bootstrap, so
# workers crash before they can register with the local raylet. Default to the
# current interpreter unless the user explicitly opts back into uv propagation.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
_RAY_TMPDIR_ENV_VAR = "METRIC_PREDICTION_RAY_TMPDIR"
_RAY_SOCKET_PATH_MAX_LENGTH = 107
_RAY_SOCKET_PATH_SUFFIX = "session_2000-01-01_00-00-00_000000_00000/sockets/plasma_store"

from data.dataset_splitter import DatasetSplit
from data.feature_preprocessing import FeaturePreprocessor
from models import create_regressor
from utils.metrics import compute_regression_metrics

log = logging.getLogger(__name__)
_FAILED_TRIAL_RMSE = 1.0e18
_FAILED_TRIAL_MAE = 1.0e18
_FAILED_TRIAL_R2 = -1.0e18


def run_ray_tune(
    *,
    model_cfg: Mapping[str, Any],
    tune_cfg: Mapping[str, Any],
    split: DatasetSplit,
    fill_strategy: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run ray tune.

    Args:
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        tune_cfg (Mapping[str, Any]): Configuration object used by the function.
        split (DatasetSplit): Dataset split to process.
        fill_strategy (str): str value used by the function.
        output_dir (Path): Output directory for artifacts.

    Returns:
        Dict[str, Any]: Result of the function.
    """
    ray, air, tune = _import_ray_modules()
    search_space = _build_search_space(tune_cfg.get("search_space", {}), tune_module=tune)
    if not search_space:
        raise ValueError("Ray Tune is enabled but tune.search_space is empty.")

    # Fail fast on local dependency issues before Ray starts trial workers.
    create_regressor(model_cfg)

    resources = dict(tune_cfg.get("resources_per_trial", {}))
    model_cfg = _limit_model_parallelism(model_cfg, resources_per_trial=resources)
    metric_name = str(tune_cfg["metric"])
    mode = str(tune_cfg["mode"])
    max_concurrent_trials = tune_cfg.get("max_concurrent_trials")

    log.info(
        "Initializing Ray Tune: experiment=%s metric=%s mode=%s num_samples=%d",
        tune_cfg["experiment_name"],
        metric_name,
        mode,
        int(tune_cfg["num_samples"]),
    )
    log.info("Ray Tune search space keys: %s", sorted(search_space))
    ray_temp_dir = _resolve_ray_temp_dir(output_dir)
    log.info("Using Ray temp directory: %s", ray_temp_dir)

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        _temp_dir=str(ray_temp_dir),
    )
    try:
        trainable = tune.with_parameters(
            _objective,
            split=split,
            model_cfg=model_cfg,
            fill_strategy=fill_strategy,
        )
        if resources:
            trainable = tune.with_resources(trainable, resources=resources)

        tune_config = tune.TuneConfig(
            metric=metric_name,
            mode=mode,
            num_samples=int(tune_cfg["num_samples"]),
            max_concurrent_trials=None if max_concurrent_trials is None else int(max_concurrent_trials),
        )
        run_config = air.RunConfig(
            name=str(tune_cfg["experiment_name"]),
            storage_path=str((output_dir / "ray_results").resolve()),
        )

        tuner = tune.Tuner(
            trainable=trainable,
            tune_config=tune_config,
            run_config=run_config,
            param_space=search_space,
        )
        results = tuner.fit()
        if results.num_errors:
            log.warning("Ray Tune recorded %d errored trial(s)", results.num_errors)
        best_result = results.get_best_result(metric=metric_name, mode=mode)
        if int(best_result.metrics.get("fit_failed", 0)):
            raise RuntimeError(
                f"All Ray Tune trials failed for experiment '{tune_cfg['experiment_name']}'. "
                "Inspect the Ray trial error logs for the underlying model failures."
            )
    finally:
        ray.shutdown()
        log.info("Ray runtime shut down")

    best_params = {key: _coerce_scalar(value) for key, value in best_result.config.items()}
    summary = {
        "best_params": best_params,
    }
    for metric_key in ("rmse", "mae", "r2"):
        if metric_key in best_result.metrics:
            summary[metric_key] = float(best_result.metrics[metric_key])
    log.info("Ray Tune best result: %s", summary)
    return summary


def _objective(
    sampled_params: Dict[str, Any],
    *,
    split: DatasetSplit,
    model_cfg: Mapping[str, Any],
    fill_strategy: str,
) -> None:
    """Objective.

    Args:
        sampled_params (Dict[str, Any]): Dict[str, Any] value used by the function.
        split (DatasetSplit): Dataset split to process.
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        fill_strategy (str): str value used by the function.

    Returns:
        None: None.
    """
    _, _, tune = _import_ray_modules()
    try:
        resolved_model_cfg = deepcopy(dict(model_cfg))
        resolved_model_cfg["params"] = deepcopy(dict(resolved_model_cfg.get("params", {})))
        resolved_model_cfg["params"].update(sampled_params)

        preprocessor = FeaturePreprocessor(strategy=fill_strategy)
        X_train = preprocessor.fit_transform(split.X_train)
        X_val = preprocessor.transform(split.X_val)

        model = create_regressor(resolved_model_cfg)
        model.fit(X_train, split.y_train)
        predictions = model.predict(X_val)
        metrics = compute_regression_metrics(split.y_val.to_numpy(), predictions)
        tune.report(
            {
                **metrics,
                "fit_failed": 0,
            }
        )
    except Exception as exc:
        log.warning(
            "Ray Tune trial failed for model '%s' with sampled params %s: %s",
            model_cfg.get("name", "unknown"),
            sampled_params,
            exc,
        )
        tune.report(
            {
                "rmse": _FAILED_TRIAL_RMSE,
                "mae": _FAILED_TRIAL_MAE,
                "r2": _FAILED_TRIAL_R2,
                "fit_failed": 1,
            }
        )


def _build_search_space(
    raw_search_space: Mapping[str, Mapping[str, Any]],
    *,
    tune_module: Any,
) -> Dict[str, Any]:
    """Build search space.

    Args:
        raw_search_space (Mapping[str, Mapping[str, Any]]): Mapping[str, Mapping[str, Any]] value used by the function.
        tune_module (Any): Value used by the function.

    Returns:
        Dict[str, Any]: Result of the function.
    """
    return {
        str(param_name): _make_domain(param_name, spec, tune_module=tune_module)
        for param_name, spec in raw_search_space.items()
    }


def _make_domain(param_name: str, spec: Mapping[str, Any], *, tune_module: Any) -> Any:
    """Make domain.

    Args:
        param_name (str): str value used by the function.
        spec (Mapping[str, Any]): Mapping[str, Any] value used by the function.
        tune_module (Any): Value used by the function.

    Returns:
        Any: Result of the function.
    """
    sampler = str(spec["sampler"]).lower()

    if sampler == "choice":
        return tune_module.choice(list(spec["values"]))
    if sampler == "uniform":
        return tune_module.uniform(float(spec["lower"]), float(spec["upper"]))
    if sampler == "loguniform":
        return tune_module.loguniform(float(spec["lower"]), float(spec["upper"]))
    if sampler == "randint":
        return tune_module.randint(int(spec["lower"]), int(spec["upper"]))

    raise ValueError(f"Unsupported Ray Tune sampler '{sampler}' for parameter '{param_name}'.")


def _coerce_scalar(value: Any) -> Any:
    """Coerce scalar.

    Args:
        value (Any): Value to process.

    Returns:
        Any: Result of the function.
    """
    if hasattr(value, "item"):
        return value.item()
    return value


def _limit_model_parallelism(
    model_cfg: Mapping[str, Any],
    *,
    resources_per_trial: Mapping[str, Any],
) -> Dict[str, Any]:
    """Limit model parallelism.

    Args:
        model_cfg (Mapping[str, Any]): Configuration object used by the function.
        resources_per_trial (Mapping[str, Any]): Mapping[str, Any] value used by the function.

    Returns:
        Dict[str, Any]: Result of the function.
    """
    resolved_model_cfg = deepcopy(dict(model_cfg))
    params = deepcopy(dict(resolved_model_cfg.get("params", {})))
    cpu_budget = resources_per_trial.get("cpu")
    if cpu_budget is None:
        resolved_model_cfg["params"] = params
        return resolved_model_cfg

    try:
        bounded_threads = max(1, int(cpu_budget))
    except (TypeError, ValueError):
        resolved_model_cfg["params"] = params
        return resolved_model_cfg

    for parameter_name in ("n_jobs", "nthread", "num_threads"):
        if parameter_name in params:
            params[parameter_name] = min(int(params[parameter_name]), bounded_threads)

    resolved_model_cfg["params"] = params
    return resolved_model_cfg


def _import_ray_modules() -> tuple[Any, Any, Any]:
    """Import ray modules.

    Returns:
        tuple[Any, Any, Any]: Result of the function.
    """
    import ray
    from ray import air, tune

    return ray, air, tune


def _resolve_ray_temp_dir(output_dir: Path) -> Path:
    """Resolve ray temp dir.

    Args:
        output_dir (Path): Output directory for artifacts.

    Returns:
        Path: Resolved path.
    """
    configured_path = os.environ.get(_RAY_TMPDIR_ENV_VAR, "").strip()
    if configured_path:
        ray_temp_dir = _absolute_path(Path(configured_path).expanduser())
        _validate_ray_temp_dir_length(ray_temp_dir)
        ray_temp_dir.mkdir(parents=True, exist_ok=True)
        return ray_temp_dir

    resolved_output_dir = _absolute_path(output_dir)
    candidate_suffix = ".ray_mp"
    candidates = [ancestor / candidate_suffix for ancestor in reversed(resolved_output_dir.parents)]
    candidates.extend(Path(base_dir) / candidate_suffix for base_dir in ("/dev/shm", "/tmp"))
    seen_candidates: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen_candidates:
            continue
        seen_candidates.add(candidate_str)
        if not _ray_temp_dir_has_safe_socket_path(candidate):
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate

    fallback_candidate = resolved_output_dir / candidate_suffix
    _validate_ray_temp_dir_length(fallback_candidate)
    fallback_candidate.mkdir(parents=True, exist_ok=True)
    return fallback_candidate


def _absolute_path(path: Path) -> Path:
    """Absolute path.

    Args:
        path (Path): Path to process.

    Returns:
        Path: Resolved path.
    """
    return Path(os.path.abspath(os.fspath(path)))


def _ray_temp_dir_has_safe_socket_path(path: Path) -> bool:
    """Ray temp dir has safe socket path.

    Args:
        path (Path): Path to process.

    Returns:
        bool: Result of the function.
    """
    return len(str(path / _RAY_SOCKET_PATH_SUFFIX)) <= _RAY_SOCKET_PATH_MAX_LENGTH


def _validate_ray_temp_dir_length(path: Path) -> None:
    """Validate ray temp dir length.

    Args:
        path (Path): Path to process.

    Returns:
        None: None.
    """
    if _ray_temp_dir_has_safe_socket_path(path):
        return

    raise OSError(
        f"Ray temp directory path is too long for Unix socket limits: {path}. "
        f"Set {_RAY_TMPDIR_ENV_VAR} to a shorter path."
    )
