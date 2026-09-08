from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from runner import tuning


class _FakeRay:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.shutdown_called = False

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeRunConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeTuneConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeBestResult:
    config = {"max_depth": 7}
    metrics = {"rmse": 0.12, "mae": 0.08, "r2": 0.91}


class _FakeResults:
    num_errors = 0

    def get_best_result(self, *, metric: str, mode: str) -> _FakeBestResult:
        assert metric == "rmse"
        assert mode == "min"
        return _FakeBestResult()


class _FakeTuner:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self) -> _FakeResults:
        return _FakeResults()


class _FakeTuneModule:
    TuneConfig = _FakeTuneConfig
    Tuner = _FakeTuner

    @staticmethod
    def choice(values: list[object]) -> tuple[str, list[object]]:
        return ("choice", values)

    @staticmethod
    def uniform(lower: float, upper: float) -> tuple[str, float, float]:
        return ("uniform", lower, upper)

    @staticmethod
    def loguniform(lower: float, upper: float) -> tuple[str, float, float]:
        return ("loguniform", lower, upper)

    @staticmethod
    def randint(lower: int, upper: int) -> tuple[str, int, int]:
        return ("randint", lower, upper)

    @staticmethod
    def with_parameters(fn: object, **kwargs: object) -> tuple[str, object, dict[str, object]]:
        return ("with_parameters", fn, kwargs)

    @staticmethod
    def with_resources(trainable: object, *, resources: dict[str, object]) -> tuple[str, object, dict[str, object]]:
        return ("with_resources", trainable, resources)

    @staticmethod
    def report(metrics: dict[str, float]) -> None:
        return None


class _AllFailedBestResult:
    config = {"max_depth": 7}
    metrics = {"rmse": 1.0e18, "mae": 1.0e18, "r2": -1.0e18, "fit_failed": 1}


class _AllFailedResults:
    num_errors = 0

    def get_best_result(self, *, metric: str, mode: str) -> _AllFailedBestResult:
        assert metric == "rmse"
        assert mode == "min"
        return _AllFailedBestResult()


class _AllFailedTuner:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self) -> _AllFailedResults:
        return _AllFailedResults()


def test_run_ray_tune_uses_output_local_temp_dir(tmp_path, monkeypatch) -> None:
    fake_ray = _FakeRay()
    fake_air = SimpleNamespace(RunConfig=_FakeRunConfig)
    fake_tune = _FakeTuneModule()
    output_dir = tmp_path / "artifacts"
    configured_temp_dir = Path("/tmp/mp-ray-test")

    monkeypatch.setattr(tuning, "_import_ray_modules", lambda: (fake_ray, fake_air, fake_tune))
    monkeypatch.setattr(tuning, "create_regressor", lambda model_cfg: object())
    monkeypatch.setenv("METRIC_PREDICTION_RAY_TMPDIR", str(configured_temp_dir))

    split = SimpleNamespace(
        X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
        X_val=pd.DataFrame({"feature": [3.0]}),
        y_train=pd.Series([1.0, 2.0]),
        y_val=pd.Series([3.0]),
    )

    result = tuning.run_ray_tune(
        model_cfg={"name": "xgboost", "params": {}},
        tune_cfg={
            "experiment_name": "xgboost_tune",
            "metric": "rmse",
            "mode": "min",
            "num_samples": 2,
            "max_concurrent_trials": 1,
            "resources_per_trial": {"cpu": 1},
            "search_space": {
                "max_depth": {
                    "sampler": "randint",
                    "lower": 3,
                    "upper": 10,
                }
            },
        },
        split=split,
        fill_strategy="mean",
        output_dir=output_dir,
    )

    expected_temp_dir = str(configured_temp_dir.resolve())
    assert fake_ray.init_calls == [
        {
            "ignore_reinit_error": True,
            "log_to_driver": False,
            "_temp_dir": expected_temp_dir,
        }
    ]
    assert configured_temp_dir.is_dir()
    assert fake_ray.shutdown_called is True
    assert result["best_params"] == {"max_depth": 7}


def test_limit_model_parallelism_respects_trial_cpu_budget() -> None:
    limited = tuning._limit_model_parallelism(
        {"name": "lightgbm", "params": {"n_jobs": 4, "learning_rate": 0.1}},
        resources_per_trial={"cpu": 1},
    )

    assert limited["params"]["n_jobs"] == 1
    assert limited["params"]["learning_rate"] == 0.1


def test_objective_reports_failed_trial_metrics(monkeypatch) -> None:
    reported_metrics: list[dict[str, float]] = []

    class _TuneReporter:
        @staticmethod
        def report(metrics: dict[str, float]) -> None:
            reported_metrics.append(metrics)

    monkeypatch.setattr(tuning, "_import_ray_modules", lambda: (object(), object(), _TuneReporter()))
    monkeypatch.setattr(
        tuning,
        "create_regressor",
        lambda model_cfg: SimpleNamespace(
            fit=lambda X, y: (_ for _ in ()).throw(ValueError("boom")),
            predict=lambda X: None,
        ),
    )

    tuning._objective(
        {"alpha": 1.0},
        split=SimpleNamespace(
            X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
            X_val=pd.DataFrame({"feature": [3.0]}),
            y_train=pd.Series([1.0, 2.0]),
            y_val=pd.Series([3.0]),
        ),
        model_cfg={"name": "ridge", "params": {}},
        fill_strategy="mean",
    )

    assert reported_metrics == [
        {
            "rmse": 1.0e18,
            "mae": 1.0e18,
            "r2": -1.0e18,
            "fit_failed": 1,
        }
    ]


def test_run_ray_tune_raises_if_all_trials_fail(tmp_path, monkeypatch) -> None:
    fake_ray = _FakeRay()
    fake_air = SimpleNamespace(RunConfig=_FakeRunConfig)
    output_dir = tmp_path / "artifacts"

    class _TuneModuleAllFailed(_FakeTuneModule):
        Tuner = _AllFailedTuner

    monkeypatch.setattr(tuning, "_import_ray_modules", lambda: (fake_ray, fake_air, _TuneModuleAllFailed()))
    monkeypatch.setattr(tuning, "create_regressor", lambda model_cfg: object())
    monkeypatch.setenv("METRIC_PREDICTION_RAY_TMPDIR", "/tmp/mp-ray-test")

    split = SimpleNamespace(
        X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
        X_val=pd.DataFrame({"feature": [3.0]}),
        y_train=pd.Series([1.0, 2.0]),
        y_val=pd.Series([3.0]),
    )

    try:
        tuning.run_ray_tune(
            model_cfg={"name": "xgboost", "params": {}},
            tune_cfg={
                "experiment_name": "xgboost_tune",
                "metric": "rmse",
                "mode": "min",
                "num_samples": 2,
                "max_concurrent_trials": 1,
                "resources_per_trial": {"cpu": 1},
                "search_space": {
                    "max_depth": {
                        "sampler": "randint",
                        "lower": 3,
                        "upper": 10,
                    }
                },
            },
            split=split,
            fill_strategy="mean",
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        assert "All Ray Tune trials failed" in str(exc)
    else:
        raise AssertionError("expected run_ray_tune to raise when all trials fail")
