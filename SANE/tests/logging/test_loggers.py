from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sane.logging import CompositeLogger, JsonLogger, MetricsLogger, WandbLogger
from sane.logging.base import MetricsLogger as MetricsLoggerProtocol


def test_json_logger_writes_jsonl_with_step_and_metrics(tmp_path: Path) -> None:
    logger = JsonLogger(log_dir=tmp_path)
    logger.log({"loss": 0.5, "acc": 0.9}, step=1)
    logger.log({"loss": 0.3, "acc": 0.95}, step=2)

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    record1 = json.loads(lines[0])
    assert record1["step"] == 1
    assert record1["loss"] == 0.5
    assert record1["acc"] == 0.9
    assert "timestamp" in record1

    record2 = json.loads(lines[1])
    assert record2["step"] == 2
    assert record2["loss"] == 0.3
    assert record2["acc"] == 0.95


def test_json_logger_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()

    logger = JsonLogger(log_dir=nested)
    logger.log({"x": 1}, step=0)

    assert (nested / "metrics.jsonl").exists()


def test_composite_logger_delegates_log_to_all_children() -> None:
    child_a = MagicMock(spec=["log", "finish"])
    child_b = MagicMock(spec=["log", "finish"])
    composite = CompositeLogger([child_a, child_b])

    metrics = {"loss": 0.1}
    composite.log(metrics, step=5)

    child_a.log.assert_called_once_with(metrics, 5)
    child_b.log.assert_called_once_with(metrics, 5)


def test_composite_logger_finish_calls_all_children() -> None:
    child_a = MagicMock(spec=["log", "finish"])
    child_b = MagicMock(spec=["log", "finish"])
    composite = CompositeLogger([child_a, child_b])

    composite.finish()

    child_a.finish.assert_called_once()
    child_b.finish.assert_called_once()


def test_json_logger_satisfies_metrics_logger_protocol() -> None:
    assert isinstance(JsonLogger.__new__(JsonLogger), MetricsLoggerProtocol)


def test_composite_logger_satisfies_metrics_logger_protocol() -> None:
    assert isinstance(CompositeLogger.__new__(CompositeLogger), MetricsLoggerProtocol)


def test_wandb_logger_logs_flattened_numeric_metrics(tmp_path: Path) -> None:
    api_key_path = tmp_path / "wandb.key"
    api_key_path.write_text("secret-key\n")

    fake_run = MagicMock()
    fake_wandb = MagicMock()
    fake_wandb.init.return_value = fake_run

    old_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        logger = WandbLogger(
            log_dir=tmp_path,
            project="proj",
            group="grp",
            api_key_file=str(api_key_path),
            config={"experiment_name": "exp"},
            run_name="trial-1",
        )
        logger.log({"loss": 0.5, "nested": {"acc": 0.9}, "flag": True}, step=3)
        logger.finish()
    finally:
        if old_wandb is None:
            del sys.modules["wandb"]
        else:
            sys.modules["wandb"] = old_wandb

    fake_wandb.login.assert_called_once_with(key="secret-key")
    fake_wandb.init.assert_called_once_with(
        project="proj",
        group="grp",
        dir=str(tmp_path),
        config={"experiment_name": "exp"},
        name="trial-1",
        reinit=True,
    )
    fake_run.log.assert_called_once_with({"loss": 0.5, "nested/acc": 0.9}, step=3)
    fake_run.finish.assert_called_once()


def test_wandb_logger_satisfies_metrics_logger_protocol() -> None:
    assert isinstance(WandbLogger.__new__(WandbLogger), MetricsLoggerProtocol)
