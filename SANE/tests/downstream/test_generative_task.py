"""Tests for the GenerativeDownstreamTask class."""

import pytest

from sane.downstream.generative import GenerativeDownstreamTask


def test_generative_task_init_custom():
    """Test custom initialization values."""
    task = GenerativeDownstreamTask(
        name="task",
        mode="sample",
        num_models=10,
        kde_bandwidth=0.5,
        num_anchors=50,
        num_samples=100,
        top_k_samples=10,
        enable_bn_conditioning=True,
        bn_iterations=100,
        verbose=False,
        encode_batch_size=32,
        decode_batch_size=128,
    )

    assert task.name == "task"
    assert task.mode == "sample"
    assert task.num_models == 10
    assert task.kde_bandwidth == 0.5
    assert task.num_anchors == 50
    assert task.num_samples == 100
    assert task.top_k_samples == 10
    assert task.enable_bn_conditioning is True
    assert task.bn_iterations == 100
    assert task.verbose is False
    assert task.encode_batch_size == 32
    assert task.decode_batch_size == 128


def test_generative_task_save_models_config():
    """Test save_sampled_models configuration."""
    task = GenerativeDownstreamTask(
        save_sampled_models=True,
        sampled_models_dir="/tmp/models"
    )
    assert task.save_sampled_models is True
    assert task.sampled_models_dir == "/tmp/models"


def test_execute_dispatch_reconstruct(monkeypatch):
    """Test that _execute dispatches to reconstruction and updates state."""
    task = GenerativeDownstreamTask(mode="reconstruct")

    def fake_run(_sane_ae):
        return {
            "mode": "reconstruct",
            "reconstructed_accuracy_mean": 85.0,
            "original_accuracy_mean": 90.0,
            "delta_mean": -5.0,
            "delta_std": 1.2,
            "per_zoo": {},
        }

    monkeypatch.setattr(task, "_run_reconstruction", fake_run)

    result = task._execute(object(), epoch=3)

    assert task._current_epoch == 3
    assert task.name in result
    assert result[task.name]["mode"] == "reconstruct"
    assert result[task.name]["recon_mean"] == 85.0
    assert result[task.name]["recon_delta"] == -5.0


def test_execute_dispatch_sample(monkeypatch):
    """Test that _execute dispatches to sampling mode."""
    task = GenerativeDownstreamTask(mode="sample")

    def fake_run(_sane_ae):
        return {
            "mode": "sample",
            "accuracy_mean": 70.0,
            "accuracy_std": 5.0,
            "accuracy_min": 60.0,
            "accuracy_max": 80.0,
            "per_zoo": {},
        }

    monkeypatch.setattr(task, "_run_sampling", fake_run)

    result = task._execute(object(), epoch=1)

    assert task._current_epoch == 1
    assert task.name in result
    assert result[task.name]["mode"] == "sample"
    assert result[task.name]["sample_mean"] == 70.0
    assert result[task.name]["sample_std"] == 5.0


def test_execute_dispatch_both(monkeypatch):
    """Test that _execute dispatches to both mode."""
    task = GenerativeDownstreamTask(mode="both")

    def fake_run(_sane_ae):
        return {
            "mode": "both",
            "reconstruction": {
                "reconstructed_accuracy_mean": 85.0,
                "original_accuracy_mean": 90.0,
                "delta_mean": -5.0,
                "delta_std": 1.2,
                "per_zoo": {},
            },
            "sampling": {
                "accuracy_mean": 70.0,
                "accuracy_std": 5.0,
                "accuracy_min": 60.0,
                "accuracy_max": 80.0,
                "per_zoo": {},
            },
        }

    monkeypatch.setattr(task, "_run_both", fake_run)

    result = task._execute(object(), epoch=2)

    assert task._current_epoch == 2
    assert task.name in result
    assert result[task.name]["mode"] == "both"
    assert result[task.name]["recon_mean"] == 85.0
    assert result[task.name]["sample_mean"] == 70.0


def test_execute_invalid_mode_raises():
    """Test that invalid mode raises a clear error."""
    task = GenerativeDownstreamTask(mode="invalid")

    with pytest.raises(ValueError, match="Unknown mode"):
        task._execute(object())
