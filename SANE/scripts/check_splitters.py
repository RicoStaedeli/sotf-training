#!/usr/bin/env python3
"""Check group-based checkpoint splitters.

By default this script builds a tiny synthetic model zoo and verifies that:

1. ModelGroupSplitter never spreads checkpoints from one model across splits.
2. HyperparameterGroupSplitter never spreads models with the same hyperparameters
   across splits when only excluded keys differ.

Pass a real model zoo root to run the same checks against a ZooDataset scan.

Run from the repository root:

    uv run python scripts/check_splitters.py
    uv run python scripts/check_splitters.py --ratios 0.7 0.15 0.15 --seed 2020
    uv run python scripts/check_splitters.py \
        --root-dir /path/to/model_zoo \
        --epoch-idx 21 22 23 24 25
    uv run python scripts/check_splitters.py \
        --root-dir /path/to/model_zoo \
        --all-epochs
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sane.data.datasets import ZooDataset  # noqa: E402
from sane.data.splitter import HyperparameterGroupSplitter, ModelGroupSplitter  # noqa: E402


@dataclass(frozen=True)
class SyntheticCheckpoint:
    model_id: str
    checkpoint_idx: int


class SyntheticCheckpointDataset:
    """Minimal dataset interface required by the group splitters."""

    def __init__(self, model_ids: Iterable[str], checkpoints_per_model: int = 2) -> None:
        self.checkpoints = [
            SyntheticCheckpoint(model_id=model_id, checkpoint_idx=checkpoint_idx)
            for model_id in model_ids
            for checkpoint_idx in range(checkpoints_per_model)
        ]

    def __len__(self) -> int:
        return len(self.checkpoints)

    def get_model_id(self, index: int) -> str:
        return self.checkpoints[index].model_id


def _split_assignments(dataset: Any, splits) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for split_idx, subset in enumerate(splits):
        split_name = f"split_{split_idx}"
        for index in subset.indices:
            if index in assignments:
                raise AssertionError(f"Dataset index {index} appears in multiple splits.")
            assignments[index] = split_name

    expected = set(range(len(dataset)))
    actual = set(assignments)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"Split indices are incomplete. Missing={missing}, extra={extra}")

    return assignments


def _assert_models_are_not_split(dataset: Any, assignments: dict[int, str]) -> None:
    model_to_splits: dict[str, set[str]] = defaultdict(set)
    for index, split_name in assignments.items():
        model_to_splits[dataset.get_model_id(index)].add(split_name)

    leaked = {model_id: splits for model_id, splits in model_to_splits.items() if len(splits) > 1}
    if leaked:
        raise AssertionError(f"Model groups leaked across splits: {leaked}")


def _hyperparameter_key(params: dict, exclude_keys: set[str]) -> str:
    filtered = {k: v for k, v in params.items() if k not in exclude_keys}
    return json.dumps(filtered, sort_keys=True)


def _assert_hyperparameters_are_not_split(
    dataset: Any,
    assignments: dict[int, str],
    params_by_model: dict[str, dict],
    exclude_keys: set[str],
) -> None:
    hyperparam_to_splits: dict[str, set[str]] = defaultdict(set)
    for index, split_name in assignments.items():
        model_id = dataset.get_model_id(index)
        key = _hyperparameter_key(params_by_model[model_id], exclude_keys)
        hyperparam_to_splits[key].add(split_name)

    leaked = {key: splits for key, splits in hyperparam_to_splits.items() if len(splits) > 1}
    if leaked:
        raise AssertionError(f"Hyperparameter groups leaked across splits: {leaked}")


def _short_model_id(model_id: str) -> str:
    path = Path(model_id)
    return path.name if path.parent != Path(".") else model_id


def _format_model_ids(model_ids: Sequence[str], limit: int) -> str:
    display = [_short_model_id(model_id) for model_id in model_ids[:limit]]
    suffix = "" if len(model_ids) <= limit else f", ... +{len(model_ids) - limit} more"
    return f"{display}{suffix}"


def _print_split_summary(title: str, dataset: Any, splits, model_id_limit: int = 20) -> None:
    print(f"\n{title}")
    for split_idx, subset in enumerate(splits):
        model_ids = sorted({dataset.get_model_id(index) for index in subset.indices})
        print(
            f"  split_{split_idx}: "
            f"{len(subset.indices):2d} checkpoints, {len(model_ids):2d} model groups"
        )
        print(f"    {_format_model_ids(model_ids, model_id_limit)}")


def check_model_group_splitter(dataset: Any, ratios: Sequence[float], seed: int, model_id_limit: int) -> None:
    splits = ModelGroupSplitter(seed=seed).split(dataset, ratios)
    assignments = _split_assignments(dataset, splits)
    _assert_models_are_not_split(dataset, assignments)
    _print_split_summary("ModelGroupSplitter", dataset, splits, model_id_limit)


def _write_params(model_dir: Path, params: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


def _read_params_by_model(model_ids: Iterable[str]) -> dict[str, dict]:
    params_by_model = {}
    for model_id in model_ids:
        params_path = Path(model_id) / "params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing params.json for model_id={model_id}: {params_path}")
        with params_path.open("r", encoding="utf-8") as handle:
            params_by_model[model_id] = json.load(handle)
    return params_by_model


def check_hyperparameter_group_splitter(
    dataset: Any,
    ratios: Sequence[float],
    seed: int,
    exclude_keys: set[str],
    model_id_limit: int,
) -> None:
    model_ids = sorted({dataset.get_model_id(index) for index in range(len(dataset))})
    params_by_model = _read_params_by_model(model_ids)

    splits = HyperparameterGroupSplitter(seed=seed, exclude_keys=list(exclude_keys)).split(dataset, ratios)
    assignments = _split_assignments(dataset, splits)
    _assert_models_are_not_split(dataset, assignments)
    _assert_hyperparameters_are_not_split(dataset, assignments, params_by_model, exclude_keys)
    _print_split_summary("HyperparameterGroupSplitter", dataset, splits, model_id_limit)


def build_synthetic_model_dataset() -> SyntheticCheckpointDataset:
    model_ids = [f"model_{idx}" for idx in range(7)]
    return SyntheticCheckpointDataset(model_ids, checkpoints_per_model=3)


def run_synthetic_checks(ratios: Sequence[float], seed: int, exclude_keys: set[str], model_id_limit: int) -> None:
    check_model_group_splitter(build_synthetic_model_dataset(), ratios, seed, model_id_limit)

    with tempfile.TemporaryDirectory(prefix="sane_splitter_check_") as tmp_dir:
        root = Path(tmp_dir)
        model_ids = []
        for model_name, params in {
            "hp_a_seed_1": {"architecture": "cnn", "lr": 0.001, "batch_size": 64, "seed": 1},
            "hp_a_seed_2": {"architecture": "cnn", "lr": 0.001, "batch_size": 64, "seed": 2},
            "hp_b_seed_1": {"architecture": "cnn", "lr": 0.01, "batch_size": 64, "seed": 1},
            "hp_b_seed_2": {"architecture": "cnn", "lr": 0.01, "batch_size": 64, "seed": 2},
            "hp_c_seed_1": {"architecture": "vit", "lr": 0.001, "batch_size": 32, "seed": 1},
            "hp_c_seed_2": {"architecture": "vit", "lr": 0.001, "batch_size": 32, "seed": 2},
        }.items():
            model_dir = root / model_name
            _write_params(model_dir, params)
            model_ids.append(str(model_dir))

        dataset = SyntheticCheckpointDataset(model_ids, checkpoints_per_model=2)
        check_hyperparameter_group_splitter(dataset, ratios, seed, exclude_keys, model_id_limit)


def discover_all_epoch_model_dirs(root_dir: str) -> tuple[list[int], list[Path]]:
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model zoo root does not exist: {root}")

    model_epoch_counts: dict[Path, int] = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        epoch_dirs = [
            path
            for path in model_dir.iterdir()
            if path.is_dir() and (path / "checkpoints").exists()
        ]
        if epoch_dirs:
            model_epoch_counts[model_dir] = len(epoch_dirs)

    if not model_epoch_counts:
        raise ValueError(f"No model checkpoint directories found below {root}.")

    min_epochs = min(model_epoch_counts.values())
    max_epochs = max(model_epoch_counts.values())
    full_length_model_dirs = [
        model_dir
        for model_dir, epoch_count in model_epoch_counts.items()
        if epoch_count == max_epochs
    ]
    if min_epochs != max_epochs:
        print(
            f"Warning: model directories have different numbers of checkpoints "
            f"(min={min_epochs}, max={max_epochs}). Using only the "
            f"{len(full_length_model_dirs)} / {len(model_epoch_counts)} model directories "
            f"with {max_epochs} checkpoints and epoch indices 0..{max_epochs - 1}."
        )

    return list(range(max_epochs)), full_length_model_dirs


def build_filtered_zoo_root(model_dirs: Sequence[Path]) -> tempfile.TemporaryDirectory:
    tmp_dir = tempfile.TemporaryDirectory(prefix="sane_full_epoch_zoo_")
    filtered_root = Path(tmp_dir.name)
    for model_dir in model_dirs:
        (filtered_root / model_dir.name).symlink_to(model_dir, target_is_directory=True)
    return tmp_dir


def build_real_zoo_dataset(args: argparse.Namespace, root_dir: str, epoch_idx: Sequence[int]) -> ZooDataset:
    print(f"Using epoch indices: {epoch_idx}")
    dataset = ZooDataset(
        root_dir=root_dir,
        epoch_idx=epoch_idx,
        drop_nan=args.drop_nan,
        max_absolute_weight=args.max_absolute_weight,
        max_checkpoints=args.max_checkpoints,
    )
    if len(dataset) == 0:
        raise ValueError("ZooDataset scan produced 0 checkpoints. Check --root-dir and --epoch-idx.")
    return dataset


def run_real_zoo_checks(args: argparse.Namespace, exclude_keys: set[str]) -> None:
    root_dir = args.root_dir
    epoch_idx = args.epoch_idx
    filtered_root_tmp = None
    if args.all_epochs:
        epoch_idx, model_dirs = discover_all_epoch_model_dirs(args.root_dir)
        filtered_root_tmp = build_filtered_zoo_root(model_dirs)
        root_dir = filtered_root_tmp.name

    dataset = build_real_zoo_dataset(args, root_dir, epoch_idx)
    unique_models = {dataset.get_model_id(index) for index in range(len(dataset))}
    print(
        f"Loaded ZooDataset from {root_dir}: "
        f"{len(dataset)} checkpoints across {len(unique_models)} model directories."
    )

    try:
        if args.splitter in {"model", "both"}:
            check_model_group_splitter(dataset, args.ratios, args.seed, args.model_id_limit)
        if args.splitter in {"hyperparameter", "both"}:
            check_hyperparameter_group_splitter(dataset, args.ratios, args.seed, exclude_keys, args.model_id_limit)
    finally:
        if filtered_root_tmp is not None:
            filtered_root_tmp.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ModelGroupSplitter and HyperparameterGroupSplitter invariants.")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.25, 0.25],
        help="Split ratios to test. They must sum to 1. Default: 0.5 0.25 0.25",
    )
    parser.add_argument("--seed", type=int, default=42, help="Splitter shuffle seed. Default: 42")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Optional real model zoo root. If omitted, runs the synthetic smoke checks.",
    )
    parser.add_argument(
        "--epoch-idx",
        type=int,
        nargs="+",
        default=[-1],
        help="Checkpoint epoch indices for a real ZooDataset. Ignored when --all-epochs is set. Default: -1",
    )
    parser.add_argument(
        "--all-epochs",
        action="store_true",
        help=(
            "Use the maximum available epoch range instead of --epoch-idx, "
            "filtering to model directories that contain that full range."
        ),
    )
    parser.add_argument("--max-checkpoints", type=int, default=None, help="Optional cap for real ZooDataset scan.")
    parser.add_argument("--drop-nan", action="store_true", help="Enable ZooDataset NaN checkpoint filtering.")
    parser.add_argument(
        "--max-absolute-weight",
        type=float,
        default=None,
        help="Optional ZooDataset max absolute weight filter.",
    )
    parser.add_argument(
        "--splitter",
        choices=["model", "hyperparameter", "both"],
        default="both",
        help="Which splitter to check. Default: both",
    )
    parser.add_argument(
        "--exclude-keys",
        nargs="+",
        default=["seed"],
        help="Keys ignored by HyperparameterGroupSplitter. Default: seed",
    )
    parser.add_argument(
        "--model-id-limit",
        type=int,
        default=20,
        help="Maximum model IDs to print per split. Default: 20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios must sum to 1, got {sum(args.ratios)}")
    if args.all_epochs and args.root_dir is None:
        raise SystemExit("--all-epochs requires --root-dir.")

    exclude_keys = set(args.exclude_keys)
    if args.root_dir is None:
        run_synthetic_checks(args.ratios, args.seed, exclude_keys, args.model_id_limit)
    else:
        run_real_zoo_checks(args, exclude_keys)

    print("\nAll splitter checks passed.")


if __name__ == "__main__":
    main()
