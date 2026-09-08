from __future__ import annotations

"""
Merge train/val/test latent representation .pt files into one complete dataset.

Usage:
    uv run python -m metric_prediction.scripts.latent_representation.merge_latent_splits \
        metric-prediction/cache/latent_representation_datasets/resnet18_tinyimagenet_encoded_checkpoints
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_SPLIT_ORDER = ("train", "val", "test")


def _parse_args() -> argparse.Namespace:
    """Parse args.

    Returns:
        argparse.Namespace: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description="Merge latent representation train/val/test .pt files into a single .pt dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing one *_train.pt, one *_val.pt, and one *_test.pt file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .pt path. Defaults to <shared-prefix>complete.pt inside dataset_dir, "
            "for example foo_latent_representations_complete.pt."
        ),
    )
    parser.add_argument(
        "--split-order",
        nargs="+",
        default=list(DEFAULT_SPLIT_ORDER),
        choices=DEFAULT_SPLIT_ORDER,
        help="Order in which split rows are concatenated.",
    )
    parser.add_argument(
        "--add-split-column",
        action="store_true",
        help="Add a split_name list containing the original split for each row.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )
    return parser.parse_args()


def _find_split_files(dataset_dir: Path, split_order: list[str]) -> dict[str, Path]:
    """Find split files.

    Args:
        dataset_dir (Path): Path value used by the function.
        split_order (list[str]): list[str] value used by the function.

    Returns:
        dict[str, Path]: Result of the function.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    split_files: dict[str, Path] = {}
    for split in split_order:
        matches = sorted(dataset_dir.glob(f"*_{split}.pt"))
        if not matches:
            raise FileNotFoundError(f"No '*_{split}.pt' file found in {dataset_dir}")
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(f"Expected exactly one '*_{split}.pt' file in {dataset_dir}, found: {names}")
        split_files[split] = matches[0]

    return split_files


def _default_output_path(split_files: dict[str, Path], dataset_dir: Path) -> Path:
    """Default output path.

    Args:
        split_files (dict[str, Path]): dict[str, Path] value used by the function.
        dataset_dir (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    train_path = split_files["train"]
    suffix = "_train.pt"
    if not train_path.name.endswith(suffix):
        return dataset_dir / "latent_representations_complete.pt"
    return dataset_dir / f"{train_path.name[: -len(suffix)]}_complete.pt"


def _load_split(path: Path) -> dict[str, Any]:
    """Load split.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any]: Result of the function.
    """
    log.info("Loading %s", path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a dict, got {type(data)}.")
    return data


def _row_count(data: dict[str, Any], path: Path) -> int:
    """Row count.

    Args:
        data (dict[str, Any]): dict[str, Any] value used by the function.
        path (Path): Path to process.

    Returns:
        int: Result of the function.
    """
    tensor_counts = [value.shape[0] for value in data.values() if isinstance(value, torch.Tensor)]
    if tensor_counts:
        return int(tensor_counts[0])

    sized_counts = [
        len(value)
        for value in data.values()
        if isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).ndim > 0
    ]
    if sized_counts:
        return int(sized_counts[0])

    raise ValueError(f"Could not determine row count for {path}.")


def _validate_row_lengths(data: dict[str, Any], row_count: int, path: Path) -> None:
    """Validate row lengths.

    Args:
        data (dict[str, Any]): dict[str, Any] value used by the function.
        row_count (int): int value used by the function.
        path (Path): Path to process.

    Returns:
        None: None.
    """
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] != row_count:
                raise ValueError(
                    f"{path}: key '{key}' has {value.shape[0]} rows, expected {row_count}."
                )
        elif isinstance(value, np.ndarray):
            if value.ndim > 0 and len(value) != row_count:
                raise ValueError(f"{path}: key '{key}' has {len(value)} rows, expected {row_count}.")
        elif isinstance(value, (list, tuple)) and len(value) != row_count:
            raise ValueError(f"{path}: key '{key}' has {len(value)} rows, expected {row_count}.")


def _merge_values(key: str, values: list[Any]) -> Any:
    """Merge values.

    Args:
        key (str): str value used by the function.
        values (list[Any]): Values to process.

    Returns:
        Any: Result of the function.
    """
    first = values[0]

    if isinstance(first, torch.Tensor):
        for value in values[1:]:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Key '{key}' mixes tensor and non-tensor values.")
            if value.shape[1:] != first.shape[1:]:
                raise ValueError(
                    f"Key '{key}' has incompatible tensor shapes: {tuple(first.shape)} vs {tuple(value.shape)}."
                )
            if value.dtype != first.dtype:
                raise ValueError(f"Key '{key}' has incompatible tensor dtypes: {first.dtype} vs {value.dtype}.")
        return torch.cat(values, dim=0)

    if isinstance(first, np.ndarray):
        for value in values[1:]:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Key '{key}' mixes ndarray and non-ndarray values.")
            if value.shape[1:] != first.shape[1:]:
                raise ValueError(
                    f"Key '{key}' has incompatible array shapes: {first.shape} vs {value.shape}."
                )
        return np.concatenate(values, axis=0)

    if isinstance(first, tuple):
        merged: list[Any] = []
        for value in values:
            if not isinstance(value, tuple):
                raise TypeError(f"Key '{key}' mixes tuple and non-tuple values.")
            merged.extend(value)
        return tuple(merged)

    if isinstance(first, list):
        merged = []
        for value in values:
            if not isinstance(value, list):
                raise TypeError(f"Key '{key}' mixes list and non-list values.")
            merged.extend(value)
        return merged

    if all(value == first for value in values):
        return first

    raise TypeError(f"Key '{key}' contains unsupported non-row values that differ across splits.")


def merge_latent_splits(
    dataset_dir: Path,
    *,
    output_path: Path | None = None,
    split_order: list[str] | None = None,
    add_split_column: bool = False,
    overwrite: bool = False,
) -> Path:
    """Merge latent splits.

    Args:
        dataset_dir (Path): Path value used by the function.
        output_path (Path | None): Path | None value used by the function.
        split_order (list[str] | None): list[str] | None value used by the function.
        add_split_column (bool): bool value used by the function.
        overwrite (bool): bool value used by the function.

    Returns:
        Path: Resolved path.
    """
    split_order = split_order or list(DEFAULT_SPLIT_ORDER)
    if set(split_order) != set(DEFAULT_SPLIT_ORDER) or len(split_order) != len(DEFAULT_SPLIT_ORDER):
        expected = ", ".join(DEFAULT_SPLIT_ORDER)
        got = ", ".join(split_order)
        raise ValueError(f"split_order must contain each required split exactly once: {expected}. Got: {got}.")

    split_files = _find_split_files(dataset_dir, split_order)
    output_path = output_path or _default_output_path(split_files, dataset_dir)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    loaded: dict[str, dict[str, Any]] = {}
    row_counts: dict[str, int] = {}
    for split in split_order:
        path = split_files[split]
        data = _load_split(path)
        row_count = _row_count(data, path)
        _validate_row_lengths(data, row_count, path)
        loaded[split] = data
        row_counts[split] = row_count
        log.info("%s: %d rows", split, row_count)

    reference_keys = set(loaded[split_order[0]])
    for split in split_order[1:]:
        keys = set(loaded[split])
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(f"{split} keys differ from reference. Missing={missing}, extra={extra}")

    merged = {
        key: _merge_values(key, [loaded[split][key] for split in split_order])
        for key in loaded[split_order[0]]
    }

    if add_split_column:
        if "split_name" in merged:
            raise ValueError("Cannot add split_name because the source files already contain that key.")
        merged["split_name"] = [
            split
            for split in split_order
            for _ in range(row_counts[split])
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)

    total_rows = sum(row_counts.values())
    log.info("Wrote %d rows to %s", total_rows, output_path)
    return output_path


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    output_path = merge_latent_splits(
        args.dataset_dir,
        output_path=args.output_path,
        split_order=args.split_order,
        add_split_column=args.add_split_column,
        overwrite=args.overwrite,
    )
    print(output_path)


if __name__ == "__main__":
    main()
