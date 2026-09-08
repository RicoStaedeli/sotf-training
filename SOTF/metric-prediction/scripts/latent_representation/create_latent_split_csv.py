from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_NAME = "latent_representation_splits.csv"
EXPECTED_SPLITS = ("train", "val", "test")
SplitColumn = Literal["model_id", "split_name"]
SplitRow = dict[SplitColumn, str]

COLUMN_NAMES: tuple[SplitColumn, ...] = ("model_id", "split_name")


def _build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        argparse.ArgumentParser: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a CSV mapping model IDs to train/val/test splits from a latent "
            "representation dataset directory."
        ),
    )
    parser.add_argument(
        "latent_dataset_dir",
        help=(
            "Directory containing the three latent representation .pt files for "
            "train, val, and test."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=(
            "Name of the CSV file to write inside latent_dataset_dir. "
            f"Defaults to {DEFAULT_OUTPUT_NAME}."
        ),
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help=(
            "Write one row per latent representation entry instead of one row per "
            "unique model ID."
        ),
    )
    return parser


def _resolve_latent_dataset_dir(raw_path: str) -> Path:
    """Resolve latent dataset dir.

    Args:
        raw_path (str): Raw path value to resolve.

    Returns:
        Path: Resolved path.
    """
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Latent dataset directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Latent dataset path is not a directory: {path}")
    return path


def _split_path(latent_dataset_dir: Path, split_name: str) -> Path:
    """Split path.

    Args:
        latent_dataset_dir (Path): Path value used by the function.
        split_name (str): str value used by the function.

    Returns:
        Path: Resolved path.
    """
    matches = sorted(latent_dataset_dir.glob(f"*_latent_representations_{split_name}.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one latent representation file for split '{split_name}' "
            f"in {latent_dataset_dir}, found {len(matches)}."
        )
    return matches[0]


def _load_split_model_ids(path: Path) -> list[str]:
    """Load split model ids.

    Args:
        path (Path): Path to process.

    Returns:
        list[str]: Result of the function.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to load latent representation .pt files.") from exc

    data: Any = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a dict, found {type(data).__name__}.")
    if "model_ids" not in data:
        raise ValueError(f"Latent representation file is missing 'model_ids': {path}")

    model_ids = data["model_ids"]
    if not isinstance(model_ids, list):
        raise ValueError(f"Expected 'model_ids' to be a list in {path}.")
    if "embeddings" in data and hasattr(data["embeddings"], "shape"):
        embedding_rows = int(data["embeddings"].shape[0])
        if len(model_ids) != embedding_rows:
            raise ValueError(
                f"Length mismatch in {path}: model_ids has {len(model_ids)} entries, "
                f"embeddings has {embedding_rows} rows."
            )

    return [str(model_id) for model_id in model_ids]


def build_split_rows(
    latent_dataset_dir: Path,
    *,
    include_duplicates: bool = False,
) -> list[SplitRow]:
    """Build split rows.

    Args:
        latent_dataset_dir (Path): Path value used by the function.
        include_duplicates (bool): bool value used by the function.

    Returns:
        list[SplitRow]: Result of the function.
    """
    rows: list[SplitRow] = []
    model_id_to_split: dict[str, str] = {}

    for split_name in EXPECTED_SPLITS:
        path = _split_path(latent_dataset_dir, split_name)
        model_ids = _load_split_model_ids(path)
        split_model_ids = model_ids if include_duplicates else sorted(set(model_ids))
        log.info(
            "Loaded %s: %d entries, %d unique model IDs",
            path.name,
            len(model_ids),
            len(set(model_ids)),
        )

        for model_id in split_model_ids:
            existing_split = model_id_to_split.get(model_id)
            if existing_split is not None and existing_split != split_name:
                raise ValueError(
                    f"Model ID appears in multiple splits: {model_id!r} "
                    f"({existing_split}, {split_name})"
                )
            model_id_to_split[model_id] = split_name
            rows.append({"model_id": model_id, "split_name": split_name})

    return rows


def write_split_csv(rows: list[SplitRow], output_path: Path) -> None:
    """Write split csv.

    Args:
        rows (list[SplitRow]): list[SplitRow] value used by the function.
        output_path (Path): Path value used by the function.

    Returns:
        None: None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMN_NAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    latent_dataset_dir = _resolve_latent_dataset_dir(args.latent_dataset_dir)
    output_path = latent_dataset_dir / args.output_name
    rows = build_split_rows(
        latent_dataset_dir,
        include_duplicates=bool(args.include_duplicates),
    )
    write_split_csv(rows, output_path)
    log.info("Wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
