#!/usr/bin/env python3
"""
Extract per-checkpoint performance metrics from a model zoo directory.

Each model subdirectory must contain a result.json file in JSONL format
(one JSON record per epoch, with training_iteration as the epoch key).

Output parquet columns:
    model_id, epoch, test_acc, train_acc, test_loss, train_loss

Columns absent from result.json are stored as NaN.

Optionally enriches an existing prepared weight-statistic dataset parquet
by joining the extracted metadata on (model_id, epoch) and adding any
metric columns that are currently missing from it.

Examples
--------
# 1. Extract only:
python extract_model_zoo_metadata.py /path/to/zoo \
    --output metric-prediction/cache/zoo_metadata/ds14_metadata.parquet

# 2. Extract and patch an existing prepared dataset in one step:
python extract_model_zoo_metadata.py /path/to/zoo \
    --output metric-prediction/cache/zoo_metadata/ds14_metadata.parquet \
    --enrich-prepared-dataset metric-prediction/cache/prepared_datasets/14_phase_...__prepared_dataset.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

log = logging.getLogger(__name__)

METRIC_COLUMNS = ("test_acc", "train_acc", "test_loss", "train_loss")
EPOCH_KEY = "training_iteration"


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_zoo_metadata(zoo_root: Path) -> pd.DataFrame:
    """Read every result.json under zoo_root and return a long (model_id, epoch, metrics) table."""
    result_files = sorted(zoo_root.glob("*/result.json"))
    if not result_files:
        raise FileNotFoundError(f"No result.json files found under {zoo_root}")

    log.info("Found %d model directories in %s", len(result_files), zoo_root)

    records: list[dict] = []
    for result_file in result_files:
        model_id = result_file.parent.name
        with result_file.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSON in %s line %d: %s", result_file, lineno, exc)
                    continue

                epoch = record.get(EPOCH_KEY)
                if epoch is None:
                    log.debug("No '%s' key in %s line %d — skipping", EPOCH_KEY, result_file, lineno)
                    continue

                row: dict = {"model_id": model_id, "epoch": int(epoch)}
                for col in METRIC_COLUMNS:
                    value = record.get(col)
                    row[col] = float(value) if value is not None else float("nan")
                records.append(row)

    if not records:
        log.warning("No records extracted from %s", zoo_root)
        return pd.DataFrame(columns=["model_id", "epoch", *METRIC_COLUMNS])

    df = pd.DataFrame(records)
    df["epoch"] = df["epoch"].astype(int)
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["model_id", "epoch"]).reset_index(drop=True)

    n_models = int(df["model_id"].nunique())
    log.info(
        "Extracted %d rows from %d models (~%d epochs/model)",
        len(df),
        n_models,
        len(df) // max(n_models, 1),
    )

    # Warn about columns that are entirely NaN (not present in result.json)
    for col in METRIC_COLUMNS:
        if df[col].isna().all():
            log.warning("Column '%s' is entirely NaN — it is not present in result.json for this zoo.", col)

    return df


# ---------------------------------------------------------------------------
# Enrichment helper (also called from generate_prepared_dataset.py)
# ---------------------------------------------------------------------------

def enrich_prepared_dataset(prepared_path: Path, metadata: pd.DataFrame) -> None:
    """Join metadata onto an existing prepared parquet, adding any missing metric columns.

    Joins on (model_id, epoch) where model_id is matched by basename so that
    full-path model_ids in the prepared dataset are handled correctly.
    """
    frame = pd.read_parquet(prepared_path)

    missing_cols = [col for col in METRIC_COLUMNS if col not in frame.columns]
    available_cols = [col for col in missing_cols if col in metadata.columns and not metadata[col].isna().all()]
    if not available_cols:
        log.info(
            "Nothing to add to %s — columns already present or all-NaN in metadata.",
            prepared_path,
        )
        return

    # Normalise model_id to basename on both sides
    frame_key = pd.DataFrame({
        "model_id": frame["model_id"].astype(str).apply(lambda p: Path(p).name),
        "epoch": pd.to_numeric(frame["epoch"], errors="coerce").astype(int),
    })
    meta_join = metadata[["model_id", "epoch", *available_cols]].copy()
    meta_join["model_id"] = meta_join["model_id"].astype(str).apply(lambda p: Path(p).name)
    meta_join["epoch"] = pd.to_numeric(meta_join["epoch"], errors="coerce").astype(int)

    merged = frame_key.merge(meta_join, on=["model_id", "epoch"], how="left")

    for col in available_cols:
        frame[col] = pd.to_numeric(merged[col].values, errors="coerce")

    frame.to_parquet(prepared_path, index=False)
    log.info("Enriched %s — added columns: %s", prepared_path, available_cols)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-checkpoint performance metrics (test_acc, train_acc, "
            "test_loss, train_loss) from a model zoo and save as a parquet table."
        )
    )
    parser.add_argument(
        "zoo_root",
        help=(
            "Root directory of the model zoo. "
            "Each immediate subdirectory should contain a result.json file."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PARQUET",
        help=(
            "Output path for the metadata parquet. "
            "Defaults to <zoo_root_name>_metadata.parquet in the current directory."
        ),
    )
    parser.add_argument(
        "--enrich-prepared-dataset",
        default=None,
        metavar="PREPARED_PARQUET",
        help=(
            "Path to an existing prepared weight-statistic dataset parquet. "
            "If given, any metric columns that are missing from it are joined in "
            "from the extracted metadata (matched on model_id × epoch)."
        ),
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_parser().parse_args()

    zoo_root = Path(args.zoo_root).expanduser().resolve()
    if not zoo_root.exists():
        raise SystemExit(f"Zoo root does not exist: {zoo_root}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path.cwd() / f"{zoo_root.name}_metadata.parquet"
    )

    metadata = extract_zoo_metadata(zoo_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output_path, index=False)
    log.info("Wrote metadata parquet: %s (%d rows)", output_path, len(metadata))

    if args.enrich_prepared_dataset:
        prepared_path = Path(args.enrich_prepared_dataset).expanduser().resolve()
        if not prepared_path.exists():
            raise SystemExit(f"Prepared dataset does not exist: {prepared_path}")
        enrich_prepared_dataset(prepared_path, metadata)


if __name__ == "__main__":
    main()
