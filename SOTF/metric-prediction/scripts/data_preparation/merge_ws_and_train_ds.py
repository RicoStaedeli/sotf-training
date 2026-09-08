#!/usr/bin/env python3
"""
Join selected columns from parquet 2 into parquet 1 without loading either file fully into memory.

Usage:
    python join_parquets.py /path/to/parquet1.parquet /path/to/parquet2.parquet /path/to/save_dir

Inputs:
    1 = path to parquet 1 (Dataframe with weight statistics)
    2 = path to parquet 2 (Dataframe with validation metrics per model and epoch)
    3 = save dir

Join keys:
    parquet1.epoch    <-> parquet2.training_iteration
    parquet1.model_id <-> parquet2.model_name_full

Columns copied from parquet 2 into parquet 1:
    - train_loss
    - test_loss
    - test_acc
    - train_acc
    - hyperparameter_key

Behavior:
    - If one of these metric columns already exists in parquet1, it is NOT added again.
    - Existing columns in parquet1 are kept unchanged.
    - The output file is saved in save_dir with the same file name as parquet1.
"""

import argparse
import sys
from pathlib import Path

import duckdb


METRIC_COLUMNS = ["train_loss", "test_loss", "test_acc", "train_acc"]
HYPERPARAMETER_KEY_COLUMN = "hyperparameter_key"
NON_HYPERPARAMETER_CONFIG_COLUMNS = {
    "config.seed",
    "config.cuda",
    "config.dataset::dump",
    "config.wandb.project",
    "config.wandb.api_key_file",
    "config.wandb.log_config",
    "config.training::checkpoint_dir",
}


def parse_args():
    """Parse args.

    Returns:
        Any: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description="Join columns from one parquet file to another in an out-of-core way."
    )
    parser.add_argument("parquet1", help="Path to full Dataframe with weight statistics")
    parser.add_argument("parquet2", help="Path to Dataframe with validation metrics per model and epoch")
    parser.add_argument("save_dir", help="Directory where the output parquet file will be saved")
    return parser.parse_args()


def validate_input(path_str: str, label: str) -> Path:
    """Validate input.

    Args:
        path_str (str): str value used by the function.
        label (str): str value used by the function.

    Returns:
        Path: Resolved path.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def get_columns(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> set[str]:
    """Get columns.

    Args:
        con (duckdb.DuckDBPyConnection): duckdb.DuckDBPyConnection value used by the function.
        parquet_path (Path): Path to the parquet.

    Returns:
        set[str]: Result of the function.
    """
    return {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        ).fetchall()
    }


def check_required_columns(con: duckdb.DuckDBPyConnection, parquet1: Path, parquet2: Path):
    """Check required columns.

    Args:
        con (duckdb.DuckDBPyConnection): duckdb.DuckDBPyConnection value used by the function.
        parquet1 (Path): Path value used by the function.
        parquet2 (Path): Path value used by the function.

    Returns:
        Any: Result of the function.
    """
    required_p1 = {"epoch", "model_id"}
    required_p2 = {"training_iteration", "model_name_full", *METRIC_COLUMNS}

    p1_cols = get_columns(con, parquet1)
    p2_cols = get_columns(con, parquet2)

    missing_p1 = required_p1 - p1_cols
    missing_p2 = required_p2 - p2_cols

    if missing_p1:
        raise ValueError(f"Parquet 1 is missing required columns: {sorted(missing_p1)}")
    if missing_p2:
        raise ValueError(f"Parquet 2 is missing required columns: {sorted(missing_p2)}")


def check_no_duplicate_keys(con: duckdb.DuckDBPyConnection, parquet2: Path):
    """Check no duplicate keys.

    Args:
        con (duckdb.DuckDBPyConnection): duckdb.DuckDBPyConnection value used by the function.
        parquet2 (Path): Path value used by the function.

    Returns:
        Any: Result of the function.
    """
    duplicate_count_query = f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                training_iteration,
                model_name_full,
                COUNT(*) AS cnt
            FROM read_parquet('{parquet2.as_posix()}')
            GROUP BY training_iteration, model_name_full
            HAVING COUNT(*) > 1
        ) t
    """
    duplicate_groups = con.execute(duplicate_count_query).fetchone()[0]

    if duplicate_groups > 0:
        sample_duplicates_query = f"""
            SELECT
                training_iteration,
                model_name_full,
                COUNT(*) AS cnt
            FROM read_parquet('{parquet2.as_posix()}')
            GROUP BY training_iteration, model_name_full
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC, training_iteration, model_name_full
            LIMIT 20
        """
        sample_rows = con.execute(sample_duplicates_query).fetchall()

        lines = [
            "Parquet 2 contains duplicate join keys (training_iteration, model_name_full).",
            f"Number of duplicated key groups: {duplicate_groups}",
            "Sample duplicated keys:",
        ]
        for training_iteration, model_name_full, cnt in sample_rows:
            lines.append(
                f"  training_iteration={training_iteration!r}, model_name_full={model_name_full!r}, count={cnt}"
            )

        raise ValueError("\n".join(lines))


def _quote_identifier(identifier: str) -> str:
    """Quote identifier.

    Args:
        identifier (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    return '"' + identifier.replace('"', '""') + '"'


def get_hyperparameter_columns(
    con: duckdb.DuckDBPyConnection,
    parquet2: Path,
) -> list[str]:
    """Get hyperparameter columns.

    Args:
        con (duckdb.DuckDBPyConnection): duckdb.DuckDBPyConnection value used by the function.
        parquet2 (Path): Path value used by the function.

    Returns:
        list[str]: Result of the function.
    """
    parquet2_columns = get_columns(con, parquet2)
    hyperparameter_columns = sorted(
        column
        for column in parquet2_columns
        if column.startswith("config") and column not in NON_HYPERPARAMETER_CONFIG_COLUMNS
    )
    if not hyperparameter_columns:
        raise ValueError(
            "Parquet 2 does not contain any config* columns that can be used to build "
            f"'{HYPERPARAMETER_KEY_COLUMN}'."
        )
    return hyperparameter_columns


def build_hyperparameter_key_expression(
    hyperparameter_columns: list[str],
    *,
    table_alias: str,
) -> str:
    """Build hyperparameter key expression.

    Args:
        hyperparameter_columns (list[str]): list[str] value used by the function.
        table_alias (str): str value used by the function.

    Returns:
        str: Resolved string value.
    """
    key_parts = []
    for column in hyperparameter_columns:
        display_name = column.removeprefix("config.")
        key_parts.append(
            f"'{display_name}=' || COALESCE(CAST({table_alias}.{_quote_identifier(column)} AS VARCHAR), '__null__')"
        )

    return " || '|' || ".join(key_parts)


def build_select_query(con: duckdb.DuckDBPyConnection, parquet1: Path, parquet2: Path):
    """Build select query.

    Args:
        con (duckdb.DuckDBPyConnection): duckdb.DuckDBPyConnection value used by the function.
        parquet1 (Path): Path value used by the function.
        parquet2 (Path): Path value used by the function.

    Returns:
        Any: Result of the function.
    """
    p1_cols = get_columns(con, parquet1)
    hyperparameter_columns = get_hyperparameter_columns(con, parquet2)

    metrics_to_add = [col for col in METRIC_COLUMNS if col not in p1_cols]
    additional_selects = [f"p2.{col}" for col in metrics_to_add]
    added_columns = list(metrics_to_add)

    if HYPERPARAMETER_KEY_COLUMN not in p1_cols:
        hyperparameter_key_expr = build_hyperparameter_key_expression(
            hyperparameter_columns,
            table_alias="p2",
        )
        additional_selects.append(f"{hyperparameter_key_expr} AS {HYPERPARAMETER_KEY_COLUMN}")
        added_columns.append(HYPERPARAMETER_KEY_COLUMN)

    if additional_selects:
        additional_select = ",\n                ".join(additional_selects)
        select_clause = f"""SELECT
                p1.*,
                {additional_select}
            FROM read_parquet('{parquet1.as_posix()}') AS p1
            LEFT JOIN read_parquet('{parquet2.as_posix()}') AS p2
              ON p1.epoch = p2.training_iteration
             AND p1.model_id = p2.model_name_full"""
    else:
        select_clause = f"""SELECT
                *
            FROM read_parquet('{parquet1.as_posix()}')"""

    return select_clause, added_columns


def main():
    """Main.

    Returns:
        Any: Result of the function.
    """
    args = parse_args()

    parquet1 = validate_input(args.parquet1, "Parquet 1")
    parquet2 = validate_input(args.parquet2, "Parquet 2")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / parquet1.name

    con = duckdb.connect(database=":memory:")

    try:
        con.execute("PRAGMA threads=4;")
        con.execute(f"PRAGMA temp_directory='{save_dir.as_posix()}';")

        check_required_columns(con, parquet1, parquet2)
        check_no_duplicate_keys(con, parquet2)

        select_query, added_columns = build_select_query(con, parquet1, parquet2)

        copy_query = f"""
        COPY (
            {select_query}
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """

        con.execute(copy_query)

        if added_columns:
            print(f"Added columns: {added_columns}")
        else:
            print(
                "No new columns to add; parquet1 already contains all metric columns "
                f"and '{HYPERPARAMETER_KEY_COLUMN}'."
            )

        print(f"Done. Output written to: {output_path}")

    finally:
        con.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
