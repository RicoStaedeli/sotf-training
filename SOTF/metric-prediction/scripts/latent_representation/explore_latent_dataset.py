from __future__ import annotations

"""
Exploratory analysis of the autoencoder latent representation dataset.

Usage:
    uv run python -m metric_prediction.scripts.latent_representation.explore_latent_dataset \
        [--path /path/to/latent_z_vectors.pt] \
        [--target-parquet /path/to/prepared_dataset.parquet]
"""

import argparse
from collections import Counter, defaultdict
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "Times New Roman"

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).parents[4] / "cache/latent_representation_datasets/latent_z_vectors.pt"


def load_latent_dataset(path: Path) -> dict:
    """Load latent dataset.

    Args:
        path (Path): Path to process.

    Returns:
        dict: Result of the function.
    """
    log.info(f"Loading latent dataset from {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def inspect_structure(data: dict) -> None:
    """Inspect structure.

    Args:
        data (dict): dict value used by the function.

    Returns:
        None: None.
    """
    print("\n=== Dataset Structure ===")
    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: Tensor shape={val.shape}, dtype={val.dtype}")
            elif isinstance(val, (list, np.ndarray)):
                try:
                    arr = np.array(val)
                    print(f"  {key}: array-like shape={arr.shape}, dtype={arr.dtype}")
                except ValueError:
                    print(f"  {key}: list len={len(val)}, inhomogeneous shapes, e.g. {getattr(val[0], 'shape', type(val[0]))}")
            else:
                print(f"  {key}: {type(val)} = {val!r:.200}")
    elif isinstance(data, torch.Tensor):
        print(f"Single tensor: shape={data.shape}, dtype={data.dtype}")
    else:
        print(f"Unknown structure: {data}")


def show_example_row(data: dict) -> None:
    """Show example row.

    Args:
        data (dict): dict value used by the function.

    Returns:
        None: None.
    """
    vec_keys = {k for k in data if any(x in k.lower() for x in ("z", "latent", "embed", "vec", "repr"))}

    print("\n=== Example Row (index 0, embeddings excluded) ===")
    for key, val in data.items():
        if key in vec_keys:
            continue
        if isinstance(val, torch.Tensor):
            entry = val[0]
            print(f"  {key}: {entry} (shape={tuple(entry.shape)}, dtype={entry.dtype})" if entry.ndim > 0 else f"  {key}: {entry.item()}")
        elif isinstance(val, (list, np.ndarray)):
            print(f"  {key}: {val[0]!r}")
        else:
            print(f"  {key}: {val!r:.200}")

    # Highlight epochs specifically
    epoch_keys = [k for k in data if "epoch" in k.lower()]
    if epoch_keys:
        print("\n--- epochs column detail ---")
        for k in epoch_keys:
            val = data[k]
            arr = val.numpy() if isinstance(val, torch.Tensor) else np.array(val)
            print(f"  {k}: dtype={arr.dtype}, shape={arr.shape}")
            print(f"    first 10 values: {arr[:10]}")
            print(f"    unique values:   {np.unique(arr)}")


def _as_list(values) -> list:
    """As list.

    Args:
        values (Any): Values to process.

    Returns:
        list: Result of the function.
    """
    if isinstance(values, torch.Tensor):
        return values.cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def _as_numpy(values) -> np.ndarray | None:
    """Convert common dataset containers to numpy arrays."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, list):
        return np.asarray(values)
    return None


def _as_2d_vectors(values, key: str) -> np.ndarray | None:
    """Convert latent vectors to a 2D matrix for stats/PCA."""
    # Handle inhomogeneous list of 2D tensors/arrays (e.g. windows mode: each
    # entry is (N_tokens, D) with variable N_tokens) — mean-pool over tokens.
    if isinstance(values, list) and len(values) > 0:
        first = values[0]
        if isinstance(first, torch.Tensor) and first.ndim == 2:
            pooled = np.stack([t.detach().cpu().float().mean(dim=0).numpy() for t in values])
            print(f"Mean-pooled '{key}' from list of (N_i, {first.shape[-1]}) tensors → {pooled.shape}")
            return pooled
        if isinstance(first, np.ndarray) and first.ndim == 2:
            pooled = np.stack([a.mean(axis=0) for a in values])
            print(f"Mean-pooled '{key}' from list of (N_i, {first.shape[-1]}) arrays → {pooled.shape}")
            return pooled

    vectors = _as_numpy(values)
    if vectors is None:
        return None

    if vectors.ndim == 3 and vectors.shape[1] == 1:
        vectors = vectors[:, 0, :]
        print(f"Squeezed '{key}' from (n, 1, d) to {vectors.shape}")
    elif vectors.ndim > 2:
        original_shape = vectors.shape
        vectors = vectors.reshape(vectors.shape[0], -1)
        print(f"Flattened '{key}' from {original_shape} to {vectors.shape}")

    if vectors.ndim != 2:
        return None
    return vectors


def _format_value(value) -> str:
    """Format value.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    if isinstance(value, torch.Tensor):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _checkpoint_label(path_value) -> str:
    """Checkpoint label.

    Args:
        path_value (Any): Value used by the function.

    Returns:
        str: Resolved string value.
    """
    path = Path(_format_value(path_value))
    if path.name == "checkpoints" and path.parent.name:
        return path.parent.name
    return path.name


def _find_key(data: dict, candidates: tuple[str, ...], required_terms: tuple[str, ...] = ()) -> str | None:
    """Find key.

    Args:
        data (dict): dict value used by the function.
        candidates (tuple[str, ...]): tuple[str, ...] value used by the function.
        required_terms (tuple[str, ...]): tuple[str, ...] value used by the function.

    Returns:
        str | None: Result of the function.
    """
    lower_to_key = {key.lower(): key for key in data}
    for candidate in candidates:
        if candidate in lower_to_key:
            return lower_to_key[candidate]

    for key in data:
        key_lower = key.lower()
        if all(term in key_lower for term in required_terms):
            return key
    return None


def check_checkpoint_consistency(data: dict) -> None:
    """Check checkpoint consistency.

    Args:
        data (dict): dict value used by the function.

    Returns:
        None: None.
    """
    print("\n=== Checkpoint Consistency ===")
    if not isinstance(data, dict):
        print("Skipping: dataset is not a dict.")
        return

    model_key = _find_key(
        data,
        candidates=("model_id", "model_ids", "model", "models"),
        required_terms=("model",),
    )
    checkpoint_key = _find_key(
        data,
        candidates=("checkpoint_path", "checkpoint_paths", "checkpoint", "checkpoints"),
        required_terms=("checkpoint",),
    )
    epoch_key = _find_key(
        data,
        candidates=("epoch", "epochs"),
        required_terms=("epoch",),
    )

    if model_key is None:
        print("Skipping: could not detect a model-id column.")
        return
    if checkpoint_key is None and epoch_key is None:
        print("Skipping: could not detect checkpoint or epoch columns.")
        return

    item_key = checkpoint_key or epoch_key
    item_name = "checkpoint" if checkpoint_key else "epoch"

    model_ids = [Path(_format_value(model_id)).name for model_id in _as_list(data[model_key])]
    raw_items = _as_list(data[item_key])
    if len(model_ids) != len(raw_items):
        print(
            f"Skipping: '{model_key}' has {len(model_ids)} entries but "
            f"'{item_key}' has {len(raw_items)} entries."
        )
        return

    items_by_model: dict[str, set[str]] = defaultdict(set)
    row_counts_by_model: Counter[str] = Counter()
    duplicate_counts: Counter[tuple[str, str]] = Counter()

    for model_id, raw_item in zip(model_ids, raw_items):
        item = _checkpoint_label(raw_item) if checkpoint_key else _format_value(raw_item)
        items_by_model[model_id].add(item)
        row_counts_by_model[model_id] += 1
        duplicate_counts[(model_id, item)] += 1

    signatures = Counter(frozenset(items) for items in items_by_model.values())
    reference_items = set(signatures.most_common(1)[0][0])
    mismatches = {
        model_id: items
        for model_id, items in items_by_model.items()
        if items != reference_items
    }
    duplicate_rows = {
        (model_id, item): count
        for (model_id, item), count in duplicate_counts.items()
        if count > 1
    }

    print(f"Using '{model_key}' grouped by '{item_key}'")
    print(f"Models: {len(items_by_model)}")
    print(f"Expected unique {item_name}s per model: {len(reference_items)}")

    if not mismatches:
        print(f"OK: all models have the same set of {item_name}s.")
    else:
        print(f"Mismatch: {len(mismatches)} / {len(items_by_model)} models differ.")
        for model_id, items in list(mismatches.items())[:10]:
            missing = sorted(reference_items - items)
            extra = sorted(items - reference_items)
            print(f"  {model_id}: {len(items)} unique {item_name}s")
            print(f"    missing ({len(missing)}): {missing[:10]}")
            print(f"    extra   ({len(extra)}): {extra[:10]}")
        if len(mismatches) > 10:
            print(f"  ... {len(mismatches) - 10} more mismatched models omitted")

    row_count_values = Counter(row_counts_by_model.values())
    print(f"Rows per model distribution: {dict(sorted(row_count_values.items()))}")
    if duplicate_rows:
        print(f"Duplicate model/{item_name} rows: {len(duplicate_rows)}")
        for (model_id, item), count in list(duplicate_rows.items())[:10]:
            print(f"  {model_id} / {item}: {count} rows")
        if len(duplicate_rows) > 10:
            print(f"  ... {len(duplicate_rows) - 10} more duplicate pairs omitted")


def extract_model_ids_and_vectors(data: dict) -> tuple[list, np.ndarray]:
    """Extract model ids and vectors.

    Args:
        data (dict): dict value used by the function.

    Returns:
        tuple[list, np.ndarray]: Result of the function.
    """
    # Try common key patterns
    id_keys = [k for k in data if "id" in k.lower() or "model" in k.lower()]
    vec_keys = [k for k in data if any(x in k.lower() for x in ("z", "latent", "embed", "vec", "repr"))]

    print(f"\nLikely ID keys:     {id_keys}")
    print(f"Likely vector keys: {vec_keys}")

    model_ids = None
    for k in id_keys:
        model_ids = data[k]
        print(f"Using '{k}' as model IDs — {len(model_ids)} entries, sample: {list(model_ids)[:5]}")
        break

    vectors = None
    for k in vec_keys:
        vectors = _as_2d_vectors(data[k], k)
        if vectors is not None:
            print(f"Using '{k}' as latent vectors — shape: {vectors.shape}")
            break

    if model_ids is None or vectors is None:
        print("\nCould not auto-detect keys. Dumping all array-like values:")
        for k, v in data.items():
            arr = _as_numpy(v)
            if arr is not None:
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        raise ValueError("Could not identify model ID and vector keys. Check key names above.")

    return list(model_ids), vectors


def basic_statistics(vectors: np.ndarray) -> None:
    """Basic statistics.

    Args:
        vectors (np.ndarray): np.ndarray value used by the function.

    Returns:
        None: None.
    """
    print("\n=== Latent Vector Statistics ===")
    print(f"Shape:           {vectors.shape}  ({vectors.shape[0]} models × {vectors.shape[1]} dims)")
    print(f"Dtype:           {vectors.dtype}")
    print(f"NaN count:       {np.isnan(vectors).sum()}")
    print(f"Inf count:       {np.isinf(vectors).sum()}")
    print(f"Global mean:     {vectors.mean():.4f}")
    print(f"Global std:      {vectors.std():.4f}")
    print(f"Per-dim mean range:  [{vectors.mean(axis=0).min():.4f}, {vectors.mean(axis=0).max():.4f}]")
    print(f"Per-dim std range:   [{vectors.std(axis=0).min():.4f}, {vectors.std(axis=0).max():.4f}]")

    # Dimension collapse check
    dead_dims = (vectors.std(axis=0) < 1e-6).sum()
    print(f"Near-zero-std dims:  {dead_dims} / {vectors.shape[1]}")


def pairwise_similarity(vectors: np.ndarray) -> None:
    """Pairwise similarity.

    Args:
        vectors (np.ndarray): np.ndarray value used by the function.

    Returns:
        None: None.
    """
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, np.nan)
    print(f"\n=== Pairwise Cosine Similarity ===")
    print(f"Mean:   {np.nanmean(sim_matrix):.4f}")
    print(f"Std:    {np.nanstd(sim_matrix):.4f}")
    print(f"Max:    {np.nanmax(sim_matrix):.4f}")
    print(f"Min:    {np.nanmin(sim_matrix):.4f}")


def pca_analysis(vectors: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Pca analysis.

    Args:
        vectors (np.ndarray): np.ndarray value used by the function.
        n_components (int): int value used by the function.

    Returns:
        np.ndarray: Result of the function.
    """
    n = min(n_components, vectors.shape[1], vectors.shape[0])
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(vectors)
    evr = pca.explained_variance_ratio_

    print(f"\n=== PCA ({n} components) ===")
    print(f"Explained variance: {evr[:5].round(3)} ...")
    print(f"Cumulative (5 / 10 / {n}): "
          f"{evr[:5].sum():.3f} / {evr[:10].sum():.3f} / {evr.sum():.3f}")
    return reduced


def join_with_targets(
    model_ids: list,
    vectors: np.ndarray,
    parquet_path: Path,
) -> pd.DataFrame:
    """Join with targets.

    Args:
        model_ids (list): list value used by the function.
        vectors (np.ndarray): np.ndarray value used by the function.
        parquet_path (Path): Path to the parquet.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    print(f"\n=== Joining with target parquet: {parquet_path} ===")
    df_target = pd.read_parquet(parquet_path, columns=["model_id", "epoch", "test_acc", "train_acc"])
    # Use last available epoch per model as target
    df_last = df_target.sort_values("epoch").groupby("model_id").last().reset_index()

    # Latent dataset stores full paths; parquet uses only the basename
    model_ids_normalized = [Path(mid).name for mid in model_ids]
    n_changed = sum(a != b for a, b in zip(model_ids, model_ids_normalized))
    if n_changed:
        print(f"Normalized {n_changed} model IDs to basename (was full path)")
        print(f"  Example: .../{model_ids_normalized[0]}")

    df_latent = pd.DataFrame(vectors, columns=[f"z_{i}" for i in range(vectors.shape[1])])
    df_latent["model_id"] = model_ids_normalized

    print(f"Sample parquet model_ids:  {df_last['model_id'].iloc[:3].tolist()}")
    print(f"Sample latent model_ids:   {model_ids_normalized[:3]}")

    merged = df_latent.merge(df_last[["model_id", "test_acc", "train_acc"]], on="model_id", how="inner")
    print(f"Matched {len(merged)} / {len(model_ids)} latent vectors to target values")
    print(merged[["model_id", "test_acc", "train_acc"]].describe())
    return merged


def plot_pca_2d(reduced: np.ndarray, targets: np.ndarray | None = None, out_dir: Path | None = None) -> None:
    """Plot pca 2d.

    Args:
        reduced (np.ndarray): np.ndarray value used by the function.
        targets (np.ndarray | None): np.ndarray | None value used by the function.
        out_dir (Path | None): Path | None value used by the function.

    Returns:
        None: None.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=targets if targets is not None else "steelblue",
        cmap="viridis" if targets is not None else None,
        alpha=0.6, s=15,
    )
    if targets is not None:
        plt.colorbar(sc, ax=ax, label="test_acc")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Latent Space — PCA (2D)")
    plt.tight_layout()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "latent_pca_2d.png", dpi=150)
        print(f"Saved PCA plot to {out_dir / 'latent_pca_2d.png'}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Main.

    Returns:
        None: None.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Explore the autoencoder latent dataset")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Path to .pt latent dataset")
    parser.add_argument(
        "--target-parquet", type=Path, default=None,
        help="Optional: path to prepared_dataset.parquet to join target metrics",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save plots")
    args = parser.parse_args()

    data = load_latent_dataset(args.path)
    inspect_structure(data)
    show_example_row(data)
    check_checkpoint_consistency(data)
    model_ids, vectors = extract_model_ids_and_vectors(data)
    basic_statistics(vectors)
    pairwise_similarity(vectors)
    reduced = pca_analysis(vectors)

    targets = None
    if args.target_parquet:
        merged = join_with_targets(model_ids, vectors, args.target_parquet)
        # Align targets with PCA order using normalized IDs
        id_to_target = dict(zip(merged["model_id"], merged["test_acc"]))
        model_ids_normalized = [Path(mid).name for mid in model_ids]
        targets = np.array([id_to_target.get(mid, np.nan) for mid in model_ids_normalized])

    plot_pca_2d(reduced, targets=targets, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
