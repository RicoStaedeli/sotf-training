"""Utilities for reporting and saving results from generative tasks."""

from pathlib import Path
from typing import Dict, Any, List, Optional

from ray import logger
import torch
import numpy as np


def _round_metric(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float, np.floating, np.integer)):
        return round(float(value), 2)
    return value


def save_plots_for_results(
    results: Dict[str, Any],
    mode: str,
    save_sampled_plots: bool,
    zoo_paths: List[str],
    is_multi_zoo: bool,
    get_save_root) -> None:
    """Save accuracy plots for any mode using provided save-root resolver."""
    if not save_sampled_plots:
        return

    per_zoo = results.get("per_zoo", {})
    zoo_path_by_name = {Path(p).name: p for p in zoo_paths}

    if is_multi_zoo and per_zoo:
        for zoo_name, stats in per_zoo.items():
            zoo_path = zoo_path_by_name.get(zoo_name, zoo_name)
            save_root = get_save_root(zoo_path)
            if save_root is None:
                continue
            save_accuracy_plot(
                output_dir=save_root,
                zoo_name=zoo_name,
                original_accs=stats.get("original_accuracies"),
                reconstructed_accs=stats.get("reconstructed_accuracies"),
                sampled_accs=stats.get("accuracies"),
                num_sampled_total=None)
    else:
        zoo_path = zoo_paths[0] if zoo_paths else ""
        save_root = get_save_root(zoo_path)
        if save_root is not None:
            save_accuracy_plot(
                output_dir=save_root,
                original_accs=results.get("original_accuracies"),
                reconstructed_accs=results.get("reconstructed_accuracies"),
                sampled_accs=results.get("accuracies"),
                num_sampled_total=results.get("num_models_sampled") if mode == "sample" else None)

def save_accuracy_plot(
    output_dir: Path,
    zoo_name: Optional[str] = None,
    original_accs: Optional[List[float]] = None,
    reconstructed_accs: Optional[List[float]] = None,
    sampled_accs: Optional[List[float]] = None,
    num_sampled_total: Optional[int] = None) -> Optional[str]:
    """Save accuracy distribution plot for any combination of original/reconstructed/sampled models."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except Exception as exc:
        logger.warning(f"Plotting dependencies missing; skipping plot save: {exc}")
        return None

    # Convert to lists, filter None
    original_accs = list(original_accs) if original_accs else []
    reconstructed_accs = list(reconstructed_accs) if reconstructed_accs else []
    sampled_accs = list(sampled_accs) if sampled_accs else []

    if not any([original_accs, reconstructed_accs, sampled_accs]):
        return None

    # Save accuracies to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = f"unordered_accuracies_{zoo_name}.csv" if zoo_name else "unordered_accuracies.csv"
    csv_path = output_dir / csv_filename

    # Determine max length for padding
    max_len = max(len(original_accs), len(reconstructed_accs), len(sampled_accs))

    with open(csv_path, 'w') as f:
        f.write("original_acc,reconstructed_acc,sampled_acc\n")
        for i in range(max_len):
            orig = f"{original_accs[i]:.4f}" if i < len(original_accs) else ""
            recon = f"{reconstructed_accs[i]:.4f}" if i < len(reconstructed_accs) else ""
            samp = f"{sampled_accs[i]:.4f}" if i < len(sampled_accs) else ""
            f.write(f"{orig},{recon},{samp}\n")

    fig, ax = plt.subplots(figsize=(14, 7))
    bins, alpha = np.linspace(0, 100, 40), 0.6
    x = np.linspace(0, 100, 200)

    def plot_dist(accs, color, edge_color, label):
        if not accs:
            return
        ax.hist(accs, bins=bins, alpha=alpha, color=color, label=label,
                density=True, edgecolor=edge_color, linewidth=1.5)
        if len(accs) > 1:
            accs_array = np.array(accs, dtype=float)
            if np.std(accs_array) > 0:
                try:
                    ax.plot(x, gaussian_kde(accs_array)(x), color=edge_color, linewidth=2.5, linestyle="-", alpha=0.8)
                except Exception as exc:
                    logger.warning(f"Skipping KDE plot due to error: {exc}")
        ax.axvline(np.mean(accs), color=edge_color, linestyle="--", linewidth=2, alpha=0.7)

    if original_accs:
        plot_dist(original_accs, "red", "darkred",
                  f"Original (n={len(original_accs)}, μ={np.mean(original_accs):.1f}%)")
    if reconstructed_accs:
        plot_dist(reconstructed_accs, "gold", "orange",
                  f"Reconstructed (n={len(reconstructed_accs)}, μ={np.mean(reconstructed_accs):.1f}%)")
    if sampled_accs:
        n_total = num_sampled_total or len(sampled_accs)
        lbl = (f"Sampled - Top {len(sampled_accs)} of {n_total} (μ={np.mean(sampled_accs):.1f}%)"
               if n_total > len(sampled_accs)
               else f"Sampled (n={len(sampled_accs)}, μ={np.mean(sampled_accs):.1f}%)")
        plot_dist(sampled_accs, "blue", "darkblue", lbl)

    ax.set_xlabel("Test Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")

    # Dynamic title based on what's plotted
    parts = []
    if original_accs: parts.append("Original")
    if reconstructed_accs: parts.append("Reconstructed")
    if sampled_accs: parts.append("Sampled")
    title = f"Model Accuracy Distribution: {' vs '.join(parts)}"
    if zoo_name:
        title = f"{title} [{zoo_name}]"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 100)
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (f"accuracy_plot_{zoo_name}.png" if zoo_name else "accuracy_plot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def _flatten_per_zoo_stats(per_zoo: Dict, prefix: str, is_recon: bool) -> Dict[str, Any]:
    """Flatten per-zoo statistics into a minimal flat dictionary for Ray/WandB."""
    result = {}
    for zoo_name, stats in per_zoo.items():
        safe_name = zoo_name.replace("/", "_").replace("-", "_").replace(".", "_")
        if is_recon:
            recon_accs = stats.get("reconstructed_accuracies") or []
            orig_accs = stats.get("original_accuracies") or []
            recon_mean = stats.get("reconstructed_mean")
            if recon_mean is None and recon_accs:
                recon_mean = float(np.mean(recon_accs))
            orig_mean = stats.get("original_mean")
            if orig_mean is None and orig_accs:
                orig_mean = float(np.mean(orig_accs))
            result[f"{prefix}_{safe_name}_mean"] = _round_metric(recon_mean)
            result[f"{prefix}_{safe_name}_std"] = _round_metric(float(np.std(recon_accs)) if recon_accs else None)
            result[f"{prefix}_{safe_name}_delta"] = _round_metric(recon_mean - orig_mean if recon_mean is not None and orig_mean is not None else None)
        else:
            sample_accs = stats.get("accuracies") or []
            orig_accs = stats.get("original_accuracies") or []
            sample_mean = stats.get("accuracy_mean")
            if sample_mean is None and sample_accs:
                sample_mean = float(np.mean(sample_accs))
            orig_mean = float(np.mean(orig_accs)) if orig_accs else None
            result[f"{prefix}_{safe_name}_mean"] = _round_metric(sample_mean)
            result[f"{prefix}_{safe_name}_std"] = _round_metric(stats.get("accuracy_std"))
            if result[f"{prefix}_{safe_name}_std"] is None and sample_accs:
                result[f"{prefix}_{safe_name}_std"] = _round_metric(float(np.std(sample_accs)))
            result[f"{prefix}_{safe_name}_delta"] = _round_metric(sample_mean - orig_mean if sample_mean is not None and orig_mean is not None else None)
    return result


def extract_minimal_metrics(full_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract minimal metrics for Ray Tune tracking.
    Flattens per_zoo stats so they appear as individual metrics in WandB/Ray.
    """
    mode = full_results.get("mode")
    minimal = {"mode": mode, "dstsk_total_time_s": _round_metric(full_results.get("dstsk_total_time_s", 0))}

    if mode == "reconstruct":
        recon_accs = full_results.get("reconstructed_accuracies") or []
        orig_accs = full_results.get("original_accuracies") or []
        recon_mean = full_results.get("reconstructed_accuracy_mean")
        if recon_mean is None and recon_accs:
            recon_mean = float(np.mean(recon_accs))
        orig_mean = full_results.get("original_accuracy_mean")
        if orig_mean is None and orig_accs:
            orig_mean = float(np.mean(orig_accs))
        recon_std = float(np.std(recon_accs)) if recon_accs else None
        orig_std = float(np.std(orig_accs)) if orig_accs else None
        minimal.update({
            "recon_mean": _round_metric(recon_mean),
            "recon_std": _round_metric(recon_std),
            "recon_delta": _round_metric(recon_mean - orig_mean if recon_mean is not None and orig_mean is not None else None),
            "recon_delta_std": _round_metric(recon_std - orig_std if recon_std is not None and orig_std is not None else None)})
        minimal.update(_flatten_per_zoo_stats(full_results.get("per_zoo", {}), "recon", True))

    elif mode == "sample":
        sample_accs = full_results.get("accuracies") or []
        orig_accs = full_results.get("original_accuracies") or []
        sample_mean = full_results.get("accuracy_mean")
        if sample_mean is None and sample_accs:
            sample_mean = float(np.mean(sample_accs))
        orig_mean = float(np.mean(orig_accs)) if orig_accs else None
        sample_std = full_results.get("accuracy_std")
        if sample_std is None and sample_accs:
            sample_std = float(np.std(sample_accs))
        orig_std = float(np.std(orig_accs)) if orig_accs else None
        minimal.update({
            "sample_mean": _round_metric(sample_mean),
            "sample_std": _round_metric(sample_std),
            "sample_delta": _round_metric(sample_mean - orig_mean if sample_mean is not None and orig_mean is not None else None),
            "sample_delta_std": _round_metric(sample_std - orig_std if sample_std is not None and orig_std is not None else None)})
        minimal.update(_flatten_per_zoo_stats(full_results.get("per_zoo", {}), "sample", False))

    elif mode == "both":
        recon = full_results.get("reconstruction", {})
        sample = full_results.get("sampling", {})
        recon_accs = recon.get("reconstructed_accuracies") or []
        recon_orig_accs = recon.get("original_accuracies") or []
        recon_mean = recon.get("reconstructed_accuracy_mean")
        if recon_mean is None and recon_accs:
            recon_mean = float(np.mean(recon_accs))
        recon_orig_mean = recon.get("original_accuracy_mean")
        if recon_orig_mean is None and recon_orig_accs:
            recon_orig_mean = float(np.mean(recon_orig_accs))
        recon_std = float(np.std(recon_accs)) if recon_accs else None
        recon_orig_std = float(np.std(recon_orig_accs)) if recon_orig_accs else None
        sample_accs = sample.get("accuracies") or []
        sample_orig_accs = sample.get("original_accuracies") or []
        sample_mean = sample.get("accuracy_mean")
        if sample_mean is None and sample_accs:
            sample_mean = float(np.mean(sample_accs))
        sample_orig_mean = float(np.mean(sample_orig_accs)) if sample_orig_accs else None
        sample_std = sample.get("accuracy_std")
        if sample_std is None and sample_accs:
            sample_std = float(np.std(sample_accs))
        sample_orig_std = float(np.std(sample_orig_accs)) if sample_orig_accs else None
        minimal.update({
            "recon_mean": _round_metric(recon_mean),
            "recon_std": _round_metric(recon_std),
            "recon_delta": _round_metric(recon_mean - recon_orig_mean if recon_mean is not None and recon_orig_mean is not None else None),
            "recon_delta_std": _round_metric(recon_std - recon_orig_std if recon_std is not None and recon_orig_std is not None else None),
            "sample_mean": _round_metric(sample_mean),
            "sample_std": _round_metric(sample_std),
            "sample_delta": _round_metric(sample_mean - sample_orig_mean if sample_mean is not None and sample_orig_mean is not None else None),
            "sample_delta_std": _round_metric(sample_std - sample_orig_std if sample_std is not None and sample_orig_std is not None else None)})

        minimal.update(_flatten_per_zoo_stats(recon.get("per_zoo", {}), "recon", True))
        minimal.update(_flatten_per_zoo_stats(sample.get("per_zoo", {}), "sample", False))

    return minimal


def _save_sampled_models(state_dicts: List[Dict], accuracies: List[float], checkpoint_dir: Optional[str] = None, sampled_models_dir: Optional[str] = None) -> str:
    """
    Save sampled model state_dicts to disk.

    Args:
        state_dicts: List of model state dictionaries
        accuracies: List of model accuracies
        checkpoint_dir: Ray Tune checkpoint directory (e.g., trial_dir/checkpoint_000100)
                   Models will be saved to trial_dir/sampled_models/
        sampled_models_dir: Explicitly configured directory (takes precedence)

    Returns:
        Path where models were saved
    """
    # Determine output directory
    if sampled_models_dir:
        # Use explicitly specified directory
        output_dir = Path(sampled_models_dir)
    elif checkpoint_dir:
        # Extract trial directory from checkpoint path (go up one level from checkpoint_NNNNNN)
        # checkpoint_dir format: /path/to/trial_dir/checkpoint_000100
        checkpoint_path = Path(checkpoint_dir)
        trial_dir = checkpoint_path.parent
        output_dir = trial_dir / "sampled_models"
    else:
        raise ValueError("Either checkpoint_dir or sampled_models_dir must be provided")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each model with its accuracy in the filename
    logger.info(f"Saving {len(state_dicts)} sampled models to {output_dir}...")

    for i, (state_dict, acc) in enumerate(zip(state_dicts, accuracies)):
        # Format: model_001_acc_45.67.pt
        filename = f"model_{i+1:03d}_acc_{acc:.2f}.pt"
        filepath = output_dir / filename

        torch.save(state_dict, filepath)

    logger.info(f"Saved {len(state_dicts)} models to {output_dir}")

    return str(output_dir)