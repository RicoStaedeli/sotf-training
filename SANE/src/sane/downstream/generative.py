from typing import Optional, Dict, Any, List, Union
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
from tqdm import tqdm

from sane.downstream.downstream_task import DownstreamTask
from sane.downstream.generative_task import (
    PerTokenKDESampler, encode_anchors, group_by_architecture,
    decode_embeddings, sample_and_decode,
    evaluate_model,
    extract_minimal_metrics, _save_sampled_models, save_plots_for_results)
from sane.data.datasets import CachedWindowedDataset, WindowedDataset, ZooDataset, CombinedCheckpointsDataset
from sane.model.autoencoder import SANEAutoEncoder

logger = logging.getLogger(__name__)

class GenerativeDownstreamTask(DownstreamTask):
    """
    Generative downstream task that evaluates model generation quality.

    Supports three modes:
    - 'reconstruct': Encode validation models and decode them back, then evaluate
    - 'sample': Sample new models from anchor distribution, then evaluate
    - 'both': Run both reconstruction and sampling

    Args:
        name: Task name for logging
        mode: Generation mode ('reconstruct', 'sample', or 'both')
        num_models: Models to reconstruct per zoo (e.g., 5 per zoo with 4 zoos = 20 total)
        max_checkpoints: Max models to consider from valset (limits evaluation cost)
        batch_size: Batch size for evaluation
        num_workers: Number of data loader workers
        kde_bandwidth: KDE bandwidth for sampling
        num_anchors: Anchor models for KDE fitting per zoo
        num_samples: Models to sample per zoo
        top_k_samples: Only report top-k best sampled models (None = report all)
        enable_bn_conditioning: Enable batch normalization conditioning after decoding
        bn_iterations: Number of forward passes for BN conditioning (default 200)
        verbose: Whether to print progress messages during execution
        encode_batch_size: Batch size for encoding models (default 64)
        decode_batch_size: Batch size for decoding embeddings (default 64)
        epochs: Epochs when task should run (None = every epoch)
        frequency: Run every N epochs (None = use epochs list)
        trainset: Training dataset (not used for zero-shot eval)
        valset: Validation dataset (provides tokenizer and checkpoints)
        testset: Test dataset (optional)
        save_sampled_models: Whether to save sampled model state_dicts to disk
        save_sampled_plots: Whether to save accuracy comparison plots alongside sampled models
        sampled_models_dir: Directory to save sampled models (if None, uses checkpoint_dir/sampled_models)
    """

    def __init__(self, name: str = "generative_dstk", mode: str = "reconstruct", num_models: int = 5,
                 max_checkpoints: Optional[int] = None, batch_size: int = 10,
                 num_workers: int = 4, kde_bandwidth: float = 0.1, num_anchors: int = 100, num_samples: int = 50,
                 top_k_samples: Optional[int] = None, enable_bn_conditioning: bool = False, bn_iterations: int = 200,
                 verbose: bool = True, encode_batch_size: int = 64, decode_batch_size: int = 64,
                 epochs: Optional[Union[int, List[int]]] = None, frequency: Optional[int] = None,
                 trainset: Optional[WindowedDataset] = None, valset: Optional[WindowedDataset] = None,
                 testset: Optional[WindowedDataset] = None, save_sampled_models: bool = False,
                 save_sampled_plots: bool = False,
                 sampled_models_dir: Optional[str] = None):
        super().__init__(name, epochs, frequency, trainset, valset, testset)

        self.mode = mode
        self.num_models = num_models
        self.max_checkpoints = max_checkpoints
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kde_bandwidth = kde_bandwidth
        self.num_anchors = num_anchors
        self.num_samples = num_samples
        self.top_k_samples = top_k_samples if top_k_samples is not None else num_samples
        self.save_sampled_models = save_sampled_models
        self.save_sampled_plots = save_sampled_plots
        self.sampled_models_dir = sampled_models_dir

        self.enable_bn_conditioning = enable_bn_conditioning
        self.bn_iterations = bn_iterations
        self.verbose = verbose
        self.encode_batch_size = encode_batch_size
        self.decode_batch_size = decode_batch_size

        # Cached data
        self._test_loaders = {}  # Cache per zoo-path test loaders
        self._tokens_dataset: Optional[CachedWindowedDataset] = None  # Cache tokens dataset for reuse across epochs

        # Tracking
        self._current_epoch = None
        self.original_accs = None  # Cache original accuracies

        # Multi-zoo support
        self._is_multi_zoo = False
        self._zoo_paths = []  # Store unique zoo paths for multi-zoo setups

        # Could maybe instead be added globally in trainer to enable TF32 for all tasks, but adding here to ensure it's set for generative task which can be slow without it
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _execute(self, sane_ae: SANEAutoEncoder, *args, **kwargs) -> Dict[str, Any]:
        """Execute the generative downstream task."""
        start_time = time.time()
        self._current_epoch = kwargs.get('epoch', None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._detect_multi_zoo()

        if self.verbose:
            logger.info(f"[{self.name}] Starting {self.mode} evaluation (epoch {self._current_epoch})")

        mode_runners = {
            "reconstruct": self._run_reconstruction,
            "sample": self._run_sampling,
            "both": self._run_both}

        if self.mode not in mode_runners:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'reconstruct', 'sample', or 'both'.")

        full_results = mode_runners[self.mode](sane_ae)
        full_results["dstsk_total_time_s"] = time.time() - start_time

        # Return minimal metrics for Ray Tune tracking
        minimal_results = extract_minimal_metrics(full_results)
        return {self.name: minimal_results}

    def _detect_multi_zoo(self):
        """Detect if we're using a multi-zoo setup and extract unique zoo paths."""
        if self.valset is None:
            self._is_multi_zoo = False
            return

        checkpoints_ds = self.valset.checkpoints_dataset
        if isinstance(checkpoints_ds, Subset):
            checkpoints_ds = checkpoints_ds.dataset

        if isinstance(checkpoints_ds, CombinedCheckpointsDataset):
            self._is_multi_zoo = True
            self._zoo_paths = [ds.root_dir for ds in checkpoints_ds.datasets if isinstance(ds, ZooDataset) and hasattr(ds, 'root_dir') and ds.root_dir]
            self._zoo_paths = list(dict.fromkeys(self._zoo_paths))  # Remove duplicates, preserve order
            if self.verbose:
                logger.info(f"Detected multi-zoo setup with {len(self._zoo_paths)} zoos")
        else:
            self._is_multi_zoo = False
            self._zoo_paths = [checkpoints_ds.root_dir] if isinstance(checkpoints_ds, ZooDataset) and hasattr(checkpoints_ds, 'root_dir') else []

    def _load_dataloader(self, zoo_path: str, split: str = 'testset', shuffle: bool = False) -> Optional[DataLoader]:
        """Load a dataloader from zoo's dataset.pt file with caching."""
        key = f"{split}_{zoo_path}"
        if key not in self._test_loaders:
            try:
                dataset_dict = torch.load(str(Path(zoo_path) / "dataset.pt"), weights_only=False)
                dataset = dataset_dict.get(split) or dataset_dict.get('testset')
                if dataset is None:
                    return None
                self._test_loaders[key] = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
            except Exception as e:
                logger.warning(f"Failed to load {split} from {zoo_path}: {e}")
                return None
        return self._test_loaders[key]

    def _run_reconstruction(self, sane_ae: SANEAutoEncoder) -> Dict[str, Any]:
        """Reconstruction mode: Encode validation models and decode them back."""
        valset_checkpoints = self.valset.checkpoints_dataset
        if self.max_checkpoints and len(valset_checkpoints) > self.max_checkpoints:
            valset_checkpoints = Subset(valset_checkpoints, list(range(self.max_checkpoints)))

        if self.verbose:
            logger.info(f"Using {len(valset_checkpoints)} validation models for reconstruction.")

        if self.original_accs is None:
            self.original_accs = self._extract_accuracies(valset_checkpoints)

        indices_by_zoo = self._group_indices_by_zoo(len(valset_checkpoints))

        all_results = []
        for zoo_path in self._zoo_paths:
            zoo_indices = indices_by_zoo[zoo_path]
            if zoo_indices:
                all_results.append(self._reconstruct_single_zoo(sane_ae, valset_checkpoints, zoo_path, zoo_indices))

        # Aggregate results
        per_zoo_stats = {}
        all_original, all_reconstructed = [], []

        for zoo_name, zoo_results in all_results:
            orig, recon = zoo_results['original'], zoo_results['reconstructed']
            all_original.extend(orig)
            all_reconstructed.extend(recon)
            per_zoo_stats[zoo_name] = {
                'original_accuracies': orig,
                'reconstructed_accuracies': recon}

        # Compute overall stats inline (mean-minus-mean for delta, std-minus-std for delta_std)
        orig_mean = float(np.mean(all_original)) if all_original else None
        orig_std = float(np.std(all_original)) if all_original else None
        recon_mean = float(np.mean(all_reconstructed)) if all_reconstructed else None
        recon_std = float(np.std(all_reconstructed)) if all_reconstructed else None
        result = {"mode": "reconstruct",
            "original_accuracy_mean": orig_mean,
            "reconstructed_accuracy_mean": recon_mean,
            "delta_mean": recon_mean - orig_mean if recon_mean is not None and orig_mean is not None else None,
            "delta_std": recon_std - orig_std if recon_std is not None and orig_std is not None else None,
            "original_accuracies": all_original,
            "reconstructed_accuracies": all_reconstructed}

        # Add per_zoo stats only for multi-zoo
        if self._is_multi_zoo and len(self._zoo_paths) > 1:
            result["per_zoo"] = per_zoo_stats

        if self.verbose:
            logger.info(f"[{self.name}] Reconstruction: {result['reconstructed_accuracy_mean']:.2f}% "
                       f"(Δmean {result['delta_mean']:+.2f}%, Δstd {result['delta_std']:+.2f}%)")

        save_plots_for_results(
            results=result,
            mode="reconstruct",
            save_sampled_plots=self.save_sampled_plots,
            zoo_paths=self._zoo_paths,
            is_multi_zoo=self._is_multi_zoo,
            get_save_root=self._get_sampled_models_root)
        return result


    def _encode_by_arch(self, sane_ae, valset_checkpoints, selected_indices, budget):
        """Group indices by architecture, select top-`budget` per group by accuracy, encode only those.

        Returns (embeddings, anchor_mask, anchor_pos, arch_groups, accs) where accs[i] is the original accuracy for model i.
        """
        all_accs = [self.original_accs[i] for i in selected_indices]
        raw_ckpts = [self._unwrap_checkpoint(Subset(valset_checkpoints, selected_indices)[i]) for i in range(len(selected_indices))]

        # Select top-budget per arch group
        encode_indices = []
        for group in group_by_architecture(raw_ckpts).values():
            ranked = sorted(group['indices'], key=lambda i: all_accs[i], reverse=True)[:budget]
            encode_indices.extend(ranked)

        final_indices = [selected_indices[i] for i in encode_indices]
        final_accs = [all_accs[i] for i in encode_indices]

        zoo_subset = Subset(valset_checkpoints, final_indices)
        tokens_subset = Subset(self._create_tokens_dataset(valset_checkpoints), final_indices)
        embeddings, anchor_mask, anchor_pos, checkpoints = encode_anchors(
            sane_ae=sane_ae, tokens_dataset=tokens_subset, zoo_dataset=zoo_subset,
            batch_size=self.encode_batch_size, verbose=self.verbose)

        return embeddings, anchor_mask, anchor_pos, group_by_architecture(checkpoints), final_accs

    def _reconstruct_single_zoo(self, sane_ae: SANEAutoEncoder, valset_checkpoints, zoo_path: str, selected_indices: List[int]) -> tuple:
        """Reconstruct models from a single zoo, grouped by architecture."""
        embeddings, anchor_mask, anchor_pos, arch_groups, accs = self._encode_by_arch(sane_ae, valset_checkpoints, selected_indices, self.num_models)

        if self.verbose:
            logger.info(f"[{self.name}] Reconstructing {len(accs)} models from {Path(zoo_path).name}")
            if len(arch_groups) > 1:
                logger.info(f"  Found {len(arch_groups)} architecture groups")

        bn_dataloader = self._load_dataloader(zoo_path, 'trainset', shuffle=True) if self.enable_bn_conditioning else None
        device = next(sane_ae.parameters()).device
        test_loader = self._load_dataloader(zoo_path, 'testset', shuffle=False)
        if test_loader is None:
            raise RuntimeError(f"Failed to load testset from {zoo_path}")

        original, reconstructed = [], []
        for group in arch_groups.values():
            group_indices = group['indices']
            group_template = group['template']

            sd_list = decode_embeddings(
                sane_ae=sane_ae, sampled_embeddings=embeddings[group_indices],
                anchor_checkpoint=group_template, anchor_mask=anchor_mask,
                anchor_positions=anchor_pos, tokenizer=self.valset.tokenizer,
                batch_size=self.decode_batch_size, verbose=self.verbose,
                enable_bn_conditioning=self.enable_bn_conditioning,
                bn_dataloader=bn_dataloader, bn_iterations=self.bn_iterations, device=device)

            template_model = group_template.model if hasattr(group_template, 'model') else group_template
            for local_idx, pos in enumerate(group_indices):
                if sd_list[local_idx] is not None:
                    original.append(float(accs[pos]))
                    reconstructed.append(float(evaluate_model(sd_list[local_idx], template_model, test_loader, device)))

        return Path(zoo_path).name, {'original': original, 'reconstructed': reconstructed}

    def _sample_from_kde(self, sane_ae: SANEAutoEncoder, sampler: PerTokenKDESampler,
                        anchor_checkpoint, anchor_mask: torch.Tensor, anchor_pos: torch.Tensor,
                        zoo_path: str, num_samples: int) -> tuple:
        """Core sampling logic: sample from KDE, decode, and evaluate.

        Returns:
            Tuple of (state_dicts, accuracies) - state_dicts needed for saving.
        """
        bn_dataloader = self._load_dataloader(zoo_path, 'trainset', shuffle=True) if self.enable_bn_conditioning else None

        state_dicts, template_model = sample_and_decode(
            sane_ae=sane_ae, kde_sampler=sampler, num_samples=num_samples,
            anchor_checkpoint=anchor_checkpoint, anchor_mask=anchor_mask, anchor_positions=anchor_pos,
            tokenizer=self.valset.tokenizer, batch_size=self.decode_batch_size,
            verbose=self.verbose, enable_bn_conditioning=self.enable_bn_conditioning,
            bn_dataloader=bn_dataloader, bn_iterations=self.bn_iterations)

        # Evaluate on zoo's testset
        test_loader = self._load_dataloader(zoo_path, 'testset', shuffle=False)
        if test_loader is None:
            raise RuntimeError(f"Failed to load testset from {zoo_path}")

        device = next(sane_ae.parameters()).device
        eval_iter = tqdm(state_dicts, desc="Evaluating", unit="model", leave=False) if self.verbose else state_dicts
        accuracies = [evaluate_model(sd, template_model, test_loader, device) for sd in eval_iter]

        return state_dicts, accuracies

    def _sample_single_zoo(self, sane_ae: SANEAutoEncoder, valset_checkpoints, zoo_path: str,
                           selected_indices: List[int], num_samples: int) -> tuple:
        """Sample models from a single zoo: fit KDE per architecture group, sample, decode, and evaluate."""
        embeddings, anchor_mask, anchor_pos, arch_groups, accs = self._encode_by_arch(sane_ae, valset_checkpoints, selected_indices, self.num_anchors)

        if self.verbose:
            logger.info(f"[{self.name}] Sampling {num_samples} models from {Path(zoo_path).name} ({len(accs)} anchors)")
            if len(arch_groups) > 1:
                logger.info(f"  Found {len(arch_groups)} architecture groups")

        all_state_dicts, all_accuracies, all_anchor_accs = [], [], []

        for group_idx, (arch_sig, group) in enumerate(arch_groups.items()):
            group_indices = group['indices']
            group_template = group['template']
            all_anchor_accs.extend(accs[i] for i in group_indices)

            if self.verbose and len(arch_groups) > 1:
                logger.info(f"  Group {group_idx+1}: {len(group_indices)} anchors, sampling {num_samples}")

            sampler = PerTokenKDESampler(bandwidth=self.kde_bandwidth)
            sampler.fit(embeddings[group_indices])

            state_dicts, accuracies = self._sample_from_kde(
                sane_ae=sane_ae, sampler=sampler, anchor_checkpoint=group_template,
                anchor_mask=anchor_mask, anchor_pos=anchor_pos,
                zoo_path=zoo_path, num_samples=num_samples)

            if self.top_k_samples < len(accuracies):
                top_idx = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)[:self.top_k_samples]
                accuracies = [accuracies[i] for i in top_idx]
                state_dicts = [state_dicts[i] for i in top_idx]

            all_state_dicts.extend(state_dicts)
            all_accuracies.extend(accuracies)

        # Save sampled models if enabled
        if self.save_sampled_models and all_state_dicts:
            save_root = self._get_sampled_models_root(zoo_path)
            if save_root is not None:
                _save_sampled_models(
                    state_dicts=all_state_dicts,
                    accuracies=all_accuracies,
                    checkpoint_dir=None,
                    sampled_models_dir=str(save_root))

        return Path(zoo_path).name, all_accuracies, all_anchor_accs

    def _get_sampled_models_root(self, zoo_path: str) -> Optional[Path]:
        """Determine the directory to save sampled models and plots for a given zoo."""
        if self.sampled_models_dir is None:
            logger.warning("save_sampled_models/save_sampled_plots=True but no sampled_models_dir available. Skipping save.")
            return None

        save_root = Path(self.sampled_models_dir)
        if self._current_epoch is not None:
            # Use actual epoch number so different frequencies don't overwrite each other
            checkpoint_name = f"checkpoint_{self._current_epoch:06d}"
            save_root = save_root / "sampled_models" / checkpoint_name

        if self._is_multi_zoo:
            save_root = save_root / Path(zoo_path).name
        return save_root

    def _run_sampling(self, sane_ae: SANEAutoEncoder) -> Dict[str, Any]:
        """Sampling mode: Sample new models from anchor distribution."""
        valset_checkpoints = self.valset.checkpoints_dataset
        if self.max_checkpoints and len(valset_checkpoints) > self.max_checkpoints:
            valset_checkpoints = Subset(valset_checkpoints, list(range(self.max_checkpoints)))

        if self.verbose:
            logger.info(f"Using {len(valset_checkpoints)} validation models for sampling.")

        if self.original_accs is None:
            self.original_accs = self._extract_accuracies(valset_checkpoints)

        indices_by_zoo = self._group_indices_by_zoo(len(valset_checkpoints))

        all_results = []
        for zoo_path in self._zoo_paths:
            if self.verbose:
                logger.info(f"Sampling from zoo: {Path(zoo_path).name} ({len(indices_by_zoo[zoo_path])} models)")

            zoo_indices = indices_by_zoo[zoo_path]
            if zoo_indices:
                all_results.append(self._sample_single_zoo(sane_ae, valset_checkpoints, zoo_path, zoo_indices, self.num_samples))

        # Aggregate results
        per_zoo_stats = {}
        all_accuracies = []
        all_anchor_accs = []

        for zoo_name, accuracies, anchor_accs in all_results:
            all_anchor_accs.extend(anchor_accs)
            all_accuracies.extend(accuracies)

            per_zoo_stats[zoo_name] = {
                'accuracies': accuracies,
                'original_accuracies': anchor_accs}

        result = {
            "mode": "sample",
            "accuracy_mean": float(np.mean(all_accuracies)) if all_accuracies else None,
            "accuracy_std": float(np.std(all_accuracies)) if all_accuracies else None,
            "accuracies": all_accuracies,
            "num_models_sampled": sum(len(r[1]) for r in all_results),
            "num_models_plotted": len(all_accuracies),
            "original_accuracies": all_anchor_accs}

        # Add per_zoo stats only for multi-zoo
        if self._is_multi_zoo and len(self._zoo_paths) > 1:
            result["per_zoo"] = per_zoo_stats

        if self.verbose:
            anchor_mean = float(np.mean(all_anchor_accs)) if all_anchor_accs else 0.0
            anchor_std = float(np.std(all_anchor_accs)) if all_anchor_accs else 0.0
            delta_mean = result['accuracy_mean'] - anchor_mean if result['accuracy_mean'] else 0.0
            delta_std = result['accuracy_std'] - anchor_std if result['accuracy_std'] else 0.0
            logger.info(f"[{self.name}] Sampling: {result['accuracy_mean']:.2f}% "
                       f"(Δmean {delta_mean:+.2f}%, Δstd {delta_std:+.2f}%)")

        save_plots_for_results(
            results=result,
            mode="sample",
            save_sampled_plots=self.save_sampled_plots,
            zoo_paths=self._zoo_paths,
            is_multi_zoo=self._is_multi_zoo,
            get_save_root=self._get_sampled_models_root)
        return result

    def _run_both(self, sane_ae: SANEAutoEncoder) -> Dict[str, Any]:
        """Both modes: Run reconstruction and sampling, then save combined plot."""
        # Temporarily disable plots in sub-methods to avoid duplicates
        save_plots = self.save_sampled_plots
        self.save_sampled_plots = False

        recon_results = self._run_reconstruction(sane_ae)
        sample_results = self._run_sampling(sane_ae)

        self.save_sampled_plots = save_plots

        # Merge results for combined plotting (original + reconstructed + sampled)
        if save_plots:
            merged = {
                "original_accuracies": recon_results.get("original_accuracies"),
                "reconstructed_accuracies": recon_results.get("reconstructed_accuracies"),
                "accuracies": sample_results.get("accuracies"),
                "num_models_sampled": sample_results.get("num_models_sampled"),
                "num_models_plotted": sample_results.get("num_models_plotted")}

            if "per_zoo" in recon_results and "per_zoo" in sample_results:
                merged["per_zoo"] = {}
                for zoo_name in recon_results["per_zoo"]:
                    recon_stats = recon_results["per_zoo"][zoo_name]
                    sample_stats = sample_results["per_zoo"].get(zoo_name, {})
                    merged["per_zoo"][zoo_name] = {
                        "original_accuracies": recon_stats.get("original_accuracies"),
                        "reconstructed_accuracies": recon_stats.get("reconstructed_accuracies"),
                        "accuracies": sample_stats.get("accuracies")}

            save_plots_for_results(
                results=merged,
                mode="both",
                save_sampled_plots=self.save_sampled_plots,
                zoo_paths=self._zoo_paths,
                is_multi_zoo=self._is_multi_zoo,
                get_save_root=self._get_sampled_models_root)

        if self.verbose:
            # Use anchor accuracies from sample_results for consistent delta
            sample_orig_accs = sample_results.get('original_accuracies', [])
            anchor_mean = float(np.mean(sample_orig_accs)) if sample_orig_accs else 0.0
            anchor_std = float(np.std(sample_orig_accs)) if sample_orig_accs else 0.0
            sample_delta_mean = sample_results['accuracy_mean'] - anchor_mean if sample_results['accuracy_mean'] else 0.0
            sample_delta_std = sample_results['accuracy_std'] - anchor_std if sample_results['accuracy_std'] else 0.0
            logger.info(f"[{self.name}] Both: "
                       f"Reconstruction Δmean {recon_results['delta_mean']:+.2f}%, Δstd {recon_results['delta_std']:+.2f}% | "
                       f"Sampling Δmean {sample_delta_mean:+.2f}%, Δstd {sample_delta_std:+.2f}%")

        return {
            "mode": "both",
            "reconstruction": recon_results,
            "sampling": sample_results}

    def _group_indices_by_zoo(self, num_available: int) -> Dict[str, List[int]]:
        """Group model indices by their zoo path."""
        valset_checkpoints = self.valset.checkpoints_dataset
        underlying = valset_checkpoints.dataset if isinstance(valset_checkpoints, Subset) else valset_checkpoints
        is_combined = isinstance(underlying, CombinedCheckpointsDataset)

        indices_by_zoo = {zoo_path: [] for zoo_path in self._zoo_paths}

        for idx in range(num_available):
            if not is_combined:
                indices_by_zoo[self._zoo_paths[0]].append(idx)
                continue

            # Get the result from underlying dataset (handling Subset)
            global_idx = valset_checkpoints.indices[idx] if isinstance(valset_checkpoints, Subset) else idx
            checkpoint = self._unwrap_checkpoint(underlying[global_idx])
            zoo_path = checkpoint.zoo_root_dir or self._zoo_paths[0]
            indices_by_zoo.get(zoo_path, indices_by_zoo[self._zoo_paths[0]]).append(idx)

        return indices_by_zoo

    def _unwrap_checkpoint(self, result):
        """Extract a single Checkpoint from wrapper formats (list from augmentations)."""
        return result[0] if isinstance(result, list) else result

    def _extract_accuracies(self, zoo_dataset: Any) -> List[float]:
        """Extract test accuracies from checkpoint metadata."""
        accuracies = []
        for i in range(len(zoo_dataset)):
            try:
                checkpoint = self._unwrap_checkpoint(zoo_dataset[i])
                acc = checkpoint.metadata.get('eval_results', {}).get('test_acc', 0)
                accuracies.append(acc * 100.0 if acc < 1.0 else acc)
            except (IndexError, KeyError, AttributeError) as e:
                logger.warning(f"Could not extract accuracy for model {i}: {e}")
                accuracies.append(0)
        return accuracies

    def _create_tokens_dataset(self, zoo_dataset: ZooDataset) -> CachedWindowedDataset:
        """Create or return cached windowed dataset for anchor encoding."""
        if self._tokens_dataset is not None:
            return self._tokens_dataset

        self._tokens_dataset = CachedWindowedDataset(
            checkpoints_dataset=zoo_dataset,
            tokenizer=self.valset.tokenizer,
            window_size=self.valset.window_size,
            num_windows_per_model=1,
            pad_windows=self.valset.pad_windows,
            allow_overlapping_windows=False,
            split_name="anchors",
        )
        return self._tokens_dataset
