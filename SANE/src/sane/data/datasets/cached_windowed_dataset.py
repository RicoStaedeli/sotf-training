import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple, Union

import torch
import yaml

from sane.data.cache import DiskCache
from sane.data.datasets.windowed_dataset import WindowedDataset

logger = logging.getLogger(__name__)


class CachedWindowedDataset(WindowedDataset):
    """Windowed dataset with optional disk caching.

    - ``"none"``: Do not read or write disk cache entries. Windows are generated in
      memory on each access. This is useful for one-pass export jobs that call
      ``__getmodel__`` and would otherwise pay a disk write/read round trip.

    - ``"per_model"``: All windows for a checkpoint are stored together as a single cache entry.
      Simple and fast for small-to-medium models; reading one window deserializes the full list.

    - ``"per_window"``: Each window is stored as its own cache entry. Reading one window only
      deserializes that window — ~N× less I/O for checkpoints with N windows.

    By default the entire cache is preloaded on initialization to avoid lazy-generation
    races in multiprocessing DataLoader workers. This can be disabled for sequential
    workloads that benefit from generating cache entries on demand.

    Args:
        checkpoints_dataset: The underlying dataset of checkpoints.
        tokenizer: Tokenizer used to convert state dicts to token sequences.
        window_size: Number of tokens per window.
        num_windows_per_model: Windows to sample per checkpoint. "auto" = full coverage.
        num_windows_per_layer: Windows per layer (takes precedence over num_windows_per_model).
        pad_windows: Pad short windows to exactly window_size.
        allow_overlapping_windows: Allow overlapping window placement.
        window_distribution_strategy: "consecutive" or "distributed".
        cache_mode: ``"none"``, ``"per_model"`` or ``"per_window"``.
        cache_dir: Root directory for cache files.
        preload_cache: Whether to materialize cache entries for all checkpoints during
            initialization. When ``False``, cache entries are generated lazily on first
            access.
        split_name: Split identifier (e.g. "train"). Used as a subdirectory to avoid
            key collisions across splits.
        dataset_name: Dataset identifier (e.g. "cifar10_resnet18"). Used as a subdirectory
            prefix to avoid collisions across datasets.
        data_config_for_hash: Full data config dict. When provided the entire dict is hashed
            so any config change auto-invalidates the cache.
    """

    def __init__(
        self,
        checkpoints_dataset,
        tokenizer,
        window_size: int,
        num_windows_per_model: int | Literal["auto"] = 1,
        num_windows_per_layer: int | None = None,
        pad_windows: bool = False,
        window_distribution_strategy: Literal["consecutive", "distributed"] = "consecutive",
        cache_mode: Literal["none", "per_model", "per_window"] = "per_model",
        cache_dir: Optional[Union[str, Path]] = "/local/cache/tokens",
        global_cache: bool = True,
        preload_cache: bool = True,
        split_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        data_config_for_hash: Optional[dict] = None,
        allow_overlapping_windows: bool = False,
        **kwargs,
    ):
        super().__init__(
            checkpoints_dataset=checkpoints_dataset,
            tokenizer=tokenizer,
            window_size=window_size,
            num_windows_per_model=num_windows_per_model,
            num_windows_per_layer=num_windows_per_layer,
            pad_windows=pad_windows,
            allow_overlapping_windows=allow_overlapping_windows,
            window_distribution_strategy=window_distribution_strategy,
            **kwargs,
        )

        self._cache_mode = cache_mode
        self.split_name = split_name
        self.dataset_name = dataset_name
        self._data_config_for_hash = data_config_for_hash
        self._preload_cache = preload_cache

        self._cache_key_suffix = self._generate_cache_key_suffix()
        self._cached_checkpoints: Set[int] = set()
        self._actual_window_counts: dict[int, int] = {}
        self._disk_cache: DiskCache | None = None

        self._global_cache = global_cache
        self._base_cache_dir = Path(cache_dir or "/local/cache/tokens")
        dataset_part = self.dataset_name or "default"
        split_part = self.split_name or "default"

        if cache_mode == "none":
            logger.info(
                "Token cache disabled: mode=none, hash=%s, dataset=%s, split=%s",
                self._cache_key_suffix,
                dataset_part,
                split_part,
            )
        else:
            subdir = "checkpoint" if cache_mode == "per_model" else "window"
            config_dir = self._base_cache_dir / f"{dataset_part}_{self._cache_key_suffix}"
            if not global_cache:
                instance_id = uuid.uuid4().hex[:12]
                config_dir = config_dir / f"instance_{instance_id}"
            effective_cache_dir = config_dir / subdir / split_part

            self._disk_cache = DiskCache(
                cache_dir=effective_cache_dir,
                create_dirs=True,
            )
            logger.info(
                "Token cache setup: mode=%s, hash=%s, dataset=%s, split=%s, dir=%s",
                cache_mode,
                self._cache_key_suffix,
                dataset_part,
                split_part,
                effective_cache_dir,
            )

            if self._data_config_for_hash is not None:
                config_file = (
                    self._base_cache_dir
                    / f"{dataset_part}_{self._cache_key_suffix}"
                    / "data_config.yaml"
                )
                if not config_file.exists():
                    config_file.write_text(
                        yaml.dump(self._data_config_for_hash, default_flow_style=False)
                    )

            self._load_cache_metadata()
            self._log_cache_state()
            if self._preload_cache:
                self.preload_cache()
            else:
                logger.info("Token cache preload disabled; cache entries will be generated on demand")

        if self._actual_window_counts:
            min_count = min(self._actual_window_counts.values())
            if min_count != self._windows_per_checkpoint:
                logger.warning(
                    "Checkpoints have varying window counts (min=%d, probe=%d). "
                    "Using minimum to ensure all indices are valid.",
                    min_count,
                    self._windows_per_checkpoint,
                )
                self._windows_per_checkpoint = min_count

    def _get_window(
        self, checkpoint_idx: int, window_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cache_mode == "none":
            return self._generate_windows(checkpoint_idx)[window_idx]

        if checkpoint_idx not in self._cached_checkpoints:
            self._ensure_checkpoint_cached(checkpoint_idx)

        if self._cache_mode == "per_model":
            assert self._disk_cache is not None
            windows = self._disk_cache.get(self._get_cache_key(checkpoint_idx))
            if windows is None:
                raise RuntimeError(
                    f"Cache miss for checkpoint {checkpoint_idx} after cache population"
                )
            return windows[window_idx]
        else:
            assert self._disk_cache is not None
            window = self._disk_cache.get(
                self._get_window_cache_key(checkpoint_idx, window_idx)
            )
            if window is None:
                raise RuntimeError(
                    f"Cache miss for checkpoint {checkpoint_idx} window {window_idx} after cache population"
                )
            return window

    def _get_all_windows(
        self, checkpoint_idx: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._cache_mode == "none":
            return self._generate_windows(checkpoint_idx)

        if checkpoint_idx not in self._cached_checkpoints:
            self._ensure_checkpoint_cached(checkpoint_idx)

        if self._cache_mode == "per_model":
            assert self._disk_cache is not None
            windows = self._disk_cache.get(self._get_cache_key(checkpoint_idx))
            if windows is None:
                raise RuntimeError(
                    f"Cache miss for checkpoint {checkpoint_idx} after cache population"
                )
            return windows
        else:
            assert self._disk_cache is not None
            windows = [
                self._disk_cache.get(self._get_window_cache_key(checkpoint_idx, w))
                for w in range(self._windows_per_checkpoint)
            ]
            if any(w is None for w in windows):
                raise RuntimeError(
                    f"Incomplete cache for checkpoint {checkpoint_idx} after cache population"
                )
            return windows

    def _generate_cache_key_suffix(self) -> str:
        config_for_metadata = {
            "tokenizer_mode": self.tokenizer.mode,
            "tokenizer_size": self.tokenizer.tokensize,
            "tokenizer_padding": self.tokenizer.padding,
            "window_size": self.window_size,
            "num_windows_per_model": self.num_windows_per_model,
            "num_windows_per_layer": self.num_windows_per_layer,
            "pad_windows": self.pad_windows,
            "allow_overlapping": self.allow_overlapping_windows,
            "window_strategy": self.window_distribution_strategy,
        }
        self._config_params = {
            **config_for_metadata,
            "split_name": self.split_name or "none",
            "dataset_name": self.dataset_name or "none",
        }

        if self._data_config_for_hash is not None:
            config_str = json.dumps(
                self._data_config_for_hash, sort_keys=True, default=str
            )
        else:
            config_str = "_".join(f"{k}_{v}" for k, v in config_for_metadata.items())

        self._config_string = config_str
        logger.info(f"Cache key config: {config_str}")
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def _get_cache_key(self, checkpoint_idx: int) -> str:
        return f"checkpoint_{checkpoint_idx}"

    def _get_window_cache_key(self, checkpoint_idx: int, window_idx: int) -> str:
        return f"checkpoint_{checkpoint_idx}_window_{window_idx}"

    def _get_metadata_path(self) -> Path:
        if self._disk_cache is None:
            raise RuntimeError("Cache metadata is unavailable when cache_mode='none'")
        return self._disk_cache.cache_dir / "cache_metadata.json"

    def _load_cache_metadata(self):
        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            checkpoints_meta = metadata.get("checkpoints", {})
            self._cached_checkpoints = set(int(idx) for idx in checkpoints_meta.keys())
            for idx_str, info in checkpoints_meta.items():
                num_windows = info.get("num_windows")
                if num_windows is not None:
                    self._actual_window_counts[int(idx_str)] = int(num_windows)
            logger.info(
                f"Loaded metadata: {len(self._cached_checkpoints)} checkpoints cached"
            )
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}. Starting fresh.")
            self._cached_checkpoints = set()

    def _save_cache_metadata(self):
        metadata_path = self._get_metadata_path()
        try:
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata: dict = {
                    "config": {k: str(v) for k, v in self._config_params.items()},
                    "checkpoints": {},
                }

            for idx in self._cached_checkpoints:
                actual = self._actual_window_counts.get(idx, self._windows_per_checkpoint)
                if str(idx) not in metadata["checkpoints"]:
                    metadata["checkpoints"][str(idx)] = {
                        "num_windows": actual,
                        "cached_at": datetime.now().isoformat(),
                    }
                else:
                    metadata["checkpoints"][str(idx)]["num_windows"] = actual

            tmp = metadata_path.parent / f"{metadata_path.name}.tmp"
            with open(tmp, "w") as f:
                json.dump(metadata, f, indent=2)
            tmp.replace(metadata_path)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _log_cache_state(self):
        if self._disk_cache is None:
            logger.info(
                "Token cache state: mode=none, cached_files=0, cached_checkpoints=0"
            )
            return

        metadata_path = self._get_metadata_path()
        config_dir = self._disk_cache.cache_dir.parent.parent
        config_file = config_dir / "data_config.yaml"
        cache_file_count = len(list(self._disk_cache.cache_dir.glob("cache_*.pt")))

        logger.info(
            "Token cache state: metadata=%s (%s), config=%s (%s), cached_files=%d, cached_checkpoints=%d",
            metadata_path,
            "exists" if metadata_path.exists() else "missing",
            config_file,
            "exists" if config_file.exists() else "missing",
            cache_file_count,
            len(self._cached_checkpoints),
        )

    def get_cache_debug_info(self) -> dict[str, object]:
        if self._disk_cache is None:
            return {
                "mode": self._cache_mode,
                "hash": self._cache_key_suffix,
                "dataset": self.dataset_name or "default",
                "split": self.split_name or "default",
                "cache_dir": None,
                "config_dir": None,
                "metadata_path": None,
                "metadata_exists": False,
                "config_path": None,
                "config_exists": False,
                "cached_files": 0,
                "cached_checkpoints": 0,
                "windows_per_checkpoint": self._windows_per_checkpoint,
            }

        metadata_path = self._get_metadata_path()
        config_dir = self._disk_cache.cache_dir.parent.parent
        config_file = config_dir / "data_config.yaml"
        cache_files = list(self._disk_cache.cache_dir.glob("cache_*.pt"))

        return {
            "mode": self._cache_mode,
            "hash": self._cache_key_suffix,
            "dataset": self.dataset_name or "default",
            "split": self.split_name or "default",
            "cache_dir": str(self._disk_cache.cache_dir),
            "config_dir": str(config_dir),
            "metadata_path": str(metadata_path),
            "metadata_exists": metadata_path.exists(),
            "config_path": str(config_file),
            "config_exists": config_file.exists(),
            "cached_files": len(cache_files),
            "cached_checkpoints": len(self._cached_checkpoints),
            "windows_per_checkpoint": self._windows_per_checkpoint,
        }

    def _ensure_checkpoint_cached(self, checkpoint_idx: int, save_metadata: bool = True):
        """Cache all windows for a checkpoint if not already done."""
        if self._cache_mode == "none":
            return

        if checkpoint_idx in self._cached_checkpoints:
            return

        windows = self._generate_windows(checkpoint_idx)
        if not windows or not isinstance(windows, list):
            raise ValueError(
                f"Generated invalid windows for checkpoint {checkpoint_idx}"
            )

        if self._cache_mode == "per_model":
            assert self._disk_cache is not None
            self._disk_cache.put(self._get_cache_key(checkpoint_idx), windows)
        else:
            assert self._disk_cache is not None
            for window_idx, window in enumerate(windows):
                self._disk_cache.put(
                    self._get_window_cache_key(checkpoint_idx, window_idx), window
                )

        self._cached_checkpoints.add(checkpoint_idx)
        self._actual_window_counts[checkpoint_idx] = len(windows)
        if save_metadata:
            self._save_cache_metadata()

        logger.debug(f"Cached {len(windows)} windows for checkpoint {checkpoint_idx}")

    def clear_cache(self):
        """Remove all cached entries for this dataset."""
        if self._cache_mode == "none":
            logger.info("Cache disabled; nothing to clear")
            return

        if self._cache_mode == "per_model":
            assert self._disk_cache is not None
            for idx in range(len(self.checkpoints_dataset)):
                self._disk_cache.remove(self._get_cache_key(idx))
        else:
            assert self._disk_cache is not None
            for idx in range(len(self.checkpoints_dataset)):
                for w in range(self._windows_per_checkpoint):
                    self._disk_cache.remove(self._get_window_cache_key(idx, w))
            meta = self._get_metadata_path()
            if meta.exists():
                try:
                    meta.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete metadata file: {e}")

        self._cached_checkpoints.clear()
        logger.info("Cache cleared")

    def preload_cache(self, checkpoint_indices: Optional[List[int]] = None):
        """Preload cache for the given checkpoints (all if None)."""
        if self._cache_mode == "none":
            logger.info("Cache disabled; skipping preload")
            return

        if checkpoint_indices is None:
            checkpoint_indices = list(range(len(self.checkpoints_dataset)))

        pending = [
            idx for idx in checkpoint_indices if idx not in self._cached_checkpoints
        ]
        if not pending:
            logger.info("Cache already complete, skipping preload")
            return

        logger.info(
            f"Preloading cache ({self._cache_mode}) for {len(pending)} checkpoints..."
        )
        for i, idx in enumerate(pending):
            self._ensure_checkpoint_cached(idx, save_metadata=False)
            if (i + 1) % 10 == 0:
                logger.info(f"Preloaded {i + 1}/{len(pending)} checkpoints")

        self._save_cache_metadata()
        logger.info("Cache preload complete")
