from pathlib import Path
import os
import json
import random
import copy
import torch
from torch.utils.data import Dataset
import tqdm

from .dataset_auxiliaries import (
    test_checkpoint_for_nan,
    test_checkpoint_with_threshold,
)


class ModelDatasetBase(Dataset):
    """
    Lazy-loading dataset for model checkpoints.

    __init__ only scans files and builds an index of samples.
    Actual checkpoint loading happens in __getitem__.
    """

    def __init__(
        self,
        root,
        layer_lst=None,
        epoch_lst=10,
        use_bias=True,
        train_val_test="train",
        ds_split=[0.7, 0.3],
        max_samples=None,
        weight_threshold=float("inf"),
        filter_function=None,
        property_keys=None,
        num_threads=4,   # kept for API compatibility, unused
        shuffle_path=True,
        verbosity=0,
        preload_properties=False,
        validate_checkpoints=False,
    ):
        self.layer_lst = layer_lst
        self.use_bias = use_bias
        self.verbosity = verbosity
        self.weight_threshold = weight_threshold
        self.property_keys = copy.deepcopy(property_keys)
        self.train_val_test = train_val_test
        self.ds_split = ds_split
        self.num_threads = num_threads
        self.preload_properties = preload_properties
        self.validate_checkpoints = validate_checkpoints

        if not isinstance(root, list):
            root = [root]

        self.root = []
        for rdx in root:
            if isinstance(rdx, (str, os.PathLike, Path)):
                self.root.append(Path(rdx))
            else:
                raise TypeError(f"root entries must be path-like, got {type(rdx)}: {rdx!r}")

        if not isinstance(epoch_lst, list):
            epoch_lst = [epoch_lst]
        self.epoch_lst = epoch_lst

        self.path_list = []
        for rdx in self.root:
            if not rdx.exists():
                raise FileNotFoundError(f"Root path does not exist: {rdx}")
            self.path_list.extend([f for f in rdx.iterdir() if f.is_dir()])

        if shuffle_path:
            rng = random.Random(42)
            rng.shuffle(self.path_list)

        if max_samples is not None:
            self.path_list = self.path_list[:max_samples]

        self.path_list = self._split_path_list(self.path_list, train_val_test, ds_split)

        self.samples = self._build_sample_index(
            filter_function=filter_function,
            validate_checkpoints=validate_checkpoints,
        )

        self.properties = None
        if self.property_keys is not None and preload_properties:
            self._preload_properties()

        if self.verbosity > 0:
            print(
                f"Initialized lazy dataset with {len(self.samples)} samples "
                f"for split '{self.train_val_test}'."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        path = sample["path"]
        epoch = sample["epoch"]

        x = self._load_checkpoint(path, epoch)
        self._validate_checkpoint_or_raise(x, path, epoch)

        return x, x

    def __get_weights__(self):
        """
        Materialize the full dataset as a list of (x, x) tuples.
        This is expensive and defeats the purpose of lazy loading.
        """
        return [self[idx] for idx in tqdm.tqdm(range(len(self)))]

    def _split_path_list(self, path_list, split_name, ds_split):
        if split_name in (None, "full", "all"):
            if self.verbosity > 0:
                print("No dataset split requested. Loading full dataset.")
            return path_list

        if not ds_split:
            if self.verbosity > 0:
                print("No valid ds_split provided. Loading full dataset.")
            return path_list

        if abs(sum(ds_split) - 1.0) > 1e-8:
            raise AssertionError(f"dataset splits do not equal 1: {ds_split}")

        n = len(path_list)

        if len(ds_split) == 2:
            idx1 = int(ds_split[0] * n)
            if split_name == "train":
                return path_list[:idx1]
            elif split_name == "test":
                return path_list[idx1:]
            else:
                raise NotImplementedError(
                    "validation split requested, but only two splits provided."
                )

        elif len(ds_split) == 3:
            idx1 = int(ds_split[0] * n)
            idx2 = idx1 + int(ds_split[1] * n)

            if split_name == "train":
                return path_list[:idx1]
            elif split_name == "val":
                return path_list[idx1:idx2]
            elif split_name == "test":
                return path_list[idx2:]
            else:
                raise ValueError(
                    f"train_val_test must be one of ['train', 'val', 'test', 'full'], got {split_name}"
                )

        raise ValueError(f"Invalid ds_split: {ds_split}")

    def _build_sample_index(self, filter_function=None, validate_checkpoints=False):
        samples = []

        total_steps = len(self.path_list) * len(self.epoch_lst)
        iterator = tqdm.tqdm(
            total=total_steps,
            desc="Indexing dataset",
            disable=(self.verbosity < 0),
        )

        for path in self.path_list:
            if filter_function is not None:
                try:
                    if filter_function(path):
                        for _ in self.epoch_lst:
                            iterator.update(1)
                        continue
                except Exception as e:
                    if self.verbosity > 0:
                        print(f"filter_function failed on {path}: {e}")
                    for _ in self.epoch_lst:
                        iterator.update(1)
                    continue

            for epoch in self.epoch_lst:
                try:
                    self._resolve_checkpoint_path(path, epoch)
                except FileNotFoundError:
                    iterator.update(1)
                    continue

                if validate_checkpoints:
                    try:
                        chk = self._load_checkpoint(path, epoch)
                        if not self._is_valid_checkpoint(chk):
                            iterator.update(1)
                            continue
                    except Exception as e:
                        if self.verbosity > 5:
                            print(f"Validation failed for {path}, epoch {epoch}: {e}")
                        iterator.update(1)
                        continue

                label = f"{path}#_#epoch_{epoch}#_#layer_{self.layer_lst}"

                samples.append(
                    {
                        "path": path,
                        "epoch": epoch,
                        "label": label,
                    }
                )
                iterator.update(1)

        iterator.close()
        return samples

    def _resolve_checkpoint_path(self, path, epoch):
        p1 = path / f"checkpoint_{epoch}" / "checkpoints"
        if p1.exists():
            return p1

        p2 = path / f"checkpoint_{epoch:06d}" / "checkpoints"
        if p2.exists():
            return p2

        raise FileNotFoundError(
            f"Could not find checkpoint for epoch {epoch} in {path}. "
            f"Tried {p1} and {p2}"
        )

    def _load_checkpoint(self, path, epoch):
        chkpth = self._resolve_checkpoint_path(path, epoch)
        return torch.load(str(chkpth), map_location=torch.device("cpu"))

    def _is_valid_checkpoint(self, chkpoint):
        if self.layer_lst is None:
            return True

        if test_checkpoint_for_nan(chkpoint, self.layer_lst, self.use_bias):
            return False

        if test_checkpoint_with_threshold(
                chkpoint,
                self.layer_lst,
                self.use_bias,
                self.weight_threshold,
        ):
            return False

        return True

    def _validate_checkpoint_or_raise(self, chkpoint, path, epoch):
        if self.layer_lst is None:
            return

        if test_checkpoint_for_nan(chkpoint, self.layer_lst, self.use_bias):
            raise ValueError(f"Found NaN values in checkpoint {path}, epoch {epoch}")

        if test_checkpoint_with_threshold(
                chkpoint,
                self.layer_lst,
                self.use_bias,
                self.weight_threshold,
        ):
            raise ValueError(
                f"Found values above threshold {self.weight_threshold} "
                f"in checkpoint {path}, epoch {epoch}"
            )

    def _preload_properties(self):
        result_keys = self.property_keys.get("result_keys", [])
        config_keys = self.property_keys.get("config_keys", [])

        last_error = None
        for idx_offset in (1, 0):
            try:
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=idx_offset,
                )
                if self.verbosity > 0:
                    print(f"Loaded properties with idx_offset={idx_offset}")
                return
            except (AssertionError, IndexError, KeyError) as e:
                last_error = e
                if self.verbosity > 0:
                    print(f"Property loading failed with idx_offset={idx_offset}: {e}")

        raise RuntimeError(
            f"Failed to preload properties for all tested idx_offset values. "
            f"Last error: {last_error}"
        )

    def read_properties(self, results_key_list, config_key_list, idx_offset=1):
        results_key_list = [key for key in results_key_list]

        properties = {}
        for key in results_key_list:
            properties[key] = []
        for key in config_key_list:
            properties[key] = []

        read_ggap = False
        if "ggap" in results_key_list:
            results_key_list.remove("ggap")
            read_ggap = True

        iterator = tqdm.tqdm(
            self.samples,
            total=len(self.samples),
            desc="Loading properties",
            disable=(self.verbosity < 0),
        )

        for iidx, sample in enumerate(iterator):
            ppdx = sample["path"]
            eedx = sample["epoch"]

            res_tmp = read_properties_from_path(ppdx, eedx, idx_offset=idx_offset)

            for key in results_key_list:
                properties[key].append(res_tmp[key])

            for key in config_key_list:
                properties[key].append(res_tmp["config"][key])

            if read_ggap:
                gap = res_tmp["train_acc"] - res_tmp["test_acc"]
                properties["ggap"].append(gap)

            if iidx == 0:
                train_it = int(res_tmp["training_iteration"])
                assert int(eedx) == train_it, (
                    f"training iteration {train_it} and epoch {eedx} don't match."
                )

        self.properties = properties


def read_properties_from_path(path, idx, idx_offset):
    """
    Reads path/result.json
    Returns the dict for training_iteration=idx
    idx_offset=0 if checkpoint_0 was written, else idx_offset=1
    """
    fname = Path(path).joinpath("result.json")
    results = []

    try:
        with fname.open() as f:
            for line in f:
                results.append(json.loads(line))
    except Exception as e:
        print(f"error loading {fname}")
        print(e)
        raise

    jdx = idx - idx_offset
    if jdx < 0 or jdx >= len(results):
        raise IndexError(
            f"Requested result index {jdx} for epoch idx={idx} with offset={idx_offset}, "
            f"but result.json has only {len(results)} entries in {fname}"
        )

    return results[jdx]