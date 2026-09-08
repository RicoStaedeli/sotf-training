from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Sequence
import hashlib
import json
import random
import re

from torch.utils.data import random_split, Subset

from sane.data.datasets.checkpoints_dataset import CheckpointsDataset
from sane.utils.generic_utils import parse_json


class Splitter(ABC):
    """ The dataset splitter mutator base class. """
    def __call__(self, *args, **kwargs)-> List[Subset[CheckpointsDataset]]:
        return self.split(*args, **kwargs)

    @abstractmethod
    def split(self, dataset: CheckpointsDataset)-> List[Subset[CheckpointsDataset]]:
        pass

class RandomSplitter(Splitter):
    """ A mutator class for splitting a dataset using ratios.

    Using this splitter is, for most cases, a bad choice. Since the splitting is done at random, it can lead to leakage between train and test sets.
    For example, if the dataset contains checkpoints of the same model, the train and test sets may contain checkpoints of the same model.
    It can also be the case the multiple checkpoints trained with the same configuration but different seed are split into train and test sets, albeit not being significantly different.
    Use this mutator only if you are aware of the risks and limitations.

    Args:
        dataset (CheckpointsDataset): The dataset to split.
        ratios (Sequence[int | float]): The ratios to split the dataset into. The sum of the ratios must be equal to 1.
    """
    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        return random_split(dataset, ratios)


class ModelGroupSplitter(Splitter):
    """Splits a dataset such that all checkpoints of the same model stay in the same split.

    For each checkpoint the dataset must expose a ``get_model_id(index) -> str`` method
    (implemented by ``ZooDataset``).  The splitter groups indices by model identity, then
    assigns entire model groups to train / val / test so no single model ever appears in
    more than one split.

    Args:
        seed (int): Random seed for reproducibility. Default: 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        """Split the dataset by model groups.

        Args:
            dataset (CheckpointsDataset): Dataset that implements ``get_model_id(index)``.
            ratios (Sequence[int | float]): Per-split proportions. Must sum to 1.

        Returns:
            List[Subset[CheckpointsDataset]]: One ``Subset`` per ratio entry.

        Raises:
            AttributeError: If *dataset* does not implement ``get_model_id``.
            ValueError: If *ratios* do not sum to 1 (within floating-point tolerance).
        """
        if not hasattr(dataset, "get_model_id"):
            raise AttributeError(
                f"{type(dataset).__name__} does not implement get_model_id(index). "
                "ModelGroupSplitter requires this method to identify model identity per checkpoint."
            )

        total = sum(ratios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1, got {total}.")

        # Group checkpoint indices by model id
        model_to_indices: dict[str, List[int]] = defaultdict(list)
        for i in range(len(dataset)):
            model_id = dataset.get_model_id(i)
            model_to_indices[model_id].append(i)

        model_groups = list(model_to_indices.values())
        rng = random.Random(self.seed)
        rng.shuffle(model_groups)

        # Assign model groups to splits greedily based on ratios
        n_models = len(model_groups)
        split_indices: List[List[int]] = [[] for _ in ratios]
        assigned = 0
        for split_idx, ratio in enumerate(ratios):
            # Last split gets all remaining models to avoid rounding gaps
            if split_idx == len(ratios) - 1:
                target = n_models - assigned
            else:
                target = round(ratio * n_models)
            for group in model_groups[assigned: assigned + target]:
                split_indices[split_idx].extend(group)
            assigned += target

        return [Subset(dataset, indices) for indices in split_indices]


class HyperparameterGroupSplitter(Splitter):
    """Splits a dataset such that all models sharing the same hyperparameter configuration
    (regardless of random seed) are always in the same split.

    For each checkpoint the dataset must expose a ``get_model_id(index) -> str`` method and
    each model directory must contain a ``params.json`` file.  The splitter reads that file,
    removes ``exclude_keys`` (default: ``["seed"]``), and uses a hash of the remaining config
    as the group key.  All models with the same hash are assigned to the same split together.

    Args:
        seed (int): Random seed for reproducibility when shuffling hyperparameter groups. Default: 42.
        exclude_keys (List[str]): Keys to strip from params.json before hashing. Default: ["seed"].
    """

    def __init__(self, seed: int = 42, exclude_keys: List[str] | None = None) -> None:
        self.seed = seed
        self.exclude_keys = set(exclude_keys) if exclude_keys is not None else {"seed"}

    def _hyperparam_hash(self, model_id: str) -> str:
        params_path = f"{model_id}/params.json"
        params = parse_json(params_path)
        filtered = {k: v for k, v in params.items() if k not in self.exclude_keys}
        canonical = json.dumps(filtered, sort_keys=True)
        return hashlib.md5(canonical.encode()).hexdigest()

    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        if not hasattr(dataset, "get_model_id"):
            raise AttributeError(
                f"{type(dataset).__name__} does not implement get_model_id(index). "
                "HyperparameterGroupSplitter requires this method."
            )

        total = sum(ratios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1, got {total}.")

        # Group checkpoint indices by model id, then group model ids by hyperparam hash
        model_to_indices: dict[str, List[int]] = defaultdict(list)
        for i in range(len(dataset)):
            model_to_indices[dataset.get_model_id(i)].append(i)

        hyperparam_to_models: dict[str, List[str]] = defaultdict(list)
        for model_id in model_to_indices:
            h = self._hyperparam_hash(model_id)
            hyperparam_to_models[h].append(model_id)

        hyperparam_groups = list(hyperparam_to_models.values())
        rng = random.Random(self.seed)
        rng.shuffle(hyperparam_groups)

        n_groups = len(hyperparam_groups)
        split_indices: List[List[int]] = [[] for _ in ratios]
        assigned = 0
        for split_idx, ratio in enumerate(ratios):
            if split_idx == len(ratios) - 1:
                target = n_groups - assigned
            else:
                target = round(ratio * n_groups)
            for model_id in (mid for group in hyperparam_groups[assigned: assigned + target] for mid in group):
                split_indices[split_idx].extend(model_to_indices[model_id])
            assigned += target

        return [Subset(dataset, indices) for indices in split_indices]


class ModelWidthGroupSplitter(Splitter):
    """Splits by model identity while ensuring train sees every model width.

    This splitter is intended for zoos that mix architectures with different widths
    (for example ResNet zoos with ``model_width=16,32,64,...`` in ``params.json``).
    It keeps all checkpoints of one model together, like ``ModelGroupSplitter``, but
    stratifies model groups by width and guarantees that the train split receives at
    least one model from every observed width bucket.

    Width is resolved from ``<model_id>/params.json`` using the first exact match for
    ``model_width`` or ``width`` (searched recursively), with a fallback to parsing the
    model directory name for patterns such as ``model_width=128``.

    Args:
        seed (int): Random seed for reproducible shuffling. Default: 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _find_width_in_obj(self, obj) -> str | None:
        if isinstance(obj, dict):
            for key in ("model_width", "width"):
                if key in obj and obj[key] is not None:
                    return str(obj[key])
            for value in obj.values():
                width = self._find_width_in_obj(value)
                if width is not None:
                    return width
        elif isinstance(obj, list):
            for value in obj:
                width = self._find_width_in_obj(value)
                if width is not None:
                    return width
        return None

    def _extract_width(self, model_id: str) -> str:
        params_path = f"{model_id}/params.json"
        params = parse_json(params_path)
        width = self._find_width_in_obj(params)
        if width is not None:
            return width

        match = re.search(r"(?:^|[,/_])model_width=([^,/_]+)", model_id)
        if match is None:
            match = re.search(r"(?:^|[,/_])width=([^,/_]+)", model_id)
        if match is not None:
            return match.group(1)

        raise ValueError(
            f"Could not determine model width for model_id={model_id}. "
            f"Expected a width key in {params_path} or a model_width=... pattern in the path."
        )

    def _allocate_counts(self, n_groups: int, ratios: Sequence[int | float]) -> List[int]:
        if n_groups <= 0:
            return [0 for _ in ratios]

        desired = [ratio * n_groups for ratio in ratios]
        counts = [int(value) for value in desired]
        remainder = n_groups - sum(counts)

        fractional_order = sorted(
            range(len(ratios)),
            key=lambda idx: (desired[idx] - counts[idx], -idx),
            reverse=True,
        )
        for idx in fractional_order[:remainder]:
            counts[idx] += 1

        if counts[0] == 0:
            donor_candidates = [idx for idx in range(1, len(counts)) if counts[idx] > 0]
            if donor_candidates:
                donor = max(donor_candidates, key=lambda idx: counts[idx])
                counts[donor] -= 1
            counts[0] += 1

        return counts

    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        if not hasattr(dataset, "get_model_id"):
            raise AttributeError(
                f"{type(dataset).__name__} does not implement get_model_id(index). "
                "ModelWidthGroupSplitter requires this method to identify model identity per checkpoint."
            )

        total = sum(ratios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1, got {total}.")

        model_to_indices: dict[str, List[int]] = defaultdict(list)
        for i in range(len(dataset)):
            model_to_indices[dataset.get_model_id(i)].append(i)

        width_to_models: dict[str, List[str]] = defaultdict(list)
        for model_id in model_to_indices:
            width_to_models[self._extract_width(model_id)].append(model_id)

        rng = random.Random(self.seed)
        width_keys = list(width_to_models)
        rng.shuffle(width_keys)

        split_indices: List[List[int]] = [[] for _ in ratios]
        for width in width_keys:
            model_ids = list(width_to_models[width])
            rng.shuffle(model_ids)
            counts = self._allocate_counts(len(model_ids), ratios)

            start = 0
            for split_idx, count in enumerate(counts):
                for model_id in model_ids[start: start + count]:
                    split_indices[split_idx].extend(model_to_indices[model_id])
                start += count

        return [Subset(dataset, indices) for indices in split_indices]
