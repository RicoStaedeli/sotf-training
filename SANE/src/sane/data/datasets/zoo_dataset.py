import torch
from typing import List, Optional, Union

from sane.data.datasets.checkpoints_dataset import CheckpointsDataset
from sane.data.checkpoint import Checkpoint, parseCheckpointInfo
from sane.data.checkpoint.checkpoint_augmentation import CheckpointAugmentation
from sane.data.checkpoint.checkpoint_filters import WeightMagnitudeCheckpointFilter
from sane.data.datasets.zoo_dataset_models import get_model, vit_pytorch_legacy_from_state_dict
from sane.utils.generic_utils import parse_json, parse_jsonl

class ZooDataset(CheckpointsDataset):
    """
    ZooDataset class supports ModelZoos compiled by the AIML lab of
    the University of St. Gallen.

    Usage:
        To instantiate a ZooDataset a path to a model zoo's root directory is required.
        Additionally, the epoch index(s) can be specified to include the checkpoint(s) 
        of a model of the specified epoch.

        As this class inherits from the standard Pytorch Dataset class both __init__ and 
        __get_item__ methods are supported.

    Indicative model zoo directory structure:

    zoo_root_dir/
    ├── model_id_dir/
    │   ├── checkpoint_<1>/
    │   ├── ...
    │   ├── checkpoint_<epoch_no>/
    │       ├── checkpoints # model's state_dict
    │       ├── checkpoints_training [optional]
    │   ├── ...
    │   ├── checkpoint_<N>/
    │   ├── params.json # model's config dictionary (specifies the architecture, optimizer and scheduler)
    │   ├── result.json # model's evaluation results - contains a line for each epoch (JSONL format)
    ├── model_id_dir/
    │   ├── checkpoint_<1>/
    │   ├── ...
    │   ├── checkpoint_<epoch_no>/
    │       ├── checkpoints # model's state_dict
    │       ├── checkpoints_training [optional]
    │   ├── ...
    │   ├── checkpoint_<N>/
    │   ├── params.json # model's config dictionary (specifies the architecture, optimizer and scheduler)
    │   ├── result.json # model's evaluation results - contains a line for each epoch (JSONL format)
    ...
    ├── config.json # grid search dictionary
    ├── dataset.pt # image dataset used for training serialized 
    """
    def __init__(self, root_dir:str, epoch_idx:Union[int, List[int]]=-1, drop_nan:bool=False, max_absolute_weight:Optional[float]=None, max_checkpoints:Optional[int]=None, augmentations:Optional[CheckpointAugmentation]=None):
        """
        Args:
        root_dir (str): Model zoo directory path.
        epoch_idx (int|List[int]): The epoch index/indices of the checkpoint(s) of a model.
        drop_nan (bool, optional): If True, checkpoints containing NaN values are filtered out. Default: False
        max_absolute_weight (float, optional): The maximum allowed absolute value for any weight. If None, this check is skipped. Default: None
        max_checkpoints (int, optional): Maximum number of checkpoints to load. If None, all checkpoints are loaded. Note that this number is taken after loading multiple epochs, it is not to be multiplied by the number of epochs loaded. Default: None
        augmentations (CheckpointAugmentation, optional): A function/transform to apply to the dataset. Default: None
        """
        super().__init__()
        if isinstance(epoch_idx, int):
            epoch_idx = [epoch_idx]

        if drop_nan or max_absolute_weight is not None:
            checkpoint_filter = WeightMagnitudeCheckpointFilter(
                max_absolute_weight=max_absolute_weight,
                drop_nan=drop_nan
            )
        else:
            checkpoint_filter = None

        self.__ckpt_info_list = parseCheckpointInfo(root_dir, epoch_idx, checkpoint_filter, max_checkpoints)
        self.epoch_idx = epoch_idx
        self.augmentations = augmentations
        self.root_dir = root_dir  # Store root_dir for accessing dataset.pt

    def __getitem__(self, index: int) -> Union[Checkpoint, List[Checkpoint]]:
        ckpt_info = self.__ckpt_info_list[index]
        
        model_config = parse_json(ckpt_info.model_config_path)
        eval_results = parse_jsonl(ckpt_info.eval_results_path)[ckpt_info.epoch_idx]

        state_dict = torch.load(ckpt_info.state_dict_path, map_location='cpu')
        if all(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        if any(k.startswith('transformer.layers') for k in state_dict):
            model = vit_pytorch_legacy_from_state_dict(state_dict)
        else:
            model = get_model(model_config)
        model.load_state_dict(state_dict)

        checkpoint = Checkpoint(
              model
            , metadata = {'model_config': model_config, 'eval_results': eval_results}
        )

        # If we use augmentations, we return a list of augmented checkpoints
        return self.augmentations(checkpoint) if self.augmentations is not None else checkpoint

    def __len__(self) -> int:
        return len(self.__ckpt_info_list)

    def get_model_id(self, index: int) -> str:
        """Returns the model directory path as a unique model identifier for the checkpoint at the given index.

        The model directory is the parent of the epoch directory in the zoo structure:
        ``root_dir/model_id_dir/epoch_dir/checkpoints``

        Args:
            index (int): Index of the checkpoint.

        Returns:
            str: Absolute path to the model directory, unique per model.
        """
        import os
        state_dict_path = self.__ckpt_info_list[index].state_dict_path
        return os.path.dirname(os.path.dirname(state_dict_path))