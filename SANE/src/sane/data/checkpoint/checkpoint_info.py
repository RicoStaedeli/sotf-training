import os
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

from sane.utils.generic_utils import resolve_abs_path

from typing import Dict, List, Optional, Union
from collections.abc import Callable

@dataclass
class CheckpointInfo():
    """
    A dataclass for retaining checkpoint related information.

    Attributes:
        epoch_idx (int): the epoch index of the checkpoint.
        state_dict_path (str): path to the pytorch state_dict of the checkpoint.
        model_config_path (str): path to the model configuration file.
        eval_results_path (str): path to the evaluation results file. 
    """
    epoch_idx: int
    state_dict_path: str
    model_config_path: Optional[str] = None
    eval_results_path: Optional[str] = None


def parseCheckpointInfo(root_dir:str, epoch_idx:List[int], checkpoint_filter: Optional[Callable[[CheckpointInfo], bool]] = None, max_checkpoints: Optional[int] = None) -> List[CheckpointInfo]:
    """
    Parses the model zoo directory structure creating a CheckpointInfo instance per checkpoint.

    Args:
        root_dir (str): The path to a model zoo directory.
        epoch_idx (int|List[int]): The epoch index/indices of the checkpoint(s) of a model.
        checkpoint_filter (Callable[CheckpointInfo, bool], optional): A function to filter the checkpoints. This function should return True for checkpoints to be kept, False otherwise.
        max_checkpoints (int, optional): Maximum number of checkpoints to load. If None, all checkpoints are loaded. Default: None.

    Returns:
        List[CheckpointInfo]: A list of CheckpointInfo instances, one per checkpoint.
    """

    root_dir = resolve_abs_path(root_dir)

    checkpoint_paths_list = []
    model_dirs = [
        fname for fname in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, fname))
    ]
    total_models = len(model_dirs)
    filter_enabled = checkpoint_filter is not None
    if filter_enabled:
        logger.info(
            "Scanning model zoo %s with checkpoint filtering enabled: %d model directories, %d requested epoch(s) per model",
            root_dir,
            total_models,
            len(epoch_idx),
        )
    else:
        logger.info(
            "Scanning model zoo %s: %d model directories, %d requested epoch(s) per model",
            root_dir,
            total_models,
            len(epoch_idx),
        )

    candidate_checkpoints = 0
    skipped_models = 0
    progress_interval = 25

    for model_idx, fname in enumerate(model_dirs, start=1):
        # ModelZoo contents (1st level)

        model_dir = os.path.join(root_dir, fname)
        model_config = None
        eval_results = None
        epoch_ckpt_list = []
        for sub_fname in os.listdir(model_dir):
            # Model contents (2nd level)

            sub_path = os.path.join(model_dir, sub_fname)
            if not os.path.isdir(sub_path):
                # sub_path points to a model meta file
                # (1) params.json
                # (2) results.json
                filename = os.path.basename(sub_path)
                if filename == "params.json":
                    model_config = sub_path
                elif filename == "result.json":
                    eval_results = sub_path
                else:
                    # ignore other model meta files.
                    pass

                continue
            
            epoch_ckpt_list.append(sub_path)

        # If we fail to load metadata, directory was not a model directory.
        if len(epoch_ckpt_list) == 0:
            continue
        
        epoch_key = lambda p: os.path.basename(p).split('_')[-1]
        epoch_ckpt_list.sort(key=epoch_key)

        if len(epoch_ckpt_list) <= max(epoch_idx):
            logger.warning(f"Skipping {model_dir}: requested epoch index {max(epoch_idx)} but only {len(epoch_ckpt_list)} checkpoints available.")
            skipped_models += 1
            continue

        for e in epoch_idx:
            state_dict_dir = epoch_ckpt_list[e]
            state_dict = os.path.join(state_dict_dir, 'checkpoints')
            # checkpoints_training is not stored/processed for now.

            # convert negative indices to positive ones.
            if e < 0:
                e = e + len(epoch_ckpt_list)

            checkpoint_info = CheckpointInfo(
                epoch_idx=e,
                state_dict_path=state_dict,
                model_config_path=model_config,
                eval_results_path=eval_results
            )
            candidate_checkpoints += 1

            if checkpoint_filter is None or checkpoint_filter(checkpoint_info):
                checkpoint_paths_list.append(checkpoint_info)

        if model_idx % progress_interval == 0 or model_idx == total_models:
            if filter_enabled:
                logger.info(
                    "Checkpoint filtering progress: %d/%d models scanned, %d candidate checkpoints, %d kept, %d models skipped",
                    model_idx,
                    total_models,
                    candidate_checkpoints,
                    len(checkpoint_paths_list),
                    skipped_models,
                )
            else:
                logger.info(
                    "Checkpoint scan progress: %d/%d models scanned, %d checkpoints discovered, %d models skipped",
                    model_idx,
                    total_models,
                    candidate_checkpoints,
                    skipped_models,
                )

    # Apply max_checkpoints limit if specified
    if max_checkpoints is not None and max_checkpoints > 0:
        if max_checkpoints > len(checkpoint_paths_list):
            logger.warning(f"Requested max_checkpoints={max_checkpoints} is greater than available checkpoints={len(checkpoint_paths_list)}. Returning all available checkpoints.")
        else:
            checkpoint_paths_list = checkpoint_paths_list[:max_checkpoints]

    return checkpoint_paths_list
