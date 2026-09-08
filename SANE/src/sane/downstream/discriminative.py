from typing import Optional, Any, List, Union
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import warnings

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning

from sane.downstream.downstream_task import DownstreamTask
from sane.data.datasets import CheckpointsDataset, WindowedDataset, CachedWindowedDataset
from sane.model.autoencoder import SANEAutoEncoder

import logging
logger = logging.getLogger(__name__)

class DiscriminativeDownstreamTask(DownstreamTask):
    r"""Performs discriminative downstream tasks (regression and classification) using features extracted from a SANEAutoEncoder.

    This class is designed to evaluate the quality of representations learned by a SANEAutoEncoder
    by performing regression and classification on specified targets. It supports both regression (e.g., test accuracy) with least squares
    and classification (e.g., model architecture properties) with logistic regression tasks.

    Args:
        name (str, optional): Name of the downstream task. Default: ``"discriminative_dstk"``
        batch_size (int, optional): Number of samples per batch. If ``None``, uses the default batch size.
        regression_targets (List[str], optional): List of target names for regression tasks. Default: ``["test_acc"]``
        classification_targets (List[str], optional): List of target names for classification tasks. Default: ``["model::nlin"]``
        epochs (Optional[Union[int, List[int]]]): Epochs at which the task should be executed. If an `int` is provided,
            it will be wrapped in a list. Default: ``None``.
        frequency (Optional[int]): Frequency (in epochs) at which the task should be executed. If both `epochs` and
            `frequency` are provided, `frequency` is ignored. If neither is provided, the task is executed every epoch.
            Default: ``None``.
        trainset (Optional[WindowedDataset]): Training dataset. Default: ``None``.
        valset (Optional[WindowedDataset]): Validation dataset. Default: ``None``.
        testset (Optional[WindowedDataset]): Test dataset. Default: ``None``.

    Note:
        If both `epochs` and `frequency` are provided, `frequency` is ignored.
        If neither is provided, the task is executed every epoch by default.
    """
    def __init__(
        self,
        name: str = "discriminative_dstk",
        batch_size: Optional[int] = None,
        regression_targets: List[str] = ["test_acc"],
        classification_targets: List[str] = ["model::nlin"],
        epochs: Optional[Union[int, List[int]]] = None,
        frequency: Optional[int] = None,
        trainset: Optional["WindowedDataset"] = None,
        valset: Optional["WindowedDataset"] = None,
        testset: Optional["WindowedDataset"] = None,
    ):
        super(DiscriminativeDownstreamTask, self).__init__(name, epochs, frequency, trainset, valset, testset)
        self.regression_targets = regression_targets
        self.classification_targets = classification_targets
        self.y_dict = None  # To store the targets for train, val, test splits
        self.batch_size = batch_size

        self.trainloader = None
        self.valloader = None
        self.testloader = None

    def _execute(self, sane_ae: SANEAutoEncoder, *args, **kwargs) -> Any:
        r"""Executes the discriminative downstream task using the provided SANEAutoEncoder.

        Args:
            sane_ae (SANEAutoEncoder): The autoencoder used to compute features.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Mapping[str, Any]: A dictionary containing the results for each target and split, as well as the running time.
        """
        # Timer
        start_time = time.time()

        # Loading the data if not already loaded
        self._prepare_data()
        self._load_targets()

        # Compute features using the autoencoder
        X = self._compute_features(sane_ae)

        # Perform the regression task here
        results = dict()
        for target in self.regression_targets:
            results[target] = self._regression(X, target)
        for target in self.classification_targets:
            results[target] = self._classification(X, target)

        results["running_time"] = time.time() - start_time

        return {self.name: results}

    def _load_targets(self):
        r"""Loads the targets for each split (train, val, test) from the checkpoints datasets."""
        if self.y_dict is None:
            self.y_dict = dict()

            self.y_dict['train'] = self._load_targets_split(self.trainset.checkpoints_dataset)
            if self.valset is not None:
                self.y_dict['val'] = self._load_targets_split(self.valset.checkpoints_dataset)
            if self.testset is not None:
                self.y_dict['test'] = self._load_targets_split(self.testset.checkpoints_dataset)

    def _load_targets_split(self, checkpoints_dataset: CheckpointsDataset):
        r"""Loads the targets for a single split from the checkpoints dataset.

        Args:
            checkpoints_dataset (CheckpointsDataset): The dataset containing the checkpoints.

        Returns:
            Mapping[str, List[Any]]: A dictionary mapping target names to lists of target values.

        Raises:
            KeyError: If a target is not found in the checkpoint metadata.
        """
        y_dict_split = {target: [] for target in self.regression_targets + self.classification_targets}

        for checkpoint in checkpoints_dataset:
            # If the checkpoint is augmented, take the canonical one
            checkpoint = checkpoint[0] if isinstance(checkpoint, Sequence) else checkpoint

            metadata = dict(checkpoint.metadata['model_config'], **checkpoint.metadata['eval_results'])
            for target in y_dict_split.keys():
                if target == 'ggap':
                    # Special handling for ggap target
                    try:
                        ggap = metadata['train_acc'] - metadata['test_acc']
                        y_dict_split[target].append(ggap)
                    except KeyError as e:
                        raise KeyError(f"Required fields for 'ggap' target (`train_acc` and `test_acc`) not found in checkpoint metadata. Available metadata fields: {list(metadata.keys())}") from e
                else:
                    try:
                        y_dict_split[target].append(metadata[target])
                    except KeyError as e:
                        raise KeyError(f"Target '{target}' not found in checkpoint metadata. List of available metadata fields: {list(metadata.keys()) + ['ggap']}") from e

        return y_dict_split

    def _prepare_data(self):
        r"""Prepares the data loaders for each split (train, val, test)."""
        if self.trainloader is None:
            self.trainloader = self._prepare_dataset_split(self.trainset, 'train')
        
        if self.valset is not None and self.valloader is None:
            self.valloader = self._prepare_dataset_split(self.valset, 'val')

        if self.testset is not None and self.testloader is None:
            self.testloader = self._prepare_dataset_split(self.testset, 'test')

    def _prepare_dataset_split(self, dataset: WindowedDataset, split_name: str) -> DataLoader:
        r"""Prepares a DataLoader for a single split.

        Args:
            dataset (BaseWindowedDataset): The dataset for the split.
            split_name (str): The name of the split (train, val, test).

        Returns:
            DataLoader: The DataLoader for the split.

        Raises:
            ValueError: If the dataset type is not supported.
        """
        class PlaceHolderDataset(torch.utils.data.Dataset):
            def __init__(self, dataset: CachedWindowedDataset):
                self.dataset = dataset

            def __len__(self):
                return self.dataset.__len_models__()

            def __getitem__(self, index: int):
                return self.dataset.__getmodel__(index)

        if dataset.num_windows_per_model != "auto":
            logger.warning(f"In {self.name}, for split '{split_name}', num_windows_per_model is set to {dataset.num_windows_per_model}, model may not be completely covered.")

        if not dataset.pad_windows:
            logger.warning(f"In {self.name}, for split '{split_name}', pad_windows is set to False, which may lead to inconsistent input sizes.")
        
        if dataset.allow_overlapping_windows:
            logger.warning(f"In {self.name}, for split '{split_name}', allow_overlapping_windows is set to True, which may lead to redundant information.")

        if dataset.window_distribution_strategy != "consecutive":
            logger.warning(f"In {self.name}, for split '{split_name}', window_distribution_strategy is set to '{dataset.window_distribution_strategy}', which may lead to uneven coverage of the model.")

        if isinstance(dataset, CachedWindowedDataset):
            dataset = PlaceHolderDataset(dataset)
        else:
            raise ValueError(f"Unsupported dataset type for split '{split_name}': {type(dataset)}")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def _compute_features(self, sane_ae: SANEAutoEncoder):
        r"""Computes features for each split using the provided SANEAutoEncoder.

        Args:
            sane_ae (SANEAutoEncoder): The autoencoder used to compute features.

        Returns:
            Mapping[str, np.ndarray]: A dictionary mapping split names to feature arrays.
        """
        X = dict()
        sane_ae.eval()

        X['train'] = self._compute_features_split(sane_ae, self.trainloader)
        if self.valset is not None:
            X['val'] = self._compute_features_split(sane_ae, self.valloader)
        if self.testset is not None:
            X['test'] = self._compute_features_split(sane_ae, self.testloader)

        return X

    def _compute_features_split(self, sane_ae: SANEAutoEncoder, dataloader: DataLoader):
        r"""Computes features for a single split using the provided SANEAutoEncoder and DataLoader.

        Args:
            sane_ae (SANEAutoEncoder): The autoencoder used to compute features.
            dataloader (DataLoader): The DataLoader for the split.

        Returns:
            np.ndarray: The computed feature array for the split.

        Raises:
            ValueError: If the DataLoader output format is unexpected.
        """
        embeddings = []

        device = next(sane_ae.parameters()).device

        for batch_idx, data in enumerate(dataloader):
            if isinstance(data, Sequence) and isinstance(data[0], Sequence):
                batch_embeddings = []
                for tensors in data:
                    if len(tensors) != 3:
                        raise ValueError("Tensors must be a tuple of (tokens, mask, position) but has length {} and dimensions {}.".format(len(tensors), [tensor.dim() for tensor in tensors]))

                    # If the tokens have 4 dimensions, it means we have different views.
                    # In that case, dimensions are: (batch_size, num_views, num_tokens, token_size)
                    # We always take the first view for downstream tasks.
                    if len(tensors[0].shape) == 4:
                        tokens, mask, position = tensors[0][:, 0].to(device), tensors[1][:, 0].to(device), tensors[2][:, 0].to(device)
                    # If not using multiple views, just move the tensors to the device
                    elif len(tensors[0].shape) == 3:
                        tokens, mask, position = [tensor.to(device) for tensor in tensors]
                    
                    with torch.no_grad():
                        batch_embeddings.append(sane_ae.encoder(tokens, position))

                # We concatenate the embeddings of all models in the batch along dimension 1 to stitch the windows back together
                batch_embeddings = torch.cat(batch_embeddings, dim=1)

                # We finalise the embeddings by passing them through the embedder, which should transform the list of tokens into a fixed-size representation
                with torch.no_grad():
                    batch_embeddings = sane_ae.embedder(batch_embeddings)

                embeddings.append(batch_embeddings.cpu().numpy())
            else:
                raise ValueError(f"DataLoader should output lists of tuples of tokens representing a specific model, but got {type(data)} containing elements of type {type(data[0])}.")

        return np.concatenate(embeddings)

    def _regression(self, X: Mapping, target: str):
        r"""Performs regression for a given target and feature set.

        Args:
            X (Mapping[str, np.ndarray]): A dictionary mapping split names to feature arrays.
            target (str): The target name for regression.

        Returns:
            Mapping[str, Mapping[str, float]]: A dictionary mapping split names to regression metrics (mse, r2).
        """
        y_train = self.y_dict['train'][target]
        if self.valset is not None:
            y_val = self.y_dict['val'][target]
        if self.testset is not None:
            y_test = self.y_dict['test'][target]

        model = LinearRegression()
        model.fit(X['train'], y_train)

        results = dict()
        results['train'] = self._evaluate_regression(y_train, model.predict(X['train']))
        if self.valset is not None:
            results['val'] = self._evaluate_regression(y_val, model.predict(X['val']))
        if self.testset is not None:
            results['test'] = self._evaluate_regression(y_test, model.predict(X['test']))

        return results

    def _evaluate_regression(self, y_true, y_pred):
        r"""Evaluates regression performance using MSE and R2 metrics.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            Mapping[str, float]: A dictionary containing MSE and R2 scores.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'mse': mse, 'r2': r2}

    def _classification(self, X: Mapping, target: str):
        r"""Performs classification for a given target and feature set.

        Args:
            X (Mapping[str, np.ndarray]): A dictionary mapping split names to feature arrays.
            target (str): The target name for classification.

        Returns:
            Mapping[str, Mapping[str, float]]: A dictionary mapping split names to classification metrics (accuracy, f1, precision, recall).
        """
        y_train = self.y_dict['train'][target]
        if self.valset is not None:
            y_val = self.y_dict['val'][target]
        if self.testset is not None:
            y_test = self.y_dict['test'][target]

        # If all targets are the same, always predict that class
        if len(set(y_train)) == 1:
            model = DummyClassifier(strategy="most_frequent")
            model.fit(X['train'], y_train)
        else:
            model = LogisticRegression()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X['train'], y_train)

        results = dict()
        results['train'] = self._evaluate_classification(y_train, model.predict(X['train']))
        if self.valset is not None:
            results['val'] = self._evaluate_classification(y_val, model.predict(X['val']))
        if self.testset is not None:
            results['test'] = self._evaluate_classification(y_test, model.predict(X['test']))

        return results

    def _evaluate_classification(self, y_true, y_pred):
        r"""Evaluates classification performance using accuracy, F1, precision, and recall metrics.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            Mapping[str, float]: A dictionary containing accuracy, F1, precision, and recall scores.
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}