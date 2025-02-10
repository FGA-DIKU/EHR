import copy
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.modules.preparation.dataset import PatientDataset


def split_pids_into_pt_ft_test(
    pids: list, pretrain: float, finetune: float, test: float
) -> Tuple[list, list, list]:
    """
    Splits the given list of patient IDs into pretrain, finetune and test lists.
    pretrain + finetune + test must be equal to 1

    :param pids: List of patient IDs to split.
    :param pretrain: Fraction of IDs to use for pretraining.
    :param finetune: Fraction of IDs to use for finetuning.
    :param test: Fraction of IDs to use for tests.

    :return: A tuple of lists:
        [0]: Pretrain pids
        [1]: Finetune pids
        [2]: Test pids
    """
    if not np.isclose(pretrain + finetune + test, 1.0):
        raise ValueError("The sum of pretrain, finetune, and test should be 1.")
    n = len(pids)
    pretrain_pids = pids[: int(n * pretrain)]
    finetune_pids = pids[int(n * pretrain) : int(n * (pretrain + finetune))]
    test_pids = pids[int(n * (pretrain + finetune)) :]
    return pretrain_pids, finetune_pids, test_pids


def split_test(pids: list, test_ratio: float) -> Tuple[list, list]:
    """Split patient IDs into train and test sets.

    Args:
        pids: List of patient IDs to split
        test_ratio: Fraction of patients to use for test set (between 0 and 1)

    Returns:
        Tuple containing:
            - train_pids: List of patient IDs for training
            - test_pids: List of patient IDs for testing
    """
    n_test = int(len(pids) * test_ratio)
    test_pids = pids[:n_test]
    train_pids = pids[n_test:]
    return train_pids, test_pids


def split_pids_into_train_val(
    data: PatientDataset, val_split: float
) -> Tuple[PatientDataset, PatientDataset]:
    """
    Splits data into train and val.
    Returns two PatientDatasets.
    """
    assert val_split < 1 and val_split > 0, "Split must be between 0 and 1"
    train_split = 1 - val_split
    random.seed(42)
    pids = copy.deepcopy(data.get_pids())
    random.shuffle(pids)
    train_pids = pids[: int(len(pids) * train_split)]
    val_pids = pids[int(len(pids) * train_split) :]
    train_data = data.filter_by_pids(train_pids)
    val_data = data.filter_by_pids(val_pids)
    return train_data, val_data


def get_n_splits_cv_pids(
    n_splits: int, train_val_pids: List[str], val_split: float = 0.2, seed: int = 42
):
    """Split patient IDs into n cross-validation folds.

    Args:
        n_splits: Number of CV folds (must be >= 1)
        train_val_pids: List of patient IDs to split
        val_split: Fraction of patients to use for validation set (between 0 and 1)
        only used if
        seed: Random seed for reproducibility

    Returns:
        List of (train_pids, val_pids) tuples for each fold
    """

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    train_val_pids = np.array(train_val_pids)

    if n_splits == 1:
        # For single split, use 80-20 split by default
        split_idx = int((1 - val_split) * len(train_val_pids))
        rng = np.random.RandomState(seed)
        rng.shuffle(train_val_pids)
        yield train_val_pids[:split_idx].tolist(), train_val_pids[split_idx:].tolist()
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(train_val_pids):
            train_pids = train_val_pids[train_idx].tolist()
            val_pids = train_val_pids[val_idx].tolist()
            yield train_pids, val_pids


def split_into_test_and_train_val_pids(pids: list, test_split: float):
    """Split patient IDs into test and train/validation sets.

    Args:
        pids: List of patient IDs to split
        test_split: Fraction of patients to use for test set (between 0 and 1)

    Returns:
        Tuple of (test_pids, train_val_pids) containing the split patient IDs
    """
    test_pids = np.random.choice(pids, size=int(len(pids) * test_split), replace=False)
    set_test_pids = set(test_pids)
    train_val_pids = [pid for pid in pids if pid not in set_test_pids]
    return test_pids, train_val_pids


def create_folds(pids: list, num_folds: int, seed: int = 42) -> List[Dict[str, list]]:
    """
    Create k folds from a list of PIDs.
    example output:

    {
        "fold_0": {
            "train": [1, 2, 3],
            "val": [4, 5, 6]
        }
    }

    Args:
        pids (list): List of patient IDs.
        num_folds (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of folds with train and val PIDs.
    """
    rng = np.random.default_rng(seed)
    pids_array = np.array(pids)
    rng.shuffle(pids_array)  # Shuffle before splitting

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = [{TRAIN_KEY: [], VAL_KEY: []} for _ in range(num_folds)]

    for i, (train_idx, val_idx) in enumerate(kf.split(pids_array)):
        folds[i][TRAIN_KEY] = [pids_array[idx] for idx in train_idx]
        folds[i][VAL_KEY] = [pids_array[idx] for idx in val_idx]

    return folds
