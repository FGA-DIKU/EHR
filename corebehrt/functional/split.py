import copy
import random
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from corebehrt.classes.dataset import PatientDataset
from corebehrt.functional.load import load_predefined_splits


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


def split_pids_into_train_val(
    data: PatientDataset, split: float
) -> Tuple[PatientDataset, PatientDataset]:
    """
    Splits data into train and val. Returns two dataframes.
    """
    assert split < 1 and split > 0, "Split must be between 0 and 1"
    random.seed(42)
    pids = copy.deepcopy(data.get_pids())
    random.shuffle(pids)
    train_pids = pids[: int(len(pids) * split)]
    val_pids = pids[int(len(pids) * split) :]
    train_data = data.filter_by_pids(train_pids)
    val_data = data.filter_by_pids(val_pids)
    return train_data, val_data


def load_train_val_split(
    data: PatientDataset, split_path: str
) -> Tuple[PatientDataset, PatientDataset]:
    """
    Load the train/val split from the given split path and return the corresponding data.
    """
    splits = ["train", "val"]
    pids = load_predefined_splits(split_path, splits)
    train_pids, val_pids = pids
    train_dataset = data.filter_by_pids(train_pids)
    val_dataset = data.filter_by_pids(val_pids)
    return train_dataset, val_dataset


def get_n_splits_cv_pids(n_splits: int, train_val_pids: List[str]):
    """Split patient IDs into n cross-validation folds.

    Args:
        dataset: PatientDataset containing all patients
        n_splits: Number of CV folds
        train_val_pids: List of patient IDs to split

    Returns:
        List of (train_pids, val_pids) tuples for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_val_pids = np.array(train_val_pids)
    for train_idx, val_idx in kf.split(train_val_pids):
        train_pids = train_val_pids[train_idx].tolist()
        val_pids = train_val_pids[val_idx].tolist()
        yield train_pids, val_pids
