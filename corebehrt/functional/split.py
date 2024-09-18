from typing import Tuple
import numpy as np

import dask.dataframe as dd
import random

from corebehrt.functional.utils import filter_table_by_pids
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


def split_pids_into_train_val(data: dd.DataFrame, split: float) -> Tuple[list, list]:
    """
    Splits data into train and val. Returns two dataframes.
    """
    assert split < 1 and split > 0, "Split must be between 0 and 1"
    random.seed(42)
    pids = data["PID"].unique().compute().tolist()
    random.shuffle(pids)
    train_pids = pids[: int(len(pids) * split)]
    val_pids = pids[int(len(pids) * split) :]
    train_data = filter_table_by_pids(data, train_pids)
    val_data = filter_table_by_pids(data, val_pids)
    return train_data, val_data


def load_train_val_split(data: dd.DataFrame, split_path: str) -> dd.DataFrame:
    """
    Load the train/val split from the given split path and return the corresponding data.
    """
    splits = ["train", "val"]
    pids = load_predefined_splits(split_path, splits)
    train_pids, val_pids = pids
    train_data = filter_table_by_pids(data, train_pids)
    val_data = filter_table_by_pids(data, val_pids)
    return train_data, val_data
