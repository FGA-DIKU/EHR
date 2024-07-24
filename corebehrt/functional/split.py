from typing import Tuple
import numpy as np


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
