import os
import numpy as np
import pandas as pd
from typing import Tuple
from os.path import join
from datetime import datetime

from torch.utils.data import WeightedRandomSampler

from corebehrt.data.dataset import PatientDataset


def get_sampler(cfg, train_dataset, outcomes):
    """Get sampler for training data.
    sample_weight: float. Adjusts the number of samples in the positive class.
    """
    if cfg.trainer_args["sampler"]:
        labels = pd.Series(outcomes).notna().astype(int)
        value_counts = labels.value_counts()
        label_weight = inverse_sqrt(value_counts)
        weights = labels.map(label_weight).values
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True
        )
        return sampler
    else:
        return None


def inverse_sqrt(x):
    return 1 / np.sqrt(x)


def compute_and_save_scores_mean_std(
    n_splits: int, finetune_folder: str, mode="val"
) -> None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    scores = []
    for fold in range(1, n_splits + 1):
        fold_checkpoints_folder = join(finetune_folder, f"fold_{fold}", "checkpoints")
        last_epoch = max(
            [
                int(f.split("_")[-2].split("epoch")[-1])
                for f in os.listdir(fold_checkpoints_folder)
                if f.startswith("checkpoint_epoch")
            ]
        )
        table_path = join(fold_checkpoints_folder, f"{mode}_scores_{last_epoch}.csv")
        if not os.path.exists(table_path):
            continue
        fold_scores = pd.read_csv(
            join(fold_checkpoints_folder, f"{mode}_scores_{last_epoch}.csv")
        )
        scores.append(fold_scores)
    scores = pd.concat(scores)
    scores_mean_std = scores.groupby("metric")["value"].agg(["mean", "std"])
    date = datetime.now().strftime("%Y%m%d-%H%M")
    scores_mean_std.to_csv(join(finetune_folder, f"{mode}_scores_mean_std_{date}.csv"))


def split_into_test_data_and_train_val_indices(
    cfg, data: PatientDataset
) -> Tuple[PatientDataset, list]:
    """Split data into test and train_val indices. And save test set."""
    indices = list(range(len(data.get_pids())))
    test_split = cfg.data.get("test_split", None)
    if test_split is not None:
        test_indices, train_val_indices = split_test_set(indices, test_split)
    else:
        test_indices = []
        train_val_indices = indices
    test_data = (
        PatientDataset()
        if len(test_indices) == 0
        else PatientDataset([data.patients[i] for i in test_indices], data.vocabulary)
    )
    return test_data, train_val_indices


def split_test_set(indices: list, test_split: float) -> Tuple[list, list]:
    """Split intro test and train_val indices"""
    np.random.seed(42)
    test_indices = np.random.choice(
        indices, size=int(len(indices) * test_split), replace=False
    )
    test_indices_set = set(test_indices)
    train_val_indices = [i for i in indices if i not in test_indices_set]
    return test_indices, train_val_indices
