import os
import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime
from typing import List
from torch.utils.data import WeightedRandomSampler


def get_sampler(cfg, outcomes: List[int]):
    """Get sampler for training data.
    sample_weight: float. Adjusts the number of samples in the positive class.
    """
    if cfg.trainer_args["sampler"]:
        _, counts = np.unique(np.array(outcomes), return_counts=True)
        label_weight = inverse_sqrt(counts)
        sampler = WeightedRandomSampler(
            weights=label_weight, num_samples=len(outcomes), replacement=True
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


def split_into_test_and_train_val_pids(pids: list, test_split: float):
    test_pids = np.random.choice(pids, size=int(len(pids) * test_split), replace=False)
    set_test_pids = set(test_pids)
    train_val_pids = [pid for pid in pids if pid not in set_test_pids]
    return test_pids, train_val_pids
