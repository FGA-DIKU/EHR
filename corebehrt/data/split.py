from typing import List

import numpy as np
from sklearn.model_selection import KFold


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
