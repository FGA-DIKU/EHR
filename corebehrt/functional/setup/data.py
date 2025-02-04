import os
from os.path import join
from typing import Tuple

import torch
from corebehrt.modules.preparation.dataset import PatientDataset
import logging

logger = logging.getLogger(__name__)


def get_pids_file(split_dir: str, mode: str) -> str:
    """Get pids file from predefined splits.
    The file can be named pids_{mode}.pt or {mode}_pids.pt."""
    if os.path.exists(join(split_dir, f"{mode}_pids.pt")):
        return join(split_dir, f"{mode}_pids.pt")
    elif os.path.exists(join(split_dir, f"pids_{mode}.pt")):
        return join(split_dir, f"pids_{mode}.pt")
    else:
        raise ValueError(f"No pids file found for mode {mode} in {split_dir}")


def load_and_select_splits(
    split_dir: str, data: PatientDataset
) -> Tuple[PatientDataset, PatientDataset]:
    """Load and select pids from predefined splits"""
    logger.info("Load and select pids")
    train_pids = torch.load(get_pids_file(split_dir, "train"))
    val_pids = torch.load(get_pids_file(split_dir, "val"))
    train_data = data.filter_by_pids(train_pids)
    val_data = data.filter_by_pids(val_pids)
    return train_data, val_data
