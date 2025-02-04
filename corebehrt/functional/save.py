import logging
import os
from os.path import join

import torch

from corebehrt.modules.dataset import PatientDataset
from corebehrt.common.setup import VOCABULARY_FILE

logger = logging.getLogger(__name__)


def save_pids_splits(
    train_data: PatientDataset, val_data: PatientDataset, save_dir: str
) -> None:
    """
    Save train and val data to a folder.
    Assumes that the data has a column named PID.
    """
    os.makedirs(save_dir, exist_ok=True)
    train_pids = train_data.get_pids()
    val_pids = val_data.get_pids()
    torch.save(train_pids, join(save_dir, "pids_train.pt"))
    torch.save(val_pids, join(save_dir, "pids_val.pt"))


def save_vocabulary(vocab: dict, dir_: str):
    """
    Save a vocabulary to the given directory.
    """
    os.makedirs(dir_, exist_ok=True)
    torch.save(vocab, join(dir_, VOCABULARY_FILE))
