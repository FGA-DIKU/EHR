from os.path import exists, join
from typing import Tuple

import torch

from corebehrt.constants.paths import CHECKPOINTS_DIR, FOLDS_FILE
from corebehrt.functional.features.split import TRAIN_KEY, VAL_KEY
from corebehrt.functional.setup.model import get_last_checkpoint_epoch
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.loader import ModelLoader


def load_checkpoint_and_epoch(model_dir: str, checkpoint_epoch: str = None) -> Tuple:
    """Load checkpoint and epoch from config."""
    checkpoint = ModelLoader(
        model_dir, checkpoint_epoch=checkpoint_epoch
    ).load_checkpoint()
    if checkpoint is not None:
        epoch = checkpoint["epoch"]
    else:
        epoch = get_last_checkpoint_epoch(join(model_dir, CHECKPOINTS_DIR))
    return checkpoint, epoch


def load_train_val_split(
    data: PatientDataset, split_path: str
) -> Tuple[PatientDataset, PatientDataset]:
    """Load predefined train/validation split from disk and filter data accordingly.

    Args:
        data: PatientDataset containing all patients
        split_path: Path to directory containing train_pids.pt and val_pids.pt files

    Returns:
        Tuple containing:
            - train_dataset: PatientDataset filtered to only include training patients
            - val_dataset: PatientDataset filtered to only include validation patients

    The split files should be PyTorch tensors containing patient IDs for each split.
    The function expects files named 'train_pids.pt' and 'val_pids.pt' in the split_path directory.
    """
    splits = torch.load(split_path)[0]
    train_pids, val_pids = splits[TRAIN_KEY], splits[VAL_KEY]
    train_dataset = data.filter_by_pids(train_pids)
    val_dataset = data.filter_by_pids(val_pids)
    return train_dataset, val_dataset


def get_splits_path(path_cfg) -> str:
    """Get the path to the predefined train/validation splits file.

    This function constructs the path to the splits file by joining the cohort directory
    with the FOLDS_FILE constant. It validates that both the cohort directory is specified
    and the splits file exists at the expected location.

    Args:
        path_cfg: Configuration object containing paths information.
                 Must have a 'paths' attribute with an optional 'cohort' key.

    Returns:
        str: Full path to the splits file.

    Raises:
        ValueError: If cohort_dir is not specified in the config or if the splits file
                   does not exist at the expected location.
    """
    cohort_dir = path_cfg.get("cohort", None)
    if cohort_dir is None:
        raise ValueError("cohort_dir is required if predefined_splits is true")
    splits_path = join(cohort_dir, FOLDS_FILE)
    if not exists(splits_path):
        raise ValueError(f"Splits file not found at {splits_path}")
    return splits_path
