"""
Input: Formatted Data
- Load patients_info, exposures, outcomes
- Exclude by sex, age, dead before index date, outcome before index date
- Split data
"""

import logging
from os.path import join
from typing import Tuple

import numpy as np
import torch

from corebehrt.classes.loader import FormattedDataLoader
from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args

CONFIG_PATH = "./corebehrt/configs/select_cohort.yaml"


def main_select_cohort(config_path):
    """
    Loads data
    Finds outcomes
    Creates features
    Handles wrong data
    Excludes patients with <k concepts
    Splits data
    Tokenizes
    Saves
    """
    cfg = load_config(config_path)

    DirectoryPreparer(cfg).setup_select_cohort()

    logger = logging.getLogger("select_cohort")
    logger.info("Starting cohort selection")

    pids, index_dates = select_cohort(cfg)
    logger.info("Finished cohort selection")

    logger.info("Split pids")
    splits = split_pids_into_train_val_test(pids, **cfg.split_ratios)
    logger.info("Save splits")
    torch.save(splits, join(cfg.paths.data, "splits.pt"))
    torch.save(index_dates, join(cfg.paths.data, "index_dates.pt"))


def split_pids_into_train_val_test(
    pids: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Split pids into train, val, test
    Return dict with keys train, val, test and values as lists of pids

    Parameters
    ----------
    pids : list
        List of patient IDs to split
    train_ratio : float, optional
        Ratio of patients to use for training, by default 0.7
    val_ratio : float, optional
        Ratio of patients to use for validation, by default 0.15
    test_ratio : float, optional
        Ratio of patients to use for testing, by default 0.15

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test' and values as lists of pids
    """
    # Check that ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle pids
    pids = np.array(pids)
    np.random.shuffle(pids)

    # Calculate split indices
    n_samples = len(pids)
    train_idx = int(n_samples * train_ratio)
    val_idx = train_idx + int(n_samples * val_ratio)

    # Split pids
    train_pids = pids[:train_idx].tolist()
    val_pids = pids[train_idx:val_idx].tolist()
    test_pids = pids[val_idx:].tolist()

    return {"train": train_pids, "val": val_pids, "test": test_pids}


def select_cohort(cfg) -> Tuple[list, dict]:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    concepts, patients_info = FormattedDataLoader(
        cfg.paths.data,
        cfg.loader.concept_types,
        include_values=(getattr(cfg.loader, "include_values", [])),
    ).load()

    #########################################################
    # Load outcomes/ exposures and index events/ set index dates
    # sex exclusion: via patients_info select sex
    # age exclusion: calculate age at index date
    # dead before index date: via patients_info select dead
    # outcome before index date: via patients_info select outcome
    # exposure selection: e.g. pregnant only
    # save remaining pids and index dates
    #########################################################


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
