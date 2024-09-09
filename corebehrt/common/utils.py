import glob
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
import torch

from corebehrt.common.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module


def iter_patients(features: dict) -> Generator[dict, None, None]:
    """Iterate over patients in a features dict."""
    for i in range(len(features["concept"])):
        yield {key: values[i] for key, values in features.items()}


def check_patient_counts(
    concepts: pd.DataFrame, patients_info: pd.DataFrame, logger
) -> None:
    """Check that the number of patients in concepts and patients_info match."""
    if concepts.PID.nunique() != patients_info.PID.nunique():
        logger.warning(
            f"patients info contains {patients_info.PID.nunique()} patients != \
                        {concepts.PID.nunique()} unique patients in concepts"
        )


def check_existing_splits(data_dir: str) -> bool:
    """Check if train, val, and test splits already exist in data_dir."""
    if (
        os.path.exists(join(data_dir, "train_pids.pt"))
        and os.path.exists(join(data_dir, "val_pids.pt"))
        and os.path.exists(join(data_dir, "test_pids.pt"))
        and os.path.exists(join(data_dir, "train_file_ids.pt"))
        and os.path.exists(join(data_dir, "val_file_ids.pt"))
        and os.path.exists(join(data_dir, "test_file_ids.pt"))
    ):
        return True
    else:
        return False


def check_directory_for_features(dir_: str) -> bool:
    """Check if features already exist in directory."""
    features_dir = join(dir_, "features")
    if os.path.exists(features_dir):
        if len(glob.glob(join(features_dir, "features*.pt"))) > 0:
            logger.warning(f"Features already exist in {features_dir}.")
            logger.warning(f"Skipping feature creation.")
        return True
    else:
        return False


def split_path(path_str: str) -> list:
    """Split path into its components."""
    directories = []
    while path_str:
        path_str, directory = os.path.split(path_str)
        # If we've reached the root directory
        if directory:
            directories.append(directory)
        elif path_str:
            break
    return directories[::-1]  # Reverse the list to get original order


def convert_epochs_to_steps(
    cfg: Config, key: str, num_patients: int, batch_size: int
) -> None:
    """Convert number of epochs to number of steps based on number of patients and batch size"""
    logger.info(f"Computing number of steps from {key}")
    num_epochs = cfg.scheduler[key]
    num_steps = int(num_patients / batch_size * num_epochs)
    logger.info(f"Number of steps for {key}: {num_steps}")
    cfg.scheduler[key.replace("_epochs", "_steps")] = num_steps
    del cfg.scheduler[key]


def compute_number_of_warmup_steps(cfg: Config, num_patients: int) -> None:
    """Compute number of warmup steps based on number of patients and batch size"""
    batch_size = cfg.trainer_args.batch_size
    epochs_keys = [key for key in cfg.scheduler if key.endswith("_epochs")]
    for key in epochs_keys:
        convert_epochs_to_steps(cfg, key, num_patients, batch_size)
