"""Train XGBoost model on encoded EHR data."""

import logging
import torch
import numpy as np
from os.path import join
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main.helper.xgboost_cv import cv_loop
from corebehrt.functional.io_operations.load import load_vocabulary


CONFIG_PATH = "./corebehrt/configs/xgboost.yaml"
from corebehrt.constants.paths import (
    FOLDS_FILE,
    TEST_PIDS_FILE,
    PREPARED_ALL_PATIENTS,
)
import os

def prepare_data_for_xgboost(dataset):
    """Convert encoded dataset to XGBoost format."""
    all_data = [dataset[i] for i in range(len(dataset))]
    
    # For each patient, take the mean of their concept encodings to get a fixed-length vector
    X = np.array([np.mean(d['concepts'], axis=0) for d in all_data])
    y = np.array([d['outcome'] for d in all_data])
    
    return X, y


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_xgboost()

    logger = logging.getLogger("xgboost")

    loaded_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    data = PatientDataset(loaded_data)
    vocab = load_vocabulary(cfg.paths.prepared_data)
    test_data = PatientDataset([])

    # Initialize test and train/val pid lists
    test_pids = []
    train_val_pids = data.get_pids()

    # If evaluation is desired, then:
    #   - If test_pids are present in the prepared data directory, use them
    if cfg.get("evaluate", False):
        if os.path.exists(join(cfg.paths.prepared_data, TEST_PIDS_FILE)):
            test_pids = torch.load(join(cfg.paths.prepared_data, TEST_PIDS_FILE))
            test_data = data.filter_by_pids(test_pids)
    train_val_data = data.filter_by_pids(train_val_pids)


    # Use folds from prepared data
    folds_path = join(cfg.paths.prepared_data, FOLDS_FILE)
    folds = torch.load(folds_path)
    check_for_overlap(folds, test_pids, logger)
    n_folds = len(folds)
    logger.info(f"Using {n_folds} predefined folds")

    cv_loop(
        cfg,
        logger,
        cfg.paths.model,
        train_val_data,
        vocab,
        folds,
        test_data,
    )



if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_train(config_path)
