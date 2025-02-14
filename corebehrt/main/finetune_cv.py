import logging
import os
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.paths import FOLDS_FILE, PROCESSED_DATA_DIR, TEST_PIDS_FILE
from corebehrt.constants.train import DEFAULT_CV_FOLDS, DEFAULT_VAL_SPLIT
from corebehrt.functional.features.split import split_into_test_and_train_val_pids
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import cv_loop, get_n_folds
from corebehrt.main.helper.pretrain import get_splits_path
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_and_save_scores_mean_std,
)
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/finetune.yaml"


def main_finetune(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_finetune()

    # Logger
    logger = logging.getLogger("finetune_cv")

    data: PatientDataset = DatasetPreparer(cfg).prepare_finetune_data()

    test_data = PatientDataset([])
    if cfg.paths.get("test_pids", None) is not None:
        logger.info("Using predefined test data")
        test_pids = torch.load(cfg.paths.test_pids)
        train_val_pids = [pid for pid in data.get_pids() if pid not in test_pids]
    else:
        logger.info("Randomly splitting test data")
        test_pids, train_val_pids = split_into_test_and_train_val_pids(
            data.get_pids(), cfg.data.get("test_split", 0)
        )
    processed_data_dir = join(cfg.paths.model, PROCESSED_DATA_DIR)
    os.makedirs(processed_data_dir, exist_ok=True)
    torch.save(test_pids, join(processed_data_dir, TEST_PIDS_FILE))

    test_data = data.filter_by_pids(test_pids)
    train_val_data = data.filter_by_pids(train_val_pids)

    if cfg.data.get("predefined_folds", False):
        folds_path = get_splits_path(cfg.paths)
        folds = torch.load(folds_path)
        n_folds = len(folds)
        logger.info(f"Using {n_folds} predefined folds")

    else:
        n_folds = cfg.data.get("cv_folds", DEFAULT_CV_FOLDS)
        val_split = cfg.data.get("val_split", DEFAULT_VAL_SPLIT)
        logger.info(f"Using cross validation with {n_folds} folds")
        folds = get_n_folds(n_folds, train_val_pids, val_split)
    torch.save(folds, join(processed_data_dir, FOLDS_FILE))
    cv_loop(
        cfg,
        logger,
        cfg.paths.model,
        train_val_data,
        folds,
        test_data,
    )

    compute_and_save_scores_mean_std(n_folds, cfg.paths.model, mode="val")
    if len(test_data) > 0:
        compute_and_save_scores_mean_std(n_folds, cfg.paths.model, mode="test")

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
