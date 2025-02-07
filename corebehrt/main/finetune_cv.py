import logging
from os.path import join

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from corebehrt.constants.paths import TEST_PIDS_FILE
from corebehrt.constants.train import DEFAULT_CV_SPLITS
from corebehrt.functional.features.split import split_into_test_and_train_val_pids
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import cv_loop, cv_loop_predefined_splits
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
            data.get_pids(), cfg.data.get("test_split", None)
        )
        torch.save(test_pids, join(cfg.paths.model, TEST_PIDS_FILE))

    test_data = data.filter_by_pids(test_pids)
    train_val_data = data.filter_by_pids(train_val_pids)

    if cfg.data.get("predefined_folds", False):
        folds_path = get_splits_path(cfg.paths)
        folds = torch.load(folds_path)
        logger.info(f"Using {len(folds)} predefined folds")
        cv_folds = cv_loop_predefined_splits(
            cfg,
            logger,
            cfg.paths.model,
            train_val_data,
            folds,
            test_data,
        )

    else:
        cv_folds = cfg.data.get("cv_folds", DEFAULT_CV_SPLITS)
        logger.info(f"Using cross validation with {cv_folds} folds")
        cv_loop(cfg, logger, cfg.paths.model, train_val_data, test_data, cv_folds)
    compute_and_save_scores_mean_std(cv_folds, cfg.paths.model, mode="val")
    if len(test_data) > 0:
        compute_and_save_scores_mean_std(cv_folds, cfg.paths.model, mode="test")
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
