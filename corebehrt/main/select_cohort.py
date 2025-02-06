import logging
from os.path import join

import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.features.split import split_test, create_folds
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.select_cohort import select_cohort
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/select_cohort.yaml"


def main_select_cohort(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_select_cohort()

    logger = logging.getLogger("select_cohort")

    logger.info("Starting cohort selection")
    pids, index_dates = select_cohort(cfg, logger)
    logger.info("Saving cohort")
    torch.save(pids, join(cfg.paths.cohort, PID_FILE))
    index_dates.to_csv(join(cfg.paths.cohort, INDEX_DATES_FILE))

    test_ratio = cfg.get("test_ratio", 0)
    train_pids, test_pids = split_test(pids, test_ratio)

    if len(test_pids) > 0:
        torch.save(test_pids, join(cfg.paths.cohort, TEST_PIDS_FILE))

    if len(train_pids) > 0:
        folds = create_folds(train_pids, cfg.get("cv_folds", 1), cfg.get("seed", 42))
        torch.save(folds, join(cfg.paths.cohort, FOLDS_FILE))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
