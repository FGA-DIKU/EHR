import logging
from os.path import join

import torch

from corebehrt.constants.paths import (
    FOLDS_FILE,
    INDEX_DATES_FILE,
    PID_FILE,
    TEST_PIDS_FILE,
)
from corebehrt.functional.features.split import create_folds
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
    path_cfg = cfg.paths
    pids, index_dates, train_val_pids, test_pids = select_cohort(
        path_cfg,
        cfg.selection,
        cfg.index_date,
        test_ratio=cfg.test_ratio,
        logger=logger,
    )
    logger.info("Saving cohort")
    torch.save(pids, join(path_cfg.cohort, PID_FILE))
    index_dates.to_csv(join(path_cfg.cohort, INDEX_DATES_FILE))

    if len(test_pids) > 0:
        torch.save(test_pids, join(path_cfg.cohort, TEST_PIDS_FILE))

    if len(train_val_pids) > 0:
        folds = create_folds(
            train_val_pids, cfg.get("cv_folds", 1), cfg.get("seed", 42)
        )
        torch.save(folds, join(path_cfg.cohort, FOLDS_FILE))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
