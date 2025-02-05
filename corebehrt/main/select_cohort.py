import logging
from os.path import join

import torch

from corebehrt.constants.paths import INDEX_DATES_FILE, PID_FILE
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


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
