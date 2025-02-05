import logging
import os
from os.path import join

from corebehrt.constants.train import DEFAULT_CV_SPLITS
from corebehrt.functional.features.split import split_into_test_and_train_val_pids
from corebehrt.functional.io_operations.load import load_pids
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import (
    cv_loop,
    cv_loop_predefined_splits,
    finetune_without_cv,
)
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_and_save_scores_mean_std,
)
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/finetune.yaml"


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main_finetune(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_finetune()

    # Logger
    logger = logging.getLogger("finetune_cv")

    data: PatientDataset = DatasetPreparer(cfg).prepare_finetune_data()

    if "predefined_splits" in cfg.paths:
        logger.info("Using predefined splits")

        test_pid_path = join(cfg.paths.predefined_splits, "test_pids.pt")
        test_pids = (
            list(load_pids(test_pid_path)) if os.path.exists(test_pid_path) else []
        )
        test_data = data.filter_by_pids(test_pids)

        if cfg.get("save_test_data", False):
            test_data.save(join(cfg.paths.model, "test_data"))

        cv_splits = cv_loop_predefined_splits(
            cfg,
            logger,
            cfg.paths.model,
            data,
            cfg.paths.predefined_splits,
            test_data,
        )

    else:
        logger.info("Splitting data")
        cv_splits = cfg.get("cv_splits", DEFAULT_CV_SPLITS)
        test_pids, train_val_pids = split_into_test_and_train_val_pids(
            data.get_pids(), cfg.data.get("test_split", None)
        )
        test_data = data.filter_by_pids(test_pids)
        if cfg.get("save_test_data", False):
            test_data.save(join(cfg.paths.model, "test_data"))
        if cv_splits > 1:
            cv_loop(
                cfg,
                logger,
                cfg.paths.model,
                data,
                train_val_pids,
                test_data,
            )
        else:
            finetune_without_cv(
                cfg,
                logger,
                cfg.paths.model,
                data,
                train_val_pids,
                test_data,
            )

    compute_and_save_scores_mean_std(cv_splits, cfg.paths.model, mode="val")
    if len(test_data) > 0:
        compute_and_save_scores_mean_std(cv_splits, cfg.paths.model, mode="test")

    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
