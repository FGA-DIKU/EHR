import logging
import os
from os.path import join, split

import torch

from corebehrt.common.config import load_config
from corebehrt.common.initialize import ModelManager
from corebehrt.common.loader import load_and_select_splits
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.classes.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.data.prepare_data import DatasetPreparer
from corebehrt.data.split import get_n_splits_cv_pids
from corebehrt.evaluation.utils import (
    compute_and_save_scores_mean_std,
    split_into_test_and_train_val_pids,
)
from corebehrt.functional.load import load_pids
from corebehrt.functional.trainer_utils import replace_steps_with_epochs
from corebehrt.trainer.trainer import EHRTrainer

CONFIG_PATH = "./corebehrt/configs/finetune.yaml"

DEFAULT_CV_SPLITS = 2  # You can change this to desired value
DEFAULT_VAL_SPLIT = 0.2

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


def cv_loop_predefined_splits(
    cfg,
    logger,
    finetune_folder: str,
    data: PatientDataset,
    predefined_splits_dir: str,
    test_data: PatientDataset,
) -> int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [
        join(predefined_splits_dir, d)
        for d in os.listdir(predefined_splits_dir)
        if os.path.isdir(os.path.join(predefined_splits_dir, d)) and "fold_" in d
    ]
    cv_splits = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split("_")[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        train_data, val_data = load_and_select_splits(fold_dir, data)
        finetune_fold(
            cfg, logger, finetune_folder, train_data, val_data, fold, test_data
        )
    return cv_splits


def cv_loop(
    cfg,
    logger,
    finetune_folder: str,
    data: PatientDataset,
    train_val_pids: list,
    test_data: PatientDataset,
) -> None:
    """Loop over cross validation folds."""
    cv_splits = cfg.get("cv_splits", DEFAULT_CV_SPLITS)
    for fold, (train_pids, val_pids) in enumerate(
        get_n_splits_cv_pids(cv_splits, train_val_pids)
    ):
        fold += 1
        logger.info(f"Training fold {fold}/{cv_splits}")
        logger.info("Splitting data")
        train_data = data.filter_by_pids(train_pids)
        val_data = data.filter_by_pids(val_pids)
        finetune_fold(
            cfg,
            logger,
            finetune_folder,
            train_data,
            val_data,
            fold,
            test_data,
        )


def finetune_without_cv(
    cfg,
    logger,
    finetune_folder: str,
    data: PatientDataset,
    train_val_pids: list,
    test_data: PatientDataset = None,
) -> None:
    val_split = cfg.data.get("val_split", DEFAULT_VAL_SPLIT)
    logger.info(
        f"Splitting train_val of length {len(train_val_pids)} into train and val with val_split={val_split}"
    )
    train_pids = train_val_pids[: int(len(train_val_pids) * (1 - val_split))]
    val_pids = train_val_pids[int(len(train_val_pids) * (1 - val_split)) :]
    train_data = data.filter_by_pids(train_pids)
    val_data = data.filter_by_pids(val_pids)
    finetune_fold(
        cfg,
        logger,
        finetune_folder,
        train_data,
        val_data,
        1,
        test_data,
    )


def finetune_fold(
    cfg,
    logger,
    finetune_folder: str,
    train_data: PatientDataset,
    val_data: PatientDataset,
    fold: int,
    test_data: PatientDataset = None,
) -> None:
    """Finetune model on one fold"""
    if "scheduler" in cfg:
        logger.info("Replacing steps with epochs in scheduler config")
        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.batch_size, len(train_data)
        )

    fold_folder = join(finetune_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Saving pids")
    torch.save(train_data.get_pids(), join(fold_folder, "train_pids.pt"))
    torch.save(val_data.get_pids(), join(fold_folder, "val_pids.pt"))
    if len(test_data) > 0:
        torch.save(test_data.get_pids(), join(fold_folder, "test_pids.pt"))

    logger.info("Initializing datasets")

    train_dataset = BinaryOutcomeDataset(train_data.patients)
    val_dataset = BinaryOutcomeDataset(val_data.patients)
    test_dataset = (
        BinaryOutcomeDataset(test_data.patients) if len(test_data) > 0 else None
    )

    modelmanager = ModelManager(cfg, fold)
    checkpoint = modelmanager.load_checkpoint()
    model = modelmanager.initialize_finetune_model(checkpoint)

    outcomes = train_data.get_outcomes()  # needed for sampler/ can be made optional
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(
        model, outcomes
    )
    epoch = modelmanager.get_epoch()

    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,  # test only after training
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        scheduler=scheduler,
        cfg=cfg,
        logger=logger,
        accumulate_logits=True,
        run_folder=fold_folder,
        last_epoch=epoch,
    )
    trainer.train()

    logger.info("Load best finetuned model to compute test scores")
    modelmanager_trained = ModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint()
    model = modelmanager_trained.initialize_finetune_model(checkpoint)
    trainer.model = model
    trainer.test_dataset = test_dataset
    trainer._evaluate(epoch, mode="test")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_finetune(args.config_path)
