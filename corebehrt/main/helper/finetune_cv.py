import os
from os.path import join, split

import torch

from corebehrt.constants.train import DEFAULT_CV_SPLITS, DEFAULT_VAL_SPLIT
from corebehrt.functional.features.split import get_n_splits_cv_pids
from corebehrt.functional.io_operations.load import load_and_select_splits
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.azure import log_metrics_to_dir


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
    with log_metrics_to_dir(f"Fold {fold}"):
        trainer.train()

    logger.info("Load best finetuned model to compute test scores")
    modelmanager_trained = ModelManager(cfg, fold)
    checkpoint = modelmanager_trained.load_checkpoint()
    model = modelmanager_trained.initialize_finetune_model(checkpoint)
    trainer.model = model
    trainer.test_dataset = test_dataset
    trainer._evaluate(epoch, mode="test")
