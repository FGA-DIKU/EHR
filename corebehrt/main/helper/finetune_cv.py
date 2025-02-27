import os
from os.path import join
from typing import List

import torch

from corebehrt.azure import log_metrics, setup_metrics_dir
from corebehrt.constants.data import TRAIN_KEY, VAL_KEY
from corebehrt.constants.train import DEFAULT_VAL_SPLIT
from corebehrt.functional.features.split import get_n_splits_cv_pids
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.trainer import EHRTrainer


def cv_loop(
    cfg,
    logger,
    finetune_folder: str,
    data: PatientDataset,
    folds: list,
    test_data: PatientDataset,
) -> None:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    for fold, fold_dict in enumerate(folds):
        fold += 1  # 1-indexed
        train_pids = fold_dict[TRAIN_KEY]
        val_pids = fold_dict[VAL_KEY]
        logger.info(f"Training fold {fold}/{len(folds)}")

        train_data = data.filter_by_pids(train_pids)
        val_data = data.filter_by_pids(val_pids)

        with setup_metrics_dir(f"Fold {fold}"):
            finetune_fold(
                cfg, logger, finetune_folder, train_data, val_data, fold, test_data
            )


def get_n_folds(
    n_folds: int, train_val_pids: list, val_split: float = DEFAULT_VAL_SPLIT
) -> list:
    """
    Generate cross-validation folds from a list of patient IDs.
     Args:
        n_folds (int): Number of cross-validation folds to generate.
        train_val_pids (list): List of patient IDs to split into folds.
        val_split (float, optional): Fraction of data to use for validation in each fold.
            Defaults to DEFAULT_VAL_SPLIT. Only used if n_folds > 1.

    Returns:
        list: List of dictionaries, where each dictionary contains train and validation
            patient IDs for a fold. Keys are TRAIN_KEY and VAL_KEY.
    """
    folds_iter = get_n_splits_cv_pids(
        n_folds,
        train_val_pids,
        val_split=val_split,
    )
    folds = [{TRAIN_KEY: fold[0], VAL_KEY: fold[1]} for fold in folds_iter]
    return folds


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

    if len(test_data) > 0:
        test_loss, test_metrics = trainer._evaluate(epoch, mode="test")
        log_best_metrics(test_loss, test_metrics, "test")


def log_best_metrics(loss: float, metrics: dict, split: str) -> None:
    """
    Logs a dict of metrics, where each metric is prepended by 'best.<split>.'.
    Example: 'val_loss' -> 'best.val.val_loss'
    """
    row = {f"{split}_loss": loss, **metrics}
    prefixed = {f"best.{split}.{k}": v for k, v in row.items()}
    log_metrics(prefixed)


def check_for_overlap(folds: List[dict], test_pids: list, logger) -> None:
    """
    Check for overlap between test and train/validation patient IDs.
    """
    fold = folds[0]  # all folds have same pids in total, we use fold 0 as example

    train_pids = set(fold[TRAIN_KEY])
    val_pids = set(fold[VAL_KEY])
    test_pids = set(test_pids)
    if train_pids & test_pids or val_pids & test_pids:
        logger.warning(
            "Found overlap between test and train/validation patient IDs. "
            "This means some patients appear in both test and training/validation sets, "
            "which may lead to data leakage and overly optimistic results. "
            "Please verify this overlap is intentional for your use case."
        )
