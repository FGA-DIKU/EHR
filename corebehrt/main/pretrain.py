"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

import logging

from corebehrt.common.config import load_config
from corebehrt.common.initialize import Initializer, ModelManager
from corebehrt.common.loader import (
    load_checkpoint_and_epoch,
    load_model_cfg_from_checkpoint,
)
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.save import save_pids_splits
from corebehrt.functional.split import load_train_val_split, split_pids_into_train_val
from corebehrt.functional.trainer_utils import replace_steps_with_epochs
from corebehrt.modules.dataset import MLMDataset
from corebehrt.modules.prepare_data import DatasetPreparer
from corebehrt.modules.trainer.trainer import EHRTrainer

CONFIG_PATH = "./corebehrt/configs/pretrain.yaml"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_pretrain()

    logger = logging.getLogger("pretrain")

    # Are we restarting training from checkpoint?
    restart_path = cfg.paths.get("restart_model")

    if not restart_path and ModelManager.check_checkpoints(cfg.paths.model):
        # No restart path provided, but model @ cfg.paths.model has checkpoints
        # so we restart from them
        restart_path = cfg.paths.model

    # Check if we are training from checkpoint, if so, update model config
    if restart_path:
        cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # Prepare data
    data = DatasetPreparer(cfg).prepare_pretrain_data()

    # Splitting data
    predefined_splits_path = cfg.paths.get("predefined_splits", False)
    if predefined_splits_path:
        train_data, val_data = load_train_val_split(data, predefined_splits_path)
    else:
        train_data, val_data = split_pids_into_train_val(
            data, cfg.data.get("val_ratio", 0.2)
        )

    vocab = load_vocabulary(cfg.paths.tokenized)
    # Initialize datasets
    train_dataset = MLMDataset(train_data.patients, vocab, **cfg.data.dataset)
    val_dataset = MLMDataset(val_data.patients, vocab, **cfg.data.dataset)

    # Save split
    save_pids_splits(train_data, val_data, cfg.paths.model)

    if "scheduler" in cfg:
        logger.info("Replacing steps with epochs in scheduler config")
        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.batch_size, len(train_dataset)
        )

    checkpoint, epoch = None, None
    if restart_path:
        checkpoint, epoch = load_checkpoint_and_epoch(
            restart_path, cfg.paths.get("checkpoint_epoch")
        )

    logger.info(f"Continue training from epoch {epoch}")
    initializer = Initializer(cfg, checkpoint=checkpoint, model_path=restart_path)
    model = initializer.initialize_pretrain_model(train_dataset)
    logger.info("Initializing optimizer")
    optimizer = initializer.initialize_optimizer(model)
    scheduler = initializer.initialize_scheduler(optimizer)

    logger.info("Initialize trainer")
    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        last_epoch=epoch,
    )
    logger.info("Start training")
    trainer.train()
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_train(config_path)
