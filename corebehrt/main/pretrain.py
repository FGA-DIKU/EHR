"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

from os.path import join
import logging

from corebehrt.common.config import load_config
from corebehrt.common.initialize import Initializer
from corebehrt.common.loader import (
    load_checkpoint_and_epoch,
    load_model_cfg_from_checkpoint,
)
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.common.utils import compute_number_of_warmup_steps
from corebehrt.data.prepare_data import DatasetPreparer
from corebehrt.trainer.trainer import EHRTrainer

CONFIG_PATH = "./corebehrt/configs/pretrain.yaml"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main_train(config_path):
    cfg = load_config(config_path)

    # Setup directories
    DirectoryPreparer(cfg).setup_pretrain()

    logger = logging.getLogger("pretrain")

    # Are we restarting training from checkpoint?
    restarted = hasattr(cfg.paths, "restart_model")

    # Check if we are training from checkpoint, if so, reload config
    if restarted:
        cfg.model = load_model_cfg_from_checkpoint(
            cfg.paths.restart_model, "pretrain_config"
        )

    # Prepare dataset
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_mlm_dataset()

    if "scheduler" in cfg:
        logger.info("Computing number of warmup steps")
        compute_number_of_warmup_steps(cfg, len(train_dataset))

    checkpoint, epoch = None, None
    if restarted:
        checkpoint, epoch = load_checkpoint_and_epoch(cfg)

    logger.info(f"Continue training from epoch {epoch}")
    initializer = Initializer(cfg, checkpoint=checkpoint)
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
