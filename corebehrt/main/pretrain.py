"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

from os.path import join

from corebehrt.common.azure import AzurePathContext, save_to_blobstore
from corebehrt.common.config import load_config
from corebehrt.common.initialize import Initializer
from corebehrt.common.loader import (
    load_checkpoint_and_epoch,
    load_model_cfg_from_checkpoint,
)
from corebehrt.common.setup import DirectoryPreparer, copy_data_config, get_args
from corebehrt.functional.trainer_utils import replace_steps_with_epochs
from corebehrt.data.prepare_data import DatasetPreparer
from corebehrt.trainer.trainer import EHRTrainer

CONFIG_PATH = "./corebehrt/configs/pretrain.yaml"
BLOBSTORE = "PHAIR"

args = get_args(CONFIG_PATH)
config_path = args.config_path
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main_train(config_path):
    cfg = load_config(config_path)

    cfg, run, mount_context = AzurePathContext(
        cfg, dataset_name=BLOBSTORE
    ).adjust_paths_for_azure_pretrain()

    logger, run_folder = DirectoryPreparer.setup_run_folder(cfg)
    copy_data_config(cfg, run_folder)

    loaded_from_checkpoint = load_model_cfg_from_checkpoint(
        cfg, "pretrain_config.yaml"
    )  # if we are training from checkpoint, we need to load the old config
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_mlm_dataset()

    if "scheduler" in cfg:
        logger.info("Replacing steps with epochs in scheduler config")
        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.batch_size, len(train_dataset)
        )

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
        run=run,
        last_epoch=epoch,
    )
    logger.info("Start training")
    trainer.train()
    if cfg.env == "azure":
        save_to_blobstore(
            cfg.paths.run_name,
            join(BLOBSTORE, "models", cfg.paths.type, cfg.paths.run_name),
            overwrite=loaded_from_checkpoint,
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    main_train(config_path)
