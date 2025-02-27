"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

import logging

from corebehrt.functional.features.split import split_pids_into_train_val
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.io_operations.save import save_pids_splits
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.setup.model import load_model_cfg_from_checkpoint
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.main.helper.pretrain import (
    load_checkpoint_and_epoch,
    load_train_val_split,
    get_splits_path,
)
from corebehrt.modules.preparation.dataset import MLMDataset
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.initializer import Initializer
from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.trainer.trainer import EHRTrainer

CONFIG_PATH = "./corebehrt/configs/prepare_pretrain.yaml"


def main_prepare_data(config_path):
    cfg = load_config(config_path)

    if cfg.data.type == "pretrain":
        # Setup directories
        DirectoryPreparer(cfg).setup_prepare_pretrain()
        logger = logging.getLogger("prepare pretrain data")
        # Prepare data
        data = DatasetPreparer(cfg).prepare_pretrain_data()

    elif cfg.data.type == "finetune":
        # Setup directories
        DirectoryPreparer(cfg).setup_prepare_finetune()
        logger = logging.getLogger("prepare finetune data")
        # Prepare data
        data = DatasetPreparer(cfg).prepare_finetune_data()

    else:
        raise ValueError(f"Unsupported data type: {cfg.data.type}")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_prepare_data(config_path)
