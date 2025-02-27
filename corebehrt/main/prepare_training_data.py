"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

import logging

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/prepare_pretrain.yaml"


def main_prepare_data(config_path):
    cfg = load_config(config_path)

    if cfg.data.type == "pretrain":
        # Setup directories
        DirectoryPreparer(cfg).setup_prepare_pretrain()
        logger = logging.getLogger("prepare pretrain data")
        logger.info("Preparing pretrain data")
        # Prepare data
        _ = DatasetPreparer(cfg).prepare_pretrain_data()

    elif cfg.data.type == "finetune":
        # Setup directories
        DirectoryPreparer(cfg).setup_prepare_finetune()
        logger = logging.getLogger("prepare finetune data")
        logger.info("Preparing finetune data")
        # Prepare data
        _ = DatasetPreparer(cfg).prepare_finetune_data()

    else:
        raise ValueError(f"Unsupported data type: {cfg.data.type}")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_prepare_data(config_path)
