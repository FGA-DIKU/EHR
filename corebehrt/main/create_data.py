"""
Input: Formatted Data
- Load concepts
- Handle wrong data
- Exclude patients with <k concepts
- Split data
- Tokenize
- truncate train and val
"""

import logging
import os
from os.path import join

import dask.dataframe as dd
import torch
from dask.diagnostics import ProgressBar

from corebehrt.functional.features.split import split_pids_into_pt_ft_test
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.features.excluder import Excluder
from corebehrt.modules.features.tokenizer import EHRTokenizer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.main.helper.create_data import (
    load_tokenize_and_save,
    create_and_save_features,
)

CONFIG_PATH = "./corebehrt/configs/create_data.yaml"


def main_data(config_path):
    """
    Loads data
    Finds outcomes
    Creates features
    Handles wrong data
    Excludes patients with <k concepts
    Splits data
    Tokenizes
    Saves
    """
    cfg = load_config(config_path)

    DirectoryPreparer(cfg).setup_create_data()

    logger = logging.getLogger("create_data")
    logger.info("Initialize Processors")
    logger.info("Starting feature creation and processing")

    # TODO: temporary fix/check until we split the script into two.
    # As cfg.paths.features is always set, its value cannot be used to decide
    # if features are present.
    if os.path.exists(join(cfg.paths.features, "held_out", "0.parquet")):
        logger.info("Reusing existing features")
    else:
        logger.info("Create and process features")
        create_and_save_features(
            Excluder(**cfg.excluder),  # Excluder is the new Handler and old Excluder
            cfg,
        )
        logger.info("Finished feature creation and processing")

    # logger.info("Get all pids")
    # df = dd.read_parquet(cfg.paths.features)
    # pids = df.PID.unique().compute().tolist()

    # logger.info("Split pids")
    # pretrain_pids, finetune_pids, test_pids = split_pids_into_pt_ft_test(
    #     pids, **cfg.split_ratios
    # )

    # logger.info("Tokenizing")

    # vocabulary = None
    # if "vocabulary" in cfg.paths:
    #     logger.info(f"Loading vocabulary from {cfg.paths.vocabulary}")
    #     vocabulary = load_vocabulary(cfg.paths.vocabulary)
    # tokenizer = EHRTokenizer(vocabulary=vocabulary, **cfg.tokenizer)

    # features_path = cfg.paths.features
    # tokenized_path = cfg.paths.tokenized

    # with ProgressBar(dt=1):
    #     logger.info("Tokenizing pretrain")
    #     load_tokenize_and_save(
    #         features_path, tokenizer, tokenized_path, "pretrain", pretrain_pids
    #     )
    #     tokenizer.freeze_vocabulary()
    #     logger.info("Tokenizing finetune")
    #     load_tokenize_and_save(
    #         features_path, tokenizer, tokenized_path, "finetune", finetune_pids
    #     )
    #     logger.info("Tokenizing test")
    #     load_tokenize_and_save(
    #         features_path, tokenizer, tokenized_path, "test", test_pids
    #     )

    # torch.save(tokenizer.vocabulary, join(tokenized_path, "vocabulary.pt"))
    # logger.info("Finished tokenizing")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
