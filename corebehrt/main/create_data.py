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

from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.split import split_pids_into_pt_ft_test
from corebehrt.modules.excluder import Excluder
from corebehrt.modules.features import FeatureCreator
from corebehrt.modules.loader import FormattedDataLoader
from corebehrt.modules.tokenizer import EHRTokenizer
from corebehrt.modules.values import ValueCreator

CONFIG_PATH = "./corebehrt/configs/create_data.yaml"

SCHEMA = {
    "PID": "str",
    "age": "float32",
    "abspos": "float64",
    "segment": "int32",
}

FEATURES_SCHEMA = {**SCHEMA, "concept": "str"}
TOKENIZED_SCHEMA = {**SCHEMA, "concept": "int32"}


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
    if os.path.exists(join(cfg.paths.features, "part.0.parquet")):
        logger.info("Reusing existing features")
    else:
        logger.info("Create and process features")
        create_and_save_features(
            Excluder(**cfg.excluder),  # Excluder is the new Handler and old Excluder
            cfg,
        )
        logger.info("Finished feature creation and processing")

    logger.info("Get all pids")
    df = dd.read_parquet(cfg.paths.features)
    pids = df.PID.unique().compute().tolist()

    logger.info("Split pids")
    pretrain_pids, finetune_pids, test_pids = split_pids_into_pt_ft_test(
        pids, **cfg.split_ratios
    )

    logger.info("Tokenizing")

    vocabulary = None
    if "vocabulary" in cfg.paths:
        logger.info(f"Loading vocabulary from {cfg.paths.vocabulary}")
        vocabulary = load_vocabulary(cfg.paths.vocabulary)
    tokenizer = EHRTokenizer(vocabulary=vocabulary, **cfg.tokenizer)

    features_path = cfg.paths.features
    tokenized_path = cfg.paths.tokenized

    with ProgressBar(dt=10):
        logger.info("Tokenizing pretrain")
        load_tokenize_and_save(
            features_path, tokenizer, tokenized_path, "pretrain", pretrain_pids
        )
        tokenizer.freeze_vocabulary()
        logger.info("Tokenizing finetune")
        load_tokenize_and_save(
            features_path, tokenizer, tokenized_path, "finetune", finetune_pids
        )
        logger.info("Tokenizing test")
        load_tokenize_and_save(
            features_path, tokenizer, tokenized_path, "test", test_pids
        )

    torch.save(tokenizer.vocabulary, join(tokenized_path, "vocabulary.pt"))
    logger.info("Finished tokenizing")


def load_tokenize_and_save(
    features_path: str,
    tokenizer: EHRTokenizer,
    tokenized_path: str,
    split: str,
    pids: list,
):
    """
    Load df for selected pids, tokenize and write to tokenized_path.
    """
    df = dd.read_parquet(features_path, filters=[("PID", "in", set(pids))]).set_index(
        "PID"
    )
    df = tokenizer(df).reset_index()
    df.to_parquet(
        join(tokenized_path, f"features_{split}"),
        write_index=False,
        schema=TOKENIZED_SCHEMA,
    )
    torch.save(pids, join(tokenized_path, f"pids_{split}.pt"))


def create_and_save_features(excluder: Excluder, cfg) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    concepts, patients_info = FormattedDataLoader(
        cfg.paths.data,
        cfg.loader.concept_types,
        include_values=(getattr(cfg.loader, "include_values", [])),
    ).load()

    with ProgressBar(dt=10):
        if "values" in cfg.features:
            value_creator = ValueCreator(**cfg.features.values)
            concepts = value_creator(concepts)
            cfg.features.pop("values")

        feature_creator = FeatureCreator(**cfg.features)
        features = feature_creator(patients_info, concepts)

        features = excluder.exclude_incorrect_events(features)

        features.to_parquet(
            cfg.paths.features, write_index=False, schema=FEATURES_SCHEMA
        )


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
