"""
Input: Formatted Data
- Load concepts
- Handle wrong data
- Exclude patients with <k concepts
- Split data
- Tokenize
- truncate train and val
"""

import os
from os.path import join

import logging
import dask.dataframe as dd
import torch
from dask.diagnostics import ProgressBar

from corebehrt.classes.excluder import Excluder
from corebehrt.classes.features import FeatureCreator
from corebehrt.classes.tokenizer import EHRTokenizer
from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.functional.split import split_pids_into_pt_ft_test
from corebehrt.classes.loader import FormattedDataLoader
from corebehrt.classes.values import ValueCreator

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

    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

    with Profiler() as _, ResourceProfiler() as _, CacheProfiler() as _, ProgressBar(
        dt=1
    ) as _:
        import time

        start = time.time()
        logger.info("Split into pretrain and finetune.")
        # load only necessary rows
        print(f"Time taken to load pretrain: {time.time() - start} seconds")
        logger.info("Tokenizing")
        vocabulary = None
        if "vocabulary" in cfg.paths:
            logger.info(f"Loading vocabulary from {cfg.paths.vocabulary}")
            vocabulary = torch.load(cfg.paths.vocabulary)
        tokenizer = EHRTokenizer(vocabulary=vocabulary, **cfg.tokenizer)

        start = time.time()
        schema = {
            "PID": "str",
            "age": "float32",
            "abspos": "float64",
            "concept": "int32",
            "segment": "int32",
        }

        load_tokenize_and_save(
            cfg.paths.features,
            tokenizer,
            cfg.paths.tokenized,
            "pretrain",
            schema,
            pretrain_pids,
        )
        tokenizer.freeze_vocabulary()
        torch.save(tokenizer.vocabulary, join(cfg.paths.tokenized, "vocabulary.pt"))
        # load_tokenize_and_save(cfg.paths.features, tokenizer, cfg.paths.tokenized, "finetune", schema, finetune_pids)
        # load_tokenize_and_save(cfg.paths.features, tokenizer, cfg.paths.tokenized, "test", schema, test_pids)
        print(f"Time taken to tokenize: {time.time() - start} seconds")

    # visualize([prof, rprof, cprof], filename = "profiling/tokenization_100k.html", show=False)


def load_tokenize_and_save(
    features_path: str,
    tokenizer: EHRTokenizer,
    tokenized_path: str,
    split: str,
    schema: dict,
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
        schema=schema,
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

    if "values" in cfg.features:
        value_creator = ValueCreator(**cfg.features.values)
        concepts = value_creator(concepts)
        cfg.features.pop("values")

    feature_creator = FeatureCreator(**cfg.features)
    features = feature_creator(patients_info, concepts)

    features = excluder.exclude_incorrect_events(features)

    schema = {
        "PID": "str",
        "age": "float32",
        "abspos": "float64",
        "concept": "str",
        "segment": "int32",
    }
    with ProgressBar(dt=1):
        features.to_parquet(cfg.paths.features, write_index=False, schema=schema)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
