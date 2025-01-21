# Description: This script is used to create and process features from the raw data.
import os
from os.path import join
from pathlib import Path
from collections import defaultdict

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
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.utils import convert_to_datetime

CONFIG_PATH = "./corebehrt/configs/create_data.yaml"

SCHEMA = {
    "PID": "str",
    "age": "float32",
    "abspos": "float64",
    "segment": "int32",
}

FEATURES_SCHEMA = {**SCHEMA, "concept": "str"}
TOKENIZED_SCHEMA = {**SCHEMA, "concept": "int32"}

default_dtypes = defaultdict(lambda: "string[pyarrow]")


def main_data(config_path):
    cfg = load_config(config_path)

    DirectoryPreparer(cfg).setup_create_data()

    logger = logging.getLogger("create_data")
    logger.info("Loading formatted features")

    # Load formatted data and create background
    raw_dir = Path(cfg.paths.data)
    concepts = dd.concat(
        [
            dd.read_parquet(
                raw_dir / f"concept.{concept_type}.parquet", dtype=default_dtypes
            )
            for concept_type in cfg.loader.concept_types
        ]
    )
    # parse_date did not play nice, so we do conversion manually
    concepts["TIMESTAMP"] = convert_to_datetime(concepts["TIMESTAMP"])

    patients_info = dd.read_parquet(
        raw_dir / "patients_info.parquet",
        dtype=default_dtypes,
    )
    # parse_date did not play nice, so we do conversion manually
    patients_info["BIRTHDATE"] = convert_to_datetime(patients_info["BIRTHDATE"])
    patients_info["DEATHDATE"] = convert_to_datetime(patients_info["DEATHDATE"])

    # Initialize tokenizer
    vocabulary = None
    if "vocabulary" in cfg.paths:
        logger.info(f"Loading vocabulary from {cfg.paths.vocabulary}")
        vocabulary = load_vocabulary(cfg.paths.vocabulary)
    tokenizer = EHRTokenizer(vocabulary=vocabulary)

    # Load or create features
    if os.path.exists(join(cfg.paths.features, "part.0.parquet")):
        logger.info("Reusing existing features")
    else:
        logger.info("Create and process features")
        with ProgressBar(dt=1):
            feature_creator = FeatureCreator(**cfg.features)
            features = feature_creator(patients_info, concepts)

            excluder = Excluder(**cfg.excluder)
            features = excluder.exclude_incorrect_events(features)

            features.to_parquet(
                cfg.paths.features, write_index=False, schema=FEATURES_SCHEMA
            )
            logger.info("Finished feature creation and processing")
    features = dd.read_parquet(cfg.paths.features)

    logger.info("Split pids")
    pids = features.PID.unique().compute().tolist()
    pretrain_pids, finetune_pids, test_pids = split_pids_into_pt_ft_test(
        pids, **cfg.split_ratios
    )

    logger.info("Tokenizing")
    with ProgressBar(dt=1):
        for split, pids in zip(
            ["pretrain", "finetune", "test"], [pretrain_pids, finetune_pids, test_pids]
        ):
            df = features[features.PID.isin(pids)]
            df = tokenizer(df)
            torch.save(pids, join(cfg.paths.tokenized, f"pids_{split}.pt"))
            df.to_parquet(
                join(cfg.paths.tokenized, f"features_{split}"),
                write_index=False,
                schema=TOKENIZED_SCHEMA,
            )
            if split == "pretrain":
                tokenizer.freeze_vocabulary()

    torch.save(tokenizer.vocabulary, join(cfg.paths.tokenized, "vocabulary.pt"))
    logger.info("Finished tokenizing")


def split_tokenize_and_save(
    features: dd.DataFrame,
    tokenizer: EHRTokenizer,
    tokenized_path: str,
    split: str,
    pids: list,
):
    """
    Load df for selected pids, tokenize and write to tokenized_path.
    """
    df = features[features.PID.isin(pids)]
    df = tokenizer(df)
    df.to_parquet(
        join(tokenized_path, f"features_{split}"),
        write_index=False,
        schema=TOKENIZED_SCHEMA,
    )
    torch.save(pids, join(tokenized_path, f"pids_{split}.pt"))


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
