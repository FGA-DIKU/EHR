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
import shutil
from os.path import join

import torch
import dask.dataframe as dd

from corebehrt.common.azure import AzurePathContext, save_to_blobstore
from corebehrt.common.config import load_config
from corebehrt.common.logger import TqdmToLogger
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.common.utils import check_directory_for_features
from corebehrt.data.concept_loader import ConceptLoaderLarge
from tqdm import tqdm

# New stuff
from corebehrt.classes.features import FeatureCreator
from corebehrt.classes.excluder import Excluder
from corebehrt.classes.tokenizer import EHRTokenizer

from corebehrt.functional.split import split_pids_into_pt_ft_test
from corebehrt.functional.convert import convert_to_sequences

CONFIG_PATH = "./corebehrt/configs/create_data.yaml"
BLOBSTORE = "PHAIR"


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
    cfg, _, mount_context = AzurePathContext(
        cfg, dataset_name=BLOBSTORE
    ).azure_data_pretrain_setup()

    logger = DirectoryPreparer(config_path).prepare_directory(cfg)
    logger.info("Mount Dataset")

    logger.info("Initialize Processors")
    logger.info("Starting feature creation and processing")
    if not check_directory_for_features(cfg.loader.data_dir):
        create_and_save_features(
            ConceptLoaderLarge(**cfg.loader),
            Excluder(**cfg.excluder),  # Excluder is the new Handler and old Excluder
            cfg,
            logger,
        )
        logger.info("Finished feature creation and processing")
    else:
        logger.info("Using existing features!")

    df = dd.read_csv(
        join(cfg.output_dir, "features", f"features.csv"), dtype={"concept": "str"}
    )
    pids = df.PID.unique().compute().tolist()
    pretrain_pids, finetune_pids, test_pids = split_pids_into_pt_ft_test(
        pids, **cfg.split_ratios
    )
    df_pt = df[df["PID"].isin(pretrain_pids)]
    df_ft_and_test = df[df["PID"].isin(finetune_pids + test_pids)]

    logger.info("Tokenizing")
    vocabulary = None
    if "vocabulary" in cfg.paths:
        logger.info(f"Loading vocabulary from {cfg.paths.vocabulary}")
        vocabulary = torch.load(cfg.paths.vocabulary)
    tokenizer = EHRTokenizer(vocabulary=vocabulary, **cfg.tokenizer)
    tokenized_dir_name = cfg.get("tokenized_dir_name", "tokenized")
    check_and_clear_directory(cfg, logger, tokenized_dir_name=tokenized_dir_name)

    # TODO: config file is already copied by DirectoryPreparer but deleted again by check_and_clear_directory
    shutil.copy(config_path, join(cfg.output_dir, tokenized_dir_name, "data_cfg.yaml"))

    # Train tokenizer and tokenzie pt
    df_pt = tokenizer(df_pt)

    # Freeze vocab
    tokenizer.freeze_vocabulary()

    # Tokenize finetune and test and split them
    df_ft_and_test = tokenizer(df_ft_and_test)
    df_ft = df_ft_and_test[df_ft_and_test["PID"].isin(finetune_pids)]
    df_test = df_ft_and_test[df_ft_and_test["PID"].isin(test_pids)]

    # Convert to sequences for downstream pipeline steps
    feats_pt, pids_pt = convert_to_sequences(df_pt)
    feats_ft, pids_ft = convert_to_sequences(df_ft)
    feats_test, pids_test = convert_to_sequences(df_test)

    # Save all sequences
    torch.save(
        tokenizer.vocabulary, join(cfg.output_dir, tokenized_dir_name, "vocabulary.pt")
    )

    torch.save(
        feats_pt, join(cfg.output_dir, tokenized_dir_name, "features_pretrain.pt")
    )
    torch.save(pids_pt, join(cfg.output_dir, tokenized_dir_name, "pids_pretrain.pt"))

    torch.save(
        feats_ft, join(cfg.output_dir, tokenized_dir_name, "features_finetune.pt")
    )
    torch.save(pids_ft, join(cfg.output_dir, tokenized_dir_name, "pids_finetune.pt"))

    torch.save(feats_test, join(cfg.output_dir, tokenized_dir_name, "features_test.pt"))
    torch.save(pids_test, join(cfg.output_dir, tokenized_dir_name, "pids_test.pt"))
    logger.info("Finished tokenizing")

    if cfg.env == "azure":
        features_dir_name = cfg.paths.get("save_features_dir_name", cfg.paths.run_name)
        save_to_blobstore(
            local_path="data/",
            remote_path=join(BLOBSTORE, "features", features_dir_name),
        )
        mount_context.stop()
    logger.info("Finished")


# TODO: Move to functional.tokenize (appears to be specific to tokenize)
def check_and_clear_directory(cfg, logger, tokenized_dir_name="tokenized"):
    tokenized_dir = join(cfg.output_dir, tokenized_dir_name)
    tokenized_files = os.listdir(tokenized_dir)
    if len(tokenized_files) > 0:
        # TODO: config file is copied by DirectoryPreparer, so this warning is always raised.
        logger.warning(f"The directory {tokenized_dir} is not empty.")
        logger.warning(f"Deleting tokenized files.")
        for file in tokenized_files:
            file_path = join(tokenized_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)


def create_and_save_features(conceptloader, excluder: Excluder, cfg, logger) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    pids = []
    for i, (concept_batch, patient_batch) in enumerate(
        tqdm(conceptloader(), desc="Batch Process Data", file=TqdmToLogger(logger))
    ):
        feature_maker = FeatureCreator(
            **cfg.features
        )  # Otherwise appended to old features
        concept_batch = feature_maker(concept_batch, patient_batch)
        concept_batch.drop(
            columns=["TIMESTAMP", "ADMISSION_ID"], inplace=True, errors="ignore"
        )
        concept_batch = excluder.exclude_incorrect_events(concept_batch)
        concept_batch, pids_batch = excluder.exclude_short_sequences(concept_batch)
        concept_batch.to_csv(
            join(cfg.output_dir, "features", f"features.csv"),
            index=False,
            mode="a" if i > 0 else "w",
            header=i == 0,
        )
        pids.extend(pids_batch)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
