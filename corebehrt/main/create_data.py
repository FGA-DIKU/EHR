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

import dask.dataframe as dd
import torch
from dask.diagnostics import ProgressBar

from corebehrt.classes.excluder import Excluder
from corebehrt.classes.features import FeatureCreator
from corebehrt.classes.tokenizer import EHRTokenizer
from corebehrt.common.azure import AzurePathContext, save_to_blobstore
from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.functional.split import split_pids_into_pt_ft_test
from corebehrt.classes.loader import FormattedDataLoader

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

    if cfg.loader.get("features_dir", None) is None:
        logger.info("Create and process features")
        create_and_save_features(
            Excluder(**cfg.excluder),  # Excluder is the new Handler and old Excluder
            cfg,
        )
        logger.info("Finished feature creation and processing")
        features_dir = join(cfg.output_dir, cfg.paths.save_features_dir_name)
    else:
        features_dir = cfg.loader.features_dir

    logger.info(f"Load features from {features_dir}")
    df = dd.read_csv(join(features_dir, "*.part"), dtype={"concept": "str"})
    pids = df.PID.unique().compute().tolist()
    logger.info("Split into pretrain and finetune.")
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
    logger.info("Save tokenized features")
    df_pt.to_csv(
        join(cfg.output_dir, tokenized_dir_name, "features_pretrain", "*.csv"),
        index=False,
    )
    df_ft.to_csv(
        join(cfg.output_dir, tokenized_dir_name, "features_finetune", "*.csv"),
        index=False,
    )
    df_test.to_csv(
        join(cfg.output_dir, tokenized_dir_name, "features_test", "*.csv"), index=False
    )
    torch.save(
        df_pt.compute()["PID"].unique().tolist(),
        join(cfg.output_dir, tokenized_dir_name, "pids_pretrain.pt"),
    )
    torch.save(
        df_ft.compute()["PID"].unique().tolist(),
        join(cfg.output_dir, tokenized_dir_name, "pids_finetune.pt"),
    )
    torch.save(
        df_test.compute()["PID"].unique().tolist(),
        join(cfg.output_dir, tokenized_dir_name, "pids_test.pt"),
    )
    torch.save(
        tokenizer.vocabulary, join(cfg.output_dir, tokenized_dir_name, "vocabulary.pt")
    )

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


def create_and_save_features(excluder: Excluder, cfg) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    save_path = join(cfg.output_dir, cfg.paths.save_features_dir_name)
    concepts, patients_info = FormattedDataLoader(
        cfg.loader.data_dir, cfg.loader.concept_types
    ).load()

    feature_creator = FeatureCreator(**cfg.features)
    features = feature_creator(patients_info, concepts)
    # !TODO: Excluder should be adapted to dask
    # features = excluder.exclude_incorrect_events(features)
    # features = excluder.exclude_short_sequences(features)

    result = features.groupby("PID").apply(
        lambda x: x.sort_values("abspos"), meta=features
    )
    with ProgressBar():
        result.to_csv(save_path, index=False)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
