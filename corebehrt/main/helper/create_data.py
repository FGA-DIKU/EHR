import os
from os.path import join

import pandas as pd
import pyarrow as pa
import torch

from corebehrt.constants.data import FEATURES_SCHEMA, PID_COL, TOKENIZED_SCHEMA
from corebehrt.functional.features.exclude import exclude_incorrect_event_ages
from corebehrt.modules.features.features import FeatureCreator
from corebehrt.modules.features.loader import FormattedDataLoader
from corebehrt.modules.features.tokenizer import EHRTokenizer
from corebehrt.modules.features.values import ValueCreator


def load_tokenize_and_save(
    features_path: str,
    tokenizer: EHRTokenizer,
    tokenized_path: str,
    split: str,
):
    """
    Load df for split, tokenize and write to tokenized_path.
    """
    pids = []
    os.makedirs(join(tokenized_path, f"features_{split}"), exist_ok=True)
    for shard in os.listdir(join(features_path, split)):
        shard_path = join(features_path, split, shard)
        shard_n = shard.split(".")[0]
        df = pd.read_parquet(shard_path).set_index(PID_COL)

        df = tokenizer(df).reset_index()
        df.to_parquet(
            join(tokenized_path, f"features_{split}", f"{shard_n}.parquet"),
            index=False,
            schema=pa.schema(TOKENIZED_SCHEMA),
        )
        pids.extend(df[PID_COL].unique().tolist())
    torch.save(set(pids), join(tokenized_path, f"pids_{split}.pt"))  # save pids as ints


def create_and_save_features(cfg, splits=["train", "tuning", "held_out"]) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    combined_patient_info = pd.DataFrame()
    for split_name in splits:
        path_name = f"{cfg.paths.data}/{split_name}"
        if not os.path.exists(path_name):
            if split_name == "held_out":
                continue
            raise ValueError(f"Path {path_name} does not exist")

        split_save_path = f"{cfg.paths.features}/{split_name}"
        os.makedirs(split_save_path, exist_ok=True)
        shards = [
            shard for shard in os.listdir(path_name) if not shard.startswith(".")
        ]  # MEDS on azure makes hidden files
        for shard in shards:
            shard_path = f"{path_name}/{shard}"
            shard_n = shard.split(".")[0]

            concepts = FormattedDataLoader(
                shard_path,
            ).load()
            concepts = handle_numeric_values(concepts, cfg.get("features"))
            exclude_regex = cfg.get("features", {}).get("exclude_regex", None)
            feature_creator = FeatureCreator(exclude_regex=exclude_regex)
            features, patient_info = feature_creator(concepts)
            combined_patient_info = pd.concat([combined_patient_info, patient_info])
            features = exclude_incorrect_event_ages(features)
            features.to_parquet(
                f"{split_save_path}/{shard_n}.parquet",
                index=False,
                schema=pa.schema(FEATURES_SCHEMA),
            )
    patient_info_path = f"{cfg.paths.features}/patient_info.parquet"
    combined_patient_info.to_parquet(patient_info_path, index=False)


def handle_numeric_values(
    concepts: pd.DataFrame, features_cfg: dict = None
) -> pd.DataFrame:
    """
    Process numeric values in concepts DataFrame based on configuration.
    Either bins the values or drops the numeric_value column.

    Parameters:
        concepts: DataFrame containing concepts data
        features_cfg: Configuration object containing features settings
    """
    if "numeric_value" not in concepts.columns:
        return concepts

    if features_cfg and "values" in features_cfg:
        num_bins = features_cfg.values.value_creator_kwargs.get("num_bins", 100)
        return ValueCreator.bin_results(concepts, num_bins=num_bins)

    return concepts.drop(columns=["numeric_value"])
