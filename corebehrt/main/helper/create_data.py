from os.path import join

import dask.dataframe as dd
import torch
from dask.diagnostics import ProgressBar

from corebehrt.constants.data import FEATURES_SCHEMA, TOKENIZED_SCHEMA
from corebehrt.modules.features.excluder import Excluder

from corebehrt.modules.features.features import FeatureCreator
from corebehrt.modules.features.loader import FormattedDataLoader
from corebehrt.modules.features.tokenizer import EHRTokenizer
from corebehrt.modules.features.values import ValueCreator
import os
import pyarrow as pa

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

    for split_name in ["train", "tuning", "held_out"]:
        path_name = f"{cfg.paths.data}/{split_name}"
        if not os.path.exists(path_name):
            ValueError(f"Path {path_name} does not exist")
        
        split_save_path = f"{cfg.paths.features}/{split_name}"
        os.makedirs(split_save_path, exist_ok=True)
        for shard in os.listdir(path_name):
            shard_path = f"{path_name}/{shard}"
            shard_n = shard.split('.')[0]

            # if int(shard_n) > 5:
            #     continue

            print(shard_path)
            
            concepts = FormattedDataLoader(
                shard_path,
            ).load()

            if "values" in cfg.features:
                concepts = ValueCreator.bin_results(
                    concepts, num_bins=cfg.features.values.value_creator_kwargs.get("num_bins", 100)
                )
            features_args = {k: v for k, v in cfg.features.items() if k != "values"}
            feature_creator = FeatureCreator(**features_args)
            features = feature_creator(concepts)

            features = excluder.exclude_incorrect_events(features)
            features.to_parquet(
                f'{split_save_path}/{shard_n}.parquet', index=False, schema=pa.schema(FEATURES_SCHEMA)
            )