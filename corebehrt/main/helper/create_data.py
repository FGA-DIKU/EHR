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

    with ProgressBar(dt=1):
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
