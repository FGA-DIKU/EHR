import os
from os.path import join

import pandas as pd
import pyarrow as pa
import torch

from corebehrt.constants.data import (
    FEATURES_SCHEMA,
    PID_COL,
    TOKENIZED_SCHEMA,
    CONCEPT_COL,
    TIMESTAMP_COL,
)
from corebehrt.functional.features.exclude import exclude_incorrect_event_ages
from corebehrt.modules.features.features import FeatureCreator
from corebehrt.modules.features.loader import FormattedDataLoader
from corebehrt.modules.features.tokenizer import EHRTokenizer
from corebehrt.modules.features.values import ValueCreator
from corebehrt.functional.preparation.utils import aggregate_rows


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


def create_and_save_features(cfg) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    combined_patient_info = pd.DataFrame()
    for split_name in ["train", "tuning", "held_out"]:
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
            concepts = handle_aggregations(
                concepts,
                agg_type=cfg.get("features", {}).get("agg_func", None),
                agg_window=cfg.get("features", {}).get("agg_window", None),
            )
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


def handle_aggregations(
    concepts: pd.DataFrame, agg_type: str = None, agg_window: int = None
) -> pd.DataFrame:
    """
    Performs aggregation based on PID, TIMESTAMP, and CONCEPT columns if agg_type is provided.
    Keeps NaN values in the timestamps columns to preseve BG codes.
    If agg_window is provided, aggregates values within the specified time window.
    """
    if agg_type:
        if agg_window:
            # Create a new column for the time window grouping
            concepts[TIMESTAMP_COL] = pd.to_datetime(concepts[TIMESTAMP_COL])
            min_time = concepts[TIMESTAMP_COL].min()
            normalized_timestamps = (
                concepts[TIMESTAMP_COL] - min_time
            ).dt.total_seconds()
            normalized_timestamps = normalized_timestamps.fillna(-1)
            concepts["TIME_GROUP"] = (
                normalized_timestamps // (agg_window * 3600)
            ).astype(int)

            aggregated_concepts = aggregate_rows(
                concepts,
                cols=[PID_COL, "TIME_GROUP", CONCEPT_COL],
                agg_type=agg_type,
                keep_nans=[TIMESTAMP_COL],
            )
            return aggregated_concepts.drop(columns=["TIME_GROUP"])
        else:
            aggregated_concepts = aggregate_rows(
                concepts,
                cols=[PID_COL, TIMESTAMP_COL, CONCEPT_COL],
                agg_type=agg_type,
                keep_nans=[TIMESTAMP_COL],
            )
            return aggregated_concepts
    return concepts


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
