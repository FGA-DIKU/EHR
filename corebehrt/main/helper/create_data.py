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
from corebehrt.functional.preparation.utils import is_valid_regex
from corebehrt.functional.preparation.filter import filter_rows_by_regex


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


def create_and_save_features(cfg, splits, logger) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    combined_patient_info = pd.DataFrame()
    for split_name in splits:
        logger.info(f"Creating features for {split_name}")
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

        logger.info(f"Found {len(shards)} shards to process in {split_name}")

        # Initialize counters for this split
        total_concepts_loaded = 0
        total_concepts_after_agg = 0
        total_concepts_after_exclusion = 0
        total_concepts_after_incorrect = 0

        for shard_idx, shard in enumerate(shards, 1):
            logger.info(f"Processing shard {shard_idx}/{len(shards)}: {shard}")
            shard_path = f"{path_name}/{shard}"
            shard_n = shard.split(".")[0]

            concepts = FormattedDataLoader(
                shard_path,
            ).load()
            total_concepts_loaded += len(concepts)

            agg_kwargs = cfg.get("features", {}).get("agg_kwargs", {})
            if agg_kwargs:
                concepts = handle_aggregations(
                    concepts,
                    agg_type=agg_kwargs.get("agg_type", None),
                    agg_window=agg_kwargs.get("agg_window", None),
                    regex=agg_kwargs.get("regex", None),
                )
                total_concepts_after_agg += len(concepts)
            else:
                total_concepts_after_agg = total_concepts_loaded

            concepts = exclude_concepts(
                concepts,
                cfg.get("features", {}).get("exclude_regex", None),
            )
            total_concepts_after_exclusion += len(concepts)

            concepts = handle_numeric_values(concepts, cfg.get("features"))
            feature_creator = FeatureCreator()
            features, patient_info = feature_creator(concepts)
            combined_patient_info = pd.concat([combined_patient_info, patient_info])
            features = exclude_incorrect_event_ages(features)
            total_concepts_after_incorrect += len(features)

            features.to_parquet(
                f"{split_save_path}/{shard_n}.parquet",
                index=False,
                schema=pa.schema(FEATURES_SCHEMA),
            )

        # Log final statistics for this split
        logger.info(f"Total concepts loaded: {total_concepts_loaded}")
        logger.info(f"Total concepts after aggregation: {total_concepts_after_agg}")
        logger.info(f"Total concepts after exclusion: {total_concepts_after_exclusion}")
        logger.info(
            f"Total concepts after incorrect age removal: {total_concepts_after_incorrect}"
        )

    patient_info_path = f"{cfg.paths.features}/patient_info.parquet"
    combined_patient_info.to_parquet(patient_info_path, index=False)
    logger.info(
        f"Total number of patients across all splits: {len(combined_patient_info)}"
    )


def exclude_concepts(concepts, exclude_regex):
    if exclude_regex is not None:
        if not is_valid_regex(exclude_regex):
            raise ValueError(f"Invalid regex: {exclude_regex}")
        concepts = filter_rows_by_regex(concepts, col=CONCEPT_COL, regex=exclude_regex)
        concepts = concepts.copy()  # to avoid SettingWithCopyWarning
    return concepts


def handle_aggregations(
    concepts: pd.DataFrame,
    agg_type: str = None,
    agg_window: int = None,
    regex: str = ".*",
) -> pd.DataFrame:
    """
    Aggregates rows in the DataFrame based on PID, TIMESTAMP, and CONCEPT columns.
    Filters rows based on the provided regex before aggregation and concatenates excluded rows back after aggregation.
    Keeps NaN values in TIMESTAMP column to preserve background codes.
    Optionally aggregates values within a specified time window.

    Args:
        concepts: DataFrame to aggregate.
        agg_type: Aggregation type (e.g., 'first', 'sum', 'mean', etc.). If None, no aggregation is performed.
        agg_window: Time window in hours for aggregation. If None, no time window aggregation is performed.
        regex: Regular expression to filter rows based on the CONCEPT_COL before aggregation.

    Returns:
        Aggregated DataFrame with specified rows.
    """
    if agg_type is None:
        return concepts

    matching_rows = concepts[concepts[CONCEPT_COL].astype(str).str.match(regex)]
    non_matching_rows = concepts[~concepts[CONCEPT_COL].astype(str).str.match(regex)]
    nan_rows = matching_rows[matching_rows[[TIMESTAMP_COL]].isna().any(axis=1)]
    non_nan_rows = matching_rows.dropna(subset=[TIMESTAMP_COL])

    if agg_window:
        min_time = non_nan_rows[TIMESTAMP_COL].min()
        normalized_timestamps = (
            non_nan_rows[TIMESTAMP_COL] - min_time
        ).dt.total_seconds()
        normalized_timestamps = normalized_timestamps.fillna(-1)
        non_nan_rows["TIME_GROUP"] = (
            normalized_timestamps // (agg_window * 3600)
        ).astype(int)

        aggregated_df = (
            non_nan_rows.groupby([PID_COL, "TIME_GROUP", CONCEPT_COL])
            .agg(agg_type)
            .reset_index()
        )
        aggregated_df = aggregated_df.drop(columns="TIME_GROUP")
    else:
        aggregated_df = (
            non_nan_rows.groupby([PID_COL, TIMESTAMP_COL, CONCEPT_COL])
            .agg(agg_type)
            .reset_index()
        )

    # Concatenate aggregated rows with NaN rows and non-matching rows
    concatted_df = pd.concat(
        [aggregated_df, nan_rows, non_matching_rows], ignore_index=True
    )
    return concatted_df


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
        add_prefix = features_cfg.values.value_creator_kwargs.get("add_prefix", False)
        prefix_regex = features_cfg.values.value_creator_kwargs.get("prefix_regex", r"^([^/]+)/")
        return ValueCreator.bin_results(concepts, num_bins=num_bins, add_prefix=add_prefix, prefix_regex=prefix_regex)

    return concepts.drop(columns=["numeric_value"])
