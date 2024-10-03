""" Random utils, should be structered later """

import logging
from datetime import datetime
from typing import Callable, List, Tuple, Union, Set

# New stuff
import dask.dataframe as dd
import pandas as pd

logger = logging.getLogger(__name__)
import random


def normalize_segments(x: Union[pd.Series, pd.DataFrame, list, dict]):
    if isinstance(x, pd.Series):
        return normalize_segments_series(x)
    elif isinstance(x, dd.DataFrame):
        return normalize_segments_dask(x)
    else:
        raise TypeError(
            "Invalid type for x, only pd.DataFrame, list, and dict are supported."
        )


def normalize_segments_dask(df: dd.DataFrame) -> dd.DataFrame:
    normalized_df = df.map_partitions(_normalize_group, meta=df)
    return normalized_df


def _normalize_group(partition):
    partition["segment"] = partition.groupby("PID")["segment"].transform(
        normalize_segments_series
    )
    return partition


def normalize_segments_series(series: pd.Series) -> pd.Series:
    # Convert to string to ensure consistent types and avoid warnings
    series = series.astype(str)
    return series.factorize(use_na_sentinel=False)[0]


def normalize_segments_list(segments: list) -> list:
    segment_set = sorted(set(segments))
    correct_segments = list(range(len(segment_set)))
    converter = {k: v for (k, v) in zip(segment_set, correct_segments)}

    return [converter[segment] for segment in segments]


def normalize_segments_dict(features: dict) -> dict:
    for idx, segments in enumerate(features["segment"]):
        features["segment"][idx] = normalize_segments_list(segments)
    return features


def get_background_length(features: dict, vocabulary) -> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = set([v for k, v in vocabulary.items() if k.startswith("BG_")])
    example_concepts = features["concept"][
        0
    ]  # Assume that all patients have the same background length
    background_length = len(set(example_concepts) & background_tokens)

    return background_length + 2  # +2 for [CLS] and [SEP] tokens


def get_background_length_dd(features: dd.DataFrame, vocabulary) -> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = set([v for k, v in vocabulary.items() if k.startswith("BG_")])
    first_pid_value = features["PID"].compute().iloc[0]
    first_pid = features[features["PID"] == first_pid_value]
    all_concepts_first_pid = first_pid["concept"].compute().tolist()
    background_length = len(set(all_concepts_first_pid) & background_tokens)

    return background_length + 2  # +2 for [CLS] and [SEP] tokens


def get_abspos_from_origin_point(
    timestamps: Union[pd.Series, List[datetime]], origin_point: datetime
) -> Union[pd.Series, List[float]]:
    """Get the absolute position in hours from the origin point"""
    if isinstance(timestamps, dd.Series):  # len can cause issues with dask
        return (timestamps - origin_point).dt.total_seconds() / 60 / 60
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([])
        return (timestamps - origin_point).dt.total_seconds() / 60 / 60
    elif isinstance(timestamps, list):
        return [
            (timestamp - origin_point).total_seconds() / 60 / 60
            for timestamp in timestamps
        ]
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series and list are supported."
        )


def get_time_difference(now: pd.Series, then: pd.Series) -> pd.Series:
    """Get the time difference in hours"""
    if len(now) == 0:
        return pd.Series([])
    return (now - then).dt.days / 365.25


def convert_df_to_feature_dict(concepts: pd.DataFrame) -> Tuple[dict, list]:
    return (
        concepts.groupby("PID").agg(list).to_dict("list"),
        concepts["PID"].sort_values().unique().tolist(),
    )


def filter_table_by_pids(df: pd.DataFrame, pids: List[str]) -> pd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with only the rows that have a PID in pids
    """
    return df[df.PID.isin(set(pids))]


def remove_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assumes that the table has a column named TIMESTAMP.
    Returns a new table with only the rows that have a TIMESTAMP.
    """
    return df[df.TIMESTAMP.notna()]


def get_first_event_by_pid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assumes that the table has a column named PID and abspos.
    Get the first event for each PID in the table.
    """
    return df.groupby("PID")["abspos"].min()


def select_random_subset(data: dd.DataFrame, n: int) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with a random subset of n PIDs.
    """
    if n >= len(data):
        return data
    pids = data["PID"].unique().compute().tolist()
    random.seed(42)
    random.shuffle(pids)
    pids = pids[:n]
    return filter_table_by_pids(data, pids)


def truncate_patient(
    patient: pd.DataFrame, background_length: int, max_len: int, sep_token: str
) -> pd.DataFrame:
    """
    Assumes patient is pd.DataFrame.
    Truncate patient to max_len, keeping background.
    """
    if len(patient["concept"]) <= max_len:
        return patient

    truncation_length = max_len - background_length
    if patient["concept"].iloc[-truncation_length] == sep_token:
        truncation_length -= 1

    truncated_patient = pd.concat(
        [patient.iloc[:background_length], patient.iloc[-truncation_length:]],
        ignore_index=True,
    )

    return truncated_patient


def truncate_data(
    data: dd.DataFrame,
    max_len: int,
    vocabulary: dict,
    truncate_function: Callable = truncate_patient,
) -> dd.DataFrame:
    """
    Assumes table has a column named PID.
    Truncate the data to max_len. CLS and SEP tokens are kept if present.
    Uses truncate_patient as default truncate function.
    """
    background_length = get_background_length_dd(data, vocabulary)

    truncated_data = (
        data.groupby("PID")[list(data.columns)]
        .apply(
            lambda x: truncate_function(
                x, background_length, max_len, vocabulary.get("[SEP]")
            ),
            meta={col: dtype for col, dtype in data.dtypes.items()},
        )
        .reset_index(drop=True)
    )

    return truncated_data


def get_gender_token(gender: str, vocabulary: dict) -> int:
    """
    Retrieves the token for the specified gender from the vocabulary.
    Assumes that the gender starts with BG_GENDER_.
    """
    gender_key = f"BG_GENDER_{gender}"
    return vocabulary[gender_key]


def get_pids(data: dd.DataFrame) -> List[str]:
    """Get unique pids from data."""
    return data["PID"].unique().compute().tolist()


def check_concepts_columns(df: dd.DataFrame) -> None:
    """Check if required columns are present in concepts."""
    required_columns = {"PID", "CONCEPT", "TIMESTAMP", "ADMISSION_ID"}
    check_required_columns(df, required_columns, "concepts")


def check_patients_info_columns(
    df: dd.DataFrame, background_vars: Set[str] = set()
) -> None:
    """Check if required columns are present in patients_info."""
    required_columns = {"PID", "BIRTHDATE", "DEATHDATE"}.union(set(background_vars))
    check_required_columns(df, required_columns, "patients_info")


def check_required_columns(
    df: dd.DataFrame, required_columns: Set[str], type_: str
) -> None:
    if not required_columns.issubset(set(df.columns)):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in {type_}: {missing_columns}")
