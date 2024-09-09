""" Random utils, should be structered later """

import pandas as pd
from datetime import datetime
from typing import Union, List, Tuple


def normalize_segments(x: Union[pd.Series, pd.DataFrame, list, dict]):
    if isinstance(x, pd.Series):
        return normalize_segments_series(x)
    elif isinstance(x, pd.DataFrame):
        return normalize_segments_df(x)
    elif isinstance(x, list):
        return normalize_segments_list(x)
    elif isinstance(x, dict):
        return normalize_segments_dict(x)
    else:
        raise TypeError(
            "Invalid type for x, only pd.DataFrame, list, and dict are supported."
        )


def normalize_segments_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("PID")["segment"].transform(
        lambda x: normalize_segments_series(x)
    )


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


def get_abspos_from_origin_point(
    timestamps: Union[pd.Series, List[datetime]], origin_point: datetime
) -> Union[pd.Series, List[float]]:
    """Get the absolute position in hours from the origin point"""
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

def filter_table_by_pids(df: pd.DataFrame, pids: List[str])->pd.DataFrame:
    """
    Assumes that the table has a column named PID. 
    Returns a new table with only the rows that have a PID in pids
    """
    return df[df.PID.isin(pids)]

def remove_missing_timestamps(df: pd.DataFrame )->pd.DataFrame:
    """
    Assumes that the table has a column named TIMESTAMP.
    Returns a new table with only the rows that have a TIMESTAMP.
    """
    return df[df.TIMESTAMP.notna()]
