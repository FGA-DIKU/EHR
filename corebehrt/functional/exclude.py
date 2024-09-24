""" File for excluding errors """

import pandas as pd
from typing import Union, List, Tuple

# New stuff
import dask.dataframe as dd
from corebehrt.functional.load import load_pids


def exclude_incorrect_event_ages(
    df: pd.DataFrame, min_age: int = -1, max_age: int = 120
) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]


def exclude_event_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude events (row) with (any) NaNs"""
    return df.dropna()


def exclude_short_sequences(
    x: Union[pd.DataFrame, List[list], dict, dd.DataFrame],
    min_len: int = 3,
    background_length: int = 0,
) -> Tuple[Union[pd.DataFrame, List[list], dict], List[int]]:
    if isinstance(x, dd.DataFrame):
        return exclude_short_sequences_dd(x, min_len, background_length)
    elif isinstance(x, pd.DataFrame):
        return exclude_short_sequences_df(x, min_len, background_length)
    elif isinstance(x, list) and isinstance(x[0], list):
        return exclude_short_sequences_list(x, min_len, background_length)
    elif isinstance(x, dict):
        return exclude_short_sequences_dict(x, min_len, background_length)
    else:
        raise TypeError("Invalid type for x, only dd.DataFrame are supported.")


def min_len_condition(c: list, min_len: int, background_length: int) -> bool:
    return len(c) >= min_len + background_length

def exclude_short_sequences_dd(
    data: dd.DataFrame, min_len: int, background_length: int
) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID and concept.
    Returns a new table with only the rows that have a concept with a length greater than min_len.
    """
    return data.map_partitions(
        exclude_short_sequences_partition, min_len, background_length, meta=data
    )

def exclude_short_sequences_partition(
    df: pd.DataFrame, min_len: int, background_length: int
) -> pd.DataFrame:
    counts = df.groupby('PID').transform('count') 
    df[counts['concept']>(min_len+background_length)]
    return df


def exclude_short_sequences_df(
    df: dd.DataFrame, min_len: int, background_length: int
) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID and concept.
    Returns a new table with only the rows that have a concept with a length greater than min_len.
    """
    filtered_df = df.groupby("PID").filter(
        lambda x: min_len_condition(x["concept"], min_len, background_length)
    )
    kept_indices = filtered_df.index.tolist()
    return filtered_df, kept_indices


def exclude_short_sequences_list(
    x: List[list], min_len: int, background_length: int
) -> Tuple[list, list]:
    kept_indices, concepts = zip(
        *[
            (i, c)
            for i, c in enumerate(x)
            if min_len_condition(c, min_len, background_length)
        ]
    )
    return list(concepts), list(kept_indices)


def exclude_short_sequences_dict(
    x: dict, min_len: int, background_length: int
) -> Tuple[dict, list]:
    kept_indices = [
        i
        for i, c in enumerate(x["concept"])
        if min_len_condition(c, min_len, background_length)
    ]
    filtered_x = {k: [v[i] for i in kept_indices] for k, v in x.items()}
    return filtered_x, kept_indices


def filter_table_by_exclude_pids(
    data: dd.DataFrame, excluded_pids
) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with only the rows that do not have a PID in pids
    """
    data = data[~data["PID"].isin(excluded_pids)]
    return data
