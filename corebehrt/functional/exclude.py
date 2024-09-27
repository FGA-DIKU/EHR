import dask.dataframe as dd
import pandas as pd

from corebehrt.functional.utils import filter_table_by_pids, get_gender_token


def exclude_incorrect_event_ages(
    df: dd.DataFrame, min_age: int = -1, max_age: int = 120
) -> dd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]


def exclude_event_nans(df: dd.DataFrame) -> dd.DataFrame:
    """Exclude events (row) with (any) NaNs"""
    return df.dropna()


def exclude_short_sequences(
    df: dd.DataFrame,
    min_len: int = 3,
    background_length: int = 0,
) -> dd.DataFrame:
    min_len = min_len + background_length
    # we can materialize the groupby object to a dataframe
    counts_df = df.groupby("PID").size().compute().reset_index(name="count")
    valid_pids = counts_df[counts_df["count"] >= min_len]["PID"]
    return filter_table_by_pids(df, valid_pids)


def exclude_short_sequences_df(
    df: pd.DataFrame, min_len: int = 3, background_length: int = 0
) -> pd.DataFrame:
    min_len = min_len + background_length
    counts_df = df.groupby("PID").size().reset_index(name="count")
    valid_pids = counts_df[counts_df["count"] >= min_len]["PID"]
    return df[df["PID"].isin(valid_pids)]


def exclude_pids_from_data(data: dd.DataFrame, pids_to_exclude: list) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with only the rows that do not have a PID in pids
    """
    return data[~data["PID"].isin(set(pids_to_exclude))]


def filter_patients_by_gender(
    data: dd.DataFrame, vocab: dict, gender: str = None
) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID and concept.
    Returns a new table with only the rows that have a concept with
    """
    if gender is None:
        return data

    gender_token = get_gender_token(gender, vocab)
    patients = data[data.concept == gender_token].PID.unique()
    return filter_table_by_pids(data, patients)
