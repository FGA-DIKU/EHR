import operator
from typing import List, Union

import dask.dataframe as dd
import pandas as pd

from corebehrt.data.dataset import PatientData
from corebehrt.functional.exclude import filter_table_by_pids


def exclude_short_sequences(
    patients: List[PatientData], min_len: int
) -> List[PatientData]:
    return [p for p in patients if len(p.concepts) >= min_len]


def filter_patients_by_age_at_last_event(
    data: dd.DataFrame, min_age: int, max_age: int
) -> dd.DataFrame:
    """
    Filters the data to include only patients whose age at their last event falls within the specified age range.

    Args:
        data: Dask DataFrame with 'PID' and 'age' columns.
              The 'age' column represents the age of the patient at each event.
        min_age: The minimum age (inclusive) of patients to include.
        max_age: The maximum age (inclusive) of patients to include.

    Returns:
        A Dask DataFrame containing only events for patients whose age at their last event is within the specified range.
    """
    last_events = data.groupby("PID").last().reset_index()

    patients_in_age_range = last_events[
        (last_events["age"] >= min_age) & (last_events["age"] <= max_age)
    ]
    pids_in_age_range = patients_in_age_range["PID"].compute()
    return filter_table_by_pids(data, pids_in_age_range)


def censor_data(data: dd.DataFrame, censor_dates: pd.Series) -> dd.DataFrame:
    """
    Censors the data by removing all events that occur after the censor_dates.
    args:
        data: dd.DataFrame (needs to have abspos column)
        censor_dates: pd.Series (index: PID, values: censor_dates as abspos)
    """
    return filter_events_by_abspos(data, censor_dates, operator.le)


def filter_events_by_abspos(
    data: dd.DataFrame,
    abspos_series: pd.Series,
    comparison_function: callable,
) -> dd.DataFrame:
    """
    Filters the data based on a timestamp per PID using the specified comparison operator.

    Args:
        data: DataFrame with 'PID' and 'abspos' columns.
        abspos_series: Series with index 'PID' and values as abspos.
        comparison_function: callable e.g., operator.le (<=), operator.ge (>=), operator.lt (<), operator.gt (>).
    Returns:
        The filtered DataFrame.
    """

    # Convert the Series to a DataFrame
    abspos_df = abspos_series.reset_index()
    abspos_df.columns = ["PID", "abspos_ref"]

    merged_df = inner_merge_tables(data, abspos_df, "PID")
    filtered_df = merged_df[
        comparison_function(merged_df["abspos"], merged_df["abspos_ref"])
    ]

    return filtered_df.drop(columns=["abspos_ref"])


def inner_merge_tables(
    df1: Union[pd.DataFrame, dd.DataFrame], df2: pd.DataFrame, on: str
) -> dd.DataFrame:
    """
    Merges two tables on a common column using an inner join.
    The first dataframe can be either a pandas or dask DataFrame.
    """
    if isinstance(df1, pd.DataFrame):
        return pd.merge(df1, df2, on=on, how="inner")
    else:
        return dd.merge(df1, df2, on=on, how="inner")
