from bisect import bisect_right
from typing import List

import pandas as pd

from corebehrt.modules.preparation.dataset import PatientData
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL


def filter_table_by_pids(df: pd.DataFrame, pids: List[str]) -> pd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with only the rows that have a PID in pids
    """
    return df[df[PID_COL].isin(pids)]


def remove_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assumes that the table has a column named TIMESTAMP.
    Returns a new table with only the rows that have a TIMESTAMP.
    """
    return df[df[TIMESTAMP_COL].notna()]


def select_first_event(
    df: pd.DataFrame, pid_col: str, timestamp_col: str
) -> pd.DataFrame:
    """Select the first event for each PID."""
    return df.groupby(pid_col)[timestamp_col].min().reset_index()


def exclude_short_sequences(
    patients: List[PatientData], min_len: int
) -> List[PatientData]:
    return [p for p in patients if len(p.concepts) >= min_len]


def censor_patient(patient: PatientData, censor_dates: float) -> PatientData:
    censor_date = censor_dates[patient.pid]
    # Find the position where censor_date fits in the sorted abspos list
    idx = bisect_right(patient.abspos, censor_date)

    # Slice everything up to idx
    patient.concepts = patient.concepts[:idx]
    patient.abspos = patient.abspos[:idx]
    patient.segments = patient.segments[:idx]
    patient.ages = patient.ages[:idx]

    return patient


def filter_by_column_rule(df, column, include_values=None, exclude_values=None):
    """
    Filter a DataFrame based on inclusion or exclusion of values in a column.

    Args:
        df: DataFrame to filter.
        column: Column name to apply the filter on.
        include_values: List of values to include. Mutually exclusive with exclude_values.
        exclude_values: List of values to exclude. Mutually exclusive with include_values.

    Returns:
        Filtered DataFrame.
    """
    if include_values is not None and exclude_values is not None:
        raise ValueError(
            f"Cannot include and exclude values for column '{column}' simultaneously."
        )

    if include_values is not None:
        df = df[df[column].isin(include_values)]
    elif exclude_values is not None:
        df = df[~df[column].isin(exclude_values)]
    return df


def filter_rows_by_regex(df, col, regex):
    """
    Filter rows in a DataFrame based on a regex pattern applied to a specific column.
    All rows containing a match to the regex pattern will be excluded.

    Args:
        df: DataFrame to filter.
        col: Column name to apply the regex filter.
        regex: Regex pattern to filter rows by.

    Returns:
        Filtered DataFrame.
    """
    mask = df[col].astype(str).str.contains(regex, case=False, na=False, regex=True)
    return df.loc[~mask]
