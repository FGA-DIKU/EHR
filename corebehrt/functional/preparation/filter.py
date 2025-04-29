"""Functions for filtering and censoring patient data during preparation.

This module provides utilities for:
- Patient data filtering (by PID, timestamps, sequence length)
- Censoring patient data (with and without concept-specific delays)
- DataFrame filtering (by column values, regex patterns)
"""

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


def censor_patient_with_delays(
    patient: PatientData, censor_dates: pd.Series, concept_id_to_delay: dict = None
) -> PatientData:
    """
    Censor patient data with pattern-based concept-specific delays.
    Optimized for cases with few distinct delay groups.

    Args:
        patient: Patient data to censor
        censor_dates: Series mapping patient IDs to base censor dates
        concept_id_to_delay: Dictionary mapping concept IDs to their delays in hours
    """
    base_censor_date = censor_dates[patient.pid]

    # Initialize keep mask
    keep_mask = [False] * len(patient.concepts)

    # Process each concept with its appropriate delay
    for i, (concept, abspos) in enumerate(zip(patient.concepts, patient.abspos)):
        # Get delay for this concept (0 for unmapped concepts)
        delay = concept_id_to_delay.get(concept, 0)

        # Calculate effective censor date for this concept
        effective_censor_date = base_censor_date + delay

        # Keep this concept if it's before or at the effective censor date
        if abspos <= effective_censor_date:
            keep_mask[i] = True

    # Apply the mask to all patient attributes
    patient.concepts = [c for i, c in enumerate(patient.concepts) if keep_mask[i]]
    patient.abspos = [a for i, a in enumerate(patient.abspos) if keep_mask[i]]
    patient.segments = [s for i, s in enumerate(patient.segments) if keep_mask[i]]
    patient.ages = [a for i, a in enumerate(patient.ages) if keep_mask[i]]

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
