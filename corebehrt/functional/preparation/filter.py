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


def censor_patient(
    patient: PatientData, censor_dates: pd.Series, predict_token_id: int
) -> PatientData:
    """
    Censors a patient's data by truncating all attributes at the censor date,
    then appends a CLS token with the censoring information.

    The function shortens the concept, abspos, segments, and ages lists of a PatientData object so that only entries occurring before or at the patient's censor date are retained, then adds a CLS token at the end.

    Args:
        patient: The PatientData object to be censored.
        censor_dates: A mapping from patient IDs to their respective censor dates.
        cls_token_id: The concept ID to use for the CLS token.

    Returns:
        The censored PatientData object with truncated attributes and appended CLS token.
    """
    censor_date = censor_dates[patient.pid]
    # Find the position where censor_date fits in the sorted abspos list
    idx = bisect_right(patient.abspos, censor_date)

    # Slice everything up to idx
    patient.concepts = patient.concepts[:idx]
    patient.abspos = patient.abspos[:idx]
    patient.segments = patient.segments[:idx]
    patient.ages = patient.ages[:idx]

    patient = _append_predict_token(patient, predict_token_id, censor_date)

    return patient


def censor_patient_with_delays(
    patient: PatientData,
    censor_dates: pd.Series,
    predict_token_id: int,
    concept_id_to_delay: dict = None,
) -> PatientData:
    """
    Censors a patient's data using concept-specific delays applied to their censor date.

    For each concept in the patient's record, calculates an effective censor date by adding a delay (if specified) to the base censor date for the patient. Retains only those concepts and corresponding attributes whose timestamps are less than or equal to their effective censor dates.

    Args:
        patient: The patient data to censor.
        censor_dates: Series mapping patient IDs to their base censor dates.
        concept_id_to_delay: Optional dictionary mapping concept IDs to delay values (in hours). Concepts not present in the dictionary use a delay of 0.

    Returns:
        The censored PatientData object with only concepts and attributes occurring before or at their effective censor dates.
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

    patient = _append_predict_token(patient, predict_token_id, base_censor_date)

    return patient


def _append_predict_token(
    patient: PatientData, predict_token_id: int, censor_date: float
) -> PatientData:
    """
    Appends a predict token to the patient's data.
    """
    if patient.dob is None:
        raise ValueError(
            f"Patient {patient.pid} is missing date of birth (dob) required for age calculation"
        )
    patient.concepts.append(predict_token_id)
    patient.abspos.append(float(censor_date))
    patient.segments.append(1)
    age_in_years = float((censor_date - patient.dob) / (365.25 * 24))
    patient.ages.append(age_in_years)
    return patient


def filter_by_column_rule(df, column, include_values=None, exclude_values=None):
    """
    Filters a DataFrame by including or excluding specified values in a given column.

    Args:
        column: The column name on which to apply the filter.
        include_values: Values to retain in the column (cannot be used with exclude_values).
        exclude_values: Values to remove from the column (cannot be used with include_values).

    Returns:
        A DataFrame filtered according to the specified inclusion or exclusion rule.

    Raises:
        ValueError: If both include_values and exclude_values are provided.
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
