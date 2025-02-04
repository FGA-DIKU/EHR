"""Random utils, should be structered later"""

from dataclasses import asdict, fields
from typing import List

import pandas as pd

from corebehrt.modules.preparation.dataset import PatientData


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


def subset_patient_data(patient: PatientData, keep_indices: List[int]) -> PatientData:
    """
    Return a new PatientData containing only the rows at `keep_indices`
    for all list-type attributes. Non-list attributes remain unchanged.
    """
    # Convert the PatientData instance to a dictionary
    data = asdict(patient)

    # For each field in the dataclass, if the value is a list, subset it.
    # Otherwise, keep it as is.
    for f in fields(PatientData):
        val = data[f.name]
        if isinstance(val, list):
            data[f.name] = [val[i] for i in keep_indices]

    # Recreate a new PatientData from the updated dictionary
    return PatientData(**data)


def select_first_event(
    df: pd.DataFrame, pid_col: str, timestamp_col: str
) -> pd.DataFrame:
    """Select the first event for each PID."""
    return df.groupby(pid_col)[timestamp_col].min().reset_index()
