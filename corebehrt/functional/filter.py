from bisect import bisect_right
from typing import List

import pandas as pd

from corebehrt.modules.dataset import PatientData


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
