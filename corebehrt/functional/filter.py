from bisect import bisect_right
from typing import List

import pandas as pd

from corebehrt.classes.dataset import PatientData


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


def filter_events_by_abspos(
    events: pd.DataFrame,
    abspos_series: pd.Series,
    comparison_function: callable,
) -> pd.DataFrame:
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

    merged_df = pd.merge(events, abspos_df, on="PID", how="inner")
    filtered_df = merged_df[
        comparison_function(merged_df["abspos"], merged_df["abspos_ref"])
    ]

    return filtered_df.drop(columns=["abspos_ref"])
