from typing import List

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


def censor_patient(patient: PatientData, censor_dates: float) -> PatientData:
    """
    Censors a patient's data by keeping only events that occur before or at the censor date.

    Args:
        patient: The patient data to censor
        censor_date: The cutoff date (in absolute time units) after which events should be removed

    Returns:
        The censored patient data with only events before or at the censor date
    """
    censor_date = censor_dates[patient.pid]
    keep_indices = set([i for i, a in enumerate(patient.abspos) if a <= censor_date])
    patient.concepts = [patient.concepts[i] for i in keep_indices]
    patient.abspos = [patient.abspos[i] for i in keep_indices]
    patient.segments = [patient.segments[i] for i in keep_indices]
    patient.ages = [patient.ages[i] for i in keep_indices]
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
