from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd

from corebehrt.functional.features.normalize import normalize_segments_series
from corebehrt.functional.utils.time import get_abspos_from_origin_point
from pandas import NaT
import uuid

def create_abspos(concepts: pd.DataFrame, origin_point: datetime) -> pd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
        origin_point: The origin point for calculating absolute position.
    Returns:
        concepts with a new 'abspos' column
    """
    concepts["abspos"] = get_abspos_from_origin_point(
        concepts["time"], origin_point
    )
    return concepts


def create_age_in_years(concepts: pd.DataFrame) -> dd.DataFrame:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'time' and 'birthdate' columns.
    Returns:
        pd.DataFrame: concepts with a new 'age' column
    """
    concepts["age"] = (concepts["time"] - concepts["birthdate"]).dt.days // 365.25
    return concepts


def create_background(
    concepts: pd.DataFrame
) -> dd.DataFrame:
    """
    Create background concepts for each patient based on the static background variables in the dataframe.
    Sets the time of the background concepts to the birthdate of the patient.
    Expects 'DOB' concept to be present in the patients_info DataFrame.
    Returns:
        table with background concepts, including 'subject_id', 'time', 'code', and 'numeric_value' columns.
        additionally, 'birthdate' column is added to the concepts DataFrame.
    """
    dob_rows = concepts[concepts['code'] == 'DOB']
    birthdates = dict(zip(dob_rows['subject_id'], dob_rows['time']))
    concepts['birthdate'] = concepts['subject_id'].map(birthdates)

    bg_rows = concepts[concepts['time'].isna()]
    concepts.loc[bg_rows.index, 'time'] = concepts.loc[bg_rows.index, 'birthdate']
    concepts.loc[bg_rows.index, 'code'] = 'BG_' + concepts.loc[bg_rows.index, 'code']

    adm_rows = concepts[concepts['code'].str.contains('ADMISSION') | concepts['code'].str.contains('DISCHARGE')]
    concepts.loc[adm_rows.index, 'code'] = 'ADM_' + concepts.loc[adm_rows.index, 'code']

    return concepts


def assign_index_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'index' and 'order' columns to ensure correct ordering.
    - The 'index' column represents the position of each row within its partition.
    - The 'order' column can be used for additional custom ordering if needed.
    - Both columns are initialized with 0 to ensure consistent behavior across partitions.
    Parameters:
        df: dd.DataFrame with 'PID' column.
    Returns:
        df with 'index' and 'order' columns.
    """
    if "index" in df.columns and "order" in df.columns:
        df["index"] = df["index"].fillna(0)
        df["order"] = df["order"].fillna(0)
    return df


def sort_features(concepts: pd.DataFrame) -> dd.DataFrame:
    """
    Sorting all concepts by 'subject_id' and 'abspos' (and 'index' and 'order' if they exist).
    """
    concepts = _sort_partitions(concepts)
    return concepts


def _sort_partitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame by 'PID' and 'abspos' to ensure correct ordering.
    Added "index" and "order" columns to ensure correct ordering if they exist.
    """
    if "index" in df.columns and "order" in df.columns:
        df = df.sort_values(
            ["subject_id", "abspos", "index", "order"]
        )  # could maybe be done more optimally, is a bit slow
        df = df.drop(columns=["index", "order"])
    else:
        df = df.sort_values(["PID", "abspos"])

    return df

def create_segments(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign segments to the concepts DataFrame based on 'ADMISSION_ID', ensuring that
    events are ordered correctly within each 'PID'.
    Parameters:
        concepts: concepts with 'PID', 'ADMISSION_ID', and 'abspos' columns.
    Returns:
        concepts with a new 'segment' column
    """
    concepts = _assign_admission_ids(concepts)
    concepts["segment"] = np.nan

    # Assign maximum segment to 'Death' concepts
    concepts = _assign_segments(concepts)
    concepts = assign_segments_to_death(concepts)
    return concepts

@staticmethod
def _get_adm_id():
    return str(uuid.uuid4())

def _assign_admission_ids(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'admission_id' to each row in concepts based on 'ADMISSION' and 'DISCHARGE' events.
    Assigns the same 'admission_id' to all events between 'ADMISSION' and 'DISCHARGE' events.
    If no 'ADMISSION' and 'DISCHARGE' events are present, assigns a new 'admission_id' to all events if the time between them is greater than 48 hours.
    """
    concepts['admission_id'] = None
    concepts['admission_id'] = concepts['admission_id'].astype(object)

    # Assign admission_id to all events between 'ADMISSION' and 'DISCHARGE' events
    admission_mask = concepts['code'].str.contains('ADMISSION')
    discharge_mask = concepts['code'].str.contains('DISCHARGE')
    admission_indices = concepts[admission_mask].index.to_list()
    discharge_indices = concepts[discharge_mask].index.to_list()

    if admission_indices and discharge_indices:
        start_idx = 0
        while start_idx < len(admission_indices):
            start = admission_indices[start_idx]
            end = next((d for d in discharge_indices if d > start), None)

            if end:
                concepts.loc[start:end, 'admission_id'] = _get_adm_id()
            start_idx += 1  

    # Assign admission_id to concepts outside admissions
    # Concepts within 48 hours of each other are considered to be part of the same admission
    outside_segments = concepts[concepts['admission_id'].isna()]
    admission_id = _get_adm_id()

    for subject_id, group in outside_segments.groupby('subject_id'):
        current_time = None
        for index, row in group.iterrows():
            if current_time is None or (row['time'] - current_time).total_seconds() > 48 * 3600:
                admission_id = _get_adm_id()
                current_time = row['time']
            concepts.loc[index, 'admission_id'] = admission_id

    return concepts

def _assign_segments(df):
    """
    Assign segments to the concepts DataFrame based on 'admission_id'
    """
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df["segment"] = df.groupby("subject_id")["admission_id"].transform(
        normalize_segments_series
    )
    return df


def assign_segments_to_death(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the maximum segment to 'DOD' concepts within each 'subject_id'.
    Parameters:
        df with 'subject_id', 'code', and 'segment' columns.
    Returns:
        df with 'DOD' concepts assigned to the maximum segment.
    """
    # Compute the maximum segment per 'PID'
    max_segment = df.groupby("subject_id")["segment"].max().rename("max_segment").reset_index()
    # Merge and assign
    df = df.merge(max_segment, on="subject_id", how="left")
    df["segment"] = df["segment"].where(df["code"] != "DOD", df["max_segment"])
    return df.drop(columns=["max_segment"])
