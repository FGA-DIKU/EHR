from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd

from corebehrt.functional.features.normalize import normalize_segments_series
from corebehrt.functional.utils.time import get_abspos_from_origin_point
import uuid
import warnings


def create_abspos(concepts: pd.DataFrame, origin_point: datetime) -> pd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
        origin_point: The origin point for calculating absolute position.
    Returns:
        concepts with a new 'abspos' column
    """
    concepts["abspos"] = get_abspos_from_origin_point(concepts["time"], origin_point)
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

def _create_patient_info(concepts: pd.DataFrame) -> pd.DataFrame:
    # Get the patient info out
    dod_rows = concepts[concepts["code"] == "DOD"]
    deathdates = dict(zip(dod_rows["subject_id"], dod_rows["time"]))
    patient_info = pd.DataFrame(
        {
            "subject_id": concepts["subject_id"].unique(),
            "birthdate": concepts.drop_duplicates("subject_id")["birthdate"],
        }
    )
    patient_info["deathdate"] = patient_info["subject_id"].map(deathdates)
    bg_info = concepts[concepts["code"].str.startswith("BG_")][["subject_id", "code"]]
    if len(bg_info) == 0:
        warnings.warn("No background information found in concepts.")
        return concepts, patient_info
    bg_info[["column_name", "value"]] = bg_info["code"].str.split("//", expand=True)
    bg_info["column_name"] = bg_info["column_name"].str.replace("BG_", "")
    bg_info_pivot = bg_info.pivot_table(
        index="subject_id", columns="column_name", values="value", aggfunc="first"
    ).reset_index()
    merged_info = pd.merge(patient_info, bg_info_pivot, on="subject_id", how="left")
    return merged_info

def create_background(concepts: pd.DataFrame) -> dd.DataFrame:
    """
    Create background concepts for each patient based on the static background variables in the dataframe.
    Sets the time of the background concepts to the birthdate of the patient.
    Expects 'DOB' concept to be present in the patients_info DataFrame.
    Returns:
        table with background concepts, including 'subject_id', 'time', 'code', and 'numeric_value' columns.
        additionally, 'birthdate' column is added to the concepts DataFrame.
    """
    dob_rows = concepts[concepts["code"] == "DOB"]
    birthdates = dict(zip(dob_rows["subject_id"], dob_rows["time"]))
    concepts["birthdate"] = concepts["subject_id"].map(birthdates)

    bg_rows = concepts[concepts["time"].isna()]
    concepts.loc[bg_rows.index, "time"] = concepts.loc[bg_rows.index, "birthdate"]
    concepts.loc[bg_rows.index, "code"] = "BG_" + concepts.loc[bg_rows.index, "code"]

    adm_rows = concepts[
        concepts["code"].str.contains("ADMISSION")
        | concepts["code"].str.contains("DISCHARGE")
    ]
    concepts.loc[adm_rows.index, "code"] = "ADM_" + concepts.loc[adm_rows.index, "code"]

    # Get the patient info out
    patient_info = _create_patient_info(concepts)
    return concepts, patient_info


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
    if "index" in concepts.columns and "order" in concepts.columns:
        concepts = concepts.sort_values(
            ["subject_id", "abspos", "index", "order"]
        )  # could maybe be done more optimally, is a bit slow
        concepts = concepts.drop(columns=["index", "order"])
    else:
        concepts = concepts.sort_values(["subject_id", "abspos"])

    return concepts

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

def _assign_admission_ids(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'admission_id' to each row in concepts based on 'ADMISSION' and 'DISCHARGE' events.
    Assigns the same 'admission_id' to all events between 'ADMISSION' and 'DISCHARGE' events.
    If no 'ADMISSION' and 'DISCHARGE' events are present, assigns a new 'admission_id' to all events if the time between them is greater than 48 hours.
    """
    def _get_adm_id():
        return str(uuid.uuid4())
        
    concepts = concepts.reset_index(drop=True)
    concepts["admission_id"] = None
    concepts["admission_id"] = concepts["admission_id"].astype(object)

    # Assign admission_id to all events between 'ADMISSION' and 'DISCHARGE' events
    admission_mask = concepts["code"].str.contains("ADMISSION")
    discharge_mask = concepts["code"].str.contains("DISCHARGE")
    admission_indices = concepts[admission_mask].index.to_list()
    discharge_indices = concepts[discharge_mask].index.to_list()

    if admission_indices and discharge_indices:
        start_idx = 0
        while start_idx < len(admission_indices):
            start = admission_indices[start_idx]
            end = next((d for d in discharge_indices if d > start), None)

            if end:
                concepts.loc[start:end, "admission_id"] = _get_adm_id()
            start_idx += 1

    # Assign admission_id to concepts outside admissions
    # Concepts within 48 hours of each other are considered to be part of the same admission
    outside_segments = concepts[concepts["admission_id"].isna()].copy()
    outside_segments = outside_segments.sort_values(by=["subject_id", "time"])
    outside_segments["time_diff"] = (
        outside_segments.groupby("subject_id")["time"].diff().dt.total_seconds()
    )
    outside_segments["new_admission"] = (outside_segments["time_diff"] > 48 * 3600) | (
        outside_segments["time_diff"].isna()
    )
    outside_segments["admission_id"] = outside_segments["new_admission"].apply(
        lambda x: _get_adm_id() if x else None
    )
    outside_segments["admission_id"] = outside_segments["admission_id"].ffill()
    concepts.loc[outside_segments.index, "admission_id"] = outside_segments[
        "admission_id"
    ]

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
    max_segment = (
        df.groupby("subject_id")["segment"].max().rename("max_segment").reset_index()
    )
    # Merge and assign
    df = df.merge(max_segment, on="subject_id", how="left")
    df["segment"] = df["segment"].where(df["code"] != "DOD", df["max_segment"])
    return df.drop(columns=["max_segment"])
