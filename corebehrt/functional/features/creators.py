import uuid
import warnings

import numpy as np
import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    ADMISSION,
    ADMISSION_CODE,
    ADMISSION_ID_COL,
    BIRTH_CODE,
    BIRTHDATE_COL,
    CONCEPT_COL,
    DEATH_CODE,
    DEATHDATE_COL,
    DISCHARGE,
    DISCHARGE_CODE,
    PID_COL,
    SEGMENT_COL,
    TIMESTAMP_COL,
)
from corebehrt.functional.features.normalize import normalize_segments_series
from corebehrt.functional.utils.time import get_hours_since_epoch


def create_abspos(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
    Returns:
        concepts with a new 'abspos' column
    """
    concepts[ABSPOS_COL] = get_hours_since_epoch(concepts[TIMESTAMP_COL])
    return concepts


def create_age_in_years(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'time' and 'birthdate' columns.
    Returns:
        pd.DataFrame: concepts with a new 'age' column
    """
    # Try to convert columns to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(concepts[TIMESTAMP_COL]):
        print(f"\nConverting {TIMESTAMP_COL} to datetime...")
        concepts[TIMESTAMP_COL] = pd.to_datetime(
            concepts[TIMESTAMP_COL], errors="coerce"
        )

    if not pd.api.types.is_datetime64_any_dtype(concepts[BIRTHDATE_COL]):
        print(f"\nConverting {BIRTHDATE_COL} to datetime...")
        concepts[BIRTHDATE_COL] = pd.to_datetime(
            concepts[BIRTHDATE_COL], errors="coerce"
        )

    # Calculate age
    concepts["age"] = (
        concepts[TIMESTAMP_COL] - concepts[BIRTHDATE_COL]
    ).dt.days // 365.25

    return concepts


def _create_patient_info(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient information DataFrame from concepts.

    Args:
        concepts: DataFrame with patient concepts

    Returns:
        DataFrame with patient information including birthdate, deathdate, and background variables
    """
    # Get unique patients
    patients = concepts[PID_COL].unique()

    # Initialize patient info - handle empty case
    patient_info = pd.DataFrame({PID_COL: patients})

    # If no patients, return empty DataFrame with proper structure
    if len(patients) == 0:
        warnings.warn("No patients found in concepts")
        patient_info[BIRTHDATE_COL] = pd.Series([], dtype="datetime64[ns]")
        patient_info[DEATHDATE_COL] = pd.Series([], dtype="datetime64[ns]")
        return patient_info

    # Fallback: extract from DOB codes
    dob_data = concepts[concepts[CONCEPT_COL] == BIRTH_CODE]
    birthdate_map = dict(zip(dob_data[PID_COL], dob_data[TIMESTAMP_COL]))
    patient_info[BIRTHDATE_COL] = patient_info[PID_COL].map(birthdate_map)

    # Extract death dates (DOD)
    dod_data = concepts[concepts[CONCEPT_COL] == DEATH_CODE]
    deathdate_map = dict(zip(dod_data[PID_COL], dod_data[TIMESTAMP_COL]))
    patient_info[DEATHDATE_COL] = patient_info[PID_COL].map(deathdate_map)

    # Extract background variables (those that start with BG_)
    bg_concepts = concepts[concepts[CONCEPT_COL].str.startswith("BG_", na=False)]

    # Process background concepts if they exist
    if not bg_concepts.empty:
        bg_info = bg_concepts[[PID_COL, CONCEPT_COL]].copy()

        # Split BG_ concepts into column_name and value, handling cases without "//"
        split_result = bg_info[CONCEPT_COL].str.split("//", expand=True)

        # Ensure we always have at least 2 columns
        if split_result.shape[1] == 1:
            # No "//" separator found, add empty value column
            split_result[1] = None

        bg_info["column_name"] = split_result[0]
        bg_info["value"] = split_result[1]

        # Remove BG_ prefix from column names
        bg_info["column_name"] = bg_info["column_name"].str.replace("BG_", "")

        # Filter out rows without proper column names or with empty column names after cleaning
        bg_info = bg_info[
            bg_info["column_name"].notna() & (bg_info["column_name"] != "")
        ]

        if not bg_info.empty:
            # Create pivot table for background variables
            bg_info_pivot = bg_info.pivot_table(
                index=PID_COL, columns="column_name", values="value", aggfunc="first"
            ).reset_index()

            # Merge with patient_info
            patient_info = pd.merge(patient_info, bg_info_pivot, on=PID_COL, how="left")

    return patient_info


def create_background(concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create background concepts for each patient based on the static background variables in the dataframe.
    Sets the time of the background concepts to the birthdate of the patient.
    Expects 'DOB' concept to be present in the patients_info DataFrame.

    Args:
        concepts: DataFrame with columns 'subject_id', 'time', 'code'

    Returns:
        tuple: (updated_concepts_df, patient_info_df)
            - updated_concepts_df: concepts with background concepts updated and birthdate column added
            - patient_info_df: patient information with birthdate, deathdate, and background variables
    """
    # Create a copy to avoid modifying the original DataFrame
    concepts = concepts.copy()

    # Extract birthdates from DOB rows
    dob_rows = concepts[concepts[CONCEPT_COL] == BIRTH_CODE]
    birthdates = dict(zip(dob_rows[PID_COL], dob_rows[TIMESTAMP_COL]))
    concepts[BIRTHDATE_COL] = concepts[PID_COL].map(birthdates)
    if concepts[BIRTHDATE_COL].isna().any():
        raise ValueError("Some patients have no DOB")

    # Use boolean masking instead of index-based selection for background rows
    bg_mask = concepts[TIMESTAMP_COL].isna()
    concepts.loc[bg_mask, TIMESTAMP_COL] = concepts.loc[bg_mask, BIRTHDATE_COL]
    concepts.loc[bg_mask, CONCEPT_COL] = "BG_" + concepts.loc[bg_mask, CONCEPT_COL]

    # Use boolean masking for admission/discharge rows
    adm_mask = concepts[CONCEPT_COL].str.contains(ADMISSION_CODE, na=False) | concepts[
        CONCEPT_COL
    ].str.contains(DISCHARGE_CODE, na=False)
    concepts.loc[adm_mask, CONCEPT_COL] = "ADM_" + concepts.loc[adm_mask, CONCEPT_COL]

    # Get the patient info
    patient_info = _create_patient_info(concepts)
    return concepts, patient_info


def assign_index_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'index' and 'order' columns to ensure correct ordering.
    - The 'index' column represents the position of each row within its partition.
    - The 'order' column can be used for additional custom ordering if needed.
    - Both columns are initialized with 0 to ensure consistent behavior across partitions.
    Parameters:
        df: pd.DataFrame with 'PID' column.
    Returns:
        df with 'index' and 'order' columns.
    """
    if "index" in df.columns and "order" in df.columns:
        df.loc[:, "index"] = df["index"].fillna(0)
        df.loc[:, "order"] = df["order"].fillna(0)
    return df


def sort_features(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Sorting all concepts by 'subject_id' and 'abspos' (and 'index' and 'order' if they exist).
    """
    if "index" in concepts.columns and "order" in concepts.columns:
        concepts = concepts.sort_values(
            [PID_COL, ABSPOS_COL, "index", "order"]
        )  # could maybe be done more optimally, is a bit slow
        concepts = concepts.drop(columns=["index", "order"])
    else:
        concepts = concepts.sort_values([PID_COL, ABSPOS_COL])

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
    concepts[SEGMENT_COL] = np.nan

    # Assign maximum segment to 'Death' concepts
    concepts = _assign_segments(concepts)
    concepts = assign_segments_to_death(concepts)
    return concepts


def _assign_admission_ids(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'admission_id' to each row in concepts based on 'ADMISSION' and 'DISCHARGE' events.
    Assigns the same 'admission_id' to all events between 'ADMISSION' and 'DISCHARGE' events.
    If no 'ADMISSION' and 'DISCHARGE' events are present, assigns a new 'admission_id' to all events
    if the time between them is greater than 48 hours.
    """

    def _get_adm_id():
        return str(uuid.uuid4())

    # Work with a copy to avoid modifying the original
    result = concepts.copy()
    result[ADMISSION_ID_COL] = None
    result[ADMISSION_ID_COL] = result[ADMISSION_ID_COL].astype(object)

    result = result.sort_values(by=[PID_COL, TIMESTAMP_COL])

    has_admission = (result[CONCEPT_COL].str.startswith(ADMISSION, na=False)).any()
    has_discharge = (result[CONCEPT_COL].str.startswith(DISCHARGE, na=False)).any()

    if has_admission and has_discharge:
        new_result = _assign_explicit_admission_ids(result, _get_adm_id)
    else:
        new_result = _assign_time_based_admission_ids(result, _get_adm_id)

    result[ADMISSION_ID_COL] = new_result[ADMISSION_ID_COL]

    return result


def _assign_time_based_admission_ids(
    patient_data: pd.DataFrame, get_adm_id_func
) -> pd.DataFrame:
    """
    Assign admission IDs based on 48-hour time gaps.
    Admission IDs are only shared between events of the same patient.
    """
    patient_data = patient_data.copy()

    if len(patient_data) == 0:
        patient_data[ADMISSION_ID_COL] = None
        patient_data[ADMISSION_ID_COL] = patient_data[ADMISSION_ID_COL].astype(object)
        return patient_data

    # Calculate time differences using vectorized operations
    time_diff = patient_data[TIMESTAMP_COL].diff().dt.total_seconds()

    # Mark new admissions (first event or gap > 48 hours)
    new_admission = (time_diff > 48 * 3600) | time_diff.isna()

    # Also mark new admission when patient ID changes
    new_admission = new_admission | (
        patient_data[PID_COL] != patient_data[PID_COL].shift()
    )

    # Create admission groups using cumsum
    admission_groups = new_admission.cumsum()

    # Generate unique admission IDs for each group
    unique_groups = admission_groups.unique()
    group_to_id = {group: get_adm_id_func() for group in unique_groups}
    patient_data[ADMISSION_ID_COL] = admission_groups.map(group_to_id).astype(object)

    return patient_data


def _assign_explicit_admission_ids(
    patient_data: pd.DataFrame, get_adm_id_func
) -> pd.DataFrame:
    """
    Assign admission IDs based on explicit ADMISSION and DISCHARGE events.
    Events outside admission periods are grouped by 48-hour rule.
    Admission IDs are only shared between events of the same patient.
    """
    patient_data = patient_data.copy()

    if len(patient_data) == 0:
        patient_data[ADMISSION_ID_COL] = None
        patient_data[ADMISSION_ID_COL] = patient_data[ADMISSION_ID_COL].astype(object)
        return patient_data

    # Pre-process codes and timestamps to avoid repeated lookups
    codes = patient_data[CONCEPT_COL].fillna("").values
    timestamps = patient_data[TIMESTAMP_COL].values
    pids = patient_data[PID_COL].values

    # Initialize result array
    admission_ids = [None] * len(patient_data)

    # Track admission state per patient
    patient_states = {}  # pid -> (current_admission_id, current_outside_id, last_timestamp)

    # Process events using direct array iteration instead of iterrows
    for i, (code, timestamp, pid) in enumerate(zip(codes, timestamps, pids)):
        # Initialize patient state if not exists
        if pid not in patient_states:
            patient_states[pid] = (None, None, None)

        current_admission_id, current_outside_id, last_timestamp = patient_states[pid]

        if code.startswith(ADMISSION):
            # Start new admission
            current_admission_id = get_adm_id_func()
            admission_ids[i] = current_admission_id
            # Reset outside admission tracking
            current_outside_id = None
            last_timestamp = None

        elif code.startswith(DISCHARGE):
            # End current admission
            if current_admission_id is not None:
                admission_ids[i] = current_admission_id
            else:
                # Discharge without admission - assign unique ID
                admission_ids[i] = get_adm_id_func()
            current_admission_id = None
            # Reset outside admission tracking
            current_outside_id = None
            last_timestamp = None

        else:
            # Regular event
            if current_admission_id is not None:
                # Inside admission period
                admission_ids[i] = current_admission_id
            else:
                # Outside admission period - apply 48-hour rule
                if (
                    current_outside_id is None
                    or last_timestamp is None
                    or pd.isna(timestamp)
                    or pd.isna(last_timestamp)
                    or (
                        pd.Timestamp(timestamp) - pd.Timestamp(last_timestamp)
                    ).total_seconds()
                    > 48 * 3600
                ):
                    # Start new outside-admission group
                    current_outside_id = get_adm_id_func()

                admission_ids[i] = current_outside_id
                last_timestamp = timestamp

        # Update patient state
        patient_states[pid] = (current_admission_id, current_outside_id, last_timestamp)

    # Assign results using vectorized assignment
    patient_data[ADMISSION_ID_COL] = pd.Series(
        admission_ids, index=patient_data.index, dtype=object
    )

    return patient_data


def _assign_segments(df):
    """
    Assign segments to the concepts DataFrame based on 'admission_id'
    """
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df[SEGMENT_COL] = df.groupby(PID_COL)[ADMISSION_ID_COL].transform(
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
        df.groupby(PID_COL)[SEGMENT_COL].max().rename("max_segment").reset_index()
    )
    # Merge and assign
    df = df.merge(max_segment, on=PID_COL, how="left")
    df[SEGMENT_COL] = df[SEGMENT_COL].where(
        df[CONCEPT_COL] != DEATH_CODE, df["max_segment"]
    )
    return df.drop(columns=["max_segment"])
