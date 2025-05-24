import uuid
import warnings

import numpy as np
import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    ADMISSION_CODE,
    BIRTH_CODE,
    BIRTHDATE_COL,
    CONCEPT_COL,
    DEATH_CODE,
    DEATHDATE_COL,
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
    Assumes not overlapping ADMISSION and DISCHARGE events.
    """

    def _get_adm_id():
        return str(uuid.uuid4())

    # Work with a copy to avoid modifying the original
    result = concepts.copy()
    result["admission_id"] = None
    result["admission_id"] = result["admission_id"].astype(object)

    # Process each patient separately
    for patient_id in result[PID_COL].unique():
        patient_mask = result[PID_COL] == patient_id
        patient_data = result[patient_mask].copy()

        # Sort by timestamp
        patient_data = patient_data.sort_values(by=TIMESTAMP_COL)

        # Check if patient has explicit ADMISSION/DISCHARGE events
        has_admission = (
            patient_data["code"].str.startswith("ADM_ADMISSION", na=False)
        ).any()
        has_discharge = (
            patient_data["code"].str.startswith("ADM_DISCHARGE", na=False)
        ).any()

        if has_admission or has_discharge:
            # Handle explicit admission/discharge logic
            patient_data = _assign_explicit_admission_ids(patient_data, _get_adm_id)
        else:
            # Handle time-based admission logic (48-hour rule)
            patient_data = _assign_time_based_admission_ids(patient_data, _get_adm_id)

        # Update the result DataFrame using boolean indexing instead of index matching
        result.loc[patient_mask, "admission_id"] = patient_data["admission_id"].values

    return result


def _assign_explicit_admission_ids(
    patient_data: pd.DataFrame, get_adm_id_func
) -> pd.DataFrame:
    """
    Assign admission IDs based on explicit ADMISSION and DISCHARGE events.
    Events outside admission periods are grouped by 48-hour rule.
    """
    patient_data = patient_data.copy()
    patient_data["admission_id"] = None
    patient_data["admission_id"] = patient_data["admission_id"].astype(object)

    current_admission_id = None
    current_outside_admission_id = None
    last_outside_timestamp = None

    for idx, row in patient_data.iterrows():
        code = row.get("code", "")
        timestamp = row[TIMESTAMP_COL]

        if pd.isna(code):
            code = ""

        if code.startswith("ADM_ADMISSION"):
            # Start new admission
            current_admission_id = get_adm_id_func()
            patient_data.at[idx, "admission_id"] = current_admission_id
            # Reset outside admission tracking
            current_outside_admission_id = None
            last_outside_timestamp = None

        elif code.startswith("ADM_DISCHARGE"):
            # End current admission
            if current_admission_id is not None:
                patient_data.at[idx, "admission_id"] = current_admission_id
            else:
                # Discharge without admission - assign unique ID
                patient_data.at[idx, "admission_id"] = get_adm_id_func()
            current_admission_id = None
            # Reset outside admission tracking
            current_outside_admission_id = None
            last_outside_timestamp = None

        else:
            # Regular event
            if current_admission_id is not None:
                # Inside admission period
                patient_data.at[idx, "admission_id"] = current_admission_id
            else:
                # Outside admission period - apply 48-hour rule
                if (
                    current_outside_admission_id is None
                    or last_outside_timestamp is None
                    or (timestamp - last_outside_timestamp).total_seconds() > 48 * 3600
                ):
                    # Start new outside-admission group
                    current_outside_admission_id = get_adm_id_func()

                patient_data.at[idx, "admission_id"] = current_outside_admission_id
                last_outside_timestamp = timestamp

    return patient_data


def _assign_time_based_admission_ids(
    patient_data: pd.DataFrame, get_adm_id_func
) -> pd.DataFrame:
    """
    Assign admission IDs based on 48-hour time gaps.
    """
    patient_data = patient_data.copy()
    patient_data["admission_id"] = None
    patient_data["admission_id"] = patient_data["admission_id"].astype(object)

    # Calculate time differences
    patient_data["time_diff"] = patient_data[TIMESTAMP_COL].diff().dt.total_seconds()

    # Mark new admissions (first event or gap > 48 hours)
    patient_data["new_admission"] = (patient_data["time_diff"] > 48 * 3600) | (
        patient_data["time_diff"].isna()
    )

    # Assign admission IDs
    current_admission_id = None
    for idx, row in patient_data.iterrows():
        if row["new_admission"]:
            current_admission_id = get_adm_id_func()
        patient_data.at[idx, "admission_id"] = current_admission_id

    # Clean up temporary columns
    patient_data = patient_data.drop(["time_diff", "new_admission"], axis=1)

    return patient_data


def _assign_segments(df):
    """
    Assign segments to the concepts DataFrame based on 'admission_id'
    """
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df[SEGMENT_COL] = df.groupby(PID_COL)["admission_id"].transform(
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
