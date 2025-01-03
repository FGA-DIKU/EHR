from typing import List

import pandas as pd

from corebehrt.classes.dataset import PatientData


def dataframe_to_patient_list(df: pd.DataFrame) -> List[PatientData]:
    """Convert a DataFrame containing patient data into a list of PatientData objects.

    Args:
        df (pd.DataFrame): DataFrame containing patient data with columns:
            - PID: Patient ID
            - concept: Medical concepts/tokens
            - abspos: Absolute positions/timestamps
            - segment: Segment IDs
            - age: Patient ages

    Returns:
        List[PatientData]: List of PatientData objects, where each object contains:
            - pid (str): Patient ID
            - concepts (List[int]): List of medical concept tokens
            - abspos (List[float]): List of absolute positions/timestamps
            - segments (List[int]): List of segment IDs
            - ages (List[float]): List of patient ages
    """
    patients_data = []

    grouped = df.groupby("PID", sort=False)
    for pid, group in grouped:
        # Convert each column to a Python list
        concepts_list = group["concept"].tolist()
        abspos_list = group["abspos"].tolist()
        segments_list = group["segment"].tolist()
        ages_list = group["age"].tolist()

        # Create a PatientData instance
        patient = PatientData(
            pid=pid,
            concepts=concepts_list,
            abspos=abspos_list,
            segments=segments_list,
            ages=ages_list,
        )

        patients_data.append(patient)

    return patients_data
