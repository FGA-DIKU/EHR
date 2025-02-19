from typing import List

import pandas as pd
from tqdm import tqdm

from corebehrt.modules.preparation.dataset import PatientData
from corebehrt.constants.data import ABSPOS_COL, CONCEPT_COL, PID_COL, SEGMENT_COL, AGE_COL

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

    grouped = df.groupby(PID_COL, sort=False)
    loop = tqdm(
        grouped, total=len(grouped), desc="Converting to patient list", mininterval=1
    )
    for pid, group in loop:
        # Convert each column to a Python list
        concepts_list = group[CONCEPT_COL].tolist()
        abspos_list = group[ABSPOS_COL].tolist()
        segments_list = group[SEGMENT_COL].tolist()
        ages_list = group[AGE_COL].tolist()

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
