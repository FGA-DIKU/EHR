from datetime import datetime
from typing import Union

import pandas as pd

from corebehrt.functional.creators import (create_abspos, create_ages,
                                           create_background, create_death,
                                           create_segments)


class FeatureCreator:
    def __init__(self,
                 ages: bool = True, # If True, create AGE
                 segment: bool = True, # If True, create SEGMENT
                 origin_point: Union[datetime, dict] = datetime(2020, 1, 26), # If given, compute ABSPOS
                 background_vars: list = ['GENDER']): # If given, add background variables
        self.ages = ages
        self.segment = segment
        self.origin_point = datetime(**origin_point) if isinstance(origin_point, dict) else origin_point
        self.background_vars = background_vars

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame) -> pd.DataFrame:
        """Create features based on given concepts and patients_info dataframes.
        concepts: table with columns: 'PID', 'TIMESTAMP', 'CONCEPT', 'ADMISSION_ID' 
        patients_info: table with columns: 'PID', and columns holding birth and death (inferred) and background variables
        Output:
            Table with PID, concept, abspos, age, segment. We constuct background and death events, insert SEP and CLS tokens.
        """
        # !Do we need those if statements?
        self._validate_input(concepts, patients_info)
        concepts.rename(columns={'CONCEPT': 'concept'}, inplace=True)
        if self.background_vars:
            concepts = create_background(concepts, patients_info, self.background_vars)
        if self.ages:
            concepts = create_ages(concepts, patients_info)
        if self.segment:
            concepts = create_segments(concepts)
        if self.origin_point:
            concepts = create_abspos(concepts, self.origin_point)
        concepts = create_death(concepts, patients_info, self.origin_point)

        return concepts

    def _validate_input(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        """Check if required columns are present in concepts and patients_info dataframes."""
        required_concept_columns = ['PID', 'TIMESTAMP', 'CONCEPT', 'ADMISSION_ID']
        required_patient_columns = ['PID'] + self.background_vars
        
        missing_concept_columns = [col for col in required_concept_columns if col not in concepts.columns]
        if missing_concept_columns:
            raise ValueError(f"Missing required columns in concepts DataFrame: {', '.join(missing_concept_columns)}")
        
        missing_patient_columns = [col for col in required_patient_columns if col not in patients_info.columns]
        if missing_patient_columns:
            raise ValueError(f"Missing required columns in patients_info DataFrame: {', '.join(missing_patient_columns)}")