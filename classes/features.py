import pandas as pd
from datetime import datetime
from typing import Union
from functional.creators import create_ages, create_abspos, create_segments, create_background

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
        if self.background_vars:
            concepts = create_background(concepts, patients_info, self.background_vars)
        if self.ages:
            concepts = create_ages(concepts, patients_info.set_index('PID')['BIRTHDATE'])
        if self.segment:
            concepts = create_segments(concepts)
        if self.origin_point:
            concepts = create_abspos(concepts, self.origin_point)

        return concepts
