from datetime import datetime
from typing import Union
import pandas as pd
import dask.dataframe as dd

from corebehrt.functional.creators import (
    create_abspos,
    create_ages,
    create_background,
    create_death,
    create_segments,
)
from corebehrt.functional.creators_dask import (
    create_abspos_dask,
    create_age_in_years_dask,
    create_background_dask,
    create_death_dask,
    create_segments_dask_fast,
)


class FeatureCreator:
    def __init__(
        self,
        origin_point: Union[datetime, dict] = datetime(
            2020, 1, 26
        ),  # If given, compute ABSPOS
        background_vars: list = ["GENDER"],
    ):  # If given, add background variables
        self.origin_point = (
            datetime(**origin_point) if isinstance(origin_point, dict) else origin_point
        )
        self.background_vars = background_vars

    def __call__(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        concepts.rename(columns={"CONCEPT": "concept"}, inplace=True)
        # Create background
        background = create_background(patients_info, self.background_vars)
        if len(background) > 0:
            # Check and only concat if non-empty to avoid deprecation warning.
            concepts = pd.concat([concepts, background])
        # Create ages
        birthdates = patients_info.set_index("PID")["BIRTHDATE"]
        concepts["age"] = create_ages(
            concepts["TIMESTAMP"], concepts["PID"].map(birthdates)
        )
        # Create segments
        grouped_object = concepts.groupby("PID")["ADMISSION_ID"]
        concepts["segment"] = create_segments(grouped_object)
        # Create abspos
        concepts["abspos"] = create_abspos(concepts["TIMESTAMP"], self.origin_point)
        # Create death
        death = create_death(
            patients_info, concepts[["PID", "segment"]], self.origin_point
        )
        if len(death) > 0:
            # Check and only concat if non-empty to avoid deprecation warning.
            concepts = pd.concat([concepts, death])

        return concepts

class FeatureCreatorDask:
    def __init__(
        self,
        save_path: str,
        origin_point: Union[datetime, dict] = datetime(
            2020, 1, 26
        ),  # If given, compute ABSPOS
        background_vars: list = ["GENDER"],
    ):  # If given, add background variables
        self.save_path = save_path
        self.origin_point = (
            datetime(**origin_point) if isinstance(origin_point, dict) else origin_point
        )
        self.background_vars = background_vars

    def __call__(self, patients_info: dd.DataFrame, concepts: dd.DataFrame) -> dd.DataFrame:
        background = create_background_dask(patients_info, self.background_vars)

        death = create_death_dask(patients_info)

        features = dd.concat([concepts, background, death])
        features = create_age_in_years_dask(features)
        features = create_abspos_dask(features, self.origin_point)

        features = features.set_index('PID')
        features.groupby('PID').apply(lambda x: x.sort_values('abspos'), meta=features)
        features = create_segments_dask_fast(features)    
        
        result = features.groupby('PID').apply(lambda x: x.sort_values('abspos'), meta=features)
        result.to_csv(self.save_path, index=False)

