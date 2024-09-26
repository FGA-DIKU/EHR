from datetime import datetime
from typing import Union
import dask.dataframe as dd

from corebehrt.functional.creators import (
    create_abspos,
    create_age_in_years,
    create_background,
    create_death,
    create_segments,
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

    def __call__(self, patients_info: dd.DataFrame, concepts: dd.DataFrame) -> dd.DataFrame:
        background = create_background(patients_info, self.background_vars)

        death = create_death(patients_info)

        features = dd.concat([concepts, background, death])
        features = create_age_in_years(features)
        features = create_abspos(features, self.origin_point)
        features = create_segments(features)    
        
        return features

