from datetime import datetime
from typing import Union
import pandas as pd

from corebehrt.functional.creators import (
    create_abspos,
    create_ages,
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

    def __call__(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> pd.DataFrame:
        concepts.rename(columns={"CONCEPT": "concept"}, inplace=True)
        # Create background
        background = create_background(patients_info, self.background_vars)
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
        concepts = pd.concat([concepts, death])

        return concepts
