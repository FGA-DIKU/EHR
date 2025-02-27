from datetime import datetime
from typing import Union

import dask.dataframe as dd

from corebehrt.functional.features.creators import (
    create_abspos,
    create_age_in_years,
    create_background,
    create_death,
    create_segments,
    assign_index_and_order,
    sort_features,
)
from corebehrt.functional.features.exclude import exclude_event_nans
from corebehrt.functional.setup.checks import (
    check_concepts_columns,
    check_patients_info_columns,
)


class FeatureCreator:
    """
    A class to create features from patient information and concepts DataFrames.
    We create background, death, age, absolute position, and segments features.
    Parameters
    ----------
    origin_point : datetime or dict, optional
        The origin point for calculating absolute positions. If a dict is provided, it's used to create a datetime object.
    background_vars : list, optional
        List of background variable names to include.

    Examples
    --------
    >>> feature_creator = FeatureCreator(origin_point=datetime(2020, 1, 26), background_vars=["GENDER", "AGE"])
    >>> features = feature_creator(patients_info, concepts)
    """

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
        self,
        patients_info: dd.DataFrame,
        concepts: dd.DataFrame,
    ) -> dd.DataFrame:

        check_concepts_columns(concepts)
        check_patients_info_columns(patients_info, self.background_vars)

        concepts = self.prepare_concepts(concepts, patients_info)

        background = create_background(patients_info, self.background_vars)
        death = create_death(patients_info)
        features = dd.concat([concepts, background, death])
        features = create_age_in_years(features)
        features = create_abspos(features, self.origin_point)

        features = assign_index_and_order(features)
        features = exclude_event_nans(features)
        features = sort_features(features)

        features = create_segments(features)
        features = features.drop(columns=["ADMISSION_ID", "TIMESTAMP", "BIRTHDATE"])

        return features

    def prepare_concepts(
        self, concepts: dd.DataFrame, patients_info: dd.DataFrame
    ) -> dd.DataFrame:
        """Set index, add BIRTHDATE to concepts for age calculation + renaming of concepts."""
        concepts = concepts.set_index("PID")  # for merging

        concepts["ADMISSION_ID"] = concepts["ADMISSION_ID"].astype(str)
        concepts["CONCEPT"] = concepts["CONCEPT"].astype(str)

        patients_info = patients_info[["PID", "BIRTHDATE"]].compute()

        def join_with_patients_info(concepts_partition: dd.DataFrame) -> dd.DataFrame:
            """Join concepts with patients_info on PID."""
            return concepts_partition.merge(patients_info, on="PID", how="left")

        concepts = concepts.map_partitions(join_with_patients_info)
        concepts = concepts.rename(
            columns={"CONCEPT": "concept"}
        )  # use lowercase for feature names
        return concepts
