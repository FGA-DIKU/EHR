from datetime import datetime
from typing import Union

import dask.dataframe as dd
from dask import config

config.set(
    {"dataframe.convert-string": False}
)  # Avoids dask greedy string conversion of objects

from corebehrt.classes.normalizer import ValuesNormalizer
from corebehrt.functional.creators import (
    create_abspos,
    create_age_in_years,
    create_sep_tokens,
    create_background,
    create_death,
    create_segments,
    sort_features,
)
from corebehrt.functional.utils import (
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
        cls_token: bool = False,
        sep_token: bool = True,
        values: dict = None,
    ):  # If given, add background variables
        self.origin_point = (
            datetime(**origin_point) if isinstance(origin_point, dict) else origin_point
        )
        self.background_vars = background_vars
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.value_creator_kwargs = values

    def __call__(
        self,
        patients_info: dd.DataFrame,
        concepts: dd.DataFrame,
    ) -> dd.DataFrame:
        # Check columns # TODO: Move elsewhere?
        check_concepts_columns(concepts)
        check_patients_info_columns(patients_info, self.background_vars)

        # min-max-normalize the RESULT column
        if "RESULT" in concepts.columns:
            concepts = ValuesNormalizer.min_max_normalize_results(
                concepts, num_bins=self.value_creator_kwargs.get("num_bins", 100)
            )

        # Prepare dataframes
        concepts = self.prepare_concepts(concepts, patients_info)
        background = create_background(
            patients_info, self.background_vars, cls_token=self.cls_token
        )
        death = create_death(patients_info)

        # Combine dataframes and wrap event_columns into a single 'event' column
        concepts = self.combine_to_event(concepts)
        background = self.combine_to_event(background)
        death = self.combine_to_event(death)

        features = dd.concat([background, concepts, death])
        if self.sep_token:
            features = create_sep_tokens(features)

        # Create temporal features
        features["age"] = create_age_in_years(features)
        features["abspos"] = create_abspos(features, self.origin_point)

        features = features.dropna()
        features = sort_features(features)

        features = create_segments(features)

        features = features.drop(columns=["ADMISSION_ID", "TIMESTAMP", "BIRTHDATE"])

        # Explode "concept" column (list of concepts) into separate rows
        features = features.explode("concept")
        features = features.dropna(subset=["concept"])
        features["concept"] = features["concept"].astype(str)

        return features

    def prepare_concepts(
        self, concepts: dd.DataFrame, patients_info: dd.DataFrame
    ) -> dd.DataFrame:
        """Set index, add BIRTHDATE to concepts for age calculation + renaming of concepts."""
        concepts = concepts.set_index("PID")  # for merging

        patients_info = patients_info[["PID", "BIRTHDATE"]].set_index("PID").compute()

        def join_with_patients_info(concepts_partition: dd.DataFrame) -> dd.DataFrame:
            """Join concepts with patients_info on PID."""
            return concepts_partition.merge(
                patients_info, how="left", left_index=True, right_index=True
            )

        concepts = concepts.map_partitions(join_with_patients_info).reset_index()

        return concepts

    @staticmethod
    def combine_to_event(dataframe: dd.DataFrame) -> dd.DataFrame:
        """Combines non-static columns to a single 'concept' list column"""
        static_columns = ["PID", "BIRTHDATE", "TIMESTAMP", "ADMISSION_ID"]
        event_columns = dataframe.columns.difference(static_columns).tolist()

        dataframe["concept"] = dataframe[event_columns].map_partitions(
            lambda x: x.values.tolist()
        )

        return dataframe.drop(columns=event_columns)
