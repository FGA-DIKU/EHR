from datetime import datetime
from typing import Union

import pandas as pd

from corebehrt.functional.features.creators import (
    create_abspos,
    create_age_in_years,
    create_background,
    create_segments,
    assign_index_and_order,
    sort_features,
)
from corebehrt.functional.features.exclude import exclude_event_nans
from corebehrt.functional.setup.checks import (
    check_features_columns,
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
    ):
        self.origin_point = (
            datetime(**origin_point) if isinstance(origin_point, dict) else origin_point
        )

    def __call__(
        self,
        concepts: pd.DataFrame,
    ) -> pd.DataFrame:
        check_features_columns(concepts)
        features, patient_info = create_background(concepts)
        features = create_age_in_years(features)
        features = create_abspos(features, self.origin_point)

        features = assign_index_and_order(features)
        features = exclude_event_nans(features)
        features = sort_features(features)

        features = create_segments(features)
        features = features.drop(columns=["admission_id", "time", "birthdate"])
        features["subject_id"] = features["subject_id"].astype(int)

        return features, patient_info
