import pandas as pd

from corebehrt.functional.features.creators import (
    assign_index_and_order,
    create_abspos,
    create_age_in_years,
    create_background,
    create_segments,
    sort_features,
)
from corebehrt.functional.features.exclude import exclude_event_nans
from corebehrt.functional.setup.checks import check_features_columns
from corebehrt.constants.data import PID_COL


class FeatureCreator:
    """
    A class to create features from patient information and concepts DataFrames.
    We create background, death, age, absolute position, and segments features.
    """

    def __call__(
        self,
        concepts: pd.DataFrame,
    ) -> pd.DataFrame:
        check_features_columns(concepts)
        features, patient_info = create_background(concepts)
        features = create_age_in_years(features)
        features = create_abspos(features)

        features = assign_index_and_order(features)
        features = exclude_event_nans(features)
        features = sort_features(features)

        features = create_segments(features)
        features = features.drop(columns=["admission_id", "time", "birthdate"])
        features[PID_COL] = features[PID_COL].astype(int)

        return features, patient_info
