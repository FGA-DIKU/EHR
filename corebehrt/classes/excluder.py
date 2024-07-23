""" Excluder class for excluding incorrect events and patients """

import pandas as pd
from typing import Union, List
from corebehrt.functional.exclude import (
    exclude_incorrect_event_ages,
    exclude_event_nans,
    exclude_short_sequences,
)
from corebehrt.functional.utils import normalize_segments


class Excluder:
    def __init__(
        self,
        min_age: int = -1,
        max_age: int = 120,
        min_len: int = 0,
        background_length: int = 0,
    ):
        self.min_age = min_age
        self.max_age = max_age
        self.min_len = min_len
        self.background_length = background_length

    def exclude_incorrect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude events with incorrect ages (outside defined range)"""
        df = exclude_incorrect_event_ages(df, self.min_age, self.max_age)
        df = exclude_event_nans(df)
        df["segment"] = normalize_segments(df)
        return df.reset_index(drop=True)

    def exclude_short_sequences(
        self, x: Union[pd.DataFrame, List[list], dict]
    ) -> Union[
        pd.DataFrame, List[list], dict
    ]:  # TODO: Currently doesn't support outcomes
        """Exclude patients with less than k events (taken background into account)"""
        return exclude_short_sequences(x, self.min_len, self.background_length)
