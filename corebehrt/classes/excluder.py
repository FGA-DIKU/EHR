""" Excluder class for excluding incorrect events and patients """

from dask import dataframe as dd

from corebehrt.functional.exclude import exclude_incorrect_event_ages


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

    def exclude_incorrect_events(self, df: dd.DataFrame) -> dd.DataFrame:
        """Exclude events with incorrect ages (outside defined range)"""
        return exclude_incorrect_event_ages(df, self.min_age, self.max_age)
