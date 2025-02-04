from dask import dataframe as dd
from corebehrt.functional.features.exclude import exclude_incorrect_event_ages


class Excluder:
    def __init__(
        self,
        min_age: int = -1,
        max_age: int = 120,
    ):
        self.min_age = min_age
        self.max_age = max_age

    def exclude_incorrect_events(self, df: dd.DataFrame) -> dd.DataFrame:
        """Exclude events with incorrect ages (outside defined range)"""
        return exclude_incorrect_event_ages(df, self.min_age, self.max_age)
