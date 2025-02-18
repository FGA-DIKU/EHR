from corebehrt.functional.features.exclude import exclude_incorrect_event_ages
import pandas as pd

class Excluder:
    def __init__(
        self,
        min_age: int = -1,
        max_age: int = 120,
    ):
        self.min_age = min_age
        self.max_age = max_age

    def exclude_incorrect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude events with incorrect ages (outside defined range)"""
        return exclude_incorrect_event_ages(df, self.min_age, self.max_age)
