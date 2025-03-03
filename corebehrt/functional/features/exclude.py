import pandas as pd
from corebehrt.constants.data import AGE_COL


def exclude_incorrect_event_ages(
    df: pd.DataFrame, min_age: int = -1, max_age: int = 120
) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df[AGE_COL] >= min_age) & (df[AGE_COL] <= max_age)]


def exclude_event_nans(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Exclude events (row) with (any) NaNs"""
    return df.dropna(subset=columns)
