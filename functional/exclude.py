""" File for excluding errors """
import pandas as pd
from typing import Union, List

def exclude_incorrect_event_ages(df: pd.DataFrame, min_age: int = -1, max_age: int = 120) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df['age'] >= min_age) & (df['age'] <= max_age)]

def exclude_event_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude events (row) with (any) NaNs """
    return df.dropna()

def exclude_short_sequences(x: Union[pd.DataFrame, List[list], dict], min_len: int = 3, background_length: int = 0) -> Union[pd.DataFrame, list, dict]:
    if isinstance(x, pd.DataFrame):
        return x.groupby('PID').filter(lambda x: len(x) >= min_len + background_length)
    elif isinstance(x, list) and isinstance(x[0], list):
        return [x for x in x if len(x) >= min_len + background_length]
    elif isinstance(x, dict):
        return {k: v for k, v in x.items() if len(x['concept']) >= min_len + background_length}
    else:
        raise TypeError('Invalid type for x, only pd.DataFrame, list of lists, and dict are supported.')
