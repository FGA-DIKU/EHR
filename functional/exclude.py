""" File for excluding errors """
import pandas as pd
from typing import Union, List, Tuple

def exclude_incorrect_event_ages(df: pd.DataFrame, min_age: int = -1, max_age: int = 120) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df['age'] >= min_age) & (df['age'] <= max_age)]

def exclude_event_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude events (row) with (any) NaNs """
    return df.dropna()

def exclude_short_sequences(x: Union[pd.DataFrame, List[list], dict], min_len: int = 3, background_length: int = 0) -> Tuple[Union[pd.DataFrame, List[list], dict], List[int]]:
    if isinstance(x, pd.DataFrame):
        return exclude_short_sequences_df(x, min_len, background_length)
    elif isinstance(x, dict): # TODO: Only used for Data.features (should be removed when pipeline is refactored)
        return exclude_short_sequences_dict(x, min_len, background_length)
    else:
        raise TypeError('Invalid type for x, only pd.DataFrame and dict are supported.')

def exclude_short_sequences_df(x: pd.DataFrame, min_len:int, background_length:int)->pd.DataFrame:
    filtered_df = x.groupby('PID').filter(lambda x: min_len_condition(x['CONCEPT'], min_len, background_length))
    kept_indices = filtered_df.index.tolist()  
    return filtered_df, kept_indices

def exclude_short_sequences_dict(x: dict, min_len:int, background_length:int)->Tuple[dict, list]:
    kept_indices = [i for i, c in enumerate(x['concept']) if min_len_condition(c, min_len, background_length)]
    filtered_x = {k: [v[i] for i in kept_indices] for k, v in x.items()}
    return filtered_x, kept_indices

def min_len_condition(c: list, min_len:int, background_length:int)->bool:
    return len(c) >= min_len + background_length
