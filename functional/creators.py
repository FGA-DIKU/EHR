""" This module contains functions that create new columns in the dataset """
import itertools
import pandas as pd
from datetime import datetime
from functional.utils import get_abspos_from_origin_point, normalize_segments_df, get_time_difference

def create_ages(concepts: pd.DataFrame, birthdates: dict) -> pd.DataFrame:
    """Creates the AGE column"""
    concepts['AGE'] = get_time_difference(concepts['TIMESTAMP'], concepts['PID'].map(birthdates))
    return concepts

def create_abspos(concepts: pd.DataFrame, origin_point: datetime) -> pd.DataFrame:
    """Creates the ABSPOS column"""
    concepts['ABSPOS'] = get_abspos_from_origin_point(concepts['TIMESTAMP'], origin_point)
    return concepts

def create_segments(concepts: pd.DataFrame, segment_col='ADMISSION_ID') -> pd.DataFrame:
    """ Creates the SEGMENT column (the normalize segments_df can do this) """
    concepts['SEGMENT'] = normalize_segments_df(concepts, segment_col)
    return concepts

def create_background(concepts: pd.DataFrame, patients_info: pd.DataFrame, background_vars: list) -> pd.DataFrame:
    """ Creates the BACKGROUND column """
    background = pd.DataFrame({
        'PID': patients_info['PID'].tolist() * len(background_vars),
        'CONCEPT': itertools.chain.from_iterable(
                [(patients_info[col].map(lambda x: f'BG_{col}_{x}')).tolist() for col in background_vars]),
        'TIMESTAMP': patients_info['BIRTHDATE'].tolist() * len(background_vars),
        })
    
    return pd.concat([background, concepts])
