""" This module contains functions that create new columns in the dataset """
import itertools
from datetime import datetime

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy


from corebehrt.functional.utils import (get_abspos_from_origin_point,
                              get_time_difference, normalize_segments_series)

def create_ages(timestamps: pd.Series, birthdates: pd.Series) -> pd.Series:
    """Returns the AGE column - Functions as a wrapper for get_time_difference"""
    return get_time_difference(timestamps, birthdates)

def create_abspos(timestamps: pd.Series, origin_point: datetime) -> pd.Series:
    """Returns the ABSPOS column - Functions as a wrapper for get_abspos_from_origin_point"""
    return get_abspos_from_origin_point(timestamps, origin_point)

def create_segments(groupby: DataFrameGroupBy) -> pd.Series:
    """Creates the SEGMENT column - Functions as a wrapper for normalize_segments_series"""
    return groupby.transform(normalize_segments_series)

def create_background(patients_info: pd.DataFrame, background_vars: list) -> pd.DataFrame:
    """ Creates the BACKGROUND column """
    background = pd.DataFrame({
        'PID': patients_info['PID'].tolist() * len(background_vars),
        'concept': itertools.chain.from_iterable(
                [(patients_info[col].map(lambda x: f'BG_{col}_{x}')).tolist() for col in background_vars]),
        'TIMESTAMP': patients_info['BIRTHDATE'].tolist() * len(background_vars),
        })
    
    return background

def create_death(patients_info: pd.DataFrame, segments_with_pids: pd.DataFrame, origin_point: datetime)-> pd.DataFrame:
    """Creates the DEATH concept"""
    patients_info = patients_info[patients_info['DEATHDATE'].notna()] # Only consider patients with death info

    last_segments = segments_with_pids.groupby('PID')['segment'].last().to_dict()
    death_info = pd.DataFrame({
        'PID': patients_info['PID'],
        'concept': ['Death'] * len(patients_info),
        'age': create_ages(patients_info['DEATHDATE'], patients_info['BIRTHDATE']),
        'abspos': create_abspos(patients_info['DEATHDATE'], origin_point),
        'segment': patients_info['PID'].map(last_segments)
    })

    return death_info
