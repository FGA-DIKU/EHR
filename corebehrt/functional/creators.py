""" This module contains functions that create new columns in the dataset """
import itertools
from datetime import datetime

import pandas as pd

from corebehrt.functional.utils import (calculate_ages_at_death,
                              get_abspos_from_origin_point, get_last_segments,
                              get_time_difference, normalize_segments_df)


def create_ages(concepts: pd.DataFrame, patients_info: pd.DataFrame) -> pd.DataFrame:
    """Creates the AGE column"""
    birthdates = patients_info.set_index('PID')[find_column(patients_info, 'birth')]
    concepts['age'] = get_time_difference(concepts['TIMESTAMP'], concepts['PID'].map(birthdates))
    return concepts

def create_abspos(concepts: pd.DataFrame, origin_point: datetime) -> pd.DataFrame:
    """Creates the ABSPOS column"""
    concepts['abspos'] = get_abspos_from_origin_point(concepts['TIMESTAMP'], origin_point)
    return concepts

def create_segments(concepts: pd.DataFrame, segment_col='ADMISSION_ID') -> pd.DataFrame:
    """ Creates the SEGMENT column (the normalize segments_df can do this) """
    return normalize_segments_df(concepts, segment_col)

def create_background(concepts: pd.DataFrame, patients_info: pd.DataFrame, background_vars: list) -> pd.DataFrame:
    """ Creates the BACKGROUND column """
    background = pd.DataFrame({
        'PID': patients_info['PID'].tolist() * len(background_vars),
        'concept': itertools.chain.from_iterable(
                [(patients_info[col].map(lambda x: f'BG_{col}_{x}')).tolist() for col in background_vars]),
        'TIMESTAMP': patients_info['DEATHDATE'].tolist() * len(background_vars),
        })
    
    return pd.concat([background, concepts])

def create_death(concepts: pd.DataFrame, patients_info: pd.DataFrame, origin_point: datetime)-> pd.DataFrame:
    """Creates the DEATH concept"""
    patients_info = patients_info[patients_info['DEATHDATE'].notna()] # Only consider patients with death info

    death_info = {'PID': patients_info['PID'].tolist()}
    death_info['concept'] = ['Death'] * len(patients_info)
    if 'segment' in concepts.columns:
        death_info['segment'] = get_last_segments(concepts, patients_info)
    if 'age' in concepts.columns:
        death_info['age'] = calculate_ages_at_death(patients_info)
    if 'abspos' in concepts.columns:
        death_info['abspos'] = get_abspos_from_origin_point(select_column(patients_info, 'birth'), origin_point).to_list()

    # Append death info to concepts
    death_info = pd.DataFrame(death_info)
    return pd.concat([concepts, death_info])


