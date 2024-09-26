import dask.dataframe as dd
from datetime import datetime
import pandas as pd

from corebehrt.functional.utils import get_abspos_from_origin_point


def create_abspos(concepts: dd.DataFrame, origin_point: datetime) -> dd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts (dd.DataFrame): The concepts DataFrame with 'TIMESTAMP' column.
        origin_point (datetime): The origin point for calculating absolute position.
    Returns:
        dd.DataFrame: The DataFrame with a new 'abspos' column and 'TIMESTAMP'
    """
    concepts['abspos'] = get_abspos_from_origin_point(
        concepts['TIMESTAMP'], 
        origin_point)
    concepts = concepts.drop(columns=['TIMESTAMP'])
    return concepts

def create_age_in_years(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Compute age in years for each row in concepts
    """
    concepts['age'] = (concepts['TIMESTAMP'] - concepts['BIRTHDATE']).dt.days // 365.25
    concepts = concepts.drop(columns=['BIRTHDATE'])  
    return concepts

def create_death(patients_info: dd.DataFrame) -> dd.DataFrame:
    # only consider patients who have a birthdate
    patients_info = patients_info[~patients_info["BIRTHDATE"].isna()]
    # Define a function to be applied to each partition
    
    # Apply death event creation across partitions
    death_dask = patients_info.map_partitions(
        death_partition, meta={"PID": "object", "concept": "object", "TIMESTAMP": "datetime64[ns]", "ADMISSION_ID": "int64", "BIRTHDATE": "datetime64[ns]"}
    )

    return death_dask

def create_background(patients_info: dd.DataFrame, background_vars: list) -> dd.DataFrame:
    # Use map_partitions to apply the transformation across Dask partitions
    background_dask = patients_info.map_partitions(
        background_partition, background_vars, meta={"PID": "object", "concept": "object", "TIMESTAMP": "datetime64[ns]", 'ADMISSION_ID': 'int64', 'BIRTHDATE': 'datetime64[ns]'}
    )

    return background_dask

def create_segments(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Assign segments to the concepts DataFrame based on 'ADMISSION_ID', ensuring that
    events are ordered correctly within each 'PID'.

    Parameters:
        concepts (dd.DataFrame): The concepts DataFrame with 'PID', 'ADMISSION_ID', and 'abspos' columns.

    Returns:
        dd.DataFrame: The DataFrame with a new 'segment' column and 'ADMISSION_ID' dropped.
    """
    # Shuffle data by 'PID' to ensure that all data for a PID is in the same partition
    concepts = concepts.shuffle(on='PID')
    
    # Sort within partitions by 'PID' and 'abspos'
    concepts = concepts.map_partitions(sort_within_pid)
    
    # Assign segments
    concepts = concepts.map_partitions(assign_segments)

    # Assign maximum segment to 'Death' concepts
    concepts = assign_segments_to_death(concepts)
    
    # Drop unnecessary columns
    concepts = concepts.drop(columns=['ADMISSION_ID'])

    return concepts

def assign_segments_to_death(df: dd.DataFrame) -> dd.DataFrame:
    # Compute the maximum segment per 'PID'
    max_segment = df.groupby('PID')['segment'].max().reset_index()
    max_segment = max_segment.rename(columns={'segment': 'max_segment'})
    df = df.merge(max_segment, on='PID', how='left')
    # Assign maximum segment to 'Death' concepts
    df['segment'] = df['segment'].where(
        df['concept'] != 'Death', df['max_segment']
    )
    df = df.drop(columns=['max_segment'])
    return df

def sort_within_pid(df):
    # Sort by 'PID' and 'abspos' to ensure correct ordering
    return df.sort_values(['PID', 'abspos'])

def assign_segments(df):
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df['segment'] = df.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0])
    return df

def death_partition(partition):
    # Filter rows where DEATHDATE is not NaT
    death_events = partition[~partition["DEATHDATE"].isna()]
    
    # Create death event DataFrame
    death_df = pd.DataFrame({
        "PID": death_events["PID"],
        "concept": "Death",  # Static concept for death event
        "TIMESTAMP": death_events["DEATHDATE"],  # Timestamp for death event
        "ADMISSION_ID": -1,  # Segment for death event
        "BIRTHDATE": death_events["BIRTHDATE"],  # Birthdate for age calculation
    })
    
    return death_df


def background_partition(partition, background_vars):
    # Vectorized creation of background concepts using pd.concat and .str methods
    replicated_pids = pd.concat([partition["PID"]] * len(background_vars), ignore_index=True)
    replicated_timestamps = pd.concat([partition["BIRTHDATE"]] * len(background_vars), ignore_index=True)
    # Efficiently concatenate the background concepts
    background_concepts = pd.concat(
        [
            partition[col].astype(str).map(lambda x: f"BG_{col}_{x}")
            for col in background_vars
        ],
        ignore_index=True
    )
    
    # Combine into a DataFrame
    background = pd.DataFrame({
        "PID": replicated_pids,
        "concept": background_concepts,
        "TIMESTAMP": replicated_timestamps,
        'ADMISSION_ID': 0,  # Segment for background events
        'BIRTHDATE': replicated_timestamps
    })
    
    return background


