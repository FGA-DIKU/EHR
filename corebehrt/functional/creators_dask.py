import dask.dataframe as dd
from datetime import datetime
import pandas as pd


def create_abspos_dask(concepts: dd.DataFrame, origin_point: datetime) -> dd.DataFrame:
    concepts['abspos'] = (concepts['TIMESTAMP'] - origin_point).dt.days*24
    concepts = concepts.drop(columns=['TIMESTAMP'])
    return concepts

def create_age_in_years_dask(concepts: dd.DataFrame) -> dd.DataFrame:
    meta = concepts.dtypes.to_dict() 
    meta['age'] = 'timedelta64[ns]'
    concepts = concepts.map_partitions(assign_age, meta=meta) 
    concepts = concepts.drop(columns=['BIRTHDATE'])  
    return concepts

def create_death_dask(patients_info: dd.DataFrame) -> dd.DataFrame:
    # only consider patients who have a birthdate
    patients_info = patients_info[~patients_info["BIRTHDATE"].isna()]
    # Define a function to be applied to each partition
    
    # Apply death event creation across partitions
    death_dask = patients_info.map_partitions(
        death_partition, meta={"PID": "object", "concept": "object", "TIMESTAMP": "datetime64[ns]", "ADMISSION_ID": "int64", "BIRTHDATE": "datetime64[ns]"}
    )

    return death_dask

def create_background_dask(patients_info: dd.DataFrame, background_vars: list) -> dd.DataFrame:
    # Use map_partitions to apply the transformation across Dask partitions
    background_dask = patients_info.map_partitions(
        background_partition, background_vars, meta={"PID": "object", "concept": "object", "TIMESTAMP": "datetime64[ns]", 'ADMISSION_ID': 'int64', 'BIRTHDATE': 'datetime64[ns]'}
    )

    return background_dask

def create_segments_dask(concepts: dd.DataFrame) -> dd.DataFrame:
    # We need to properly sort by abspos for correct segment creation
    # Ensure Dask uses the 'tasks' shuffle method
    concepts = concepts.shuffle(on='PID')
    # Apply the function using map_partitions
    concepts = concepts.map_partitions(sort_within_pid)    
    concepts = concepts.map_partitions(assign_segments)

    concepts = assign_segments_to_death(concepts)
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

def assign_age(df):
    df['age'] = (df['TIMESTAMP'] - df['BIRTHDATE']).dt.days // 365.25
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


