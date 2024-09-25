import dask.dataframe as dd
from datetime import datetime
import pandas as pd
import numpy as np

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

def create_segments_dask(concepts: dd.DataFrame) -> dd.DataFrame:
    meta = concepts.dtypes.to_dict() 
    meta['segment'] = 'int64'
    concepts = concepts.map_partitions(assign_segment, meta=meta)   
    #concepts = concepts.drop(columns=['ADMISSION_ID'])

    # Step 2: Set 'PID' as the index for efficient merging
    concepts = concepts.set_index('PID').persist()

    # Step 3: Compute the maximum segment per 'PID'
    max_segment = concepts[['segment']].groupby('PID').max()
    max_segment = max_segment.rename(columns={'segment': 'max_segment'})

    # Step 4: Merge using the index
    concepts = concepts.merge(max_segment, left_index=True, right_index=True, how='left')

    # Step 5: Assign maximum segment to 'Death' concepts
    concepts['segment'] = concepts['segment'].where(
        concepts['concept'] != 'Death', concepts['max_segment']
    )

    # Step 6: Reset index if necessary and drop columns
    concepts = concepts.reset_index()
    concepts = concepts.drop(columns=['max_segment', 'ADMISSION_ID'])

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


def sort_within_partition(df):
    return df.sort_values('abspos')

def assign_age(df):
    df['age'] = (df['TIMESTAMP'] - df['BIRTHDATE']).dt.days // 365.25
    return df

def create_segments_dask_fast(concepts: dd.DataFrame) -> dd.DataFrame:
    # Step 1: Repartition data by 'PID'
    concepts = concepts.repartition(partition_on='PID')

    # Step 2: Assign segments within each partition
    meta = concepts._meta.copy()
    meta['segment'] = 'int64'
    concepts = concepts.map_partitions(assign_segment_fast, meta=meta)

    # Step 3: Compute the maximum segment per 'PID'
    max_segment = concepts.groupby('PID')['segment'].max().reset_index()
    max_segment = max_segment.rename(columns={'segment': 'max_segment'})

    # Step 4: Merge the maximum segment back to the original DataFrame
    concepts = concepts.merge(max_segment, on='PID', how='left')

    # Step 5: Assign maximum segment to 'Death' concepts
    concepts['segment'] = concepts['segment'].where(
        concepts['concept'] != 'Death', concepts['max_segment']
    )

    # Step 6: Drop unnecessary columns
    concepts = concepts.drop(columns=['max_segment', 'ADMISSION_ID'])

    return concepts

def assign_segment_fast(df):
    # Since data is partitioned by 'PID', all rows in df have the same 'PID'
    df = df.sort_values('abspos')
    # Assign segments by factorizing 'ADMISSION_ID' in the order of 'abspos'
    df['segment'] = pd.factorize(df['ADMISSION_ID'])[0]
    return df



def assign_segment(df):
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


def sort_partition(df):
    return df.sort_values(by=['PID', 'TIMESTAMP'])
def sort_partition_abspos(df):
    return df.sort_values(by=['PID', 'abspos'])