
import dask.dataframe as dd
import pandas as pd
# from dask.distributed import Client
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from dask.distributed import performance_report
from time import time
from datetime import datetime
from dask import config
from corebehrt.functional.utils import get_abspos_from_origin_point

config.set({'dataframe.shuffle.method': 'tasks'})
size = '10k'
BACKGROUND_VARS = ['RACE', 'ETHNICITY', 'GENDER']
ORIGIN_POINT = pd.Timestamp(2020, 1, 26)

def main(size, background_vars=BACKGROUND_VARS, origin_point=ORIGIN_POINT):
    #cluster = LocalCluster(n_workers=6, threads_per_worker=1)
    #client = Client(cluster)
    #with performance_report(filename="dask-report.html"):
    with ProgressBar():
        concepts, patients_info = load(size)  

        background = create_background_dask(patients_info, background_vars)
        death = create_death_dask(patients_info)
        features = dd.concat([concepts, background, death])

        features = create_age_in_years_dask(features)
        
        features = create_abspos_dask(features, origin_point)
        
        features = create_segments_dask(features)    
        
        print('Compute')
        features.to_csv(f'outputs/features_{size}', index=False)
        
      
def sort_within_partition(df):
    return df.sort_values('abspos')

def create_abspos_dask(concepts: dd.DataFrame, origin_point: datetime) -> dd.DataFrame:
    concepts['abspos'] = get_abspos_from_origin_point(
        concepts['TIMESTAMP'], 
        origin_point)
    concepts = concepts.drop(columns=['TIMESTAMP'])
    return concepts

def create_age_in_years_dask(concepts: dd.DataFrame) -> dd.DataFrame:
    concepts['age'] = (concepts['TIMESTAMP'] - concepts['BIRTHDATE']).dt.days // 365.25
    concepts = concepts.drop(columns=['BIRTHDATE'])  
    return concepts


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


def sort_within_pid(df):
    # Sort by 'PID' and 'abspos' to ensure correct ordering
    return df.sort_values(['PID', 'abspos'])

def assign_segments(df):
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df['segment'] = df.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0])
    return df

def create_death_dask(patients_info: dd.DataFrame) -> dd.DataFrame:
    """
    Create a Dask DataFrame containing death events for patients with known death dates.

    Parameters:
        patients_info (dd.DataFrame): Dask DataFrame containing patient information, including 'PID', 'BIRTHDATE', and 'DEATHDATE'.

    Returns:
        dd.DataFrame: Dask DataFrame with death events, including 'PID', 'concept', 'TIMESTAMP', 'ADMISSION_ID', and 'BIRTHDATE' columns.
    """
    # Filter patients who have both BIRTHDATE and DEATHDATE
    death_events = patients_info[
        (~patients_info["BIRTHDATE"].isna()) & (~patients_info["DEATHDATE"].isna())
    ][["PID", "BIRTHDATE", "DEATHDATE"]]

    # Rename 'DEATHDATE' to 'TIMESTAMP' and add necessary columns
    death_events = death_events.rename(columns={"DEATHDATE": "TIMESTAMP"})
    death_events = death_events.assign(
        concept="Death",
        ADMISSION_ID=-1,  # Segment for death event
    )

    # Reorder columns if necessary
    death_events = death_events[
        ["PID", "concept", "TIMESTAMP", "ADMISSION_ID", "BIRTHDATE"]
    ]

    return death_events


def create_background_dask(patients_info: dd.DataFrame, background_vars: list) -> dd.DataFrame:
    """
    Create background concepts for each patient based on specified background variables.

    Parameters
    ----------
    patients_info : dd.DataFrame
        Dask DataFrame containing patient information, including 'PID', 'BIRTHDATE', and background variables.
    background_vars : list
        List of background variable column names to include.

    Returns
    -------
    dd.DataFrame
        Dask DataFrame with background concepts, including 'PID', 'concept', 'TIMESTAMP', 'ADMISSION_ID', and 'BIRTHDATE' columns.
    """
    # Filter patients with non-null BIRTHDATE
    patients_info = patients_info[~patients_info["BIRTHDATE"].isna()]

    # Melt the DataFrame to transform background variables into rows
    background = patients_info[['PID', 'BIRTHDATE'] + background_vars].melt(
        id_vars=['PID', 'BIRTHDATE'],
        value_vars=background_vars,
        var_name='variable',
        value_name='value'
    )

    # Create 'concept' column
    background['concept'] = 'BG_' + background['variable'] + '_' + background['value'].astype(str)

    # Assign additional columns
    background['TIMESTAMP'] = background['BIRTHDATE']
    background['ADMISSION_ID'] = 0

    # Select and reorder the required columns
    background = background[['PID', 'concept', 'TIMESTAMP', 'ADMISSION_ID', 'BIRTHDATE']]

    return background

# def create_background_dask(patients_info: dd.DataFrame, background_vars: list) -> dd.DataFrame:
#     # Use map_partitions to apply the transformation across Dask partitions
#     background_dask = patients_info.map_partitions(
#         background_partition, background_vars, meta={"PID": "object", "concept": "object", "TIMESTAMP": "datetime64[ns]", 'ADMISSION_ID': 'int64', 'BIRTHDATE': 'datetime64[ns]'}
#     )

#     return background_dask
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


def load(size):
    #dir_ = f'example_data_large_{size}'
    dir_ = f'example_data'
    concepts = load_concepts(dir_)
    patients_info = load_patients_info(dir_)

    concepts = concepts.merge(patients_info[['PID', 'BIRTHDATE']], on='PID', how='left', broadcast=True)  # for age calculation
    return concepts, patients_info

def load_patients_info(dir_):
    patients_info = dd.read_csv(
        f'{dir_}/patients_info.csv',
        parse_dates=['BIRTHDATE'],
        dtype={'DEATHDATE': 'object'},
        assume_missing=True  # This helps with missing values in integer columns 
    )
    patients_info["DEATHDATE"] = dd.to_datetime(patients_info["DEATHDATE"], errors='coerce', infer_datetime_format=True)

    return patients_info

def load_concepts(dir_):
    diag = dd.read_csv(f'{dir_}/concept.diagnose.csv',  parse_dates=['TIMESTAMP'])#.set_index('PID')
    med = dd.read_csv(f'{dir_}/concept.medication.csv', parse_dates=['TIMESTAMP'])#.set_index('PID')
    med['TIMESTAMP'] = med['TIMESTAMP'].dt.tz_localize(None)
    diag['TIMESTAMP'] = diag['TIMESTAMP'].dt.tz_localize(None)
    concepts = dd.concat([diag, med]).set_index('PID')
    concepts = concepts.rename(columns={"CONCEPT": "concept"})
    return concepts

if __name__ == "__main__":
    start = time()
    main(size)
    print(f'Time: {time()-start}')
    print(f"Time in minutes: {round((time()-start)/60, 3)}")
