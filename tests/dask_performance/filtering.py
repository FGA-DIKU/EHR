import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
import time

# Set the size of the dataset
n_patients = 10_000_000  # Number of unique patients
events_per_patient = 5   # Number of events per patient
total_rows = n_patients * events_per_patient

# Simulate data
def create_large_dataframe(n_patients, events_per_patient):
    np.random.seed(0)
    # Generate patient IDs
    pids = np.arange(n_patients).repeat(events_per_patient)
    # Generate ages between 20 and 90
    ages = np.random.randint(20, 90, size=total_rows)
    # Generate event data
    events = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=total_rows)
    # Generate timestamps
    timestamps = np.random.randint(1_000_000, 2_000_000, size=total_rows)
    
    df = pd.DataFrame({
        'PID': pids,
        'age': ages,
        'event': events,
        'timestamp': timestamps
    })
    return df

# Create the large pandas DataFrame
df = create_large_dataframe(n_patients, events_per_patient)

# Convert to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=100)


def filter_patients_by_age_at_last_event_direct(data, min_age, max_age):
    """
    Filtering using direct .isin() method.
    """
    # Get the last event for each patient
    last_events = data.groupby('PID').last().reset_index()
    
    # Filter patients by age at last event
    patients_in_age_range = last_events[
        (last_events['age'] >= min_age) & (last_events['age'] <= max_age)
    ]['PID'].compute()
    
    # Direct filtering
    filtered_data = data[data['PID'].isin(patients_in_age_range)]
    
    filtered_data.to_parquet('temp/direct', write_index=False)

def filter_patients_by_age_at_last_event_map_partitions(data, min_age, max_age):
    """
    Filtering using map_partitions with precomputed set.
    """
    # Get the last event for each patient
    last_events = data.groupby('PID').last().reset_index()
    
    # Filter patients by age at last event
    patients_in_age_range = last_events[
        (last_events['age'] >= min_age) & (last_events['age'] <= max_age)
    ]['PID'].compute()
    
    # Convert to set for faster lookup
    pids_set = set(patients_in_age_range)
    
    # Use map_partitions to filter data
    filtered_data = data.map_partitions(
        lambda partition: partition[partition['PID'].isin(pids_set)],
        meta=data._meta
    )
    filtered_data.to_parquet('temp/map_partitions', write_index=False)


# Define age range
min_age = 50
max_age = 60

# Measure time for direct filtering
import os
os.makedirs('temp', exist_ok=True)
start_time = time.time()
# Create the large pandas DataFrame
df = create_large_dataframe(n_patients, events_per_patient)

# Convert to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=100)
with ProgressBar():
    filter_patients_by_age_at_last_event_direct(ddf, min_age, max_age)
end_time = time.time()
print(f"Direct filtering took {end_time - start_time:.2f} seconds.")

# Create the large pandas DataFrame
df = create_large_dataframe(n_patients, events_per_patient)

# Convert to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=100)
# Measure time for map_partitions filtering
start_time = time.time()
with ProgressBar():
    filter_patients_by_age_at_last_event_map_partitions(ddf, min_age, max_age)
    
end_time = time.time()
print(f"map_partitions filtering took {end_time - start_time:.2f} seconds.")
import shutil
shutil.rmtree('temp')