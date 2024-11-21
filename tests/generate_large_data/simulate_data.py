"""
Simulate large data for performance testing and profiling.
"""

import os
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_N = 10_000
DEFAULT_N_CONCEPTS = 20  # Number of concepts per patient
DEFAULT_WRITE_DIR = "tmp/example_data_large"
DEFAULT_BATCH_SIZE = 200_000


def main_write(
    n_patients,
    n_concepts,
    write_dir,
):
    os.makedirs(write_dir, exist_ok=True)
    batch_size_patients = min(DEFAULT_BATCH_SIZE, n_patients)
    # Initialize parquet writers
    writers = {}
    schema_written = False

    for i in tqdm(range(n_patients // batch_size_patients)):
        patients_info = generate_patients_info_batch(batch_size_patients)
        concepts = generate_concepts_batch(patients_info, n_concepts, prefix="D")
        concepts_m = generate_concepts_batch(patients_info, n_concepts, prefix="M")
        concepts_l = generate_concepts_batch(
            patients_info,
            n_concepts,
            prefix="LAB",
            result_col=True,
            n_unique_concepts=10,
        )

        # Dictionary mapping DataFrames to their output files
        data_mapping = {
            "patients_info": (patients_info, f"{write_dir}/patients_info.parquet"),
            "diagnose": (concepts, f"{write_dir}/concept.diagnose.parquet"),
            "medication": (concepts_m, f"{write_dir}/concept.medication.parquet"),
            "labtest": (concepts_l, f"{write_dir}/concept.labtest.parquet"),
        }

        # Write each DataFrame
        for name, (df, filepath) in data_mapping.items():
            table = pa.Table.from_pandas(df)

            if not schema_written:
                # First batch: create new file
                writers[name] = pq.ParquetWriter(filepath, table.schema)

            writers[name].write_table(table)

        schema_written = True

    # Close all writers
    for writer in writers.values():
        writer.close()


def generate_patients_info_batch(n_patients):
    # Set a random seed for reproducibility (optional)
    np.random.seed(42)

    # Define the range of birthdates (e.g., between 1940 and 2020)
    start_birthdate = np.datetime64("1940-01-01")
    end_birthdate = np.datetime64("2020-01-01")

    # Generate random birthdates between start and end dates
    birthdates = np.random.choice(
        np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"), n_patients
    )

    # Generate deathdates where some people are still alive (i.e., deathdate is NaT)
    death_prob = np.random.rand(n_patients)

    # For those with death_prob > 0.8, generate a deathdate between their birthdate and a future date (e.g., 2025)
    deathdates = np.where(
        death_prob > 0.8,
        np.array(
            [
                np.random.choice(
                    np.arange(
                        birthdate + np.timedelta64(10),
                        np.datetime64("2024-01-01"),
                        dtype="datetime64[D]",
                    )
                )
                for birthdate in birthdates
            ]
        ),
        pd.NaT,
    )
    # Generate race, ethnicity, and gender
    races = np.random.choice(
        ["white", "black", "asian", "native", "other"], size=n_patients
    )
    ethnicities = np.random.choice(["hispanic", "nonhispanic"], size=n_patients)
    genders = np.random.choice(["M", "F"], size=n_patients)

    # Generate random PIDs
    pids = np.array([str(uuid.uuid4()) for _ in range(n_patients)])

    return pd.DataFrame(
        {
            "PID": pids,
            "BIRTHDATE": birthdates,
            "DEATHDATE": deathdates,
            "RACE": races,
            "ETHNICITY": ethnicities,
            "GENDER": genders,
        }
    )


def generate_concepts_batch(
    patients_info,
    mean_records_per_pid,
    prefix="",
    result_col=False,
    n_unique_concepts=1000,
):
    # Generate random number of records for each patient using exponential distribution
    n_records_per_patient = np.random.exponential(
        scale=mean_records_per_pid, size=len(patients_info)
    ).astype(int)
    # Ensure at least 1 record per patient
    n_records_per_patient = np.maximum(n_records_per_patient, 1)

    # Create index array for repeating patient rows
    repeated_indices = np.repeat(patients_info.index.values, n_records_per_patient)

    # Repeat each row variable number of times
    repeated_patients_info = patients_info.loc[repeated_indices].reset_index(drop=True)

    # Convert BIRTHDATE and DEATHDATE to pandas datetime format
    repeated_patients_info["BIRTHDATE"] = pd.to_datetime(
        repeated_patients_info["BIRTHDATE"]
    )
    end_dates = pd.to_datetime(repeated_patients_info["DEATHDATE"]).fillna(
        pd.Timestamp(year=2025, month=1, day=1)
    )
    # Convert birthdates and deathdates to Unix timestamps (seconds since epoch)
    birthdates = repeated_patients_info["BIRTHDATE"].astype(np.int64) // 10**9
    deathdates = end_dates.astype(np.int64) // 10**9
    valid_mask = birthdates < deathdates
    deathdates[~valid_mask] = birthdates[~valid_mask] + 1
    # Generate random timestamps between birthdates and deathdates
    random_timestamps = np.random.randint(birthdates, deathdates, dtype=np.int64)
    timestamps = pd.to_datetime(random_timestamps, unit="s")

    # Generate ADMISSION_ID column using vectorized operations
    admission_ids = np.array(
        [str(uuid.uuid4()) for _ in range(len(repeated_patients_info))]
    )

    # Generate CONCEPT column using vectorized operations
    concepts = np.random.randint(0, n_unique_concepts, size=len(repeated_patients_info))
    if prefix != "":
        concepts = [f"{prefix}_{c}" for c in concepts]
    # Create the DataFrame
    concepts_data = pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "PID": repeated_patients_info["PID"],
            "ADMISSION_ID": admission_ids,
            "CONCEPT": concepts,
        }
    )

    if result_col:
        results = np.random.randint(100, 200, size=len(repeated_patients_info))
        concepts_data["RESULT"] = results

    # Filter out rows where TIMESTAMP is less than BIRTHDATE
    concepts_data = concepts_data[
        concepts_data["TIMESTAMP"] >= repeated_patients_info["BIRTHDATE"].values
    ]

    return concepts_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate large data for performance testing and profiling."
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=DEFAULT_N,
        help=f"Number of patients to generate (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--n-concepts",
        type=int,
        default=DEFAULT_N_CONCEPTS,
        help=f"Number of concepts per patient (default: {DEFAULT_N_CONCEPTS})",
    )
    parser.add_argument(
        "--write-dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help=f"Directory to write output files (default: {DEFAULT_WRITE_DIR})",
    )

    args = parser.parse_args()
    main_write(
        n_patients=args.n_patients,
        n_concepts=args.n_concepts,
        write_dir=args.write_dir,
    )
