"""
Simulate both random and correlated medical data for performance testing and ML model comparison.
"""

import os
import uuid
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

# Default parameters
DEFAULT_N_PATIENTS = 10_000
DEFAULT_N_CONCEPTS = 20
DEFAULT_WRITE_DIR = "tmp/correlated_data"
DEFAULT_INCLUDE_LABS = True
DEFAULT_DATA_TYPE = "both"


# Concept relationships and probabilities
CONCEPT_NAMING = {
    "D": "diagnosis",
    "M": "medication",
    "LAB": "labtest",
}
CONCEPT_RELATIONSHIPS = {
    "DE10": {  # Type 1 diabetes
        "base_probability": 0.4,  # Probability of selecting this as base condition
        "related_concepts": {
            "MIN01": 0.8,  # 30% Probability of creating a related concept
            "MIN02": 0.2,  
            "LAB/GL": 0.25,  
        }
    },
    "DE11": {  # Type 2 diabetes
        "base_probability": 0.4,
        "related_concepts": {
            "MME01": 0.2,  
            "MME02": 0.9,  
            "LAB/GL": 0.7,  
        }
    },
    "DO60": {  # Preterm labor
        "base_probability": 0.2,
        "related_concepts": {
            "DDZ32": 0.3,  
            "DDZ34": 0.2,  
        }
    }
}

def get_related_concepts(base_concept: str) -> List[str]:
    """Get all related concepts for a given base concept."""
    related = []
    if base_concept in CONCEPT_RELATIONSHIPS:
        for concept, prob in CONCEPT_RELATIONSHIPS[base_concept]["related_concepts"].items():
            if np.random.random() < prob:
                related.append(concept)
    return related

def generate_random_concepts_batch(
    patients_info: pd.DataFrame,
    mean_records_per_pid: int,
    prefix: str = "",
    result_col: bool = False,
    n_unique_concepts: int = 1000,
) -> pd.DataFrame:
    """
    Generate completely random medical concepts without any correlations.
    This is useful for creating baseline data to compare against correlated data.
    """
    # Generate random number of records for each patient
    n_records_per_patient = np.random.exponential(
        scale=mean_records_per_pid, size=len(patients_info)
    ).astype(int)
    n_records_per_patient = np.maximum(n_records_per_patient, 1)

    repeated_indices = np.repeat(patients_info.index.values, n_records_per_patient)
    repeated_patients_info = patients_info.loc[repeated_indices].reset_index(drop=True)

    # Convert dates to datetime
    repeated_patients_info["BIRTHDATE"] = pd.to_datetime(repeated_patients_info["BIRTHDATE"])
    end_dates = pd.to_datetime(repeated_patients_info["DEATHDATE"]).fillna(
        pd.Timestamp(year=2025, month=1, day=1)
    )

    # Generate timestamps
    birthdates = repeated_patients_info["BIRTHDATE"].astype(np.int64) // 10**9
    deathdates = end_dates.astype(np.int64) // 10**9
    valid_mask = birthdates < deathdates
    deathdates[~valid_mask] = birthdates[~valid_mask] + 1
    random_timestamps = np.random.randint(birthdates, deathdates, dtype=np.int64)
    timestamps = pd.to_datetime(random_timestamps, unit="s")

    # Generate completely random concepts
    concepts = np.random.randint(0, n_unique_concepts, size=len(repeated_patients_info))
    if prefix != "":
        concepts = [f"{prefix}{c}" for c in concepts]

    # Create the DataFrame
    concepts_data = pd.DataFrame({
        "TIMESTAMP": timestamps,
        "PID": repeated_patients_info["PID"],
        "ADMISSION_ID": [str(uuid.uuid4()) for _ in range(len(repeated_patients_info))],
        "CONCEPT": concepts,
    })

    if result_col:
        # Generate random lab results
        results = np.random.randint(100, 200, size=len(repeated_patients_info))
        concepts_data["RESULT"] = results

    # Filter out rows where TIMESTAMP is less than BIRTHDATE
    concepts_data = concepts_data[
        concepts_data["TIMESTAMP"] >= repeated_patients_info["BIRTHDATE"].values
    ]

    return concepts_data

def generate_correlated_concepts_batch(
    patients_info: pd.DataFrame,
    mean_records_per_pid: int,
    result_col: bool = False,
) -> pd.DataFrame:
    """
    Generate medical concepts with realistic correlations and relationships.
    This creates data that better represents real-world medical patterns.
    """
    # Generate random number of records for each patient
    n_records_per_patient = np.random.exponential(
        scale=mean_records_per_pid, size=len(patients_info)
    ).astype(int)
    n_records_per_patient = np.maximum(n_records_per_patient, 1)

    repeated_indices = np.repeat(patients_info.index.values, n_records_per_patient)
    repeated_patients_info = patients_info.loc[repeated_indices].reset_index(drop=True)

    # Convert dates to datetime
    repeated_patients_info["BIRTHDATE"] = pd.to_datetime(repeated_patients_info["BIRTHDATE"])
    end_dates = pd.to_datetime(repeated_patients_info["DEATHDATE"]).fillna(
        pd.Timestamp(year=2025, month=1, day=1)
    )

    # Generate timestamps
    birthdates = repeated_patients_info["BIRTHDATE"].astype(np.int64) // 10**9
    deathdates = end_dates.astype(np.int64) // 10**9
    valid_mask = birthdates < deathdates
    deathdates[~valid_mask] = birthdates[~valid_mask] + 1
    random_timestamps = np.random.randint(birthdates, deathdates, dtype=np.int64)
    timestamps = pd.to_datetime(random_timestamps, unit="s")

    # Generate concepts with relationships
    concepts = []
    for _ in range(len(repeated_patients_info)):
        # Choose a base concept based on probabilities
        base_concept = np.random.choice(
            list(CONCEPT_RELATIONSHIPS.keys()),
            p=[rel["base_probability"] for rel in CONCEPT_RELATIONSHIPS.values()]
        )
        
        # Always include the base concept
        concept_list = [base_concept]
        
        # Get related concepts based on their individual probabilities
        related_concepts = get_related_concepts(base_concept)
        if related_concepts:
            concept_list.extend(related_concepts)
            
        # Choose one concept from the list (base + related if any)
        concept = np.random.choice(concept_list)
        concepts.append(concept)

    # Create the DataFrame
    concepts_data = pd.DataFrame({
        "TIMESTAMP": timestamps,
        "PID": repeated_patients_info["PID"],
        "ADMISSION_ID": [str(uuid.uuid4()) for _ in range(len(repeated_patients_info))],
        "CONCEPT": concepts,
    })

    if result_col:
        # Generate realistic lab results based on concepts
        results = []
        for concept in concepts:
            if concept.startswith("LAB"):  # Only generate results for lab tests
                # Generate random integer between 0 and 100
                results.append(np.random.randint(0, 100))
            else:
                results.append(np.nan)  # NaN for non-lab concepts
        concepts_data["RESULT"] = results

    # Filter out rows where TIMESTAMP is less than BIRTHDATE
    concepts_data = concepts_data[
        concepts_data["TIMESTAMP"] >= repeated_patients_info["BIRTHDATE"].values
    ]

    return concepts_data

def generate_admissions_batch(patients_info, mean_records_per_pid):
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
    durations = np.random.randint(10, 700, size=len(timestamps))
    end_timestamps = timestamps + pd.to_timedelta(durations, unit="d")

    # Generate ADMISSION_ID column using vectorized operations
    admission_ids = np.array(
        [str(uuid.uuid4()) for _ in range(len(repeated_patients_info))]
    )

    # Create the DataFrame
    admissions_data = pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "PID": repeated_patients_info["PID"],
            "ADMISSION_ID": admission_ids,
            "CONCEPT": ["ADMISSION"] * len(repeated_patients_info),
        }
    )

    discharge_data = pd.DataFrame(
        {
            "TIMESTAMP": end_timestamps,
            "PID": repeated_patients_info["PID"],
            "ADMISSION_ID": admission_ids,
            "CONCEPT": ["DISCHARGE"] * len(repeated_patients_info),
        }
    )

    # Filter out rows where TIMESTAMP is less than BIRTHDATE
    admissions_data = admissions_data[
        admissions_data["TIMESTAMP"] >= repeated_patients_info["BIRTHDATE"].values
    ]

    discharge_data = discharge_data[
        discharge_data["TIMESTAMP"] >= repeated_patients_info["BIRTHDATE"].values
    ]

    comb_admissions_data = pd.concat([admissions_data, discharge_data])
    return comb_admissions_data

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

def main_write(
    n_patients: int,
    n_concepts: int,
    write_dir: str,
    include_labs: bool = True,
    data_type: str = "both",  # "random", "correlated", or "both"
):
    """
    Generate and write simulated data to parquet files.
    
    Args:
        n_patients: Number of patients to generate
        n_concepts: Average number of concepts per patient
        write_dir: Directory to write output files
        include_labs: Whether to include lab test data
        data_type: Type of data to generate ("random", "correlated", or "both")
    """
    os.makedirs(write_dir, exist_ok=True)
    batch_size_patients = min(200_000, n_patients)
    
    # Initialize parquet writers
    writers = {}
    schema_written = False

    for i in tqdm(range(n_patients // batch_size_patients)):
        patients_info = generate_patients_info_batch(batch_size_patients)
        
        # Generate data based on specified type
        if data_type in ["random", "both"]:
            random_diagnose = generate_random_concepts_batch(patients_info, n_concepts, prefix="D")
            random_medication = generate_random_concepts_batch(patients_info, n_concepts, prefix="M")
            random_admission = generate_admissions_batch(patients_info, 5)
            
            if include_labs:
                random_labtest = generate_random_concepts_batch(
                    patients_info, n_concepts, prefix="LAB", result_col=True, n_unique_concepts=10
                )
        
        if data_type in ["correlated", "both"]:
            # Generate correlated data
            correlated_data = generate_correlated_concepts_batch(patients_info, n_concepts, result_col=True)
            correlated_admission = generate_admissions_batch(patients_info, 5)
            
            # Split correlated data into different types based on CONCEPT_NAMING
            correlated_by_type = {}
            for concept_type, file_type in CONCEPT_NAMING.items():
                mask = correlated_data["CONCEPT"].str.startswith(concept_type)
                correlated_by_type[file_type] = correlated_data[mask].copy()
                
                # Remove result column for non-lab concepts
                if file_type != "labtest":
                    correlated_by_type[file_type] = correlated_by_type[file_type].drop(columns=["RESULT"])

        # Dictionary mapping DataFrames to their output files
        data_mapping = {}
        
        if data_type == "random":
            data_mapping.update({
                "patients_info": (patients_info, f"{write_dir}/patients_info.parquet"),
                "diagnose": (random_diagnose, f"{write_dir}/concept.diagnose.parquet"),
                "medication": (random_medication, f"{write_dir}/concept.medication.parquet"),
                "admission": (random_admission, f"{write_dir}/concept.admission.parquet"),
            })
            if include_labs:
                data_mapping["labtest"] = (
                    random_labtest,
                    f"{write_dir}/concept.labtest.parquet",
                )
        
        elif data_type == "correlated":
            data_mapping.update({
                "patients_info": (patients_info, f"{write_dir}/patients_info.parquet"),
                "diagnose": (correlated_by_type["diagnosis"], f"{write_dir}/concept.diagnose.parquet"),
                "medication": (correlated_by_type["medication"], f"{write_dir}/concept.medication.parquet"),
                "admission": (correlated_admission, f"{write_dir}/concept.admission.parquet"),
            })
            if include_labs:
                data_mapping["labtest"] = (
                    correlated_by_type["labtest"],
                    f"{write_dir}/concept.labtest.parquet",
                )
        
        else:  # both
            # Combine random and correlated data for each type
            combined_diagnose = pd.concat([random_diagnose, correlated_by_type["diagnosis"]])
            combined_medication = pd.concat([random_medication, correlated_by_type["medication"]])
            combined_admission = pd.concat([random_admission, correlated_admission])
            
            data_mapping.update({
                "patients_info": (patients_info, f"{write_dir}/patients_info.parquet"),
                "diagnose": (combined_diagnose, f"{write_dir}/concept.diagnose.parquet"),
                "medication": (combined_medication, f"{write_dir}/concept.medication.parquet"),
                "admission": (combined_admission, f"{write_dir}/concept.admission.parquet"),
            })
            if include_labs:
                combined_labtest = pd.concat([random_labtest, correlated_by_type["labtest"]])
                data_mapping["labtest"] = (
                    combined_labtest,
                    f"{write_dir}/concept.labtest.parquet",
                )

        # Write each DataFrame
        for name, (df, filepath) in data_mapping.items():
            table = pa.Table.from_pandas(df)

            if not schema_written:
                # First batch: create new file
                writers[name] = pq.ParquetWriter(
                    filepath, table.schema, compression_level=10, compression="ZSTD"
                )

            writers[name].write_table(table)

        schema_written = True

    # Close all writers
    for writer in writers.values():
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate both random and correlated medical data for ML model comparison."
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=DEFAULT_N_PATIENTS,
        help=f"Number of patients to generate (default: {DEFAULT_N_PATIENTS})",
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
    parser.add_argument(
        "--include-labs",
        action="store_true",
        default=DEFAULT_INCLUDE_LABS,
        help=f"Include lab test data generation (default: {DEFAULT_INCLUDE_LABS})",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["random", "correlated", "both"],
        default=DEFAULT_DATA_TYPE,
        help=f"Type of data to generate: random, correlated, or both (default: {DEFAULT_DATA_TYPE})",
    )

    args = parser.parse_args()
    main_write(
        n_patients=args.n_patients,
        n_concepts=args.n_concepts,
        write_dir=args.write_dir,
        include_labs=args.include_labs,
        data_type=args.data_type,
    ) 