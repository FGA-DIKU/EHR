"""
Simulate synthetic medical data with correlated lab values.
"""

import os
import uuid
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Default parameters
PATIENTS_INFO_PATH = "data/patients_info_20percent.parquet"
DEFAULT_WRITE_DIR = "tmp/binary_labs"

# Define lab value ranges and their probabilities for different conditions
LAB_VALUE_INFO = {
    "S/LAB1": {
        "positive": {
            "dist": "uniform",
            "range": [1],
            "n_samples": [1, 3]
        },
        "negative": {
            "mean": 100,
            "std": 25,
            "dist": "normal",
            "n_samples": [1, 3]
        },
    },
}

CONCEPT_RELATIONSHIPS = {
    "S/DIAG1": {
        "base_probability": 1.0,
        "related_concepts": {
            "S/LAB1": {
                "prob": 0.5,
                "conditions": ["positive"],
            }
        }
    },
}

def generate_lab_value(lab_name: str, condition: str) -> Optional[float]:
    """
    Generate a lab value based on the lab name and condition.
    
    Args:
        lab_name: Name of the lab test
        condition: The condition affecting the lab values
        
    Returns:
        Optional[float]: Generated lab value or None if invalid input
    """
    if lab_name not in LAB_VALUE_INFO or condition not in LAB_VALUE_INFO[lab_name]:
        return None
        
    range_info = LAB_VALUE_INFO[lab_name][condition]
    if range_info["dist"] == "uniform":
        return np.random.choice(range_info["range"])
    elif range_info["dist"] == "normal":
        return np.random.normal(range_info["mean"], range_info["std"])
    return None

def get_related_concepts(base_concept: str) -> Tuple[List[str], List[float]]:
    """
    Get all related concepts and their values for a given base concept.
    
    Args:
        base_concept: The base concept to get related concepts for
        
    Returns:
        Tuple[List[str], List[float]]: Lists of related concepts and their values
    """
    related_concepts = []
    values = []
    
    if base_concept not in CONCEPT_RELATIONSHIPS:
        return related_concepts, values
        
    for lab_name, info in CONCEPT_RELATIONSHIPS[base_concept]["related_concepts"].items():
        if np.random.random() < info["prob"]:
            # Generate condition for patient
            condition = np.random.choice(info["conditions"])
            n_samples = np.random.randint(LAB_VALUE_INFO[lab_name][condition]["n_samples"][0], LAB_VALUE_INFO[lab_name][condition]["n_samples"][1])
            
            for _ in range(n_samples):
                value = generate_lab_value(lab_name, condition)
                if value is not None:
                    related_concepts.append(lab_name)
                    values.append(value)
                    
    return related_concepts, values

def generate_lab_concepts(pids_list: List[str]) -> pd.DataFrame:
    """
    Generate lab concepts and values for a list of patient IDs.
    
    Args:
        pids_list: List of patient IDs
        
    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []
    
    for pid in pids_list:
        # Add base concept
        base_concept = np.random.choice(list(CONCEPT_RELATIONSHIPS.keys()))
        records.append({
            "PID": pid,
            "CONCEPT": base_concept,
            "RESULT": None
        })
        
        # Add related concepts and their values
        related_concepts, values = get_related_concepts(base_concept)
        for concept, value in zip(related_concepts, values):
            records.append({
                "PID": pid,
                "CONCEPT": concept,
                "RESULT": value
            })
    
    return pd.DataFrame(records)

def generate_timestamps(pids_list: List[str], patient_df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Generate timestamps for a list of patient IDs.
    
    Args:
        pids_list: List of patient IDs to generate timestamps for
        patient_df: DataFrame containing patient information with birthdate and deathdate columns
        
    Returns:
        List[pd.Timestamp]: List of generated timestamps
    """
    timestamps = []
    for pid in pids_list:
        patient_info = patient_df[patient_df["PID"] == pid].iloc[0]
        birthdate = pd.to_datetime(patient_info["birthdate"])
        
        # Handle deathdate - if NaT, use a default future date
        deathdate = pd.to_datetime(patient_info["deathdate"])
        if pd.isna(deathdate):
            deathdate = pd.Timestamp(year=2025, month=1, day=1)
            
        # Ensure deathdate is after birthdate
        if deathdate <= birthdate:
            deathdate = birthdate + pd.Timedelta(days=1)
            
        # Generate random timestamp between birthdate and deathdate
        time_diff = (deathdate - birthdate).total_seconds()
        random_seconds = np.random.randint(0, int(time_diff))
        timestamp = birthdate + pd.Timedelta(seconds=random_seconds)
        timestamps.append(timestamp)
        
    return timestamps

def generate_data(patient_df: pd.DataFrame, write_dir: str) -> None:
    """
    Generate synthetic data for patients and save to parquet.
    
    Args:
        patient_df: DataFrame containing patient information
        write_dir: Directory to write output files
    """
    # Generate concepts and lab values
    concepts_data = generate_lab_concepts(patient_df["PID"].tolist())
    
    # Create final DataFrame
    data = pd.DataFrame({
        "subject_id": concepts_data["PID"],
        "code": concepts_data["CONCEPT"],
        "numeric_value": concepts_data["RESULT"].astype(float),
    })
    
    # Generate timestamps for each record
    data["time"] = generate_timestamps(data["subject_id"].tolist(), patient_df)

    # Write to csv
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(write_dir / "synthetic_data.csv", index=False)
    
    print(f"Generated data for {len(patient_df)} patients, a total of {len(data)} records")
    print(f"Saved to {write_dir / 'synthetic_data.csv'}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical data with correlated lab values"
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files"
    )
    parser.add_argument(
        "--patients_info_path",
        type=str,
        default=PATIENTS_INFO_PATH,
        help="Path to patient information parquet file"
    )
    
    args = parser.parse_args()
    
    # Read patient data
    try:
        patient_df = pd.read_parquet(args.patients_info_path)
        patient_df = patient_df.rename(columns={"subject_id": "PID"})
    except FileNotFoundError:
        print(f"Error: Could not find patient info file at {args.patients_info_path}")
        return
    
    # Generate and save data
    generate_data(patient_df, args.write_dir)

if __name__ == "__main__":
    main()
