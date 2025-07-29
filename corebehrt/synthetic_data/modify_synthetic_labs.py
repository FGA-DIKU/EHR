"""
Modify lab values in an existing synthetic data CSV file according to new distributions.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import os
from theoretical_separation import (
    cohens_d,
    sweep_threshold_auc,
    manual_mann_whitney_u,
    scipy_mann_whitney_u,
)

# Default parameters
N = 100000
LOW_MEAN = 0.48
HIGH_MEAN = 0.52
STD = 0.10
DEFAULT_INPUT_FILE = f"../../data/vals/synthetic_data/{N}n/bn_labs_n{N}_50p_1unq.csv"
DEFAULT_WRITE_DIR = f"../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../data/vals/synthetic_data_plots/{N}n/"
SAVE_NAME = f"gaus_mean_{int(LOW_MEAN * 100)}_{int(HIGH_MEAN * 100)}_std_{int(STD * 100)}_n{N}_50p_1unq"
POSITIVE_DIAGS = ["S/DIAG1"]

# Define new lab value distributions
LAB_VALUE_INFO = {
    "S/LAB1": {
        "high": {
            "dist": "normal",
            "mean": HIGH_MEAN,
            "std": STD,
        },
        "low": {
            "dist": "normal",
            "mean": LOW_MEAN,
            "std": STD,
        },
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


def modify_lab_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Modify only the numeric values of existing lab entries while keeping everything else the same.

    Args:
        data: DataFrame containing the synthetic data

    Returns:
        pd.DataFrame: Modified DataFrame with updated lab values
    """
    # Create a copy of the data
    modified_data = data.copy()

    # Get the mask for lab entries
    lab_mask = modified_data["code"] == "S/LAB1"

    # For each lab entry, generate a new value based on the patient's condition
    for idx in modified_data[lab_mask].index:
        patient_id = modified_data.at[idx, "subject_id"]
        is_positive = modified_data.at[idx, "is_positive"]
        condition = "high" if is_positive else "low"

        # Generate new value
        new_value = generate_lab_value("S/LAB1", condition)
        if new_value is not None:
            modified_data.at[idx, "numeric_value"] = new_value

    return modified_data


def get_positive_patients(data: pd.DataFrame, positive_diags: list) -> pd.DataFrame:
    """
    Get positive patients from the data and add is_positive column.

    Args:
        data: DataFrame containing the synthetic data
        positive_diags: List of diagnosis codes that indicate positive cases

    Returns:
        pd.DataFrame: DataFrame with added is_positive column
    """
    positive_patients = set()
    for diag in positive_diags:
        positive_patients.update(data[data["code"] == diag]["subject_id"].unique())

    data["is_positive"] = data["subject_id"].isin(positive_patients)
    return data


def print_statistics(data: pd.DataFrame) -> None:
    """
    Print statistics about the lab values.

    Args:
        data: DataFrame containing the modified data
    """
    # Get lab values for positive and negative patients
    lab_mask = data["code"] == "S/LAB1"
    positive_mask = data["is_positive"]

    positive_lab_values = data[lab_mask & positive_mask]["numeric_value"]
    negative_lab_values = data[lab_mask & ~positive_mask]["numeric_value"]

    print("\nLab value statistics (positive):")
    print(f"Count: {len(positive_lab_values)}")
    print(f"Mean: {positive_lab_values.mean():.3f}")
    print(f"Std: {positive_lab_values.std():.3f}")
    print(f"Min: {positive_lab_values.min():.3f}")
    print(f"Max: {positive_lab_values.max():.3f}")

    print("\nLab value statistics (negative):")
    print(f"Count: {len(negative_lab_values)}")
    print(f"Mean: {negative_lab_values.mean():.3f}")
    print(f"Std: {negative_lab_values.std():.3f}")
    print(f"Min: {negative_lab_values.min():.3f}")
    print(f"Max: {negative_lab_values.max():.3f}")


def create_distribution_plot(
    data: pd.DataFrame, save_path: Path, perfect_roc: float
) -> None:
    """
    Create a figure showing the distribution of lab values for positive vs negative patients.

    Args:
        data: DataFrame containing the modified data
        save_path: Path to save the plot
        perfect_roc: The theoretical perfect ROC AUC value
    """
    # Get lab values for positive and negative patients
    lab_mask = data["code"] == "S/LAB1"
    lab_data = data[lab_mask].copy()

    # Create a single subplot for the histogram
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Histogram
    positive_values = lab_data[lab_data["is_positive"]]["numeric_value"]
    negative_values = lab_data[~lab_data["is_positive"]]["numeric_value"]

    ax1.hist(positive_values, bins=30, alpha=0.7, label="With Diagnosis", color="red")
    ax1.hist(negative_values, bins=30, alpha=0.7, label="No Diagnosis", color="blue")
    ax1.set_xlabel("Lab Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Lab Values by Diagnosis Status")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add perfect ROC AUC text box
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # reserve top space for text
    fig.text(
        0.5,
        0.98,  # x=center, y=near top
        f"Theoretical Perfect ROC AUC: {perfect_roc:.4f}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved to {save_path}")


def calculate_theoretical_performance(data: pd.DataFrame) -> None:
    """
    Calculate the theoretical performance of the model.

    Args:
        data: DataFrame containing the modified data
    """
    sweep_auc = sweep_threshold_auc(data)
    mann_whitney_u = None  # manual_mann_whitney_u(data)
    scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
    cohens_d_metric = cohens_d(data)
    print("\nTheoretical performance:")
    print(f"Sweep AUC: {sweep_auc}")
    print(f"Mann-Whitney U: {mann_whitney_u}")
    print(f"Scipy Mann-Whitney U: {scipy_mann_whitney_u_auc}")
    print(f"Cohen's d: {cohens_d_metric}")
    return {
        "sweep_auc": sweep_auc,
        "mann_whitney_u": mann_whitney_u,
        "scipy_mann_whitney_u_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Modify lab values in synthetic data according to new distributions"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input synthetic data CSV file",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )

    args = parser.parse_args()

    # Read input data
    try:
        data = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {args.input_file}")
        return

    print("Initial data:")
    print(data.head())

    # Get positive patients and add is_positive column
    data = get_positive_patients(data, POSITIVE_DIAGS)

    # Print initial statistics
    print("\nInitial data statistics:")
    print(f"Total records: {len(data)}")
    print(f"Total patients: {data['subject_id'].nunique()}")

    # Count unique positive and negative patients
    positive_patients = data[data["is_positive"]]["subject_id"].nunique()
    negative_patients = data[~data["is_positive"]]["subject_id"].nunique()

    print(f"Positive patients: {positive_patients}")
    print(f"Negative patients: {negative_patients}")

    # Modify lab values
    modified_data = modify_lab_values(data)

    print("\nData after modifying lab values:")
    print(modified_data.head())

    # Print statistics about modified values
    print_statistics(modified_data)
    theoretical_performance = calculate_theoretical_performance(modified_data)

    # Create distribution plot
    save_path = Path(DEFAULT_PLOT_DIR) / f"{SAVE_NAME}_distribution_plot.png"
    create_distribution_plot(
        modified_data, save_path, perfect_roc=theoretical_performance["sweep_auc"]
    )

    # Calculate theoretical AUC

    # Write to csv
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    modified_data.to_csv(write_dir / f"{SAVE_NAME}.csv", index=False)
    print(f"\nSaved modified data to {write_dir / f'{SAVE_NAME}.csv'}")

    # Min-max normalize numeric_value for S/LAB1 and save as a separate file
    normalized_data = modified_data.copy()
    lab_mask = normalized_data["code"] == "S/LAB1"
    if lab_mask.any():
        min_val = normalized_data.loc[lab_mask, "numeric_value"].min()
        max_val = normalized_data.loc[lab_mask, "numeric_value"].max()
        if max_val > min_val:
            normalized_data.loc[lab_mask, "numeric_value"] = (
                normalized_data.loc[lab_mask, "numeric_value"] - min_val
            ) / (max_val - min_val)
        else:
            normalized_data.loc[lab_mask, "numeric_value"] = 0.0
    normalized_filename = write_dir / f"{SAVE_NAME}_minmaxnorm.csv"
    normalized_data.to_csv(normalized_filename, index=False)
    # Create distribution plot
    save_path = Path(DEFAULT_PLOT_DIR) / f"{SAVE_NAME}_minmaxnorm_distribution_plot.png"
    normalized_theoretical_performance = calculate_theoretical_performance(
        normalized_data
    )
    create_distribution_plot(
        normalized_data,
        save_path,
        perfect_roc=normalized_theoretical_performance["sweep_auc"],
    )
    print(f"\nSaved min-max normalized data to {normalized_filename}")


if __name__ == "__main__":
    main()
