"""
Add noise to synthetic data by switching labels and optionally removing data.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
from theoretical_separation import (
    cohens_d,
    sweep_threshold_auc,
    manual_mann_whitney_u,
    scipy_mann_whitney_u,
)

# Default parameters
N = 100000
LOW_MEAN = 0.45
HIGH_MEAN = 0.55
STD = 0.10
DEFAULT_INPUT_FILE = f"../../../data/vals/synthetic_data/{N}n/gaus_mean_{int(LOW_MEAN * 100)}_{int(HIGH_MEAN * 100)}_std_{int(STD * 100)}_n{N}_50p_1unq.csv"
DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"

# Default noise parameters
DEFAULT_SWITCH_PERCENTAGE = 0.0  # % label switching
DEFAULT_REMOVE_PERCENTAGE = 0.25  # 0  % data removal
DEFAULT_SEED = 42  # Fixed seed for reproducibility


def add_label_noise(
    data: pd.DataFrame, switch_percentage: float = 0.05
) -> pd.DataFrame:
    """
    Add noise by switching labels for a percentage of positive and negative cases.
    This is done by swapping lab values between positive and negative patients.

    Args:
        data: DataFrame containing the synthetic data
        switch_percentage: Percentage of positive and negative cases to switch (0.0 to 1.0)

    Returns:
        pd.DataFrame: DataFrame with noisy labels
    """
    noisy_data = data.copy()

    # Get positive and negative patients
    positive_patients = set(
        noisy_data[noisy_data["is_positive"]]["subject_id"].unique()
    )
    negative_patients = set(
        noisy_data[~noisy_data["is_positive"]]["subject_id"].unique()
    )

    # Calculate number of patients to switch
    n_positive_to_switch = int(len(positive_patients) * switch_percentage)
    n_negative_to_switch = int(len(negative_patients) * switch_percentage)

    # Randomly select patients to switch
    positive_to_switch = set(
        np.random.choice(list(positive_patients), n_positive_to_switch, replace=False)
    )
    negative_to_switch = set(
        np.random.choice(list(negative_patients), n_negative_to_switch, replace=False)
    )

    print(
        f"Swapping lab values for {len(positive_to_switch)} positive and {len(negative_to_switch)} negative patients..."
    )

    # Create pairs of patients to swap lab values
    positive_switch_list = list(positive_to_switch)
    negative_switch_list = list(negative_to_switch)

    # Ensure we have equal numbers to swap (take the minimum)
    n_pairs = min(len(positive_switch_list), len(negative_switch_list))

    for i in range(n_pairs):
        pos_patient = positive_switch_list[i]
        neg_patient = negative_switch_list[i]

        # Get lab values for both patients
        pos_labs = noisy_data[
            (noisy_data["subject_id"] == pos_patient) & (noisy_data["code"] == "S/LAB1")
        ]["numeric_value"].values
        neg_labs = noisy_data[
            (noisy_data["subject_id"] == neg_patient) & (noisy_data["code"] == "S/LAB1")
        ]["numeric_value"].values

        # Swap the lab values only if both patients have lab data
        if len(pos_labs) > 0 and len(neg_labs) > 0:
            # Create masks for the lab records
            pos_mask = (noisy_data["subject_id"] == pos_patient) & (
                noisy_data["code"] == "S/LAB1"
            )
            neg_mask = (noisy_data["subject_id"] == neg_patient) & (
                noisy_data["code"] == "S/LAB1"
            )

            # Ensure we have the same number of lab records for both patients
            min_labs = min(len(pos_labs), len(neg_labs))

            # Swap the numeric values (only the first min_labs records)
            noisy_data.loc[pos_mask, "numeric_value"] = neg_labs[:min_labs]
            noisy_data.loc[neg_mask, "numeric_value"] = pos_labs[:min_labs]

    print(f"Swapped lab values for {n_pairs} patient pairs")

    return noisy_data


def remove_random_data(
    data: pd.DataFrame, remove_percentage: float = 0.10
) -> pd.DataFrame:
    """
    Remove a random percentage of lab data points (S/LAB1), preserving all diagnostic codes.

    Args:
        data: DataFrame containing the synthetic data
        remove_percentage: Percentage of lab data to remove (0.0 to 1.0)

    Returns:
        pd.DataFrame: DataFrame with random lab data removed
    """
    if remove_percentage <= 0:
        return data

    # Get only lab records (S/LAB1)
    lab_mask = data["code"] == "S/LAB1"
    lab_data = data[lab_mask]

    # Calculate number of lab records to remove
    n_lab_records_to_remove = int(len(lab_data) * remove_percentage)

    # Randomly select lab indices to remove
    lab_indices_to_remove = np.random.choice(
        lab_data.index, n_lab_records_to_remove, replace=False
    )

    # Remove the selected lab records
    cleaned_data = data.drop(lab_indices_to_remove)

    print(
        f"Removed {n_lab_records_to_remove} lab records ({remove_percentage * 100:.1f}% of lab data)"
    )
    print(f"Preserved all diagnostic codes (S/DIAG1)")

    return cleaned_data


def remove_random_patients_lab_data(
    data: pd.DataFrame, remove_percentage: float = 0.10
) -> pd.DataFrame:
    """
    Remove lab data for a random percentage of patients, so that these patients have no lab data.

    Args:
        data: DataFrame containing the synthetic data
        remove_percentage: Percentage of patients to remove lab data from (0.0 to 1.0)

    Returns:
        pd.DataFrame: DataFrame with lab data removed for selected patients
    """
    if remove_percentage <= 0:
        return data

    # Get all patients with lab data
    lab_mask = data["code"] == "S/LAB1"
    patients_with_labs = set(data[lab_mask]["subject_id"].unique())

    # Calculate number of patients to remove lab data from
    n_patients_to_remove = int(len(patients_with_labs) * remove_percentage)

    # Randomly select patients to remove lab data from
    patients_to_remove = set(
        np.random.choice(list(patients_with_labs), n_patients_to_remove, replace=False)
    )

    # Remove all lab data for selected patients
    lab_data_to_remove = (data["code"] == "S/LAB1") & (
        data["subject_id"].isin(patients_to_remove)
    )
    cleaned_data = data[~lab_data_to_remove]

    print(
        f"Removed lab data for {len(patients_to_remove)} patients ({remove_percentage * 100:.1f}% of patients with lab data)"
    )
    print(f"Preserved all diagnostic codes (S/DIAG1)")

    return cleaned_data


def print_statistics(data: pd.DataFrame, title: str = "Lab value statistics") -> None:
    """
    Print statistics about the lab values.

    Args:
        data: DataFrame containing the data
        title: Title for the statistics section
    """
    # Get lab values for positive and negative patients
    lab_mask = data["code"] == "S/LAB1"
    positive_mask = data["is_positive"]

    positive_lab_values = data[lab_mask & positive_mask]["numeric_value"]
    negative_lab_values = data[lab_mask & ~positive_mask]["numeric_value"]

    print(f"\n{title} (positive):")
    print(f"Count: {len(positive_lab_values)}")
    print(f"Mean: {positive_lab_values.mean():.3f}")
    print(f"Std: {positive_lab_values.std():.3f}")
    print(f"Min: {positive_lab_values.min():.3f}")
    print(f"Max: {positive_lab_values.max():.3f}")

    print(f"\n{title} (negative):")
    print(f"Count: {len(negative_lab_values)}")
    print(f"Mean: {negative_lab_values.mean():.3f}")
    print(f"Std: {negative_lab_values.std():.3f}")
    print(f"Min: {negative_lab_values.min():.3f}")
    print(f"Max: {negative_lab_values.max():.3f}")


def create_distribution_plot(
    data: pd.DataFrame, save_path: Path, perfect_roc: float, title_suffix: str = ""
) -> None:
    """
    Create a figure showing the distribution of lab values for positive vs negative patients.

    Args:
        data: DataFrame containing the data
        save_path: Path to save the plot
        perfect_roc: The theoretical perfect ROC AUC value
        title_suffix: Additional text for the title
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
    ax1.set_title(f"Distribution of Lab Values by Diagnosis Status{title_suffix}")
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


def create_comparison_plot(
    original_data: pd.DataFrame, noisy_data: pd.DataFrame, save_path: Path
) -> None:
    """
    Create a comparison plot showing original vs noisy distributions.

    Args:
        original_data: Original DataFrame
        noisy_data: Noisy DataFrame
        save_path: Path to save the plot
    """
    lab_mask = original_data["code"] == "S/LAB1"

    # Get original values
    original_positive = original_data[lab_mask & original_data["is_positive"]][
        "numeric_value"
    ]
    original_negative = original_data[lab_mask & ~original_data["is_positive"]][
        "numeric_value"
    ]

    # Get noisy values - fix the boolean indexing issue
    noisy_lab_mask = noisy_data["code"] == "S/LAB1"
    noisy_lab_data = noisy_data[noisy_lab_mask]
    noisy_positive = noisy_lab_data[noisy_lab_data["is_positive"]]["numeric_value"]
    noisy_negative = noisy_lab_data[~noisy_lab_data["is_positive"]]["numeric_value"]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Original distribution
    ax1.hist(original_positive, bins=30, alpha=0.7, label="With Diagnosis", color="red")
    ax1.hist(original_negative, bins=30, alpha=0.7, label="No Diagnosis", color="blue")
    ax1.set_xlabel("Lab Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Original Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Noisy distribution
    ax2.hist(noisy_positive, bins=30, alpha=0.7, label="With Diagnosis", color="red")
    ax2.hist(noisy_negative, bins=30, alpha=0.7, label="No Diagnosis", color="blue")
    ax2.set_xlabel("Lab Value")
    ax2.set_ylabel("Count")
    ax2.set_title("Noisy Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to {save_path}")


def calculate_theoretical_performance(data: pd.DataFrame) -> dict:
    """
    Calculate the theoretical performance of the model, accounting for missing lab data.

    Args:
        data: DataFrame containing the data

    Returns:
        dict: Dictionary containing performance metrics
    """
    # Get lab data
    lab_mask = data["code"] == "S/LAB1"
    lab_data = data[lab_mask]

    # Get diagnostic data to identify positive/negative patients
    diag_mask = data["code"] == "S/DIAG1"
    positive_patients = set(data[diag_mask]["subject_id"].unique())

    # Count patients with and without lab data
    all_patients = set(data["subject_id"].unique())
    patients_with_labs = set(lab_data["subject_id"].unique())
    patients_without_labs = all_patients - patients_with_labs

    # Calculate performance for patients with lab data
    if len(lab_data) > 0:
        sweep_auc = sweep_threshold_auc(data)
        scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
        cohens_d_metric = cohens_d(data)
    else:
        sweep_auc = 0.5
        scipy_mann_whitney_u_auc = 0.5
        cohens_d_metric = 0.0

    # Calculate overall performance accounting for missing data
    n_total_patients = len(all_patients)
    n_patients_with_labs = len(patients_with_labs)
    n_patients_without_labs = len(patients_without_labs)

    # For patients without labs, performance is 50/50 (AUC = 0.5)
    if n_total_patients > 0:
        # Weighted average: (patients_with_labs * lab_auc + patients_without_labs * 0.5) / total_patients
        overall_sweep_auc = (
            n_patients_with_labs * sweep_auc + n_patients_without_labs * 0.5
        ) / n_total_patients
        overall_scipy_auc = (
            n_patients_with_labs * scipy_mann_whitney_u_auc
            + n_patients_without_labs * 0.5
        ) / n_total_patients
    else:
        overall_sweep_auc = 0.5
        overall_scipy_auc = 0.5

    mann_whitney_u = None  # manual_mann_whitney_u(data)

    print("\nTheoretical performance:")
    print(
        f"Patients with lab data: {n_patients_with_labs}/{n_total_patients} ({n_patients_with_labs / n_total_patients * 100:.1f}%)"
    )
    print(
        f"Patients without lab data: {n_patients_without_labs}/{n_total_patients} ({n_patients_without_labs / n_total_patients * 100:.1f}%)"
    )
    print(f"Lab-only Sweep AUC: {sweep_auc:.4f}")
    print(f"Overall Sweep AUC (accounting for missing data): {overall_sweep_auc:.4f}")
    print(f"Mann-Whitney U: {mann_whitney_u}")
    print(f"Lab-only Scipy Mann-Whitney U: {scipy_mann_whitney_u_auc:.4f}")
    print(f"Overall Scipy Mann-Whitney U: {overall_scipy_auc:.4f}")
    print(f"Cohen's d: {cohens_d_metric:.4f}")

    return {
        "sweep_auc": overall_sweep_auc,
        "lab_only_sweep_auc": sweep_auc,
        "mann_whitney_u": mann_whitney_u,
        "scipy_mann_whitney_u_auc": overall_scipy_auc,
        "lab_only_scipy_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
        "n_patients_with_labs": n_patients_with_labs,
        "n_patients_without_labs": n_patients_without_labs,
        "n_total_patients": n_total_patients,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add noise to synthetic data by switching labels and optionally removing data"
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
    parser.add_argument(
        "--switch_percentage",
        type=float,
        default=DEFAULT_SWITCH_PERCENTAGE,
        help="Percentage of positive and negative cases to switch (0.0 to 1.0, default: 0.05)",
    )
    parser.add_argument(
        "--remove_percentage",
        type=float,
        default=DEFAULT_REMOVE_PERCENTAGE,
        help="Percentage of data to randomly remove (0.0 to 1.0, default: 0.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Read input data
    try:
        data = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {args.input_file}")
        return

    print("Input data:")
    print(data.head())

    # Check if is_positive column exists, if not add it
    if "is_positive" not in data.columns:
        print("Adding is_positive column...")
        positive_diags = ["S/DIAG1"]
        positive_patients = set()
        for diag in positive_diags:
            positive_patients.update(data[data["code"] == diag]["subject_id"].unique())
        data["is_positive"] = data["subject_id"].isin(positive_patients)

    # Print initial statistics
    print("\nInitial data statistics:")
    print(f"Total records: {len(data)}")
    print(f"Total patients: {data['subject_id'].nunique()}")

    # Count unique positive and negative patients
    positive_patients = data[data["is_positive"]]["subject_id"].nunique()
    negative_patients = data[~data["is_positive"]]["subject_id"].nunique()

    print(f"Positive patients: {positive_patients}")
    print(f"Negative patients: {negative_patients}")

    # Print original statistics
    print_statistics(data, "Original lab value statistics")
    original_performance = calculate_theoretical_performance(data)

    # Apply noise
    noisy_data = data.copy()

    # Add label noise
    if args.switch_percentage > 0:
        print(
            f"\nAdding label noise: switching {args.switch_percentage * 100:.1f}% of positive and negative cases..."
        )
        noisy_data = add_label_noise(noisy_data, args.switch_percentage)

    # Remove random data
    if args.remove_percentage > 0:
        print(
            f"\nRemoving random data: removing {args.remove_percentage * 100:.1f}% of records..."
        )
        noisy_data = remove_random_data(noisy_data, args.remove_percentage)

    print("\nData after adding noise:")
    print(noisy_data.head())

    # Print noisy statistics
    print_statistics(noisy_data, "Noisy lab value statistics")
    noisy_performance = calculate_theoretical_performance(noisy_data)

    # Create output filename based on noise parameters
    noise_suffix = ""
    if args.switch_percentage > 0:
        noise_suffix += f"_switch{int(args.switch_percentage * 100)}p"
    if args.remove_percentage > 0:
        noise_suffix += f"_remove{int(args.remove_percentage * 100)}p"

    if not noise_suffix:
        noise_suffix = "_no_noise"

    # Extract base filename from input
    input_filename = Path(args.input_file).stem
    output_filename = f"{input_filename}{noise_suffix}"

    # Create distribution plots
    save_path = Path(DEFAULT_PLOT_DIR) / f"{output_filename}_distribution_plot.png"
    create_distribution_plot(
        noisy_data,
        save_path,
        noisy_performance["sweep_auc"],
        f" (Noise: {noise_suffix})",
    )

    # Create comparison plot
    comparison_save_path = (
        Path(DEFAULT_PLOT_DIR) / f"{output_filename}_comparison_plot.png"
    )
    create_comparison_plot(data, noisy_data, comparison_save_path)

    # Write to csv
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    noisy_data.to_csv(write_dir / f"{output_filename}.csv", index=False)
    print(f"\nSaved noisy data to {write_dir / f'{output_filename}.csv'}")


if __name__ == "__main__":
    main()
