"""
Apply histogram equalisation to lab values in synthetic data CSV files.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
from scipy import stats
from theoretical_separation import cohens_d, sweep_threshold_auc, scipy_mann_whitney_u

# Default parameters
N = 10000
LOW_MEAN = 0.45
HIGH_MEAN = 0.55
STD = 0.15
N_BINS = 256
DEFAULT_INPUT_FILE = f"../../../data/vals/synthetic_data/{N}n/gaus_mean_{int(LOW_MEAN * 100)}_{int(HIGH_MEAN * 100)}_std_{int(STD * 100)}_n{N}_50p_1unq.csv"
DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"
SAVE_NAME = f"gaus_mean_{int(LOW_MEAN * 100)}_{int(HIGH_MEAN * 100)}_std_{int(STD * 100)}_n{N}_50p_1unq_histeq_n{N_BINS}"


def histogram_equalisation(values: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """
    Perform histogram equalisation on a 1D array of values.

    Args:
        values: Input array of values to equalise
        n_bins: Number of bins for histogram calculation (default: 256)

    Returns:
        np.ndarray: Equalized values
    """
    # Calculate histogram
    hist, bin_edges = np.histogram(values, bins=n_bins, density=True)

    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]

    # Create interpolation function
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Interpolate to get equalised values
    equalised_values = np.interp(values, bin_centers, cdf)

    return equalised_values


def apply_histogram_equalisation(
    data: pd.DataFrame, lab_code: str = "S/LAB1", n_bins: int = 256
) -> pd.DataFrame:
    """
    Apply histogram equalisation to lab values in the dataset.

    Args:
        data: DataFrame containing the synthetic data
        lab_code: The lab code to equalise
        n_bins: Number of bins for histogram calculation

    Returns:
        pd.DataFrame: DataFrame with equalised lab values
    """
    # Create a copy of the data
    equalised_data = data.copy()

    # Get the mask for lab entries
    lab_mask = equalised_data["code"] == lab_code

    if not lab_mask.any():
        print(f"Warning: No entries found for lab code '{lab_code}'")
        return equalised_data

    # Get the lab values
    lab_values = equalised_data.loc[lab_mask, "numeric_value"].values

    # Apply histogram equalisation
    equalised_values = histogram_equalisation(lab_values, n_bins)

    # Update the data
    equalised_data.loc[lab_mask, "numeric_value"] = equalised_values

    return equalised_data


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
    original_data: pd.DataFrame, equalised_data: pd.DataFrame, save_path: Path
) -> None:
    """
    Create a comparison plot showing original vs equalised distributions.

    Args:
        original_data: Original DataFrame
        equalised_data: Equalized DataFrame
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

    # Get equalised values
    equalised_positive = equalised_data[lab_mask & equalised_data["is_positive"]][
        "numeric_value"
    ]
    equalised_negative = equalised_data[lab_mask & ~equalised_data["is_positive"]][
        "numeric_value"
    ]

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

    # Equalised distribution
    ax2.hist(
        equalised_positive, bins=30, alpha=0.7, label="With Diagnosis", color="red"
    )
    ax2.hist(equalised_negative, bins=30, alpha=0.7, label="No Diagnosis", color="blue")
    ax2.set_xlabel("Lab Value")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram Equalised Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to {save_path}")


def calculate_theoretical_performance(data: pd.DataFrame) -> dict:
    """
    Calculate the theoretical performance of the model.

    Args:
        data: DataFrame containing the data

    Returns:
        dict: Dictionary containing performance metrics
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
        description="Apply histogram equalisation to lab values in synthetic data"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input synthetic data CSV file (non-normalized)",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--lab_code",
        type=str,
        default="S/LAB1",
        help="Lab code to apply histogram equalisation to",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=N_BINS,
        help=f"Number of bins for histogram equalisation (default: {N_BINS})",
    )

    args = parser.parse_args()

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

    # Apply histogram equalisation
    equalised_data = apply_histogram_equalisation(data, args.lab_code, args.n_bins)

    print("\nData after histogram equalisation:")
    print(equalised_data.head())

    # Print equalised statistics
    print_statistics(equalised_data, "Equalised lab value statistics")
    equalised_performance = calculate_theoretical_performance(equalised_data)

    # Create distribution plots
    save_path = Path(DEFAULT_PLOT_DIR) / f"{SAVE_NAME}_distribution_plot.png"
    create_distribution_plot(
        equalised_data,
        save_path,
        equalised_performance["sweep_auc"],
        " (Histogram Equalised)",
    )

    # Create comparison plot
    comparison_save_path = Path(DEFAULT_PLOT_DIR) / f"{SAVE_NAME}_comparison_plot.png"
    create_comparison_plot(data, equalised_data, comparison_save_path)

    # Write to csv
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    equalised_data.to_csv(write_dir / f"{SAVE_NAME}.csv", index=False)
    print(f"\nSaved histogram equalised data to {write_dir / f'{SAVE_NAME}.csv'}")


if __name__ == "__main__":
    main()
