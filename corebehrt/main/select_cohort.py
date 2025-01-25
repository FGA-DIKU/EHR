"""
Input: Formatted Data
- Load patients_info, exposures, outcomes
- Exclude by sex, age, dead before index date, outcome before index date
"""

import logging
from datetime import datetime
from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from corebehrt.classes.outcomes import OutcomeHandler
from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.functional.filter import filter_by_column_rule
from corebehrt.functional.utils import select_first_event

CONFIG_PATH = "./corebehrt/configs/select_cohort.yaml"

PID_COL = "PID"
TIMESTAMP_COL = "TIMESTAMP"


def main_select_cohort(config_path):
    """
    Loads data
    Finds outcomes
    Creates features
    Handles wrong data
    Excludes patients with <k concepts
    Splits data
    Tokenizes
    Saves
    """
    cfg = load_config(config_path)

    DirectoryPreparer(cfg).setup_select_cohort()

    logger = logging.getLogger("select_cohort")
    logger.info("Starting cohort selection")

    pids, index_dates = select_cohort(cfg)
    logger.info("Finished cohort selection")

    torch.save(pids, join(cfg.paths.cohort, "pids.pt"))
    torch.save(index_dates, join(cfg.paths.cohort, "index_dates.pt"))


def split_pids_into_train_val_test(
    pids: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Split pids into train, val, test
    Return dict with keys train, val, test and values as lists of pids

    Parameters
    ----------
    pids : list
        List of patient IDs to split
    train_ratio : float, optional
        Ratio of patients to use for training, by default 0.7
    val_ratio : float, optional
        Ratio of patients to use for validation, by default 0.15
    test_ratio : float, optional
        Ratio of patients to use for testing, by default 0.15

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test' and values as lists of pids
    """
    # Check that ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle pids
    pids = np.array(pids)
    np.random.shuffle(pids)

    # Calculate split indices
    n_samples = len(pids)
    train_idx = int(n_samples * train_ratio)
    val_idx = train_idx + int(n_samples * val_ratio)

    # Split pids
    train_pids = pids[:train_idx].tolist()
    val_pids = pids[train_idx:val_idx].tolist()
    test_pids = pids[val_idx:].tolist()

    return {"train": train_pids, "val": val_pids, "test": test_pids}


def select_cohort(cfg) -> Tuple[list, dict]:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    patients_info, outcomes, exposures, initial_pids = load_data(cfg)

    if initial_pids:
        patients_info = select_pids(patients_info, initial_pids)

    if cfg.selection.get("exposed_only", False):
        patients_info = select_pids(patients_info, exposures[PID_COL].unique())

    # Filter by sex, hospital, etc.
    if cfg.selection.get("categories", False):
        patients_info = filter_by_column_rules(patients_info, cfg)

    index_dates = get_index_dates(patients_info, cfg.index_date, exposures)

    patients_info = filter_by_age(
        patients_info,
        index_dates,
        cfg.selection.get("age_min", None),
        cfg.selection.get("age_max", None),
    )

    if cfg.selection.get("dead_before_index_date", False):
        patients_info = patients_info[patients_info["DEATHDATE"] > index_dates]

    if cfg.selection.get("outcome_before_index_date", False):
        # align outcomes with patients_info
        outcomes = pd.Series(index=outcomes[PID_COL], data=outcomes["TIMESTAMP"])
        outcomes = outcomes.reindex(patients_info[PID_COL])
        outcomes = outcomes[outcomes < index_dates]
        patients_to_exclude = set(outcomes.index)
        patients_info = patients_info[~patients_info[PID_COL].isin(patients_to_exclude)]

    final_pids = patients_info[PID_COL].unique().tolist()
    return final_pids, index_dates


def filter_by_age(
    patients_info: pd.DataFrame,
    index_dates: pd.Series,
    min_age: int = None,
    max_age: int = None,
) -> pd.DataFrame:
    if min_age:
        patients_info = patients_info[
            (index_dates - patients_info["BIRTHDATE"]) >= min_age
        ]
    if max_age:
        patients_info = patients_info[
            (index_dates - patients_info["BIRTHDATE"]) <= max_age
        ]
    return patients_info


def filter_by_column_rules(patients_info: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    for col, rule in cfg.selection.categories.items():
        patients_info = filter_by_column_rule(
            patients_info,
            column=col,
            include_values=rule.get("include"),
            exclude_values=rule.get("exclude"),
        )
    return patients_info


def select_pids(patients_info: pd.DataFrame, pids: List[str]) -> pd.DataFrame:
    pids = set(pids)
    return patients_info[patients_info[PID_COL].isin(pids)]


def load_data(cfg):
    patients_info = read_table(
        cfg.paths.patients_info, parse_dates=["BIRTHDATE", "DEATHDATE"]
    )
    outcomes = read_table(cfg.paths.outcome, parse_dates=[TIMESTAMP_COL])
    if cfg.paths.get("exposure", False):
        exposures = read_table(cfg.paths.exposure, parse_dates=[TIMESTAMP_COL])
    else:
        exposures = outcomes
    exposures = select_first_event(exposures, PID_COL, TIMESTAMP_COL)
    initial_pids = (
        set(torch.load(cfg.paths.initial_pids))
        if cfg.paths.get("initial_pids", False)
        else set()
    )
    return patients_info, outcomes, exposures, initial_pids


def get_index_dates(
    patients_info: pd.DataFrame, index_date: dict, exposures: pd.DataFrame = None
) -> pd.Series:
    pids = set(patients_info[PID_COL].unique())
    if index_date["mode"] == "absolute":
        index_dates = create_timestamp_series(pids, index_date["absolute"])
    elif index_date["mode"] == "relative":
        n_hours_from_exposure = index_date["relative"]["n_hours_from_exposure"]
        exposed_index_timestamps = get_index_timestamps_for_exposed(
            pids, n_hours_from_exposure, exposures
        )
        index_dates = OutcomeHandler.draw_index_dates_for_unexposed(
            exposed_index_timestamps, pids
        )
    # align index dates with patients_info
    index_dates = index_dates.reindex(patients_info[PID_COL])
    return index_dates


def get_index_timestamps_for_exposed(
    pids: set, n_hours_from_exposure: int, exposures: pd.DataFrame
):
    # Convert hours to timedelta
    hours_delta = pd.Timedelta(hours=n_hours_from_exposure)
    exposures = exposures[exposures[PID_COL].isin(pids)]
    index_dates = exposures["TIMESTAMP"] + hours_delta
    return index_dates


def create_timestamp_series(pids: set, timestamp: dict) -> pd.Series:
    """
    Create a Series with PID index and timestamp values.

    Args:
        pids: Set of patient IDs
        timestamp: Single timestamp value for all entries

    Returns:
        pandas Series with PID index and timestamp values
    """
    timestamp = datetime(**timestamp)
    return pd.Series(
        data=timestamp, index=pd.Index(list(pids), name="PID"), name="TIMESTAMP"
    )


def read_table(path: str, parse_dates: list[str] = []) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, parse_dates=parse_dates)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
