import logging
from os.path import join
from typing import List, Tuple

import pandas as pd
import torch

from corebehrt.classes.outcomes import IndexDateHandler
from corebehrt.classes.patient_filter import (
    apply_exclusion_filters,
    filter_by_age,
    filter_by_categories,
    filter_by_initial_pids,
)
from corebehrt.common.config import load_config
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.functional.utils import select_first_event

CONFIG_PATH = "./corebehrt/configs/select_cohort.yaml"
# Constants
PID_COL = "PID"
TIMESTAMP_COL = "TIMESTAMP"


def main_select_cohort(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_select_cohort()

    logger = logging.getLogger("select_cohort")
    logger.info("Starting cohort selection")

    pids, index_dates = select_cohort(cfg)
    logger.info("Finished cohort selection")

    torch.save(pids, join(cfg.paths.cohort, "pids.pt"))
    torch.save(index_dates, join(cfg.paths.cohort, "index_dates.pt"))


def select_cohort(cfg) -> Tuple[List[str], pd.Series]:
    """
    Select cohort by applying multiple filtering steps.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of final patient IDs and their index dates
    """
    # Load initial data
    patients_info, outcomes, exposures, initial_pids = load_data(cfg)

    # Instantiate selector and index date handler

    # Apply initial PID filters
    patients_info = filter_by_initial_pids(
        patients_info,
        initial_pids,
        exposures,
        exposed_only=cfg.selection.get("exposed_only", False),
    )

    # Apply category filters
    patients_info = filter_by_categories(patients_info, cfg.selection.get("categories"))

    # Determine index dates
    index_dates = IndexDateHandler.determine_index_dates(
        patients_info,
        cfg.index_date["mode"],
        cfg.index_date.get("absolute", cfg.index_date.get("relative")),
        exposures,
    )

    # Apply age filtering
    patients_info = filter_by_age(
        patients_info,
        index_dates,
        min_age=cfg.selection.get("age_min"),
        max_age=cfg.selection.get("age_max"),
    )

    # Apply additional exclusion filters
    patients_info = apply_exclusion_filters(
        patients_info,
        index_dates,
        outcomes,
        dead_before_index_date=cfg.selection.get("dead_before_index_date", False),
        outcome_before_index_date=cfg.selection.get("outcome_before_index_date", False),
    )

    return patients_info["PID"].unique().tolist(), index_dates


def load_data(cfg):
    """Load patient, outcomes, and exposures data."""
    patients_info = read_table(
        cfg.paths.patients_info, parse_dates=["BIRTHDATE", "DEATHDATE"]
    )
    outcomes = read_table(cfg.paths.outcome, parse_dates=[TIMESTAMP_COL])

    exposures = (
        read_table(cfg.paths.exposure, parse_dates=[TIMESTAMP_COL])
        if cfg.paths.get("exposure", False)
        else outcomes
    )

    exposures = select_first_event(exposures, PID_COL, TIMESTAMP_COL)

    initial_pids = (
        set(torch.load(cfg.paths.initial_pids))
        if cfg.paths.get("initial_pids", False)
        else set()
    )

    return patients_info, outcomes, exposures, initial_pids


def read_table(path: str, parse_dates: List[str] = []) -> pd.DataFrame:
    """Read table from CSV or Parquet file."""
    if path.endswith(".csv"):
        return pd.read_csv(path, parse_dates=parse_dates)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
