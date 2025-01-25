import logging
from datetime import datetime
from os.path import join
from typing import List, Tuple, Optional, Set

import pandas as pd
import torch

from corebehrt.classes.outcomes import OutcomeHandler
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

    # Apply initial PID filters
    patients_info = _filter_initial_pids(
        patients_info, initial_pids, cfg.selection.get("exposed_only", False), exposures
    )

    # Apply category filters
    patients_info = _filter_by_categories(patients_info, cfg)

    # Determine index dates
    index_dates = get_index_dates(patients_info, cfg.index_date, exposures)

    # Apply age filtering
    patients_info = filter_by_age(
        patients_info,
        index_dates,
        cfg.selection.get("age_min"),
        cfg.selection.get("age_max"),
    )

    # Apply additional exclusion filters
    patients_info = _apply_exclusion_filters(
        patients_info, index_dates, outcomes, cfg.selection
    )

    return patients_info[PID_COL].unique().tolist(), index_dates


def _filter_initial_pids(
    patients_info: pd.DataFrame,
    initial_pids: Set[str],
    exposed_only: bool,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    """Filter patient info by initial PIDs and exposure status."""
    if initial_pids:
        patients_info = patients_info[patients_info[PID_COL].isin(initial_pids)]

    if exposed_only:
        patients_info = patients_info[patients_info[PID_COL].isin(exposures[PID_COL])]

    return patients_info


def _filter_by_categories(patients_info: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply category-based filtering from configuration."""
    if not cfg.selection.get("categories", False):
        return patients_info

    for col, rule in cfg.selection.categories.items():
        include = rule.get("include")
        exclude = rule.get("exclude")

        if include:
            patients_info = patients_info[patients_info[col].isin(include)]
        elif exclude:
            patients_info = patients_info[~patients_info[col].isin(exclude)]

    return patients_info


def filter_by_age(
    patients_info: pd.DataFrame,
    index_dates: pd.Series,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
) -> pd.DataFrame:
    """Filter patients by age at index date."""
    if min_age is not None:
        patients_info = patients_info[
            (index_dates - patients_info["BIRTHDATE"]) >= min_age
        ]

    if max_age is not None:
        patients_info = patients_info[
            (index_dates - patients_info["BIRTHDATE"]) <= max_age
        ]

    return patients_info


def _apply_exclusion_filters(
    patients_info: pd.DataFrame,
    index_dates: pd.Series,
    outcomes: pd.DataFrame,
    selection_cfg: dict,
) -> pd.DataFrame:
    """Apply exclusion filters based on configuration."""
    # Exclude patients dead before index date
    if selection_cfg.get("dead_before_index_date", False):
        # Keep patients who are either:
        # 1. Not dead (DEATHDATE is NaN) or
        # 2. Dead after index date (DEATHDATE > index_dates)
        patients_info = patients_info[
            patients_info["DEATHDATE"].isna() | (patients_info["DEATHDATE"] > index_dates)
        ]

    # Exclude patients with outcomes before index date
    if selection_cfg.get("outcome_before_index_date", False):
        outcomes_series = pd.Series(
            index=outcomes[PID_COL], data=outcomes[TIMESTAMP_COL]
        )
        outcomes_series = outcomes_series.reindex(patients_info[PID_COL])
        patients_to_exclude = set(outcomes_series[outcomes_series < index_dates].index)
        patients_info = patients_info[~patients_info[PID_COL].isin(patients_to_exclude)]

    return patients_info


def get_index_dates(
    patients_info: pd.DataFrame, index_date: dict, exposures: pd.DataFrame = None
) -> pd.Series:
    """Determine index dates based on configuration."""
    pids = set(patients_info[PID_COL].unique())

    if index_date["mode"] == "absolute":
        index_dates = create_timestamp_series(pids, index_date["absolute"])
    elif index_date["mode"] == "relative":
        n_hours_from_exposure = index_date["relative"]["n_hours_from_exposure"]
        exposed_index_timestamps = _get_index_timestamps_for_exposed(
            pids, n_hours_from_exposure, exposures
        )
        index_dates = OutcomeHandler.draw_index_dates_for_unexposed(
            exposed_index_timestamps, pids
        )

    # Align index dates with patients_info
    return index_dates.reindex(patients_info[PID_COL])


def _get_index_timestamps_for_exposed(
    pids: Set[str], n_hours_from_exposure: int, exposures: pd.DataFrame
) -> pd.Series:
    """Get index timestamps for exposed patients."""
    hours_delta = pd.Timedelta(hours=n_hours_from_exposure)
    exposures = exposures[exposures[PID_COL].isin(pids)]
    return exposures[TIMESTAMP_COL] + hours_delta


def create_timestamp_series(pids: Set[str], timestamp: dict) -> pd.Series:
    """Create a timestamp series for given PIDs."""
    timestamp = datetime(**timestamp)
    return pd.Series(
        data=timestamp, index=pd.Index(list(pids), name=PID_COL), name=TIMESTAMP_COL
    )


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
