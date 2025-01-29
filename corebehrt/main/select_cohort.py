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
    filter_df_by_pids,
)
from corebehrt.common.config import load_config
from corebehrt.common.constants import PID_COL, TIMESTAMP_COL
from corebehrt.common.setup import (
    DirectoryPreparer,
    get_args,
    INDEX_DATES_FILE,
    PID_FILE,
)
from corebehrt.functional.utils import select_first_event
from corebehrt.data.concept_loader import ConceptLoader

logger = logging.getLogger("select_cohort")

CONFIG_PATH = "./corebehrt/configs/select_cohort.yaml"


def main_select_cohort(config_path: str):
    """Execute cohort selection and save results."""
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_select_cohort()

    logger = logging.getLogger("select_cohort")

    logger.info("Starting cohort selection")
    pids, index_dates = select_cohort(cfg)

    logger.info("Saving cohort")
    torch.save(pids, join(cfg.paths.cohort, PID_FILE))
    index_dates.to_csv(join(cfg.paths.cohort, INDEX_DATES_FILE))


def select_cohort(cfg) -> Tuple[List[str], pd.Series]:
    """
    Select cohort by applying multiple filtering steps.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of final patient IDs and their index dates
    """
    logger.info("Loading data")
    patients_info, outcomes, exposures, initial_pids = load_data(cfg)
    patients_info = patients_info.drop_duplicates(subset=PID_COL, keep="first")
    logger.info("N patients_info: %d", len(patients_info))
    logger.info("Patients in initial_pids: %d", len(initial_pids))

    if initial_pids:
        logger.info("Filtering by initial_pids")
        patients_info = filter_df_by_pids(patients_info, initial_pids)

    if cfg.selection.get("exposed_only", False):
        logger.info("Filtering by exposures")
        patients_info = filter_df_by_pids(patients_info, exposures[PID_COL])

    logger.info("Filtering by categories")
    patients_info = filter_by_categories(patients_info, cfg.selection.get("categories"))

    logger.info("Determining index dates")
    mode = cfg.index_date["mode"]
    index_dates = IndexDateHandler.determine_index_dates(
        patients_info,
        mode,
        absolute_timestamp=cfg.index_date[mode].get("absolute_timestamp"),
        n_hours_from_exposure=cfg.index_date[mode].get("n_hours_from_exposure"),
        exposures=exposures,
    )
    patients_info = patients_info.merge(
        index_dates, on=PID_COL
    )  # the TIMESTAMP column is the index date.

    logger.info("Filtering by age")
    patients_info = filter_by_age(
        patients_info,
        min_age=cfg.selection.get("age_min"),
        max_age=cfg.selection.get("age_max"),
    )

    logger.info("Applying additional exclusion filters")
    patients_info = apply_exclusion_filters(
        patients_info,
        outcomes,
        outcome_before_index_date=cfg.selection.get("exclude_prior_outcomes", False),
    )
    return patients_info[PID_COL].unique().tolist(), index_dates


def load_data(cfg):
    """Load patient, outcomes, and exposures data."""
    patients_info = ConceptLoader.read_file(cfg.paths.patients_info)
    outcomes = ConceptLoader.read_file(cfg.paths.outcome)

    exposures = (
        ConceptLoader.read_file(cfg.paths.exposure)
        if cfg.paths.get("exposure", False)
        else outcomes
    )

    exposures = select_first_event(exposures, PID_COL, TIMESTAMP_COL)

    initial_pids = (
        torch.load(cfg.paths.initial_pids)
        if cfg.paths.get("initial_pids", False)
        else []
    )

    return patients_info, outcomes, exposures, initial_pids


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_select_cohort(args.config_path)
