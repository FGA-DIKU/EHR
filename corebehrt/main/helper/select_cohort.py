from typing import List, Tuple

import pandas as pd
import torch

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.preparation.filter import select_first_event
from corebehrt.modules.cohort_handling.index_dates import IndexDateHandler
from corebehrt.modules.cohort_handling.patient_filter import (
    exclude_pids_from_df,
    filter_by_age,
    filter_by_categories,
    filter_by_death,
    filter_by_prior_outcomes,
    filter_df_by_pids,
)
from corebehrt.modules.features.loader import ConceptLoader


def select_cohort(cfg, logger) -> Tuple[List[str], pd.Series]:
    """
    Select cohort by applying multiple filtering steps.


    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of final patient IDs and their index dates
    """
    logger.info("Loading data")
    patients_info, outcomes, exposures, initial_pids, exclude_pids = load_data(cfg)
    patients_info = patients_info.drop_duplicates(subset=PID_COL, keep="first")
    logger.info("N patients_info: %d", len(patients_info))
    logger.info("Patients in initial_pids: %d", len(initial_pids))

    if initial_pids:
        logger.info("Filtering by initial_pids")
        patients_info = filter_df_by_pids(patients_info, initial_pids)
        log_patient_num(logger, patients_info)

    if exclude_pids:
        logger.info("Filtering by exclude_pids")
        patients_info = exclude_pids_from_df(patients_info, exclude_pids)
        log_patient_num(logger, patients_info)

    if cfg.selection.get("exposed_only", False):
        logger.info("Filtering by exposures")

        patients_info = filter_df_by_pids(patients_info, exposures[PID_COL])
        log_patient_num(logger, patients_info)

    if cfg.selection.get("categories", False):
        logger.info("Filtering by categories")
        patients_info = filter_by_categories(patients_info, cfg.selection.categories)
        log_patient_num(logger, patients_info)

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

    if cfg.selection.get("age", False):
        logger.info("Filtering by age")
        patients_info = filter_by_age(
            patients_info,
            min_age=cfg.selection.age.get("min_years"),
            max_age=cfg.selection.age.get("max_years"),
        )
        log_patient_num(logger, patients_info)

    logger.info("Applying additional exclusion filters")
    patients_info = filter_by_death(patients_info)
    log_patient_num(logger, patients_info)

    if cfg.selection.get("exclude_prior_outcomes", False):
        logger.info("Filtering by prior outcomes")
        patients_info = filter_by_prior_outcomes(patients_info, outcomes)
        log_patient_num(logger, patients_info)
    return patients_info[PID_COL].unique().tolist(), index_dates


def log_patient_num(logger, patients_info):
    logger.info(f"Patient number: {len(patients_info[PID_COL].unique())}")


def load_data(
    cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
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

    exclude_pids = (
        torch.load(cfg.paths.exclude_pids)
        if cfg.paths.get("exclude_pids", False)
        else []
    )

    return patients_info, outcomes, exposures, initial_pids, exclude_pids
