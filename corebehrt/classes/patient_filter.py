from typing import Optional, Set

import pandas as pd

PID_COL = "PID"
TIMESTAMP_COL = "TIMESTAMP"


def filter_by_initial_pids(
    patients_info: pd.DataFrame,
    initial_pids: Optional[Set[str]] = None,
    exposures: Optional[pd.DataFrame] = None,
    exposed_only: bool = False,
) -> pd.DataFrame:
    """Filter patients by PIDs and exposure status.

    Args:
        patients_info: DataFrame containing patient data
        initial_pids: Set of patient IDs to include
        exposures: DataFrame containing exposure records
        exposed_only: If True, keep only patients with exposures
    """
    if initial_pids:
        patients_info = patients_info[patients_info[PID_COL].isin(initial_pids)]

    if exposed_only and exposures is not None:
        patients_info = patients_info[patients_info[PID_COL].isin(exposures[PID_COL])]

    return patients_info


def filter_by_categories(
    patients_info: pd.DataFrame, category_filters: Optional[dict] = None
) -> pd.DataFrame:
    """Filter patients by categorical variables.

    Args:
        patients_info: DataFrame containing patient data
        category_filters: Dict of {column: {'include': [...]} or {'exclude': [...]}}
    """
    if not category_filters:
        return patients_info

    for col, rule in category_filters.items():
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
    """Filter patients by age range at their index date.

    Args:
        patients_info: DataFrame containing patient data
        index_dates: Series of reference dates for age calculation
        min_age: Minimum age in years (inclusive)
        max_age: Maximum age in years (inclusive)
    """
    if min_age is not None:
        patients_info = patients_info[
            (index_dates - patients_info["BIRTHDATE"]) >= min_age
        ]

    if max_age is not None:
        patients_info = patients_info[
            (index_dates - patients_info["BIRTHDATE"]) <= max_age
        ]

    return patients_info


def apply_exclusion_filters(
    patients_info: pd.DataFrame,
    index_dates: pd.Series,
    outcomes: pd.DataFrame,
    dead_before_index_date: bool = False,
    outcome_before_index_date: bool = False,
) -> pd.DataFrame:
    """Remove patients based on death and outcome criteria.

    Args:
        patients_info: DataFrame containing patient data
        index_dates: Series of reference dates for exclusion
        outcomes: DataFrame containing outcome events
        dead_before_index_date: Exclude patients who died before index
        outcome_before_index_date: Exclude patients with prior outcomes
    """
    if dead_before_index_date:
        patients_info = patients_info[
            patients_info["DEATHDATE"].isna()
            | (patients_info["DEATHDATE"] > index_dates)
        ]

    if outcome_before_index_date:
        outcomes_series = pd.Series(
            index=outcomes[PID_COL], data=outcomes[TIMESTAMP_COL]
        )
        outcomes_series = outcomes_series.reindex(patients_info[PID_COL])
        patients_to_exclude = set(outcomes_series[outcomes_series < index_dates].index)
        patients_info = patients_info[~patients_info[PID_COL].isin(patients_to_exclude)]

    return patients_info
