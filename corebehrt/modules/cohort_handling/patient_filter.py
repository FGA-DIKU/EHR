from typing import Optional

import pandas as pd

from corebehrt.constants.data import (
    PID_COL,
    TIMESTAMP_COL,
    BIRTHDATE_COL,
    DEATHDATE_COL,
)


def filter_by_categories(
    patients_info: pd.DataFrame, category_filters: Optional[dict] = None
) -> pd.DataFrame:
    """Filter patients by categorical variables.

    Args:
        patients_info: DataFrame with category columns
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
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
) -> pd.DataFrame:
    """Filter patients by age range at their index date.

    Args:
        patients_info: DataFrame with BIRTHDATE_COL and TIMESTAMP_COL (index date) columns in datetime format
        min_age: Minimum age in years (inclusive)
        max_age: Maximum age in years (inclusive)
    """
    age_in_years = (
        patients_info[TIMESTAMP_COL] - patients_info[BIRTHDATE_COL]
    ).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    if min_age is not None:
        patients_info = patients_info.loc[age_in_years >= min_age]

    if max_age is not None:
        patients_info = patients_info.loc[age_in_years <= max_age]

    return patients_info


def filter_by_death(patients_info: pd.DataFrame) -> pd.DataFrame:
    """Remove patients who died before their index date.

    Args:
        patients_info: DataFrame with DEATHDATE_COL and TIMESTAMP_COL columns in datetime format
    """
    return patients_info[
        patients_info[DEATHDATE_COL].isna()
        | (patients_info[DEATHDATE_COL] > patients_info[TIMESTAMP_COL])
    ]


def filter_by_prior_outcomes(
    patients_info: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Remove patients who had outcomes before their index date.

    Args:
        patients_info: DataFrame with PID_COL and TIMESTAMP_COL columns
        outcomes: DataFrame with PID_COL and TIMESTAMP_COL columns in datetime format
    """
    if outcomes.empty:
        return patients_info

    earliest_outcomes = outcomes.groupby(PID_COL)[TIMESTAMP_COL].min()
    earliest_outcomes = earliest_outcomes.reindex(patients_info[PID_COL].unique())

    indexed_patients_info = patients_info.set_index(PID_COL, drop=False)

    pids_to_exclude = indexed_patients_info.index[
        (earliest_outcomes < indexed_patients_info[TIMESTAMP_COL])
    ]

    return patients_info[~patients_info[PID_COL].isin(pids_to_exclude)]


def filter_df_by_pids(df: pd.DataFrame, pids: list) -> pd.DataFrame:
    """Filter using PID column in dataframe."""
    return df[df[PID_COL].isin(pids)]


def exclude_pids_from_df(df: pd.DataFrame, pids: list) -> pd.DataFrame:
    """Exclude using PID column in dataframe."""
    return df[~df[PID_COL].isin(pids)]
