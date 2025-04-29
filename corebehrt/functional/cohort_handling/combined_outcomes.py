import logging

import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    PID_COL,
    TIMESTAMP_COL,
    TIMESTAMP_SOURCE,
    WINDOW_HOURS_MAX,
    WINDOW_HOURS_MIN,
    PRIMARY,
    SECONDARY,
)

logger = logging.getLogger(__name__)


def find_matches_within_window(
    primary_events: pd.DataFrame,
    secondary_events: pd.DataFrame,
    window_hours_min: int | float,
    window_hours_max: int | float,
    timestamp_source: str = PRIMARY,
) -> pd.DataFrame:
    """Find matches between primary and secondary events within the specified time window."""

    result_rows = []

    # Process each patient
    for pid in primary_events[PID_COL].unique():
        patient_primary = primary_events[primary_events[PID_COL] == pid]
        patient_secondary = secondary_events[secondary_events[PID_COL] == pid]

        # Skip if either event type is missing for this patient
        if len(patient_primary) == 0 or len(patient_secondary) == 0:
            continue

        # For each primary event for this patient
        for _, primary_row in patient_primary.iterrows():
            matches = _get_matches_within_interval(
                primary_row[ABSPOS_COL],
                patient_secondary,
                window_hours_min,
                window_hours_max,
            )

            # If we found matches, add this primary event to the results
            if len(matches) > 0:
                result_row = _create_result_row(primary_row, matches, timestamp_source)
                result_rows.append(result_row)

    # Convert results to DataFrame
    if result_rows:
        return pd.DataFrame(result_rows)
    else:
        return create_empty_results_df()


def _get_matches_within_interval(
    primary_time: float,
    patient_secondary: pd.DataFrame,
    window_hours_min: float | int,
    window_hours_max: float | int,
) -> pd.DataFrame:
    """Get secondary events that match the time window criteria based on direction."""
    return patient_secondary[
        (patient_secondary[ABSPOS_COL] >= primary_time + window_hours_min)
        & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours_max)
    ]


def _create_result_row(
    primary_row: pd.Series, matches: pd.DataFrame, timestamp_source: str
) -> pd.Series:
    """Create a result row based on the primary event and matching secondary events."""
    result_row = primary_row.copy()
    if timestamp_source == "secondary":
        # Use the closest secondary event timestamp
        primary_time = primary_row[ABSPOS_COL]
        closest_idx = (matches[ABSPOS_COL] - primary_time).abs().idxmin()
        closest_match = matches.loc[closest_idx]
        result_row[TIMESTAMP_COL] = closest_match[TIMESTAMP_COL]
        result_row[ABSPOS_COL] = closest_match[ABSPOS_COL]

    return result_row


def create_empty_results_df():
    """Create an empty df with columns: PID_COL, TIMESTAMP_COL, ABSPOS_COL"""
    return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL])


def check_combination_args(args: dict):
    if PRIMARY not in args:
        raise ValueError(f"{PRIMARY} must be defined in the combinations dictionary")
    if SECONDARY not in args:
        raise ValueError(f"{SECONDARY} must be defined in the combinations dictionary")
    timestamp_source = args.get(TIMESTAMP_SOURCE, None)
    if timestamp_source not in {PRIMARY, SECONDARY}:
        logger.warning(
            f"Invalid timestamp_source '{timestamp_source}', falling back to 'primary'"
        )
    window_hours_min = args.get(WINDOW_HOURS_MIN, None)
    if not isinstance(window_hours_min, (int, float)):
        raise ValueError(
            f"{WINDOW_HOURS_MIN} must be a float or in got '{window_hours_min}'"
        )
    window_hours_max = args.get(WINDOW_HOURS_MAX, None)
    if not isinstance(window_hours_max, (int, float)):
        raise ValueError(
            f"{WINDOW_HOURS_MAX} must be a float or in got '{window_hours_max}'"
        )
