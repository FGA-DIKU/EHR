import logging
from typing import Dict

import pandas as pd

from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL

logger = logging.getLogger(__name__)


def find_matches_within_window(
    primary_events: pd.DataFrame, secondary_events: pd.DataFrame, combinations: Dict
) -> pd.DataFrame:
    """Find matches between primary and secondary events within the specified time window."""
    window_hours = combinations.get("window_hours", 24)  # Default: 24 hours
    direction = combinations.get("direction", "any")  # Default: "any" direction
    timestamp_source = combinations.get("timestamp_source", "primary")

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
            matches = _get_matches_by_direction(
                primary_row[ABSPOS_COL], patient_secondary, window_hours, direction
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


def _get_matches_by_direction(
    primary_time: float,
    patient_secondary: pd.DataFrame,
    window_hours: int,
    direction: str,
) -> pd.DataFrame:
    """Get secondary events that match the time window criteria based on direction."""
    if direction == "any":
        # Secondary can be before or after primary
        return patient_secondary[
            (patient_secondary[ABSPOS_COL] >= primary_time - window_hours)
            & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours)
        ]
    elif direction == "before":
        # Secondary must be before primary
        return patient_secondary[
            (patient_secondary[ABSPOS_COL] >= primary_time - window_hours)
            & (patient_secondary[ABSPOS_COL] <= primary_time)
        ]
    elif direction == "after":
        # Secondary must be after primary
        return patient_secondary[
            (patient_secondary[ABSPOS_COL] >= primary_time)
            & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours)
        ]
    else:
        logger.warning(f"Unknown direction '{direction}', defaulting to 'any'")
        return patient_secondary[
            (patient_secondary[ABSPOS_COL] >= primary_time - window_hours)
            & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours)
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
