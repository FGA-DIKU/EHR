import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
)
from corebehrt.functional.cohort_handling.matching import get_col_booleans
from corebehrt.functional.preparation.filter import (
    filter_table_by_pids,
    remove_missing_timestamps,
)
from corebehrt.functional.utils.time import get_hours_since_epoch

logger = logging.getLogger(__name__)


class OutcomeMaker:
    def __init__(self, outcomes: dict):
        self.outcomes = outcomes

    def __call__(
        self,
        concepts_plus: pd.DataFrame,
        patients_info: pd.DataFrame,
        patient_set: List[str],
    ) -> dict:
        """Create outcomes from concepts_plus and patients_info"""
        # Convert patient IDs to the right type for filtering
        patient_ids = [int(pid) for pid in patient_set]
        concepts_plus = filter_table_by_pids(concepts_plus, patient_ids)
        patients_info = filter_table_by_pids(patients_info, patient_ids)
        concepts_plus = remove_missing_timestamps(concepts_plus)

        outcome_tables = {}
        for outcome, attrs in self.outcomes.items():
            # Handle combination outcomes
            if "combinations" in attrs:
                timestamps = self.match_combinations(
                    concepts_plus, attrs["combinations"]
                )
            # Handle traditional outcomes
            else:
                types = attrs["type"]
                matches = attrs["match"]
                if types == "patients_info":
                    timestamps = self.match_patient_info(patients_info, matches)
                else:
                    timestamps = self.match_concepts(
                        concepts_plus, types, matches, attrs
                    )

            # Only process if we have data
            if len(timestamps) > 0:
                timestamps[ABSPOS_COL] = get_hours_since_epoch(
                    timestamps[TIMESTAMP_COL]
                )
                timestamps[ABSPOS_COL] = timestamps[ABSPOS_COL].astype(int)
                timestamps[PID_COL] = timestamps[PID_COL].astype(int)

            outcome_tables[outcome] = timestamps
        return outcome_tables

    def match_patient_info(
        self, patients_info: pd.DataFrame, match: str
    ) -> pd.DataFrame:
        """Get timestamps of interest from patients_info"""
        # Handle case when patients_info is empty
        if len(patients_info) == 0:
            return pd.DataFrame(columns=[PID_COL, match, TIMESTAMP_COL])
        # Ensure match column contains timestamps
        if not pd.api.types.is_datetime64_any_dtype(patients_info[match]):
            logger.warning(
                f"Column {match} does not contain timestamps, returning empty DataFrame"
            )
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL])

        # Rename match column to TIMESTAMP_COL for consistency
        patients_info = patients_info.rename(columns={match: TIMESTAMP_COL})
        return patients_info[[PID_COL, TIMESTAMP_COL]].dropna()

    def match_concepts(
        self,
        concepts_plus: pd.DataFrame,
        types: List[List],
        matches: List[List],
        attrs: Dict,
    ) -> pd.DataFrame:
        """It first goes through all the types and returns true for a row if the entry starts with any of the matches.
        We then ensure all the types are true for a row by using bitwise_and.reduce. E.g. CONCEPT==COVID_TEST AND VALUE==POSITIVE
        """
        # Handle empty DataFrame
        if len(concepts_plus) == 0:
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL])

        # Make a copy to avoid modifying the original
        filtered_concepts = concepts_plus.copy()

        if "exclude" in attrs:
            filtered_concepts = filtered_concepts[
                ~filtered_concepts[CONCEPT_COL].isin(attrs["exclude"])
            ]

        col_booleans = get_col_booleans(
            filtered_concepts,
            types,
            matches,
            attrs.get("match_how", "startswith"),
            attrs.get("case_sensitive", True),
        )

        if len(col_booleans) == 0:
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL])

        mask = np.bitwise_and.reduce(col_booleans)

        if "negation" in attrs:
            mask = ~mask

        result = filtered_concepts[mask]
        if len(result) > 0:
            return result.drop(columns=[CONCEPT_COL, VALUE_COL])
        else:
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL])

    def match_combinations(
        self,
        concepts_plus: pd.DataFrame,
        combinations: Dict,
    ) -> pd.DataFrame:
        """Match combinations of codes that occur within a specific time window of each other.

        Args:
            concepts_plus: DataFrame containing concepts
            combinations: Dictionary defining the combinations to match
                Example: {
                    "primary": {"type": ["code"], "match": [["DOD"]]},
                    "secondary": {"type": ["code"], "match": [["DI20"]]},
                    "window_hours": 24,
                    "direction": "any" # or "before", "after"
                }

        Returns:
            DataFrame with timestamps of matched combinations
        """
        # Handle empty input DataFrame
        if len(concepts_plus) == 0:
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL])

        # Get primary events
        primary_config = combinations["primary"]
        primary_events = self.match_concepts(
            concepts_plus,
            primary_config["type"],
            primary_config["match"],
            {k: v for k, v in primary_config.items() if k not in ["type", "match"]},
        )

        # Get secondary events
        secondary_config = combinations["secondary"]
        secondary_events = self.match_concepts(
            concepts_plus,
            secondary_config["type"],
            secondary_config["match"],
            {k: v for k, v in secondary_config.items() if k not in ["type", "match"]},
        )

        # Return empty DataFrame if either set of events is empty
        if len(primary_events) == 0 or len(secondary_events) == 0:
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL])

        # Add absolute positions for time window comparison
        primary_events[ABSPOS_COL] = get_hours_since_epoch(
            primary_events[TIMESTAMP_COL]
        )
        secondary_events[ABSPOS_COL] = get_hours_since_epoch(
            secondary_events[TIMESTAMP_COL]
        )

        # Define the time window
        window_hours = combinations.get(
            "window_hours", 24
        )  # Default to 24 hours if not specified
        direction = combinations.get(
            "direction", "any"
        )  # Default to "any" direction if not specified

        # Initialize an empty DataFrame for the results
        result_rows = []

        # For each patient
        for pid in primary_events[PID_COL].unique():
            patient_primary = primary_events[primary_events[PID_COL] == pid]
            patient_secondary = secondary_events[secondary_events[PID_COL] == pid]

            # Skip if either event type is missing for this patient
            if len(patient_primary) == 0 or len(patient_secondary) == 0:
                continue

            # For each primary event for this patient
            for _, primary_row in patient_primary.iterrows():
                primary_time = primary_row[ABSPOS_COL]

                # Find secondary events within the time window
                if direction == "any":
                    # Secondary can be before or after primary
                    matches = patient_secondary[
                        (patient_secondary[ABSPOS_COL] >= primary_time - window_hours)
                        & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours)
                    ]
                elif direction == "before":
                    # Secondary must be before primary
                    matches = patient_secondary[
                        (patient_secondary[ABSPOS_COL] >= primary_time - window_hours)
                        & (patient_secondary[ABSPOS_COL] <= primary_time)
                    ]
                elif direction == "after":
                    # Secondary must be after primary
                    matches = patient_secondary[
                        (patient_secondary[ABSPOS_COL] >= primary_time)
                        & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours)
                    ]
                else:
                    logger.warning(
                        f"Unknown direction '{direction}', defaulting to 'any'"
                    )
                    matches = patient_secondary[
                        (patient_secondary[ABSPOS_COL] >= primary_time - window_hours)
                        & (patient_secondary[ABSPOS_COL] <= primary_time + window_hours)
                    ]

                # If we found matches, add this primary event to the results
                if len(matches) > 0:
                    # By default, use the primary event timestamp
                    result_row = primary_row.copy()

                    # Optionally, if specified in config, use a specific timestamp
                    timestamp_source = combinations.get("timestamp_source", "primary")
                    if timestamp_source == "secondary":
                        # Use the closest secondary event timestamp
                        closest_idx = (
                            (matches[ABSPOS_COL] - primary_time).abs().idxmin()
                        )
                        closest_match = matches.loc[closest_idx]
                        result_row[TIMESTAMP_COL] = closest_match[TIMESTAMP_COL]
                        result_row[ABSPOS_COL] = closest_match[ABSPOS_COL]

                    result_rows.append(result_row)

        # Convert results to DataFrame
        if result_rows:
            return pd.DataFrame(result_rows)
        else:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL, ABSPOS_COL])
