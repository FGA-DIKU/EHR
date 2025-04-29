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
from corebehrt.functional.cohort_handling.combined_outcomes import (
    create_empty_results_df,
    find_matches_within_window,
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
            return create_empty_results_df()

        # Get primary and secondary events
        primary_events = self.get_events(concepts_plus, combinations["primary"])
        secondary_events = self.get_events(concepts_plus, combinations["secondary"])

        # Return empty DataFrame if either set of events is empty
        if len(primary_events) == 0 or len(secondary_events) == 0:
            return create_empty_results_df()

        # Add absolute positions for time window comparison
        primary_events[ABSPOS_COL] = get_hours_since_epoch(
            primary_events[TIMESTAMP_COL]
        )
        secondary_events[ABSPOS_COL] = get_hours_since_epoch(
            secondary_events[TIMESTAMP_COL]
        )

        # Find events within the time window
        return find_matches_within_window(
            primary_events, secondary_events, combinations
        )

    def get_events(self, concepts_plus: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Extract events from concepts based on configuration."""
        extra_params = {k: v for k, v in config.items() if k not in ["type", "match"]}
        return self.match_concepts(
            concepts_plus, config["type"], config["match"], extra_params
        )
