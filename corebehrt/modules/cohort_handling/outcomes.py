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
        concepts_plus = filter_table_by_pids(concepts_plus, patient_set)
        patients_info = filter_table_by_pids(patients_info, patient_set)
        concepts_plus = remove_missing_timestamps(concepts_plus)

        outcome_tables = {}
        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]
            if types == "patients_info":
                timestamps = self.match_patient_info(patients_info, matches)
            else:
                timestamps = self.match_concepts(concepts_plus, types, matches, attrs)
            timestamps[ABSPOS_COL] = get_hours_since_epoch(timestamps[TIMESTAMP_COL])
            timestamps[ABSPOS_COL] = timestamps[ABSPOS_COL].astype(int)
            timestamps[PID_COL] = timestamps[PID_COL].astype(int)
            outcome_tables[outcome] = timestamps
        return outcome_tables

    def match_patient_info(self, patients_info: dict, match: List[List]) -> pd.Series:
        """Get timestamps of interest from patients_info"""
        return patients_info[[PID_COL, match]].dropna()

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
        if "exclude" in attrs:
            concepts_plus = concepts_plus[
                ~concepts_plus[CONCEPT_COL].isin(attrs["exclude"])
            ]
        col_booleans = get_col_booleans(
            concepts_plus,
            types,
            matches,
            attrs.get("match_how", "startswith"),
            attrs.get("case_sensitive", True),
        )
        mask = np.bitwise_and.reduce(col_booleans)
        if "negation" in attrs:
            mask = ~mask
        return concepts_plus[mask].drop(columns=[CONCEPT_COL, VALUE_COL])
