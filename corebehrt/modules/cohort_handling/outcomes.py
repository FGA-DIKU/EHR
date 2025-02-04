import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from corebehrt.common.constants import PID_COL, TIMESTAMP_COL
from corebehrt.functional.matching import get_col_booleans
from corebehrt.functional.utils import (
    filter_table_by_pids,
    get_abspos_from_origin_point,
    remove_missing_timestamps,
)

logger = logging.getLogger(__name__)


class OutcomeMaker:
    def __init__(self, outcomes: dict, origin_point: dict):
        self.outcomes = outcomes
        self.origin_point = self.get_origin_point(origin_point)

    def get_origin_point(self, origin_point: dict) -> datetime:
        if isinstance(origin_point, datetime):
            return origin_point
        return datetime(**origin_point)

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
            timestamps["abspos"] = get_abspos_from_origin_point(
                timestamps[TIMESTAMP_COL], self.origin_point
            )
            timestamps["abspos"] = timestamps["abspos"].astype(int)
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
                ~concepts_plus["CONCEPT"].isin(attrs["exclude"])
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
        return concepts_plus[mask].drop(columns=["ADMISSION_ID", "CONCEPT"])
