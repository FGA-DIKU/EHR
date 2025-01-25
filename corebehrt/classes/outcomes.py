import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from corebehrt.functional.matching import get_col_booleans
from corebehrt.functional.utils import (
    filter_table_by_pids,
    get_abspos_from_origin_point,
    remove_missing_timestamps,
)

logger = logging.getLogger(__name__)

PID_COL = "PID"
TIMESTAMP_COL = "TIMESTAMP"


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


def get_binary_outcomes(
    index_dates: pd.Series,
    outcomes: pd.DataFrame,
    start_follow_up: float = 0,
    end_follow_up: float = None,
):
    """Get binary outcomes for each patient."""
    raise NotImplementedError


class IndexDateHandler:
    @staticmethod
    def create_timestamp_series(pids: Set[str], timestamp: dict) -> pd.Series:
        """Create a timestamp series for given PIDs."""
        timestamp = datetime(**timestamp)
        return pd.Series(
            data=timestamp, index=pd.Index(list(pids), name=PID_COL), name=TIMESTAMP_COL
        )

    @staticmethod
    def get_index_timestamps_for_exposed(
        pids: Set[str], n_hours_from_exposure: int, exposures: pd.DataFrame
    ) -> pd.Series:
        """Get index timestamps for exposed patients."""
        hours_delta = pd.Timedelta(hours=n_hours_from_exposure)
        exposures = filter_table_by_pids(exposures, pids)
        return exposures[TIMESTAMP_COL] + hours_delta

    @staticmethod
    def draw_index_dates_for_unexposed(
        censoring_timestamps: pd.Series, data_pids: List[str]
    ) -> pd.Series:
        """Draw censor dates for patients not in censor_timestamps."""
        np.random.seed(42)
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        random_abspos = np.random.choice(
            censoring_timestamps.values, size=len(missing_pids)
        )
        new_entries = pd.Series(random_abspos, index=missing_pids)
        return pd.concat([censoring_timestamps, new_entries])

    @classmethod
    def determine_index_dates(
        cls,
        patients_info: pd.DataFrame,
        index_date_mode: str,
        index_date_params: dict,
        exposures: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Determine index dates based on mode."""
        pids = set(patients_info[PID_COL].unique())

        if index_date_mode == "absolute":
            return cls.create_timestamp_series(pids, index_date_params)

        if index_date_mode == "relative":
            n_hours = index_date_params["n_hours_from_exposure"]
            exposed_timestamps = cls.get_index_timestamps_for_exposed(
                pids, n_hours, exposures
            )
            return cls.draw_index_dates_for_unexposed(exposed_timestamps, pids)

        raise ValueError(f"Unsupported index date mode: {index_date_mode}")
