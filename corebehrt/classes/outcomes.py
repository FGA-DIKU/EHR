import logging
import operator
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from corebehrt.functional.filter import filter_events_by_abspos
from corebehrt.functional.matching import get_col_booleans
from corebehrt.functional.utils import (
    filter_table_by_pids,
    get_abspos_from_origin_point,
    get_first_event_by_pid,
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
                timestamps["TIMESTAMP"], self.origin_point
            )
            timestamps["abspos"] = timestamps["abspos"].astype(int)
            outcome_tables[outcome] = timestamps
        return outcome_tables

    def match_patient_info(self, patients_info: dict, match: List[List]) -> pd.Series:
        """Get timestamps of interest from patients_info"""
        return patients_info[["PID", match]].dropna()

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


class OutcomeHandler:
    # !TODO: Move this to separate file and make it a constant
    ORIGIN_POINT = datetime(2020, 1, 26)

    def __init__(
        self,
        index_date: Dict[str, int] = None,
        select_patient_group: str = None,
        exclude_pre_followup_outcome_patients: bool = False,
        n_hours_start_followup: int = 0,
    ):
        """
        index_date (optional): default None (indexing based on exposures). If given use same index date for all patients
        select_patient_group (optional): select only exposed or unexposed patients
        exclude_pre_followup_outcome_patients (optional): remove patients with outcome before follow-up start
        n_hours_start_followup (optional): number of hours to start follow-up after exposure (looking for positive label)
        """
        self.index_date = index_date
        self.select_patient_group = select_patient_group
        self.exclude_pre_followup_outcome_patients = (
            exclude_pre_followup_outcome_patients
        )
        self.n_hours_start_followup = n_hours_start_followup

    def handle(
        self,
        data_pids: List[str],
        outcomes: pd.DataFrame,
        exposures: pd.DataFrame,
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        data: Patient Data
        outcomes: DataFrame with outcome timestamps
        exposures: DataFrame with exposure timestamps

        The following steps are taken:
         1. Filter outcomes and censor outcomes to only include patients which are presnet in the Data
         2. Pick earliest exposure timestamp as index_date for the exposed patients
         3. Assign index timestamp to patients without it (save which patients were actually with an exposure)
         4. Optionally select only exposed/unexposed patients
         5. Optinally remove patients with outcome(s) before start of follow-up period
         6. Select first outcome after start of follow-up for each patient
         7. Assign outcome- and index dates to data.
        """
        self.check_input(outcomes, exposures)
        # Step 1: Filter outcomes and index_dates to include only relevant patients
        outcomes = filter_table_by_pids(outcomes, data_pids)
        index_dates = filter_table_by_pids(exposures, data_pids)

        # Step 2: Pick earliest exposure ts as index date for each patient
        index_dates = get_first_event_by_pid(exposures)

        # Step 3 (Optional): Use a specific index date for all
        if self.index_date:
            index_dates = self.compute_abspos_for_index_date(data_pids)

        # Step 5: Assign censoring to patients without it (random assignment)
        logger.info(f"Number of exposed patients: {len(set(index_dates.index))}")
        index_dates = self.draw_index_dates_for_unexposed(index_dates, data_pids)

        # Step 6: Select first outcome after censoring for each patient
        outcomes = self.get_first_outcome_in_follow_up(
            outcomes, index_dates, self.n_hours_start_followup
        )

        # Step 8: Assign outcomes and censor outcomes to data
        index_dates = self.synchronize_patients(data_pids, index_dates)
        outcomes = self.synchronize_patients(data_pids, outcomes)
        return index_dates, outcomes

    def synchronize_patients(
        self, data_pids: List[str], timestamps: pd.Series
    ) -> pd.Series:
        """
        Synchronize patients in timestamps with data.
        timestamps should be indexed by PID.
        PIDs in timestamps that are not present in data will be assigned nan as timestamp.
        """
        logger.info("Synchronizing patients in data with timestamps.")
        timestamps = timestamps.reindex(data_pids)
        timestamps = timestamps.astype(pd.Float64Dtype())
        return timestamps

    def check_input(self, outcomes, exposures):
        """Check that outcomes and exposures have columns PID and abspos."""
        required_columns = {"PID", "abspos"}
        for df, name in [(outcomes, "outcomes"), (exposures, "exposures")]:
            if not required_columns.issubset(set(df.columns)):
                raise ValueError(f"{name} must have columns PID and abspos.")

    @staticmethod
    def draw_index_dates_for_unexposed(
        censoring_timestamps: pd.Series, data_pids: List[str]
    ) -> pd.Series:
        """Draw censor dates for patients that are not in the censor_timestamps."""
        np.random.seed(42)
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        random_abspos = np.random.choice(
            censoring_timestamps.values, size=len(missing_pids)
        )
        new_entries = pd.Series(random_abspos, index=missing_pids)
        censoring_timestamps = pd.concat([censoring_timestamps, new_entries])
        return censoring_timestamps

    def compute_abspos_for_index_date(self, pids: List) -> pd.Series:
        """
        Create a pandas series hlding the same abspos based on self.index_date if not None.
        """
        logger.info(f"Using {self.index_date} as index_date for all patients.")
        index_datetime = datetime(**self.index_date)
        logger.warning(
            f"Using {self.ORIGIN_POINT} as origin point. Make sure is the same as used for feature creation."
        )
        outcome_abspos = get_abspos_from_origin_point(
            [index_datetime], self.ORIGIN_POINT
        )
        return pd.Series(outcome_abspos * len(pids), index=pids)

    def get_first_outcome_in_follow_up(
        self,
        outcomes: pd.DataFrame,
        index_dates: pd.Series,
        n_hours_start_followup: int,
    ) -> pd.Series:
        """Get the first outcome event occurring at or after the follow-up date for each PID.

        Args:
            outcomes: DataFrame containing outcome events with columns PID and abspos
            index_dates: Series mapping PIDs to their index dates (in abspos)
            n_hours_start_followup: Number of hours after index date to start follow-up period

        Returns:
            Series mapping PIDs to abspos of their first outcome event during follow-up.
            Only includes PIDs that had an outcome during follow-up.
        """
        start_followup_date = index_dates + n_hours_start_followup
        filtered_outcomes = filter_events_by_abspos(
            outcomes, start_followup_date, operator.ge
        )
        first_outcome = get_first_event_by_pid(filtered_outcomes)
        return first_outcome
