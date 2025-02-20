import logging
from datetime import datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.functional.preparation.filter import filter_table_by_pids

logger = logging.getLogger(__name__)


class IndexDateHandler:
    @staticmethod
    def create_timestamp_series(pids: Set[str], timestamp: datetime) -> pd.Series:
        """Create a timestamp series for given PIDs."""
        return pd.Series(
            data=timestamp,
            index=pd.Index(list(pids), name=PID_COL),
            name=TIMESTAMP_COL,
        )

    @staticmethod
    def get_index_timestamps_for_exposed(
        pids: Set[str], n_hours_from_exposure: int, exposures: pd.DataFrame
    ) -> pd.Series:
        """Get index timestamps for exposed patients."""
        hours_delta = pd.Timedelta(hours=n_hours_from_exposure)
        exposures = filter_table_by_pids(exposures, pids)
        # Set PID as index and get timestamp series
        result = exposures.set_index(PID_COL)[TIMESTAMP_COL] + hours_delta
        result.index.name = PID_COL  # Ensure index name is set to PID_COL
        return result

    @staticmethod
    def draw_index_dates_for_unexposed(
        data_pids: List[str],
        censoring_timestamps: pd.Series,
    ) -> pd.Series:
        """Draw censor dates for patients not in censor_timestamps."""
        np.random.seed(42)
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        random_abspos = np.random.choice(
            censoring_timestamps.values, size=len(missing_pids)
        )
        new_entries = pd.Series(
            random_abspos, index=pd.Index(list(missing_pids), name=PID_COL)
        )
        result = pd.concat([censoring_timestamps, new_entries])
        result.index.name = (
            PID_COL  # Ensure the final concatenated series has PID_COL as index name
        )
        return result

    @classmethod
    def determine_index_dates(
        cls,
        patients_info: pd.DataFrame,
        index_date_mode: str,
        *,  # force keyword arguments,
        absolute_timestamp: Optional[dict] = None,
        n_hours_from_exposure: Optional[int] = None,
        exposures: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Determine index dates based on mode.
        Args:
            patients_info: pd.DataFrame with patients info
            index_date_mode: str, "absolute" or "relative"
            absolute_timestamp: dict with year, month, day (required if index_date_mode == "absolute")
            n_hours_from_exposure: int (required if index_date_mode == "relative")
            exposures: pd.DataFrame (required if index_date_mode == "relative")
        """
        pids = set(patients_info[PID_COL].unique())

        result = None
        if index_date_mode == "absolute":
            absolute_timestamp = datetime(**absolute_timestamp)
            result = cls.create_timestamp_series(pids, absolute_timestamp)
        elif index_date_mode == "relative":
            n_hours = n_hours_from_exposure or 0
            exposed_timestamps = cls.get_index_timestamps_for_exposed(
                pids, n_hours, exposures
            )
            result = cls.draw_index_dates_for_unexposed(pids, exposed_timestamps)
        else:
            raise ValueError(f"Unsupported index date mode: {index_date_mode}")

        # Ensure the series has both index name and series name
        result.index.name = PID_COL
        result.name = TIMESTAMP_COL
        return result
