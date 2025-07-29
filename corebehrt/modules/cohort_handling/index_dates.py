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
        result = exposures.set_index(PID_COL)[TIMESTAMP_COL] + hours_delta
        result.index.name = PID_COL
        return result

    @staticmethod
    def _ensure_series_format(data: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure data is in Series format with PID as index."""
        if isinstance(data, pd.DataFrame):
            return data.set_index(PID_COL)[TIMESTAMP_COL]
        return data

    @staticmethod
    def draw_index_dates_for_unexposed_w_secondary_censoring(
        data_pids: List[str],
        censoring_timestamps: pd.Series,
        secondary_censoring_timestamps: pd.Series,
    ) -> pd.Series:
        """
        Draw censor dates for patients not in censoring_timestamps.
        Uses secondary censoring timestamps for missing patients when available.
        """
        np.random.seed(42)
        
        # Ensure proper Series format
        censoring_timestamps = IndexDateHandler._ensure_series_format(censoring_timestamps)
        secondary_censoring_timestamps = IndexDateHandler._ensure_series_format(secondary_censoring_timestamps)
        
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        result = censoring_timestamps.copy()
        
        if missing_pids:
            # Get patients that have secondary censoring timestamps
            secondary_pids = set(secondary_censoring_timestamps.index) & missing_pids
            
            # Use secondary timestamps directly for patients that have them
            for pid in secondary_pids:
                result[pid] = secondary_censoring_timestamps.loc[pid]
            
            # For remaining missing patients, draw randomly from secondary censoring dates
            remaining_missing = missing_pids - secondary_pids
            if remaining_missing and len(secondary_censoring_timestamps) > 0:
                random_secondary_dates = np.random.choice(
                    secondary_censoring_timestamps.values, 
                    size=len(remaining_missing),
                    replace=True
                )
                new_entries = pd.Series(
                    random_secondary_dates, 
                    index=pd.Index(list(remaining_missing), name=PID_COL)
                )
                result = pd.concat([result, new_entries])
        
        result.index.name = PID_COL
        return result

    @staticmethod
    def draw_index_dates_for_unexposed(
        data_pids: List[str],
        censoring_timestamps: pd.Series,
        minimum_index_dates: Optional[pd.DataFrame] = None,
        maximum_index_dates: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Draw censor dates for patients not in censoring_timestamps.
        Includes validation against minimum/maximum index dates.
        """
        np.random.seed(42)
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        
        # Draw random timestamps for missing patients
        random_abspos = np.random.choice(censoring_timestamps.values, size=len(missing_pids))
        new_entries = pd.Series(random_abspos, index=pd.Index(list(missing_pids), name=PID_COL))
        result = pd.concat([censoring_timestamps, new_entries])
        result.index.name = PID_COL
        
        # Validate and redraw dates outside valid range
        if minimum_index_dates is not None or maximum_index_dates is not None:
            result = IndexDateHandler._validate_and_redraw_dates(
                result, censoring_timestamps, minimum_index_dates, maximum_index_dates
            )
        
        return result

    @staticmethod
    def _validate_and_redraw_dates(
        result: pd.Series,
        censoring_timestamps: pd.Series,
        minimum_index_dates: Optional[pd.DataFrame],
        maximum_index_dates: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Validate dates against min/max constraints and redraw if necessary."""
        min_dates_series = None
        max_dates_series = None
        
        if minimum_index_dates is not None:
            min_dates_series = minimum_index_dates.set_index(PID_COL)[TIMESTAMP_COL]
            aligned_min_dates = min_dates_series.reindex(result.index)
        
        if maximum_index_dates is not None:
            max_dates_series = maximum_index_dates.set_index(PID_COL)[TIMESTAMP_COL]
            aligned_max_dates = max_dates_series.reindex(result.index)
        
        # Find patients whose dates are outside valid range
        mask = pd.Series(False, index=result.index)
        if minimum_index_dates is not None:
            mask |= aligned_min_dates.notna() & (result < aligned_min_dates)
        if maximum_index_dates is not None:
            mask |= aligned_max_dates.notna() & (result > aligned_max_dates)
        
        patients_to_redraw = mask.sum()
        if patients_to_redraw > 0:
            logger.info(f"Redrawing {patients_to_redraw} patients whose censoring dates fall outside valid index date range")
            
            for pid in result[mask].index:
                min_date = aligned_min_dates[pid] if minimum_index_dates is not None else None
                max_date = aligned_max_dates[pid] if maximum_index_dates is not None else None
                
                valid_timestamps = censoring_timestamps.copy()
                if min_date is not None:
                    valid_timestamps = valid_timestamps[valid_timestamps >= min_date]
                if max_date is not None:
                    valid_timestamps = valid_timestamps[valid_timestamps <= max_date]
                
                if len(valid_timestamps) > 0:
                    result[pid] = np.random.choice(valid_timestamps.values)
                else:
                    if min_date is not None:
                        result[pid] = min_date
                        logger.warning(f"No valid censoring timestamps >= minimum date for patient {pid}, using minimum date")
                    elif max_date is not None:
                        result[pid] = max_date
                        logger.warning(f"No valid censoring timestamps <= maximum date for patient {pid}, using maximum date")
                    else:
                        logger.warning(f"No valid censoring timestamps in range for patient {pid}")
        
        return result

    @classmethod
    def determine_index_dates(
        cls,
        patients_info: pd.DataFrame,
        index_date_mode: str,
        *,  # force keyword arguments
        absolute_timestamp: Optional[dict] = None,
        n_hours_from_exposure: Optional[int] = None,
        exposures: Optional[pd.DataFrame] = None,
        minimum_index_dates: Optional[pd.DataFrame] = None,
        maximum_index_dates: Optional[pd.DataFrame] = None,
        n_hours_from_minimum_index_date: Optional[int] = None,
        n_hours_from_maximum_index_date: Optional[int] = None,
        secondary_censoring_timestamps: Optional[pd.DataFrame] = None,
        n_hours_from_secondary_censoring_timestamps: Optional[int] = None,
    ) -> pd.Series:
        """
        Determine index dates based on mode.
        
        Args:
            patients_info: DataFrame with patients info
            index_date_mode: "absolute" or "relative"
            absolute_timestamp: dict with year, month, day (required if mode == "absolute")
            n_hours_from_exposure: int (required if mode == "relative")
            exposures: DataFrame (required if mode == "relative")
            minimum_index_dates: DataFrame (optional)
            maximum_index_dates: DataFrame (optional)
            n_hours_from_minimum_index_date: int (optional)
            n_hours_from_maximum_index_date: int (optional)
            secondary_censoring_timestamps: DataFrame (optional)
            n_hours_from_secondary_censoring_timestamps: int (optional)
            
        Returns:
            pd.Series: Index dates for all patients
        """
        pids = set(patients_info[PID_COL].unique())

        if index_date_mode == "absolute":
            absolute_timestamp = datetime(**absolute_timestamp)
            result = cls.create_timestamp_series(pids, absolute_timestamp)
        elif index_date_mode == "relative":
            result = cls._handle_relative_mode(
                pids, n_hours_from_exposure, exposures,
                minimum_index_dates, maximum_index_dates,
                n_hours_from_minimum_index_date, n_hours_from_maximum_index_date,
                secondary_censoring_timestamps, n_hours_from_secondary_censoring_timestamps
            )
        else:
            raise ValueError(f"Unsupported index date mode: {index_date_mode}")

        result.index.name = PID_COL
        result.name = TIMESTAMP_COL
        return result

    @classmethod
    def _handle_relative_mode(
        cls,
        pids: Set[str],
        n_hours_from_exposure: Optional[int],
        exposures: Optional[pd.DataFrame],
        minimum_index_dates: Optional[pd.DataFrame],
        maximum_index_dates: Optional[pd.DataFrame],
        n_hours_from_minimum_index_date: Optional[int],
        n_hours_from_maximum_index_date: Optional[int],
        secondary_censoring_timestamps: Optional[pd.DataFrame],
        n_hours_from_secondary_censoring_timestamps: Optional[int]
    ) -> pd.Series:
        """Handle relative mode index date calculation."""
        n_hours = n_hours_from_exposure or 0
        exposed_timestamps = cls.get_index_timestamps_for_exposed(pids, n_hours, exposures)
        
        # Process minimum/maximum index dates
        if n_hours_from_minimum_index_date is not None:
            minimum_index_dates = cls.get_index_timestamps_for_exposed(
                pids, n_hours_from_minimum_index_date, minimum_index_dates
            ).reset_index()
        
        if n_hours_from_maximum_index_date is not None:
            maximum_index_dates = cls.get_index_timestamps_for_exposed(
                pids, n_hours_from_maximum_index_date, maximum_index_dates
            ).reset_index()
        
        # Handle secondary censoring if provided
        if n_hours_from_secondary_censoring_timestamps is not None:
            secondary_censoring_timestamps = cls.get_index_timestamps_for_exposed(
                pids, n_hours_from_secondary_censoring_timestamps, secondary_censoring_timestamps
            )
            return cls.draw_index_dates_for_unexposed_w_secondary_censoring(
                list(pids), exposed_timestamps, secondary_censoring_timestamps
            )
        else:
            return cls.draw_index_dates_for_unexposed(
                pids, exposed_timestamps, minimum_index_dates, maximum_index_dates
            )
