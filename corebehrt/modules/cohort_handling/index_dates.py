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
    def draw_index_dates_for_unexposed_w_secondary_censoring(
        data_pids: List[str],
        censoring_timestamps: pd.Series,
        secondary_censoring_timestamps: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Draw censor dates for patients not in censor_timestamps."""
        np.random.seed(42)
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        
        # Initialize result with existing censoring timestamps
        result = censoring_timestamps.copy()
        
        # For patients with secondary censoring timestamps, use those instead
        if secondary_censoring_timestamps is not None:
            # Get patients that have secondary censoring timestamps
            secondary_pids = set(secondary_censoring_timestamps.index) & set(data_pids)
            
            # Update result with secondary censoring timestamps for those patients
            for pid in secondary_pids:
                if pid in result.index:
                    # Replace existing censoring timestamp with secondary one
                    result[pid] = secondary_censoring_timestamps.loc[pid, TIMESTAMP_COL]
                else:
                    # Add new entry with secondary censoring timestamp
                    result[pid] = secondary_censoring_timestamps.loc[pid, TIMESTAMP_COL]
            
            # Remove these patients from missing_pids since they now have censoring dates
            missing_pids = missing_pids - secondary_pids
        
        # For remaining missing patients, draw from normal censoring timestamps
        if missing_pids:
            random_abspos = np.random.choice(
                censoring_timestamps.values, size=len(missing_pids)
            )
            new_entries = pd.Series(
                random_abspos, index=pd.Index(list(missing_pids), name=PID_COL)
            )
            result = pd.concat([result, new_entries])
        
        result.index.name = PID_COL  # Ensure the final series has PID_COL as index name
        return result

    @staticmethod
    def draw_index_dates_for_unexposed(
        data_pids: List[str],
        censoring_timestamps: pd.Series,
        minimum_index_dates: Optional[pd.DataFrame] = None,
        maximum_index_dates: Optional[pd.DataFrame] = None,
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
        
        # If minimum_index_dates or maximum_index_dates are provided, redraw dates that fall outside the valid range
        if minimum_index_dates is not None or maximum_index_dates is not None:
            min_dates_series = None
            max_dates_series = None
            
            if minimum_index_dates is not None:
                min_dates_series = minimum_index_dates.set_index(PID_COL)[TIMESTAMP_COL]
                aligned_min_dates = min_dates_series.reindex(result.index)
            
            if maximum_index_dates is not None:
                max_dates_series = maximum_index_dates.set_index(PID_COL)[TIMESTAMP_COL]
                aligned_max_dates = max_dates_series.reindex(result.index)
            
            # Find patients whose drawn dates are outside the valid range
            mask = pd.Series(False, index=result.index)
            
            if minimum_index_dates is not None:
                mask |= aligned_min_dates.notna() & (result < aligned_min_dates)
            
            if maximum_index_dates is not None:
                mask |= aligned_max_dates.notna() & (result > aligned_max_dates)
            
            patients_to_redraw = mask.sum()
            
            if patients_to_redraw > 0:
                logger.info(f"Redrawing {patients_to_redraw} patients whose censoring dates fall outside valid index date range")
                
                # For patients that need redrawing, draw new dates from the valid range
                for pid in result[mask].index:
                    min_date = aligned_min_dates[pid] if minimum_index_dates is not None else None
                    max_date = aligned_max_dates[pid] if maximum_index_dates is not None else None
                    
                    # Filter censoring timestamps to only include those within the valid range
                    valid_timestamps = censoring_timestamps.copy()
                    
                    if min_date is not None:
                        valid_timestamps = valid_timestamps[valid_timestamps >= min_date]
                    
                    if max_date is not None:
                        valid_timestamps = valid_timestamps[valid_timestamps <= max_date]
                    
                    if len(valid_timestamps) > 0:
                        # Draw from valid timestamps
                        new_date = np.random.choice(valid_timestamps.values)
                        result[pid] = new_date
                    else:
                        # If no valid timestamps, use the minimum date if available, otherwise maximum date
                        if min_date is not None:
                            result[pid] = min_date
                            logger.warning(f"No valid censoring timestamps >= minimum date for patient {pid}, using minimum date")
                        elif max_date is not None:
                            result[pid] = max_date
                            logger.warning(f"No valid censoring timestamps <= maximum date for patient {pid}, using maximum date")
                        else:
                            logger.warning(f"No valid censoring timestamps in range for patient {pid}")
        
        return result
    
    @staticmethod
    def apply_minimum_index_dates(result: pd.Series, minimum_index_dates: pd.DataFrame) -> pd.Series:
        """Apply minimum index dates to the result."""
        # Convert minimum_index_dates DataFrame to Series with PID as index
        min_dates_series = minimum_index_dates.set_index(PID_COL)[TIMESTAMP_COL]
        
        # Align the series by reindexing min_dates_series to match result's index
        # This will fill missing values with NaN for patients not in minimum_index_dates
        aligned_min_dates = min_dates_series.reindex(result.index)
        
        # Only apply minimum dates where we have valid minimum dates (not NaN)
        mask = aligned_min_dates.notna()
        result = result.copy()
        result[mask] = result[mask].where(result[mask] >= aligned_min_dates[mask], aligned_min_dates[mask])
        
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
        minimum_index_dates: Optional[pd.DataFrame] = None,
        maximum_index_dates: Optional[pd.DataFrame] = None,
        n_hours_from_minimum_index_date: Optional[int] = None,
        n_hours_from_maximum_index_date: Optional[int] = None,
        secondary_censoring_timestamps: Optional[pd.DataFrame] = None,
        n_hours_from_secondary_censoring_timestamps: Optional[int] = None,
    ) -> pd.Series:
        """Determine index dates based on mode.
        Args:
            patients_info: pd.DataFrame with patients info
            index_date_mode: str, "absolute" or "relative"
            absolute_timestamp: dict with year, month, day (required if index_date_mode == "absolute")
            n_hours_from_exposure: int (required if index_date_mode == "relative")
            exposures: pd.DataFrame (required if index_date_mode == "relative")
            minimum_index_dates: pd.DataFrame (optional if index_date_mode == "relative")
            maximum_index_dates: pd.DataFrame (optional if index_date_mode == "relative")
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
            if n_hours_from_minimum_index_date is not None:
                minimum_index_dates = cls.get_index_timestamps_for_exposed(
                    pids, n_hours_from_minimum_index_date, minimum_index_dates
                )
                # Convert Series back to DataFrame for draw_index_dates_for_unexposed
                minimum_index_dates = minimum_index_dates.reset_index()
            if n_hours_from_maximum_index_date is not None:
                maximum_index_dates = cls.get_index_timestamps_for_exposed(
                    pids, n_hours_from_maximum_index_date, maximum_index_dates
                )
                # Convert Series back to DataFrame for draw_index_dates_for_unexposed
                maximum_index_dates = maximum_index_dates.reset_index()
            
            if n_hours_from_secondary_censoring_timestamps is not None:
                secondary_censoring_timestamps = cls.get_index_timestamps_for_exposed(
                    pids, n_hours_from_secondary_censoring_timestamps, secondary_censoring_timestamps
                )
                # Convert Series back to DataFrame for draw_index_dates_for_unexposed
                secondary_censoring_timestamps = secondary_censoring_timestamps.reset_index()
                result = cls.draw_index_dates_for_unexposed(pids, secondary_censoring_timestamps)
            else:   
                result = cls.draw_index_dates_for_unexposed(pids, exposed_timestamps, minimum_index_dates, maximum_index_dates)
        else:
            raise ValueError(f"Unsupported index date mode: {index_date_mode}")

        # Ensure the series has both index name and series name
        result.index.name = PID_COL
        result.name = TIMESTAMP_COL
        return result
