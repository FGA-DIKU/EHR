from datetime import datetime
import pandas as pd
from typing import Union


def get_hours_since_epoch(
    timestamps: Union[pd.Series, datetime],
) -> Union[pd.Series, float]:
    """
    Convert timestamps to hours since Unix epoch (January 1, 1970 UTC).
    Ensures all timestamps are in UTC before conversion.

    Parameters:
        timestamps: Timestamps to convert, can be pandas Series, list of datetimes, or single datetime
    Returns:
        Hours since epoch in the same container type as input (Series, list, or float)
    """
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([])
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps, utc=True)

        hours = (timestamps.astype("int64") // 10**9) / 3600
        return hours

    elif isinstance(timestamps, datetime):
        return get_hours_since_epoch(pd.Series([timestamps])).iloc[0]
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series, list, and datetime are supported."
        )
