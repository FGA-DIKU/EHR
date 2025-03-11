from datetime import datetime
from typing import Union

import pandas as pd


def get_hours_since_epoch(
    timestamps: Union[pd.Series, datetime],
) -> Union[pd.Series, float]:
    """
    Convert timestamps to hours since Unix epoch (January 1, 1970 UTC).

    Parameters:
        timestamps: Timestamps to convert, can be pandas Series, list of datetimes, or single datetime
    Returns:
        Hours since epoch in the same container type as input (Series, list, or float)
    """
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([])
        return (timestamps.astype("int64") // 10**9) / 3600
    elif isinstance(timestamps, datetime):
        # Use same calculation method as Series
        return (pd.Series([timestamps]).astype("int64").iloc[0] // 10**9) / 3600
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series, list, and datetime are supported."
        )
