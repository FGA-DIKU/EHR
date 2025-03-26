from datetime import datetime
from typing import Union

import pandas as pd


def get_hours_since_epoch(
    timestamps: Union[pd.Series, datetime],
) -> Union[pd.Series, float]:
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([], dtype=float)
        # Convert timestamps to UTC (timezone-aware)
        timestamps = pd.to_datetime(
            timestamps, utc=True
        )  # ensure consistency across dataset
        # Remove the timezone information to get a timezone-naive series, necessary for the next step
        timestamps = timestamps.dt.tz_localize(None)
        # Cast to microsecond precision
        timestamps = timestamps.astype("datetime64[us]")
        # Convert microseconds to hours
        hours = (timestamps.astype("int64") // 10**6) / 3600
        return hours

    elif isinstance(timestamps, datetime):
        return get_hours_since_epoch(pd.Series([timestamps])).iloc[0]
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series, list, and datetime are supported."
        )
