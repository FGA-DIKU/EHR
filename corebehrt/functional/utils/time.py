from datetime import datetime
from typing import List, Union

import pandas as pd


def get_abspos_from_origin_point(
    timestamps: Union[pd.Series, List[datetime], datetime], origin_point: datetime
) -> Union[pd.Series, List[float], datetime]:
    """Get the absolute position in hours from the origin point"""
    if isinstance(timestamps, pd.Series):
        if len(timestamps) == 0:
            return pd.Series([])
        return (timestamps - origin_point).dt.total_seconds() / 60 / 60
    elif isinstance(timestamps, list):
        return [
            (timestamp - origin_point).total_seconds() / 60 / 60
            for timestamp in timestamps
        ]
    elif isinstance(timestamps, datetime):
        return (timestamps - origin_point).total_seconds() / 60 / 60
    else:
        raise TypeError(
            "Invalid type for timestamps, only pd.Series, list, and datetime are supported."
        )
