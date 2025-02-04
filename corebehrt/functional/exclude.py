import dask.dataframe as dd


def exclude_incorrect_event_ages(
    df: dd.DataFrame, min_age: int = -1, max_age: int = 120
) -> dd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]


def exclude_event_nans(df: dd.DataFrame) -> dd.DataFrame:
    """Exclude events (row) with (any) NaNs"""
    return df.dropna()
