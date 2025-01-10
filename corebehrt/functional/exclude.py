import dask.dataframe as dd


def exclude_incorrect_event_ages(
    df: dd.DataFrame, min_age: int = -1, max_age: int = 120
) -> dd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]


def exclude_pids_from_data(data: dd.DataFrame, pids_to_exclude: list) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with only the rows that do not have a PID in pids
    """
    return data[~data["PID"].isin(set(pids_to_exclude))]
