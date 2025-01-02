import dask.dataframe as dd

from corebehrt.functional.utils import filter_table_by_pids, get_gender_token


def exclude_incorrect_event_ages(
    df: dd.DataFrame, min_age: int = -1, max_age: int = 120
) -> dd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]


def exclude_event_nans(df: dd.DataFrame) -> dd.DataFrame:
    """Exclude events (row) with (any) NaNs"""
    return df.dropna()


def exclude_pids_from_data(data: dd.DataFrame, pids_to_exclude: list) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID.
    Returns a new table with only the rows that do not have a PID in pids
    """
    return data[~data["PID"].isin(set(pids_to_exclude))]


def filter_patients_by_gender(
    data: dd.DataFrame, vocab: dict, gender: str = None
) -> dd.DataFrame:
    """
    Assumes that the table has a column named PID and concept.
    Returns a new table with only the rows that have a concept with
    """
    if gender is None:
        return data

    gender_token = get_gender_token(gender, vocab)
    patients = data[data.concept == gender_token].PID.unique()
    return filter_table_by_pids(data, patients)
