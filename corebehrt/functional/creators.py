from datetime import datetime

import dask.dataframe as dd

from corebehrt.functional.utils import (
    get_abspos_from_origin_point,
    normalize_segments_series,
)
from corebehrt.functional.constants import CLS_TOKEN, SEP_TOKEN


def create_abspos(concepts: dd.DataFrame, origin_point: datetime) -> dd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
        origin_point: The origin point for calculating absolute position.
    Returns:
        dd.Series: the calculated absolute position in hours
    """
    return get_abspos_from_origin_point(concepts["TIMESTAMP"], origin_point)


def create_age_in_years(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'TIMESTAMP' and 'BIRTHDATE' columns.
    Returns:
        dd.Series: the calculated age in year
    """
    return (concepts["TIMESTAMP"] - concepts["BIRTHDATE"]).dt.days // 365.25


def create_sep_tokens(features: dd.DataFrame):
    def process_partition(partition):
        def process_group(group):
            last_event_in_admission = group["segment"].duplicated(keep="last")
            last_concepts = group.loc[~last_event_in_admission, "concept"]
            last_concepts.map(lambda x: x.append(SEP_TOKEN))  # Utilise in-place append
            return group

        return partition.groupby("PID", group_keys=False).apply(process_group)

    features = features.set_index("PID")  # Make groupby properly work
    features = features.map_partitions(process_partition, meta=features)
    features = features.reset_index()
    return features


def create_death(patients_info: dd.DataFrame) -> dd.DataFrame:
    """
    Create a Dask DataFrame containing death events for patients with known death dates.
    Parameters:
        patients_info: containing patient information, including 'PID', 'BIRTHDATE', and 'DEATHDATE'.
    Returns:
        patients_info with death events, including 'PID', 'concept', 'TIMESTAMP', 'ADMISSION_ID', and 'BIRTHDATE' columns.
    """
    # Filter patients who have both BIRTHDATE and DEATHDATE
    death_events = patients_info[
        (~patients_info["BIRTHDATE"].isna()) & (~patients_info["DEATHDATE"].isna())
    ][["PID", "BIRTHDATE", "DEATHDATE"]]

    # Rename 'DEATHDATE' to 'TIMESTAMP' and add necessary columns
    death_events = death_events.rename(columns={"DEATHDATE": "TIMESTAMP"})
    death_events = death_events.assign(
        death="Death",
        ADMISSION_ID="last",  # Segment for death event
    )
    death_events["ADMISSION_ID"] = death_events["ADMISSION_ID"].astype(
        "string[pyarrow]"
    )

    # Reorder columns if necessary
    death_events = death_events[
        ["PID", "death", "TIMESTAMP", "ADMISSION_ID", "BIRTHDATE"]
    ]

    return death_events


def create_background(
    patients_info: dd.DataFrame, background_vars: list, cls_token: bool = False
) -> dd.DataFrame:
    """
    Create background concepts for each patient based on static background variables and optionally add a '[CLS]' token.
    Parameters:
        patients_info: containing patient information, including 'PID', 'BIRTHDATE', and background variables.
        background_vars: static background variables column names to include.
        cls_token: whether to add a '[CLS]' token to the background variables.
    Returns:
        table with background concepts, including 'PID', 'concept', 'TIMESTAMP', 'ADMISSION_ID', and 'BIRTHDATE' columns.
    """
    # Filter patients with non-null BIRTHDATE
    patients_info = patients_info[~patients_info["BIRTHDATE"].isna()]

    for var in background_vars:
        patients_info[var] = patients_info[var].map(
            lambda x: f"BG_{var}_{x}", meta=(var, "string[pyarrow]")
        )

    # Add '[CLS]' token to the background variables
    if cls_token:
        background_vars = ["CLS_TOKEN"] + background_vars
        patients_info = patients_info.assign(CLS_TOKEN=CLS_TOKEN)

    # Assign additional columns
    patients_info["TIMESTAMP"] = patients_info["BIRTHDATE"]
    patients_info["ADMISSION_ID"] = "first"
    patients_info["ADMISSION_ID"] = patients_info["ADMISSION_ID"].astype(
        "string[pyarrow]"
    )

    # Select and reorder the required columns
    background = patients_info[
        ["PID", "TIMESTAMP", "ADMISSION_ID", "BIRTHDATE"] + background_vars
    ]

    return background


def sort_features(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Shuffles partitions on "PID" and sorts all concepts by 'abspos'.
    """
    concepts = concepts.shuffle(on="PID")
    concepts = concepts.map_partitions(lambda df: df.sort_values("abspos"))
    return concepts


def create_segments(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Assign segments to the concepts DataFrame based on 'ADMISSION_ID', ensuring that
    events are ordered correctly within each 'PID'.
    Parameters:
        concepts: concepts with 'PID', 'ADMISSION_ID', and 'abspos' columns.
    Returns:
        concepts with a new 'segment' column
    """
    # Assign maximum segment to 'Death' concepts
    concepts = concepts.shuffle("PID")
    # Has to be assigned inside here due to some weird dask behaviour
    concepts["segment"] = concepts.map_partitions(_assign_segments)
    return concepts


def _assign_segments(df):
    """
    Assign segments to the concepts DataFrame based on 'ADMISSION_ID'
    """
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    return df.groupby("PID")["ADMISSION_ID"].transform(normalize_segments_series)
