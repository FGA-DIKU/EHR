from datetime import datetime

import dask.dataframe as dd

from corebehrt.functional.utils import (
    get_abspos_from_origin_point,
    normalize_segments_series,
)


def create_abspos(concepts: dd.DataFrame, origin_point: datetime) -> dd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
        origin_point: The origin point for calculating absolute position.
    Returns:
        concepts with a new 'abspos' column
    """
    concepts["abspos"] = get_abspos_from_origin_point(
        concepts["TIMESTAMP"], origin_point
    )
    return concepts


def create_age_in_years(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'TIMESTAMP' and 'BIRTHDATE' columns.
    Returns:
        dd.DataFrame: concepts with a new 'age' column
    """
    concepts["age"] = (concepts["TIMESTAMP"] - concepts["BIRTHDATE"]).dt.days // 365.25
    return concepts


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
        concept="Death",
        ADMISSION_ID="last",  # Segment for death event
    )

    # Reorder columns if necessary
    death_events = death_events[
        ["PID", "concept", "TIMESTAMP", "ADMISSION_ID", "BIRTHDATE"]
    ]

    return death_events


def create_background(
    patients_info: dd.DataFrame, background_vars: list
) -> dd.DataFrame:
    """
    Create background concepts for each patient based on static background variables.
    Parameters:
        patients_info: containing patient information, including 'PID', 'BIRTHDATE', and background variables.
        background_vars: static background variables column names to include.
    Returns:
        table with background concepts, including 'PID', 'concept', 'TIMESTAMP', 'ADMISSION_ID', and 'BIRTHDATE' columns.
    """
    # Filter patients with non-null BIRTHDATE
    patients_info = patients_info[~patients_info["BIRTHDATE"].isna()]

    # Melt the DataFrame to transform background variables into rows
    background = patients_info[["PID", "BIRTHDATE"] + background_vars].melt(
        id_vars=["PID", "BIRTHDATE"],
        value_vars=background_vars,
        var_name="variable",
        value_name="value",
    )

    # Create 'concept' column
    background["concept"] = (
        "BG_" + background["variable"] + "_" + background["value"].astype(str)
    )

    # Assign additional columns
    background["TIMESTAMP"] = background["BIRTHDATE"]
    background["ADMISSION_ID"] = "first"

    # Select and reorder the required columns
    background = background[
        ["PID", "concept", "TIMESTAMP", "ADMISSION_ID", "BIRTHDATE"]
    ]

    return background


def assign_index_and_order(df: dd.DataFrame) -> dd.DataFrame:
    """
    Assign 'index' and 'order' columns to ensure correct ordering.
    - The 'index' column represents the position of each row within its partition.
    - The 'order' column can be used for additional custom ordering if needed.
    - Both columns are initialized with 0 to ensure consistent behavior across partitions.
    Parameters:
        df: dd.DataFrame with 'PID' column.
    Returns:
        df with 'index' and 'order' columns.
    """
    if "index" in df.columns and "order" in df.columns:
        df["index"] = df["index"].fillna(0)
        df["order"] = df["order"].fillna(0)
    return df


def sort_features(concepts: dd.DataFrame) -> dd.DataFrame:
    """
    Sorting all concepts by 'PID' and 'abspos' (and 'index' and 'order' if they exist).
    """
    concepts = concepts.shuffle(on="PID")
    concepts = concepts.map_partitions(_sort_partitions)
    return concepts


def _sort_partitions(df: dd.DataFrame) -> dd.DataFrame:
    """
    Sort the DataFrame by 'PID' and 'abspos' to ensure correct ordering.
    Added "index" and "order" columns to ensure correct ordering if they exist.
    """
    if "index" in df.columns and "order" in df.columns:
        df = df.sort_values(
            ["PID", "abspos", "index", "order"]
        )  # could maybe be done more optimally, is a bit slow
        df = df.drop(columns=["index", "order"])
    else:
        df = df.sort_values(["PID", "abspos"])

    return df


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
    concepts = concepts.map_partitions(_assign_segments)
    concepts = concepts.map_partitions(assign_segments_to_death)

    return concepts


def _assign_segments(df):
    """
    Assign segments to the concepts DataFrame based on 'ADMISSION_ID'
    """
    # Group by 'PID' and apply factorize to 'ADMISSION_ID'
    df["segment"] = df.groupby("PID")["ADMISSION_ID"].transform(
        normalize_segments_series
    )
    return df


def assign_segments_to_death(df: dd.DataFrame) -> dd.DataFrame:
    """
    Assign the maximum segment to 'Death' concepts within each 'PID'.
    Parameters:
        df with 'PID', 'concept', and 'segment' columns.
    Returns:
        df with 'Death' concepts assigned to the maximum segment.
    """
    # Compute the maximum segment per 'PID'
    max_segment = df.groupby("PID")["segment"].max().rename("max_segment")
    # Merge and assign
    df = df.merge(max_segment.reset_index(), on="PID", how="left")
    df["segment"] = df["segment"].where(df["concept"] != "Death", df["max_segment"])
    return df.drop(columns=["max_segment"])
