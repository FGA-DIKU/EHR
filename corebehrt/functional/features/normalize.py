from dataclasses import replace
from typing import List

import numpy as np
import pandas as pd

from corebehrt.modules.preparation.dataset import PatientData


def min_max_normalize(df: pd.DataFrame, min_count: int) -> pd.Series:
    """
    Performs min-max normalisation on a dataframe.
    """
    mask = (
        (df["count"] >= min_count)
        & ~df["count"].isna()
        & ~df["RESULT"].isna()
        & ~df["min"].isna()
        & ~df["max"].isna()
    )
    normed_result = np.where(
        mask, (df["RESULT"] - df["min"]) / (df["max"] - df["min"]), -1
    )

    return pd.Series(normed_result, index=df.index)


def normalize_segments_for_patient(patient: PatientData) -> PatientData:
    normalized_segments = normalize_segments(patient.segments)
    return replace(patient, segments=normalized_segments)


def normalize_segments(segments: List[int]) -> List[int]:
    """Normalize a list of segment IDs to be zero-based and contiguous.

    Takes a list of segment IDs that may have gaps or start from an arbitrary number,
    and normalizes them to start from 0 and increment by 1 while preserving their relative order.

    Args:
        segments (list): List of segment IDs to normalize

    Returns:
        list: Normalized list of segment IDs starting from 0 with no gaps

    Example:
        >>> normalize_segments([5, 5, 8, 10, 8])
        [0, 0, 1, 2, 1]
    """
    segment_set = sorted(set(segments))
    correct_segments = list(range(len(segment_set)))
    converter = {k: v for (k, v) in zip(segment_set, correct_segments)}

    return [converter[segment] for segment in segments]


def normalize_segments_series(series: pd.Series) -> pd.Series:
    # Convert to string to ensure consistent types and avoid warnings
    series = series.astype(str)
    return series.factorize(use_na_sentinel=False)[0]
