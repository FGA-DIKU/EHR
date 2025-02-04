import numpy as np
import pandas as pd


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
