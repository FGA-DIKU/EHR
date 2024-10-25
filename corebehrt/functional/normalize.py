import numpy as np

# def min_max_normalize(value, min_value, max_value) -> float:
#     """
#     Performs min-max normalisation on a single value.
#     """
#     return (value - min_value) / (max_value - min_value)
#     df["RESULT"] = np.where(
#             df["count"] >= min_count,
#             (df["RESULT"] - df["min"]) / (df["max"] - df["min"]),
#             -1
#         )
#         return df


def min_max_normalize(df, min_count) -> float:
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

    df["RESULT"] = np.where(
        mask, (df["RESULT"] - df["min"]) / (df["max"] - df["min"]), -1
    )
    return df
