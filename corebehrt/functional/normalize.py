import pandas as pd


def min_max_normalize(x: pd.Series) -> pd.Series:
    """
    Performs min-max normalisation on a Series. If the series only contains a single value, it will be replaced with 'UNIQUE'.
    """
    x = x.dropna()
    if x.nunique() == 1:
        return pd.Series(["UNIQUE"] * len(x), index=x.index)
    x = x[x.str.isnumeric()].astype(float) # Drop any non-numeric values
    return (x - x.min()) / (x.max() - x.min())
