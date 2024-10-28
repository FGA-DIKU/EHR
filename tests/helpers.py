import pandas as pd


def compute_column_checksum(df: pd.DataFrame, column: str) -> str:
    """
    Compute a checksum for a specified column in the DataFrame.
    """
    encoded = df[column].apply(str)
    encoded = encoded.apply(lambda x: hash(x))
    return encoded.sum()
