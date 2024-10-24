import dask.dataframe as dd
from typing import Optional, Callable
import pandas as pd


def add_binned_values(
    concepts: dd.DataFrame,
    normalise: Optional[Callable[[pd.Series], pd.Series]] = None,
) -> dd.DataFrame:
    """
    Adds binned values to the 'concepts' DataFrame for numeric columns.

    This function processes the 'RESULT' column, converting it to numeric values,
    and creates binned values scaled by 100. These are concatenated back to the original 
    DataFrame, with new concept labels prefixed by 'VAL_'.

    Parameters:
    ----------
    concepts : 
        A Dask DataFrame containing at least 'RESULT' and 'CONCEPT' columns. The 'RESULT' column
        should have numeric data to be binned.
    normalise : 
        A function that normalises the 'RESULT' column. This function should take a Pandas 
        Series and return a transformed Series. If None, no normalisation is applied.
    """
    concepts["RESULT"] = dd.to_numeric(concepts["RESULT"], errors="coerce")
    concepts["index"] = concepts.index + 1

    values = concepts.dropna(subset=["RESULT"])
    if normalise is not None and callable(normalise):
        values = normalise(values)
    values["RESULT"] = (values["RESULT"] * 100).astype(int)
    values["CONCEPT"] = "VAL_" + values["RESULT"].astype(str)

    concepts["order"] = 0
    values["order"] = 1
    concatted_df = dd.concat([concepts, values])
    return concatted_df.drop(["RESULT"], axis=1)


def add_quantile_values(
    concepts: dd.DataFrame,
) -> dd.DataFrame:
    """
    Adds quantile values to the 'concepts' DataFrame.

    This function extracts quantile values the 'RESULT' column as values starting with Q. 
    These are concatenated back to the original DataFrame, with new concept labels prefixed by 'VAL_'.

    Parameters:
    ----------
    concepts : 
        A Dask DataFrame containing at least 'RESULT' and 'CONCEPT' columns. The 'RESULT' column
        should have quantile values starting with Q. 
    """
    concepts["index"] = concepts.index + 1
    values = concepts[concepts["RESULT"].astype(str).str.startswith("Q")]
    values["CONCEPT"] = "VAL_" + values["RESULT"].astype(str)

    concepts["order"] = 0
    values["order"] = 1

    concatted_df = dd.concat([concepts, values])
    return concatted_df.drop(["RESULT"], axis=1)
