import dask.dataframe as dd
from typing import Optional, Callable
import pandas as pd

def add_binned_values(
    concepts: dd.DataFrame,
    normalise: Optional[Callable[[pd.Series], pd.Series]] = None,
) -> dd.DataFrame:
    """
    Includes binning values for the concepts DataFrame. Only includes numeric values.
    Expects 'RESULT' and 'CONCEPT' column to be present.
    Optionally normalises the 'RESULT' column using the provided normalise function.
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
    Includes quantile values for the concepts DataFrame.
    Expects 'RESULT' and 'CONCEPT' column to be present.
    Excepts values in 'RESULT' column to be quantiles starting with Q.
    """
    concepts["index"] = concepts.index + 1
    values = concepts[concepts["RESULT"].astype(str).str.startswith("Q")]
    values["CONCEPT"] = "VAL_" + values["RESULT"].astype(str)

    concepts["order"] = 0
    values["order"] = 1

    concatted_df = dd.concat([concepts, values])
    return concatted_df.drop(["RESULT"], axis=1)
