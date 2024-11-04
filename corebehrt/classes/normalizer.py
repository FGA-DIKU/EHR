from dask import dataframe as dd
from corebehrt.functional.normalize import min_max_normalize


class ValuesNormalizer:
    """
    A class to load normalise values in data frames.
    Expects 'RESULT' and 'concept' column to be present.
    """

    @staticmethod
    def apply_normalization(row, normalization_func, **kwargs):
        """
        Applies the given normalisation function if count >= min_count, else sets RESULT to -1.
        """
        min_count = kwargs.pop("min_count", 3)
        if row["count"] < min_count:
            return -1
        else:
            return normalization_func(row["RESULT"], **kwargs)

    @staticmethod
    def min_max_normalize_results(values: dd.DataFrame, min_count: int) -> dd.DataFrame:
        """
        Normalises the 'RESULT' column of the given DataFrame using min-max normalisation.
        If the count of a 'concept' is less than 3, the RESULT is set to -1.
        Expects 'RESULT' and 'concept' column to be present.
        """
        # Compute min, max, and count for each CONCEPT
        grouped = values.groupby("CONCEPT")["RESULT", "TIMESTAMP"]
        min_max_count = grouped.agg({"RESULT": ["min", "max", "count"]})
        min_max_count.columns = ["min", "max", "count"]
        values = dd.merge(
            values, min_max_count, left_on="CONCEPT", right_index=True, how="left"
        )
        
        # Apply normalisation
        values = values.map_partitions(min_max_normalize, min_count)

        # Drop auxiliary columns and values which have become NaN after normalisation
        values = values.drop(["min", "max", "count"], axis=1)
        values = values.dropna(subset=["RESULT"])
        return values
