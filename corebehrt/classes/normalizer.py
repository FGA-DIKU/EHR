from dask import dataframe as dd
from corebehrt.functional.normalize import min_max_normalize


class ValuesNormalizer:
    """
    A class to load normalise values in data frames.
    Expects 'RESULT' and 'CONCEPT' column to be present.
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
    def min_max_normalize_results(values: dd.DataFrame) -> dd.DataFrame:
        """
        Normalises the 'RESULT' column of the given DataFrame using min-max normalisation.
        If the count of a CONCEPT is less than 3, the RESULT is set to -1.
        Expects 'RESULT' and 'CONCEPT' column to be present.
        """
        # Compute min, max, and count for each CONCEPT
        grouped = values.groupby("CONCEPT")["RESULT", "TIMESTAMP"]
        min_values = (
            grouped.min()
            .compute()
            .drop(columns=["TIMESTAMP"])
            .rename(columns={"RESULT": "min"})
        )
        max_values = (
            grouped.max()
            .compute()
            .drop(columns=["TIMESTAMP"])
            .rename(columns={"RESULT": "max"})
        )
        count_values = (
            grouped.count()
            .compute()
            .drop(columns=["TIMESTAMP"])
            .rename(columns={"RESULT": "count"})
        )
        min_max_count = dd.merge(
            min_values, max_values, left_index=True, right_index=True
        ).merge(count_values, left_index=True, right_index=True)
        values = dd.merge(
            values, min_max_count, left_on="CONCEPT", right_index=True, how="left"
        )

        # Apply normalisation row-wise
        values["RESULT"] = values.apply(
            lambda row: ValuesNormalizer.apply_normalization(
                row,
                min_max_normalize,
                min_value=row["min"],
                max_value=row["max"],
                min_count=3,
            ),
            axis=1,
            meta=("RESULT", "f8"),
        )

        # Drop auxiliary columns
        values = values.drop(["min", "max", "count"], axis=1)
        return values
