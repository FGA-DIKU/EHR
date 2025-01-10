from dask import dataframe as dd
from corebehrt.functional.normalize import min_max_normalize


class ValuesNormalizer:
    """
    A class to load normalise values in data frames.
    Expects a 'result' column and 'concept' column to be present.
    """

    @staticmethod
    def min_max_normalize_results(concepts: dd.DataFrame, num_bins=100) -> dd.Series:
        concepts = concepts.shuffle("CONCEPT") # Shuffle to ensure that the same concept is in the same partition
        # Has to be be assigned inside here due to some weird dask behaviour
        concepts["RESULT"] = concepts.map_partitions(
            lambda partition: partition.groupby("CONCEPT")["RESULT"].transform(lambda series:
                ValuesNormalizer.min_max_normalize(series, num_bins=num_bins)
            )
        )
        return concepts

    @staticmethod
    def min_max_normalize(series: dd.groupby.SeriesGroupBy, num_bins=100) -> dd.Series:
        """
        Normalises the a column of the given SeriesGroupBy using min-max normalisation and converts to 100 VAL_{i} bins
        """
        normalized_values = min_max_normalize(series)
        normalized_values = normalized_values.dropna()
        val_mask = normalized_values != "UNIQUE"
        normalized_values = normalized_values.where(
            ~val_mask,
            normalized_values[val_mask].mul(num_bins).astype(int).astype(str)
        ).astype(str)
        return "VAL_" + normalized_values
