import pandas as pd

class ValueCreator:
    """
    A class to load normalise values in data frames.
    Expects a 'result' column and 'concept' column to be present.
    """

    @staticmethod
    def bin_results(concepts: pd.DataFrame, num_bins=100) -> pd.DataFrame:
        concepts["numeric_value"] = ValueCreator.bin(
            concepts["numeric_value"], num_bins=num_bins
        )

        # Add index + order
        concepts["index"] = concepts.index
        values = concepts.dropna(subset=["numeric_value"]).copy()
        values.loc[:, "code"] = values["numeric_value"]
        concepts.loc[:, "order"] = 0
        values.loc[:, "order"] = 1
        concatted = pd.concat([concepts, values])
        return concatted.drop(columns=["numeric_value"], axis=1)

    @staticmethod
    def bin(normalized_values: pd.Series, num_bins=100) -> pd.Series:
        """
        Bins the values in a series into num_bins bins. Expects the values to be normalised.
        """
        normalized_values = pd.to_numeric(normalized_values, errors="coerce")
        val_mask = normalized_values.notna()
        normalized_values[val_mask] = normalized_values[val_mask].mul(num_bins)
        normalized_values = normalized_values.astype(object)
        normalized_values[val_mask] = (
            normalized_values[val_mask].astype(int).astype(str)
        )
        normalized_values[val_mask] = "VAL_" + normalized_values[val_mask]
        return normalized_values
