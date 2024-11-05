import dask.dataframe as dd
from corebehrt.common.config import instantiate_function


class ValueCreator:
    """
    A class to create features from patient information and concepts DataFrames.
    We create background, death, age, absolute position, and segments features.
    Parameters
    ----------
    origin_point : datetime or dict, optional
        The origin point for calculating absolute positions. If a dict is provided, it's used to create a datetime object.
    background_vars : list, optional
        List of background variable names to include.

    Examples
    --------
    >>> feature_creator = FeatureCreator(origin_point=datetime(2020, 1, 26), background_vars=["GENDER", "AGE"])
    >>> features = feature_creator(patients_info, concepts)
    """

    def __init__(
        self,
        value_type: str,
        value_type_kwargs: dict = None,
        normalize_args: dict = None,
    ):
        self.value_type = value_type  # values.get("value_type", None)
        self.value_type_kwargs = (
            value_type_kwargs  # values.get("value_type_kwargs", None)
        )
        self.normalize_args = normalize_args  # values.get("normalize_args", None)
        if self.normalize_args is not None:
            self.normalize_args["func"] = instantiate_function(
                self.normalize_args["func"]
            )

    def __call__(
        self,
        concepts: dd.DataFrame,
    ) -> dd.DataFrame:
        if self.value_type == "binned":
            return self.add_binned_values(
                concepts,
                self.value_type_kwargs["multiplication_factor"],
                self.normalize_args,
            )
        elif self.value_type == "quantile":
            return self.add_quantile_values(concepts)
        else:
            raise ValueError(f"Unknown value type: {self.value_type}")

    @staticmethod
    def add_binned_values(
        concepts: dd.DataFrame,
        multiplication_factor: int,
        normalize_args: dict = None,
    ) -> dd.DataFrame:
        """
        Adds binned values to the 'concept' DataFrame for numeric columns.

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
        concepts = concepts.reset_index(drop=False)
        concepts["index"] += 1
        values = concepts.dropna(subset=["RESULT"])
        if normalize_args is not None and callable(normalize_args["func"]):
            values = normalize_args["func"](values, **normalize_args["kwargs"])

        values["RESULT"] = (values["RESULT"] * 100).astype(int)
        values["CONCEPT"] = "VAL_" + values["RESULT"].astype(str)

        concepts["order"] = 0
        values["order"] = 1
        concatted_df = dd.concat([concepts, values])
        return concatted_df.drop(["RESULT"], axis=1)

    @staticmethod
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
