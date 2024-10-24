from typing import List, Tuple, Callable

import dask.dataframe as dd

from corebehrt.functional.load import (
    load_concept,
    load_patients_info,
)
from corebehrt.functional.utils import (
    check_concepts_columns,
    check_patients_info_columns,
)
from corebehrt.functional.values import add_binned_values, add_quantile_values


class FormattedDataLoader:
    """
    A class to load concepts and patients_info from a specified folder.

    This class assumes that the folder contains formatted data files:
    - `patients_info.*` file with mandatory columns 'PID', 'BIRTHDATE', and 'DEATHDATE'.
        and optional additional columns with static variables.
    - `concept.{concept_type}.*` files with mandatory columns 'PID', 'TIMESTAMP', 'CONCEPT', and 'ADMISSION_ID'.

    Attributes:
        folder (str): The path to the folder containing data files.
        concept_types (List[str]): A list of concept types to load e.g. ['diagnosis', 'medication'].
        include_values (List[str]): A list of concept types to include values for e.g. ['labtest', 'medication']. The given concept file is excepted to have a 'RESULT' column.
        value_type (str): The type of value to add to the concepts e.g. 'binned_value', 'quantile_value'.
        normalise_func (Callable): A function to normalise values e.g. min-max normalisation.

    Methods:
        load() -> Tuple[dd.DataFrame, dd.DataFrame]:
            Loads the concepts and patients_info DataFrames from the specified folder.
            Returns a tuple containing the concepts DataFrame and the patients_info DataFrame.
    """

    def __init__(
        self,
        folder: str,
        concept_types: List[str],
        include_values: List[str] = [],
        value_type: str = None,
        normalise_func: Callable = None,
    ):
        self.folder = folder
        self.concept_types = concept_types
        self.include_values = include_values
        self.value_type = value_type
        self.normalise_func = normalise_func

    def load(self) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """Loads the concepts and patients_info DataFrames."""
        concepts = [
            (
                self._apply_values(self._load_concept(concept_type))
                if concept_type in self.include_values
                else self._load_concept(concept_type)
            )
            for concept_type in self.concept_types
        ]
        concepts = dd.concat(concepts)
        check_concepts_columns(concepts)

        patients_info = self._load_patients_info()
        check_patients_info_columns(patients_info)

        return concepts, patients_info

    def _load_patients_info(self) -> dd.DataFrame:
        """
        Load patients_info data from formatted_data_dir.
        Expects BIRTHDATE and DEATHDATE columns to be present.
        Returns a dask dataframe.
        """
        return load_patients_info(self.folder)

    def _apply_values(self, concepts: dd.DataFrame) -> dd.DataFrame:
        """
        Adds values to concepts from "RESULT" column.
        Returns a dask dataframe.
        """
        if self.value_type == "binned_value":
            return add_binned_values(concepts, self.normalise_func)
        elif self.value_type == "quantile_value":
            return add_quantile_values(concepts)
        else:
            raise NotImplementedError(
                f"Value type '{self.value_type}' is not supported."
            )

    def _load_concept(self, concept_type: str):
        """
        Load concept data from formatted_data_dir.
        Expects TIMESTAMP column to be present.
        Returns a dask dataframe.
        """
        return load_concept(self.folder, concept_type)
