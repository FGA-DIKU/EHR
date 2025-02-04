from typing import List, Tuple

import dask.dataframe as dd

from corebehrt.functional.load import (
    load_concept,
    load_patients_info,
)
from corebehrt.functional.utils import (
    check_concepts_columns,
    check_patients_info_columns,
)


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
    ):
        self.folder = folder
        self.concept_types = concept_types
        self.include_values = include_values

    def load(self) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """Loads the concepts and patients_info DataFrames."""
        concepts = [
            (
                self._remove_values(self._load_concept(concept_type))
                if concept_type not in self.include_values
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

    def _remove_values(self, concepts: dd.DataFrame) -> dd.DataFrame:
        """
        Removes 'RESULT' column from concepts if column exists.
        Returns a dask dataframe.
        """
        if "RESULT" in concepts.columns:
            return concepts.drop(columns=["RESULT"])
        return concepts

    def _load_concept(self, concept_type: str):
        """
        Load concept data from formatted_data_dir.
        Expects TIMESTAMP column to be present.
        Returns a dask dataframe.
        """
        return load_concept(self.folder, concept_type)
