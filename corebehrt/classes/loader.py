from typing import List, Tuple

import dask.dataframe as dd

from corebehrt.functional.load import load_concept, load_patients_info


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

    Methods:
        load() -> Tuple[dd.DataFrame, dd.DataFrame]:
            Loads the concepts and patients_info DataFrames from the specified folder.
            Returns a tuple containing the concepts DataFrame and the patients_info DataFrame.
    """

    def __init__(self, folder: str, concept_types: List[str]):
        self.folder = folder
        self.concept_types = concept_types

    def load(self) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """Loads the concepts and patients_info DataFrames."""
        concepts = [
            self.load_concept(concept_type) for concept_type in self.concept_types
        ]
        concepts = dd.concat(concepts)
        self.check_concepts_columns(concepts)

        patients_info = self.load_patients_info()
        self.check_patients_info_columns(patients_info)

        return concepts, patients_info

    def check_concepts_columns(self, concepts: dd.DataFrame):
        """Checks if the concepts DataFrame has the required columns."""
        required_columns = ["PID", "TIMESTAMP", "CONCEPT", "ADMISSION_ID"]
        for column in required_columns:
            if column not in concepts.columns:
                raise ValueError(f"{column} not found in concepts columns.")

    def check_patients_info_columns(self, patients_info: dd.DataFrame):
        """Checks if the patients_info DataFrame has the required columns."""
        required_columns = ["PID", "BIRTHDATE", "DEATHDATE"]
        for column in required_columns:
            if column not in patients_info.columns:
                raise ValueError(f"{column} not found in patients_info columns.")

    def load_patients_info(self) -> dd.DataFrame:
        """
        Load patients_info data from formatted_data_dir.
        Expects BIRTHDATE and DEATHDATE columns to be present.
        Returns a dask dataframe.
        """
        return load_patients_info(self.folder)

    def load_concept(self, concept_type: str):
        """
        Load concept data from formatted_data_dir.
        Expects TIMESTAMP column to be present.
        Returns a dask dataframe.
        """
        return load_concept(self.folder, concept_type)
