import glob
import os
from typing import Iterator, List, Tuple

import pandas as pd

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL, SCHEMA
from corebehrt.constants.paths import (
    CONCEPT_FORMAT,
    CSV_EXT,
    PARQUET_EXT,
    PATIENTS_INFO_FORMAT,
)
from corebehrt.functional.io_operations.load import load_concept
from corebehrt.functional.setup.checks import (
    check_concepts_columns,
)


class ConceptLoader:
    """Load concepts and patient data"""

    def __init__(
        self, concepts=["diagnose", "medication"], data_dir: str = "formatted_data"
    ):
        # Create paths to relevant files
        concepts_paths = glob.glob(os.path.join(data_dir, CONCEPT_FORMAT))
        self.concepts_paths = [
            path
            for path in concepts_paths
            if os.path.basename(path).split(".")[1] in concepts
        ]
        print("concepts_paths", self.concepts_paths)
        self.patients_info_path = glob.glob(
            os.path.join(data_dir, PATIENTS_INFO_FORMAT)
        )

    def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.process()

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process concepts"""
        concepts = pd.concat(
            [self.read_file(p) for p in self.concepts_paths], ignore_index=True
        ).drop_duplicates()
        concepts = concepts.sort_values(TIMESTAMP_COL)

        """ Process patients info """
        patients_info = self.read_file(self.patients_info_path[0])

        return concepts, patients_info

    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        """Read csv or parquet file and return a DataFrame"""
        _, file_ext = os.path.splitext(file_path)
        if file_ext == CSV_EXT:
            df = pd.read_csv(file_path)
        elif file_ext == PARQUET_EXT:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        df = ConceptLoader._handle_types_columns(df)

        return df

    @staticmethod
    def _handle_types_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Try to convert all potential types columns to the correct type"""
        filtered_schema = {
            col: dtype for col, dtype in SCHEMA.items() if col in df.columns
        }
        df = df.astype(filtered_schema)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        return df


class ShardLoader:
    def __init__(
        self,
        data_dir: str,
        splits: List[str] = ["train", "tuning", "held_out"],
        patient_info_path: str = None,
    ):
        self.data_dir = data_dir
        self.splits = splits
        self.patient_info_path = (
            patient_info_path
            if patient_info_path
            else os.path.join(data_dir, "patients_info.parquet")
        )

    def __call__(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        return self.process()

    def process(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        if os.path.exists(self.patient_info_path):
            patients_info = self.read_file(self.patient_info_path)
        else:
            patients_info = None
        for split in self.splits:
            path_name = os.path.join(self.data_dir, split)
            if not os.path.exists(path_name):
                raise ValueError(f"Path {path_name} does not exist")

            for shard in [
                file for file in os.listdir(path_name) if not file.startswith(".")
            ]:
                shard_path = os.path.join(path_name, shard)
                df = self.read_file(shard_path)
                if patients_info is not None:
                    yield df, patients_info[
                        patients_info[PID_COL].isin(df[PID_COL].unique())
                    ]
                else:
                    yield df, None

    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        """Read csv or parquet file and return a DataFrame"""
        _, file_ext = os.path.splitext(file_path)
        if file_ext == CSV_EXT:
            df = pd.read_csv(file_path)
        elif file_ext == PARQUET_EXT:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        filtered_schema = {
            col: dtype for col, dtype in SCHEMA.items() if col in df.columns
        }
        df = df.astype(filtered_schema)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        return df


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
        load() -> Tuple[pd.DataFrame, pd.DataFrame]:
            Loads the concepts and patients_info DataFrames from the specified folder.
            Returns a tuple containing the concepts DataFrame and the patients_info DataFrame.
    """

    def __init__(
        self,
        path: str,
    ):
        self.path = path

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the concepts and patients_info DataFrames."""
        concepts = self._load_concept(self.path)
        check_concepts_columns(concepts)
        return concepts

    def _load_concept(self, path: str):
        """
        Load concept data from formatted_data_dir.
        Expects TIMESTAMP column to be present.
        Returns a pandas dataframe.
        """
        return load_concept(path)
