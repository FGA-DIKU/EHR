import glob
import os
import random
from datetime import datetime
from typing import Iterator, Tuple

import dateutil
import pandas as pd
import pyarrow.parquet as pq

from corebehrt.common.constants import (
    CONCEPT_FORMAT,
    CSV_EXT,
    PARQUET_EXT,
    PATIENTS_INFO_FORMAT,
    PID_COL,
    TIMESTAMP_COL,
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

    def read_file(self, file_path: str) -> pd.DataFrame:
        """Read csv or parquet file and return a DataFrame"""
        _, file_ext = os.path.splitext(file_path)
        if file_ext == CSV_EXT:
            df = pd.read_csv(file_path)
        elif file_ext == PARQUET_EXT:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        return self._handle_datetime_columns(df)

    @classmethod
    def _handle_datetime_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Try to convert all potential datetime columns to datetime objects"""
        for col in cls._detect_date_columns(df):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].dt.tz_localize(None)
        return df

    @staticmethod
    def _detect_date_columns(df: pd.DataFrame) -> Iterator[str]:
        for col in df.columns:
            if isinstance(df[col], datetime):
                continue
            if "TIME" in col.upper() or "DATE" in col.upper():
                try:
                    first_non_na = df.loc[df[col].notna(), col].iloc[0]
                    dateutil.parser.parse(first_non_na)
                    yield col
                except:
                    continue


class ConceptLoaderLarge(ConceptLoader):
    """Load concepts and patient data in chunks"""

    def __init__(
        self,
        concepts: list = ["diagnosis", "medication"],
        data_dir: str = "formatted_data",
        chunksize=10000,
        batchsize=100000,
    ):
        super().__init__(concepts, data_dir)
        self.chunksize = chunksize
        self.batchsize = batchsize

    def __call__(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        return self.process()

    def process(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        patients_info = self.read_file(self.patients_info_path[0])
        patient_ids = patients_info[PID_COL].unique()
        random.seed(42)
        random.shuffle(patient_ids)

        for chunk_ids in self.get_patient_batch(patient_ids, self.batchsize):
            concepts_chunk = pd.concat(
                [self.read_file_chunk(p, chunk_ids) for p in self.concepts_paths],
                ignore_index=True,
            ).drop_duplicates()
            concepts_chunk = concepts_chunk.sort_values(by=[PID_COL, TIMESTAMP_COL])
            patients_info_chunk = patients_info[patients_info[PID_COL].isin(chunk_ids)]
            yield concepts_chunk, patients_info_chunk

    def read_file_chunk(self, file_path: str, chunk_ids: list = None) -> pd.DataFrame:
        chunks = []
        for chunk in self._get_iterator(file_path, self.chunksize):
            filtered_chunk = chunk[
                chunk[PID_COL].isin(chunk_ids)
            ]  # assuming 'PID' is the patient ID column in this file too
            chunks.append(filtered_chunk)
        chunks_df = pd.concat(chunks, ignore_index=True)

        return self._handle_datetime_columns(chunks_df)

    @staticmethod
    def _get_iterator(file_path: str, chunksize: int) -> Iterator[pd.DataFrame]:
        _, file_ext = os.path.splitext(file_path)
        if file_ext == CSV_EXT:
            return pd.read_csv(file_path, chunksize=chunksize)
        elif file_ext == PARQUET_EXT:
            return ParquetIterator(file_path, chunksize)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    @staticmethod
    def get_patient_batch(patient_ids: list, batch_size: int) -> Iterator[list]:
        for i in range(0, len(patient_ids), batch_size):
            yield patient_ids[i : i + batch_size]


class ParquetIterator:
    def __init__(self, filename, batch_size=100000):
        parquet_file = pq.ParquetFile(filename)
        self.batch_iterator = parquet_file.iter_batches(batch_size=batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.batch_iterator)
            return batch.to_pandas()
        except StopIteration:
            raise StopIteration
