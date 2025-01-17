""" Utils for loading data """

import os
from glob import glob
from os.path import join
from typing import Dict, List, Set, Union

import dask.dataframe as dd
import torch

from corebehrt.common.setup import VOCABULARY_FILE


# Taken from common.loader
def load_pids(files: Union[List, str]) -> Set:
    """Loads pids from multiple files or one file. Doesn't preserve order."""
    if isinstance(files, str):
        return set(torch.load(files, weights_only=True))
    pids = set()
    for file in files:
        pids.update(set(torch.load(file, weights_only=True)))
    return pids


def load_predefined_splits(split_path: str, splits: List = ["train", "val"]) -> List:
    for split in splits:
        if not os.path.exists(join(split_path, f"pids_{split}.pt")):
            raise FileNotFoundError(f"pids_{split}.pt not found in {split_path}")

    return [load_pids(join(split_path, f"pids_{split}.pt")) for split in splits]


def load_patients_info(folder: str, pattern: str = "patients_info.*") -> dd.DataFrame:
    """
    Load patients_info data from formatted_data_dir.
    Expects BIRTHDATE and DEATHDATE columns to be present.
    Returns a dask dataframe.
    """
    file = get_file_with_pattern(folder, pattern)

    kwargs = {
        "parse_dates": ["BIRTHDATE"],
        "dtype": {"DEATHDATE": "object"},
        "assume_missing": True,
    }  # This helps with missing values in integer columns

    if file.endswith(".parquet"):
        df = dd.read_parquet(file, **kwargs)
    elif file.endswith(".csv"):
        df = dd.read_csv(file, **kwargs)
    df["DEATHDATE"] = dd.to_datetime(
        df["DEATHDATE"], errors="coerce", infer_datetime_format=True
    )
    return df


def load_concept(folder: str, concept_type: str) -> dd.DataFrame:
    """
    Load concept data from formatted_data_dir.
    Expects TIMESTAMP column to be present.
    Returns a dask dataframe.
    """
    file = get_file_with_pattern(folder, f"concept.{concept_type}.*")

    if file.endswith(".parquet"):
        df = dd.read_parquet(file, parse_dates=["TIMESTAMP"])
    elif file.endswith(".csv"):
        df = dd.read_csv(file, parse_dates=["TIMESTAMP"])
    else:
        raise ValueError(f"Unknown file type: {file}")

    df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_localize(
        None
    )  # to prevent tz-naive/tz-aware issues
    return df


def load_vocabulary(dir_: str) -> Dict:
    """
    Load a vocabulary from the given directory.
    """
    return torch.load(join(dir_, VOCABULARY_FILE), weights_only=True)


def get_file_with_pattern(folder: str, pattern: str) -> List[str]:
    """
    Returns the file in folder matching the pattern
    Throws an error if no file or multiple files are found.
    """
    pattern = join(folder, pattern)
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for '{pattern}' in folder '{folder}'")
    if len(files) > 1:
        raise ValueError(f"Multiple files found for '{pattern}' in folder '{folder}'")
    return files[0]
