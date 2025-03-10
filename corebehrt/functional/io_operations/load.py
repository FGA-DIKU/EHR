"""Utils for loading data"""

import logging
from os.path import join
from typing import Dict, List, Set, Union

import pandas as pd
import torch

from corebehrt.constants.paths import VOCABULARY_FILE

logger = logging.getLogger(__name__)


def load_pids(files: Union[List, str]) -> Set:
    """Loads pids from multiple files or one file. Doesn't preserve order."""
    if isinstance(files, str):
        return set(torch.load(files, weights_only=True))
    pids = set()
    for file in files:
        pids.update(set(torch.load(file, weights_only=True)))
    return pids


def load_concept(path) -> pd.DataFrame:
    """
    Load concept data from formatted_data_dir.
    Expects time column to be present.
    Returns a pandas dataframe.
    """

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unknown file type: {path}")

    df["time"] = df["time"].dt.tz_localize(None)  # to prevent tz-naive/tz-aware issues
    return df


def load_vocabulary(dir_: str) -> Dict:
    """
    Load a vocabulary from the given directory.
    """
    return torch.load(join(dir_, VOCABULARY_FILE), weights_only=True)
