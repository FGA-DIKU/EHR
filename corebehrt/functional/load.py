""" Utils for loading data """
import os
from os.path import join
from typing import Dict, List, Set, Union
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


def load_vocabulary(dir_: str) -> Dict:
    """
    Load a vocabulary from the given directory.
    """
    return torch.load(join(dir_, VOCABULARY_FILE), weights_only=True)
