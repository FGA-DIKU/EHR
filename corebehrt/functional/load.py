""" Utils for loading data """

from typing import List, Union
import torch
import os
from os.path import join


# Taken from common.loader
def load_pids(files: Union[List, str]) -> List:
    """Loads pids from multiple files or one file. Doesn't preserve order."""
    if isinstance(files, str):
        return set(torch.load(files))
    pids = set()
    for file in files:
        pids.update(set(torch.load(file)))
    return pids


def load_predefined_pids(
    split_path: str, mode: Union[List, str] = ["train", "val"]
) -> List:
    if isinstance(split_path, List) or str(split_path).endswith(".pt"):
        pids = load_pids(split_path)
    elif os.path.exists(join(split_path, "pids.pt")):
        pids = load_pids(join(split_path, "pids.pt"))
    else:
        if isinstance(mode, str):
            mode = [mode]
        split_paths = [join(split_path, f"pids_{m}.pt") for m in mode]
        pids = load_pids(split_paths)
    return pids


def load_predefined_splits(split_path: str, splits: List = ["train", "val"]) -> List:
    for split in splits:
        if not os.path.exists(join(split_path, f"pids_{split}.pt")):
            raise FileNotFoundError(f"pids_{split}.pt not found in {split_path}")

    return [load_pids(join(split_path, f"pids_{split}.pt")) for split in splits]
