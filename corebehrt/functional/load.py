""" Utils for loading data """
from typing import Dict, List, Tuple, Union
import torch 


# Taken from common.loader
def load_pids(files: Union[List, str])->List:
    """Loads pids from multiple files or one file. Doesn't preserve order."""
    if isinstance(files, str):
        return set(torch.load(files))    
    pids = set()
    for file in files:
        pids.update(set(torch.load(file)))
    return pids