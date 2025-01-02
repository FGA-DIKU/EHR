import os
from os.path import join

import torch

VOCABULARY_FILE = "vocabulary.pt"


class Saver:
    """Save features, pids, vocabulary and sequence lengths to a folder"""

    def __init__(self, run_folder) -> None:
        self.run_folder = run_folder
        os.makedirs(self.run_folder, exist_ok=True)

    def save_train_val_pids(self, train_pids: list, val_pids: list) -> None:
        """Save train and val pids to a file."""
        torch.save(train_pids, join(self.run_folder, "pids_train.pt"))
        torch.save(val_pids, join(self.run_folder, "pids_val.pt"))

    def save_list(self, list_: list, name: str) -> None:
        """Save a list to a file"""
        torch.save(list_, join(self.run_folder, name))

    def save_vocab(self, vocabulary, name: str = VOCABULARY_FILE):
        """Save vocabulary to a file"""
        torch.save(vocabulary, join(self.run_folder, name))
