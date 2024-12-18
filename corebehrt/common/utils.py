import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch

logger = logging.getLogger(__name__)  # Get the logger for this module


@dataclass
class Data:
    features: dict = field(default_factory=dict)
    pids: list = field(default_factory=list)
    outcomes: Optional[List] = field(default=None)
    censor_outcomes: Optional[List] = field(default=None)
    vocabulary: Optional[Dict] = field(default=None)
    mode: Optional[str] = field(default=None)

    def __len__(self):
        return len(self.pids)

    def copy(self) -> "Data":
        """Create a copy of this Data object"""
        return Data(
            features=deepcopy(self.features),
            pids=deepcopy(self.pids),
            outcomes=deepcopy(self.outcomes) if self.outcomes is not None else None,
            censor_outcomes=(
                deepcopy(self.censor_outcomes)
                if self.censor_outcomes is not None
                else None
            ),
            vocabulary=deepcopy(self.vocabulary),
            mode=self.mode,
        )

    @classmethod
    def load_from_directory(cls, data_dir: str, mode: str) -> "Data":
        """Load data from data_dir."""

        def load_tensor(filename, required=False):
            """Helper function to load a tensor if it exists, otherwise return None"""
            filepath = join(data_dir, filename)
            if not os.path.exists(filepath):
                if required:
                    raise FileNotFoundError(f"{filename} not found in {data_dir}")
                else:
                    return None
            return torch.load(filepath)

        features = load_tensor(f"{mode}_features.pt", required=True)
        pids = load_tensor(f"{mode}_pids.pt", required=True)
        outcomes = load_tensor(f"{mode}_outcomes.pt")
        censor_outcomes = load_tensor(f"{mode}_censor_outcomes.pt")
        vocabulary = load_tensor("vocabulary.pt")
        return cls(features, pids, outcomes, censor_outcomes, vocabulary, mode=mode)

    def check_lengths(self):
        """Check that all features have the same length"""
        for key, values in self.features.items():
            assert len(values) == len(
                self.pids
            ), f"Length of {key} does not match length of pids"
        if self.outcomes is not None:
            assert len(self.outcomes) == len(
                self.pids
            ), "Length of outcomes does not match length of pids"
        if self.censor_outcomes is not None:
            assert len(self.censor_outcomes) == len(
                self.pids
            ), "Length of censor outcomes does not match length of pids"

    def split(self, val_split: float) -> Tuple["Data", "Data"]:
        """Split data into train and validation. Returns two Data objects"""
        train_indices, val_indices = self._get_train_val_splits(val_split)

        train_data = self.select_data_subset_by_indices(train_indices, "train")
        val_data = self.select_data_subset_by_indices(val_indices, "val")
        return train_data, val_data

    def select_data_subset_by_indices(self, indices: list, mode: str = "") -> "Data":
        return Data(
            features={
                key: [values[i] for i in indices]
                for key, values in self.features.items()
            },
            pids=[self.pids[i] for i in indices],
            outcomes=(
                [self.outcomes[i] for i in indices]
                if self.outcomes is not None
                else None
            ),
            censor_outcomes=(
                [self.censor_outcomes[i] for i in indices]
                if self.censor_outcomes is not None
                else None
            ),
            vocabulary=self.vocabulary,
            mode=mode,
        )

    def select_data_subset_by_pids(self, pids: list, mode: str = "") -> "Data":
        pid2index = {pid: index for index, pid in enumerate(self.pids)}
        if not set(pids).issubset(set(self.pids)):
            difference = len(set(pids).difference(set(self.pids)))
            logger.warning(
                "Selection pids for split {} is not a subset of the pids in the data. There are {} selection pids that are not in data pids.".format(
                    mode, difference
                )
            )
        logger.info(f"{len(pid2index)} pids in data")
        indices = [pid2index[pid] for pid in pids if pid in pid2index]
        logger.info(f"Selected {len(indices)} pids for split {mode}")
        return self.select_data_subset_by_indices(indices, mode)

    def _get_train_val_splits(self, split: float) -> Tuple[list, list]:
        """Randomly split a list of items into two lists of lengths determined by split"""
        assert split < 1 and split > 0, "Split must be between 0 and 1"
        indices = list(range(len(self.pids)))
        random.seed(42)
        random.shuffle(indices)
        split_index = int(len(indices) * (1 - split))
        return indices[:split_index], indices[split_index:]

    def exclude_pids(self, exclude_pids: List[str]) -> "Data":
        """Exclude pids from data."""
        logger.info(f"Excluding {len(exclude_pids)} pids")
        logger.info(f"Pids before exclusion: {len(self.pids)}")
        current_pids = self.pids
        data = self.select_data_subset_by_pids(
            list(set(current_pids).difference(set(exclude_pids))), mode=self.mode
        )
        logger.info(f"Pids after exclusion: {len(self.pids)}")
        return data

    def add_outcomes(self, outcomes: Union[List, Dict]):
        """Add outcomes to data"""
        self.outcomes = self._outcome_helper(outcomes)

    def add_index_dates(self, index_dates: Union[List, Dict]):
        """Add censor outcomes to data"""
        self.censor_outcomes = self._outcome_helper(index_dates)

    def _outcome_helper(self, outcomes_like: Union[List, Dict, pd.Series]) -> List:
        """Helper function to convert outcomes to list if necessary"""
        if isinstance(outcomes_like, dict):
            outcomes_like = [outcomes_like.get(pid, np.nan) for pid in self.pids]
        elif isinstance(outcomes_like, pd.Series):
            outcomes_like = [
                outcomes_like.loc[pid] if pid in outcomes_like else np.nan
                for pid in self.pids
            ]
        elif isinstance(outcomes_like, list) and (len(outcomes_like) != len(self.pids)):
            raise ValueError("Length of outcomes does not match length of pids")
        return outcomes_like
