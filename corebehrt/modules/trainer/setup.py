import os
import re
from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler


def get_sampler(cfg, outcomes: List[int]):
    """Get sampler for training data.
    sample_weight: float. Adjusts the number of samples in the positive class.

    """

    def _inverse_sqrt(x):
        return 1 / np.sqrt(x)

    if cfg.trainer_args["sampler"]:
        _, counts = np.unique(np.array(outcomes), return_counts=True)
        label_weight = _inverse_sqrt(counts)

        sampler = WeightedRandomSampler(
            weights=label_weight, num_samples=len(outcomes), replacement=True
        )
        return sampler
    else:
        return None


def get_last_checkpoint_epoch(checkpoint_folder: str) -> int:
    """Returns the epoch of the last checkpoint."""
    # Regular expression to match the pattern retry_XXX
    pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
    max_epoch = None
    for filename in os.listdir(checkpoint_folder):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if max_epoch is None or epoch > max_epoch:
                max_epoch = epoch
    # Return the folder with the maximum retry number
    if max_epoch is None:
        raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))

    return max_epoch
