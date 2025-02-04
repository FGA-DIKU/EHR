import numpy as np
from typing import List
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
