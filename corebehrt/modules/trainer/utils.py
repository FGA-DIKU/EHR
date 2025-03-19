from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler


def get_sampler(cfg, outcomes: List[int]):
    """Get sampler for training data.
    sample_weight: float. Adjusts the number of samples in the positive class.

    """

    def _inverse_sqrt(x):
        return 1 / np.sqrt(x)

    if cfg.trainer_args.get("sampler", None):
        _, counts = np.unique(np.array(outcomes), return_counts=True)
        label_weight = _inverse_sqrt(counts)

        sampler = WeightedRandomSampler(
            weights=label_weight, num_samples=len(outcomes), replacement=True
        )
        return sampler
    else:
        return None


def is_plateau(
    best_metric_value: float,
    current_metric_value: float,
    plateau_threshold: float = 0.01,
) -> bool:
    """
    Determines if training has reached a plateau by comparing current metric to best.

    Args:
        current_metric_value: The latest value of the monitored metric
        plateau_threshold: Relative improvement threshold below which we consider a plateau

    Returns:
        bool: True if a plateau is detected, False otherwise
    """
    # Calculate relative improvement (absolute value of the change divided by the best)
    relative_improvement = abs(current_metric_value - best_metric_value) / abs(
        best_metric_value
    )

    # We consider it a plateau if the relative improvement is less than the threshold
    return relative_improvement < plateau_threshold
