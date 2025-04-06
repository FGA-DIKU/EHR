from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler
import pandas as pd


def get_sampler(cfg, outcomes: List[int]):
    """Get sampler for training data.
    sample_weight: float. Adjusts the number of samples in the positive class.
    If sampler is false, no sampler is used.
    If sample_weight_function is not defined, the default is inverse_sqrt.
    Other option is effective_n_samples from the paper https://arxiv.org/pdf/1901.05555
    """

    def _inverse_sqrt(x):
        return 1 / np.sqrt(x)

    def _effective_n_samples(outcomes):
        labels = pd.Series(outcomes)
        beta = (len(outcomes) - 1) / len(outcomes)
        n0 = labels.value_counts()[0]
        n1 = labels.value_counts()[1]
        E0 = (1 - (beta**n0)) / (1 - beta)
        E1 = (1 - (beta**n1)) / (1 - beta)
        probs = [E0 / (E0 + E1), E1 / (E0 + E1)]
        weights = [probs[label] / labels.value_counts()[label] for label in labels]
        return weights

    if cfg.trainer_args.get("sampler", None):
        _, counts = np.unique(np.array(outcomes), return_counts=True)
        if (
            cfg.trainer_args.get("sample_weight_function", None)
            == "effective_n_samples"
        ):
            label_weight = _effective_n_samples(outcomes)
        else:
            label_weight = _inverse_sqrt(counts)

        sampler = WeightedRandomSampler(
            weights=label_weight, num_samples=len(outcomes), replacement=True
        )
        return sampler
    else:
        return None


def get_pos_weight(cfg, outcomes):
    """Get positive weight for loss function."""
    pos_weight = cfg.trainer_args.get("pos_weight", None)
    if pos_weight is None or len(outcomes) == 0:
        return None
    if pos_weight == "sqrt":
        outcomes_series = pd.Series(outcomes)
        num_pos = (outcomes_series == 1).sum()
        num_neg = (outcomes_series == 0).sum()
        return np.sqrt(num_neg / num_pos)
    elif pos_weight == "effective_n_samples":
        labels = pd.Series(outcomes).astype(int)
        beta = (len(outcomes) - 1) / (len(outcomes))
        n0 = labels.value_counts()[0]
        n1 = labels.value_counts()[1]
        alpha_0 = (1 - beta) / (1 - beta**n0)
        alpha_1 = (1 - beta) / (1 - beta**n1)
        pos_weight = alpha_1 / alpha_0
        return pos_weight
    elif isinstance(pos_weight, (int, float)):
        return pos_weight
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
