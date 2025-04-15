from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler
import pandas as pd
from corebehrt.modules.setup.config import instantiate_function


def get_sampler(cfg, outcomes: List[int]):
    """Get sampler for training data.
    If sampler_function is false or undefined, then no sampler is used.
    If sample_weight_function is defined then the function is used to calculate the weights.
    """
    sampler_function = cfg.trainer_args.get("sampler_function")
    if sampler_function:
        label_weight = instantiate_function(sampler_function)(outcomes)
        return WeightedRandomSampler(
            weights=label_weight, num_samples=len(outcomes), replacement=True
        )
    return None


class Sampling:
    @staticmethod
    def inverse_sqrt(x):
        """Calculate the inverse square root of a value or array."""
        return 1 / np.sqrt(x)

    @staticmethod
    def effective_n_samples(outcomes: List[int]):
        """Calculate weights using the effective number of samples method from paper https://arxiv.org/pdf/1901.05555."""
        labels = pd.Series(outcomes)
        beta = (len(outcomes) - 1) / len(outcomes)
        n0 = labels.value_counts()[0]
        n1 = labels.value_counts()[1]
        E0 = (1 - (beta**n0)) / (1 - beta)
        E1 = (1 - (beta**n1)) / (1 - beta)
        probs = [E0 / (E0 + E1), E1 / (E0 + E1)]
        weights = [probs[label] / labels.value_counts()[label] for label in labels]
        return weights


def get_loss_weight(cfg, outcomes: List[int]):
    """Get weights for weighted loss function.
    If loss_weight_function is false or undefined, then no positive weight is used.
    If loss_weight_function is defined then the function is used to calculate the weights.
    """
    if cfg.trainer_args.get("loss_weight_function") is None or len(outcomes) == 0:
        return None

    weight_func = instantiate_function(cfg.trainer_args.get("loss_weight_function"))
    return weight_func(outcomes)


class PositiveWeight:
    @staticmethod
    def sqrt(outcomes: List[int]):
        """Calculate the square root of the ratio of negative to positive samples."""
        outcomes_series = pd.Series(outcomes)
        num_pos = (outcomes_series == 1).sum()
        num_neg = (outcomes_series == 0).sum()
        return np.sqrt(num_neg / num_pos)

    @staticmethod
    def effective_n_samples(outcomes: List[int]):
        """Calculate positive weight using the effective number of samples method."""
        labels = pd.Series(outcomes).astype(int)
        beta = (len(outcomes) - 1) / (len(outcomes))
        n0 = labels.value_counts()[0]
        n1 = labels.value_counts()[1]
        alpha_0 = (1 - beta) / (1 - beta**n0)
        alpha_1 = (1 - beta) / (1 - beta**n1)
        pos_weight = alpha_1 / alpha_0
        return pos_weight


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
