from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

from corebehrt.modules.setup.config import instantiate_function


def get_sampler(cfg, outcomes: List[int]) -> Optional[WeightedRandomSampler]:
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


def compute_labels(outcomes: List[int]) -> Dict[int, int]:
    """Compute the labels for the outcomes."""
    labels = pd.Series(outcomes)
    counts = labels.value_counts()
    print(f"Class counts:\n{labels.to_string()}")

    if len(counts) < 2:
        raise ValueError(
            f"Found only {len(counts)} class(es) in the data. Multi-class sampling/weighting requires at least 2 classes."
        )
    return counts


class Sampling:
    @staticmethod
    def inverse_sqrt(outcomes: List[int]) -> List[float]:
        """Calculate the inverse square root of class frequencies.

        Args:
            outcomes: List of class labels

        Returns:
            List[float]: Weight for each sample based on inverse square root of class frequency

        Raises:
            ValueError: If data contains less than 2 classes
        """
        labels = compute_labels(outcomes)  # Will raise ValueError if < 2 classes
        weights = 1 / np.sqrt(labels)
        # Map weights back to samples
        return [weights[label] for label in outcomes]

    @staticmethod
    def effective_n_samples(outcomes: List[int]) -> List[float]:
        """Calculate weights using the effective number of samples method.

        Args:
            outcomes: List of class labels

        Returns:
            List[float]: Weight for each sample based on effective number of samples

        Raises:
            ValueError: If data contains less than 2 classes
        """
        labels = compute_labels(outcomes)  # Will raise ValueError if < 2 classes
        # Calculate beta as per the paper
        beta = (len(outcomes) - 1) / len(outcomes)

        # Calculate effective number for each class
        effective_nums = {
            label: (1 - (beta**count)) / (1 - beta) for label, count in labels.items()
        }

        # Calculate class probabilities
        total_effective = sum(effective_nums.values())
        class_probs = {
            label: eff_num / total_effective
            for label, eff_num in effective_nums.items()
        }

        # Calculate weights for each sample
        return [class_probs[outcome] / labels[outcome] for outcome in outcomes]


def get_loss_weight(cfg, outcomes: List[int]) -> Optional[List[float]]:
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
    def sqrt(outcomes: List[int]) -> float:
        """Calculate the square root of the ratio of negative to positive samples.

        Args:
            outcomes: List of binary labels (0 or 1)

        Returns:
            float: Square root of negative to positive ratio

        Raises:
            ValueError: If data contains less than 2 classes or no positive samples
        """
        labels = compute_labels(outcomes)  # Will raise ValueError if < 2 classes
        num_neg = labels.get(0, 0)
        num_pos = labels.get(1, 0)

        if num_pos == 0:
            raise ValueError("No positive samples (class 1) found in the dataset")
        if num_neg == 0:
            raise ValueError("No negative samples (class 0) found in the dataset")

        return np.sqrt(num_neg / num_pos)

    @staticmethod
    def effective_n_samples(outcomes: List[int]) -> float:
        """Calculate positive weight using the effective number of samples method.

        Args:
            outcomes: List of binary labels (0 or 1)

        Returns:
            float: Weight ratio based on effective number of samples

        Raises:
            ValueError: If data contains less than 2 classes or missing classes
        """
        labels = compute_labels(outcomes)  # Will raise ValueError if < 2 classes
        n0 = labels.get(0)
        n1 = labels.get(1)

        if n0 is None or n1 is None:
            raise ValueError(
                "Both classes (0 and 1) must be present for binary classification"
            )

        beta = (len(outcomes) - 1) / len(outcomes)
        alpha_0 = (1 - beta) / (1 - beta**n0)
        alpha_1 = (1 - beta) / (1 - beta**n1)
        return alpha_1 / alpha_0


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
