import pandas as pd
from corebehrt.functional.constants import UNKNOWN_TOKEN


def tokenize_partition(series: pd.Series, vocabulary: dict) -> pd.Series:
    """Optimized in-partition tokenization using direct dictionary mapping."""
    unk_token = vocabulary[UNKNOWN_TOKEN]
    # Direct mapping with fillna for unknown tokens
    return series.map(vocabulary).fillna(unk_token).astype(int)


def limit_concept_length_partition(series: pd.Series, cutoffs: dict) -> pd.Series:
    """Efficiently limit concept lengths within a partition.

    Args:
        series: Pandas Series containing concepts
        cutoffs: Dict mapping prefixes to max lengths, e.g. {'D': 6, 'M': 4}
            Will limit concepts starting with 'D' to 6 chars, 'M' to 4 chars.

    Example:
        With cutoffs={'D': 4}, 'D123456' becomes 'D1234'
    """
    # Create a copy to avoid modifying original
    result = series.copy()

    # Vectorized operations for each prefix
    for prefix, length in cutoffs.items():
        # Create mask for matching prefix
        mask = result.str.startswith(prefix)

        # Apply length limit only where mask is True
        if mask.any():
            result.loc[mask] = result.loc[mask].str[:length]

    return result
