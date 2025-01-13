import pandas as pd
from corebehrt.functional.constants import UNKNOWN_TOKEN


def tokenize_partition(series: pd.Series, vocabulary: dict) -> pd.Series:
    """Optimized in-partition tokenization using direct dictionary mapping."""
    unk_token = vocabulary[UNKNOWN_TOKEN]
    # Direct mapping with fillna for unknown tokens
    return series.map(vocabulary).fillna(unk_token).astype(int)
