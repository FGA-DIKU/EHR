import pandas as pd
from corebehrt.functional.constants import (
    SPECIAL_TOKEN_ABSPOS_ADJUSTMENT,
    UNKNOWN_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
)


def add_special_tokens_partition(
    df: pd.DataFrame, add_sep=True, add_cls=True
) -> pd.DataFrame:
    """Efficiently add special tokens to a partition without full sorting."""
    special_rows = []
    df = df.reset_index(drop=True)

    if add_cls:
        # Find indices of the earliest event for each PID
        earliest_indices = df.groupby("PID")["abspos"].idxmin()
        # Create [CLS] rows
        cls_rows = df.loc[earliest_indices].copy()
        cls_rows["concept"] = CLS_TOKEN
        cls_rows[
            "abspos"
        ] -= SPECIAL_TOKEN_ABSPOS_ADJUSTMENT  # Adjust position to come before earliest event
        cls_rows["segment"] = 0
        special_rows.append(cls_rows)

    if add_sep:
        # Find segment changes within same PID
        segment_changes = (df["segment"] != df["segment"].shift(-1)) & (
            df["PID"] == df["PID"].shift(-1)
        )
        sep_rows = df[segment_changes].copy()
        sep_rows["concept"] = SEP_TOKEN
        sep_rows[
            "abspos"
        ] += SPECIAL_TOKEN_ABSPOS_ADJUSTMENT  # Adjust position slightly
        special_rows.append(sep_rows)

    # Combine all rows and sort by 'PID' and 'abspos'
    if special_rows:
        df = pd.concat([df] + special_rows, ignore_index=True)
        df = df.sort_values(["PID", "abspos"]).reset_index(drop=True)

    return df


def tokenize_partition(series: pd.Series, vocabulary: dict) -> pd.Series:
    """Optimized in-partition tokenization"""
    unk_token = vocabulary[UNKNOWN_TOKEN]
    # Use vectorized operation instead of map
    return series.map(lambda x: vocabulary.get(x, unk_token))


def limit_concept_length_partition(series: pd.Series, cutoffs: dict) -> pd.Series:
    """Efficiently limit concept lengths within a partition"""
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
