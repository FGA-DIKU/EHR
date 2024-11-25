import pandas as pd


def add_special_tokens_partition(
    df: pd.DataFrame, add_sep=True, add_cls=True, sep_token="[SEP]", cls_token="[CLS]"
) -> pd.DataFrame:
    """Efficiently add special tokens to a partition"""
    special_rows = []
    df = df.reset_index(drop=True)

    if add_cls:
        # Get first row for each PID
        cls_rows = df.groupby("PID", as_index=False).first().copy()
        cls_rows["concept"] = cls_token
        cls_rows["abspos"] -= 1e-3
        cls_rows["segment"] = 0
        special_rows.append(cls_rows)  # Add CLS rows to special_rows

    if add_sep:
        # Find segment changes within same PID
        segment_changes = (df["segment"] != df["segment"].shift(-1)) & (
            df["PID"] == df["PID"].shift(-1)
        )
        sep_rows = df[segment_changes].copy()
        sep_rows["concept"] = sep_token
        sep_rows["abspos"] += 1e-3
        special_rows.append(sep_rows)

    # Combine all rows and sort
    if special_rows:
        df = pd.concat([df] + special_rows, ignore_index=True)
        df = df.sort_values(["PID", "abspos"])

    return df


def tokenize_partition(series: pd.Series, vocabulary: dict) -> pd.Series:
    """Optimized in-partition tokenization"""
    unk_token = vocabulary["[UNK]"]
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
