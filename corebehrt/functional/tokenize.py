import re

import dask.dataframe as dd
import pandas as pd


def add_special_tokens_partition(
    df: pd.DataFrame, add_sep=True, add_cls=True, sep_token="[SEP]", cls_token="[CLS]"
) -> pd.DataFrame:
    """Efficiently add special tokens to a partition"""
    # Create special token rows
    special_rows = []
    df = df.reset_index(drop=True)
    if add_cls:
        # Get first row for each PID
        cls_rows = df.groupby("PID", as_index=False).first().copy()
        cls_rows["concept"] = cls_token
        cls_rows["abspos"] -= 1e-3
        cls_rows["segment"] = 0

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

    # df = df.astype(df.dtypes.to_dict())
    return df


def tokenize(concepts: dd.Series, vocabulary: dict, frozen_vocab: bool) -> dd.Series:
    """
    Tokenize the given concepts.
    if frozen_vocab is True, the vocabulary will not be updated.
    And the tokens will be replaced with the UNK token.
    """
    if frozen_vocab:
        concepts = tokenize_frozen(concepts, vocabulary)
    else:
        concepts, vocabulary = tokenize_with_update(concepts, vocabulary)

    return concepts, vocabulary


def tokenize_with_update(
    concepts: dd.Series, vocabulary: dict
) -> tuple[dd.Series, dict]:
    # Extract unique concepts in parallel without computing immediately
    unique_concepts = concepts.drop_duplicates().compute()

    # Determine new concepts not in the existing vocabulary
    existing_concepts = set(vocabulary.keys())
    new_concepts = [c for c in unique_concepts if c not in existing_concepts]

    # Assign new indices to new concepts
    start_index = max(vocabulary.values()) + 1
    new_vocab_entries = {
        concept: idx for idx, concept in enumerate(new_concepts, start=start_index)
    }

    # Update the vocabulary
    vocabulary.update(new_vocab_entries)

    # Broadcast the vocabulary to the workers
    broadcast_vocab = vocabulary.copy()

    # Map concepts to indices using the updated vocabulary
    concepts = concepts.map(broadcast_vocab.get, meta=("concept", "int64"))

    return concepts, vocabulary


def tokenize_frozen(concepts: dd.Series, vocabulary: dict) -> dd.Series:
    """Tokenize the concepts in the features DataFrame with a frozen vocabulary."""
    return concepts.map(
        lambda x: vocabulary.get(x, vocabulary["[UNK]"]), meta=("concept", "int64")
    )


def limit_concept_length(concepts: dd.Series, cutoffs: dict) -> dd.Series:
    # Precompile regex patterns for efficiency
    patterns = {prefix: re.compile(f"^{re.escape(prefix)}") for prefix in cutoffs}

    def apply_cutoffs(series):
        for prefix, length in cutoffs.items():
            pattern = patterns[prefix]
            mask = series.str.match(pattern)
            series = series.where(~mask, series.str.slice(0, length))
        return series

    return concepts.map_partitions(apply_cutoffs, meta=concepts)
