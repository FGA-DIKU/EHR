import dask.dataframe as dd


def _get_segment_change(df: dd.DataFrame) -> dd.Series:
    """Select entries where segment changes."""
    return (df.segment != df.segment.shift(-1)) | (df.PID != df.PID.shift(-1))


def _get_first_event(df: dd.DataFrame) -> dd.Series:
    """Select the first event in the group based on abspos."""
    return (df.PID != df.PID.shift(1)).fillna(True)


def _add_token(
    df: dd.DataFrame, token: str, mask_func: callable, abspos_adjustment: float
) -> dd.DataFrame:
    """
    Insert tokens into the DataFrame based on the mask_func applied to the sorted (PID,abspos) dataframe.
    The abspos_adjustment is used to ensure the tokens will be placed correctly in the final sequences.
    """
    df = df.sort_values(["PID", "abspos"])
    df_change = df[mask_func(df)]
    df_change["concept"] = token
    df_change["abspos"] += abspos_adjustment
    return dd.concat([df, df_change]).sort_values(["PID", "abspos"])


def add_separator_token(df: dd.DataFrame, sep_token: str = "[SEP]") -> dd.DataFrame:
    """Add separator token after each segment in the dataframe"""
    return _add_token(df, sep_token, _get_segment_change, 1e-3)


def add_cls_token(df: dd.DataFrame, cls_token: str = "[CLS]"):
    """Add a classification token to the beginning of each patient's sequence"""
    return _add_token(df, cls_token, _get_first_event, -1e-3)


def tokenize(
    features: dd.DataFrame, vocabulary: dict, frozen_vocab: bool
) -> dd.DataFrame:
    """
    Tokenize the concepts in the features DataFrame.
    if frozen_vocab is True, the vocabulary will not be updated.
    And the tokens will be replaced with the UNK token.
    """
    if frozen_vocab:
        features = tokenize_frozen(features, vocabulary)
    else:
        features = tokenize_with_update(features, vocabulary)

    return features, vocabulary


def tokenize_with_update(features: dd.DataFrame, vocabulary: dict) -> dd.DataFrame:
    """Tokenize the concepts in the features DataFrame and update the vocabulary."""
    unique_concepts = features["concept"].unique().compute()
    for concept in unique_concepts:
        if concept not in vocabulary:
            vocabulary[concept] = len(vocabulary)
    features["concept"] = features["concept"].map(
        lambda x: vocabulary[x], meta=("concept", "int64")
    )
    return features


def tokenize_frozen(features: dd.DataFrame, vocabulary: dict) -> dd.DataFrame:
    """Tokenize the concepts in the features DataFrame with a frozen vocabulary."""
    features["concept"] = features["concept"].map(
        lambda x: vocabulary.get(x, vocabulary["[UNK]"]), meta=("concept", "int64")
    )
    return features


def limit_concept_length(features: dd.DataFrame, cutoffs: dict) -> dd.DataFrame:
    """Limit the length of the concepts based on cutoffs."""
    for key, value in cutoffs.items():
        type_mask = features["concept"].str.starts_with(key)
        # shorten the concepts
        features["concept"] = features.loc[type_mask, "concept"].str.slice(0, value)
    return features
