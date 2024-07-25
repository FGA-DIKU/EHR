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


def add_separator_token(
    features: dd.DataFrame, sep_token: str = "[SEP]"
) -> dd.DataFrame:
    """Add separator token after each segment in the dataframe"""
    return _add_token(features, sep_token, _get_segment_change, 1e-3)


def add_cls_token(features: dd.DataFrame, cls_token: str = "[CLS]"):
    """Add a classification token to the beginning of each patient's sequence"""
    return _add_token(features, cls_token, _get_first_event, -1e-3)


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


def tokenize_with_update(concepts: dd.Series, vocabulary: dict) -> dd.Series:
    """Tokenize the concepts in the features DataFrame and update the vocabulary."""
    vocabulary = {**vocabulary}
    unique_concepts = concepts.unique().compute()
    for concept in unique_concepts:
        if concept not in vocabulary:
            vocabulary[concept] = len(vocabulary)
    concepts = concepts.map(lambda x: vocabulary[x], meta=("concept", "int64"))
    return concepts, vocabulary


def tokenize_frozen(concepts: dd.Series, vocabulary: dict) -> dd.Series:
    """Tokenize the concepts in the features DataFrame with a frozen vocabulary."""
    return concepts.map(
        lambda x: vocabulary.get(x, vocabulary["[UNK]"]), meta=("concept", "int64")
    )


def limit_concept_length(concepts: dd.Series, cutoffs: dict) -> dd.Series:
    """Limit the length of the concepts based on cutoffs."""
    for key, value in cutoffs.items():
        mask = concepts.str.startswith(key)
        concepts = concepts.mask(mask, concepts.str.slice(0, value))
    return concepts
