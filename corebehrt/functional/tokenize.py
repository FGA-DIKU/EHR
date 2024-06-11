import dask.dataframe as dd

def get_change_of_segment(group):
    """Filter where 'segment' changes compared to the next entry."""
    sorted_group = group.sort_values('abspos')
    segment_change = sorted_group['segment'] != sorted_group['segment'].shift(-1)
    filtered_group = sorted_group[segment_change]
    return filtered_group

def get_first_event(group: dd.DataFrame):
    """Select the first event in the group."""
    return group.sort_values('abspos').head(1)

def add_token(
        features: dd.DataFrame, token: str, 
        group_func: callable, abspos_adjustment: float) -> dd.DataFrame:
    """
    Add a token to the DataFrame.
    First group on PID then apply a function to each group 
    to select the position of the token.
    The abspos_adjustment is used to ensure the tokens will be placed correctly.
    """
    features = features.sort_values(['PID', 'abspos'])
    events = features.groupby('PID').apply(group_func, meta=features)
    events['concept'] = token
    events['abspos'] = events['abspos'] + abspos_adjustment
    return dd.concat([features, events]).sort_values(['PID', 'abspos'])

def add_separator_token(df: dd.DataFrame, sep_token: str = '[SEP]') -> dd.DataFrame:
    """Add separator token after each segment in the dataframe"""
    return add_token(df, sep_token, get_change_of_segment, 1e-3)

def add_cls_token(df: dd.DataFrame, cls_token: str = '[CLS]'):
    """Add a classification token to the beginning of each patient's sequence"""
    return add_token(df, cls_token, get_first_event, -1e-3)

def tokenize(features: dd.DataFrame, vocabulary: dict, frozen: bool) -> dd.DataFrame:
    """Tokenize the features DataFrame."""
    pass
    

def limit_concept_length(features: dd.DataFrame, cutoffs: dict) -> dd.DataFrame:
    """Limit the length of the concepts."""
    for key, value in cutoffs.items():
        concept_mask = features['concept'].str.starts_with(key)
        features = features[concept_mask]
        # shorten the concepts
        features['concept'] = features['concept'].str.slice(0, value)
    return features