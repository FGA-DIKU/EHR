""" Random utils, should be structered later """
import pandas as pd
from typing import Union

def normalize_segments(x: Union[pd.DataFrame, dict], segment_col: str = 'segment'):
    if isinstance(x, pd.DataFrame):
        return normalize_segments_df(x, segment_col)
    elif isinstance(x, dict): # TODO: Only used for Data.features (should be removed when pipeline is refactored)
        return normalize_segments_dict(x, segment_col)
    else:
        raise TypeError('Invalid type for x, only pd.DataFrame and dict are supported.')

def normalize_segments_df(df: pd.DataFrame, segment_col: str = 'segment') -> pd.DataFrame:
    return df.groupby('PID')[segment_col].transform(lambda x: x.factorize()[0])

def normalize_segments_dict(features: dict, segment_col: str = 'segment') -> dict:
    for idx, segments in enumerate(features[segment_col]):
        segment_set = sorted(set(segments))
        converter = {k: i for i, k in enumerate(segment_set)}
        features[segment_col][idx] = [converter[segment] for segment in segments]
    return features

def get_background_length(features: dict, vocabulary)-> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = set([v for k, v in vocabulary.items() if k.startswith('BG_')])
    example_concepts = features['concept'][0] # Assume that all patients have the same background length
    background_length = len(set(example_concepts) & background_tokens)

    return background_length + 2 # +2 for [CLS] and [SEP] tokens
