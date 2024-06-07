""" Random utils, should be structered later """
import pandas as pd
from typing import Union

def normalize_segments(x: Union[pd.Series, pd.DataFrame, list, dict], segment_col: str = 'segment'):
    if isinstance(x, pd.Series):
        return normalize_segments_series(x)
    elif isinstance(x, pd.DataFrame):
        return normalize_segments_df(x)
    elif isinstance(x, list):
        return normalize_segments_list(x)
    elif isinstance(x, dict):
        return normalize_segments_dict(x)
    else:
        raise TypeError('Invalid type for x, only pd.DataFrame, list, and dict are supported.')

def normalize_segments_df(df: pd.DataFrame, segment_col: str = 'segment') -> pd.DataFrame:
    return df.groupby('PID')[segment_col].transform(lambda x: normalize_segments_series(x))

def normalize_segments_series(series: pd.Series) -> pd.Series:
    return series.factorize()[0]

def normalize_segments_list(segments: list) -> list:
    segment_set = sorted(set(segments))
    correct_segments = list(range(len(segment_set)))
    converter = {k: v for (k,v) in zip(segment_set, correct_segments)}

    return [converter[segment] for segment in segments]

def normalize_segments_dict(features: dict, segment_col: str = 'segment') -> dict:
    for idx, segments in enumerate(features[segment_col]):
        features[segment_col][idx] = normalize_segments_list(segments)
    return features


def get_background_length(features: dict, vocabulary)-> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = set([v for k, v in vocabulary.items() if k.startswith('BG_')])
    example_concepts = features['concept'][0] # Assume that all patients have the same background length
    background_length = len(set(example_concepts) & background_tokens)

    return background_length + 2 # +2 for [CLS] and [SEP] tokens