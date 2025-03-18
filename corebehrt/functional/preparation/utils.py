from dataclasses import asdict, fields
from typing import List

import pandas as pd

from corebehrt.modules.preparation.dataset import PatientData
from corebehrt.constants.data import CONCEPT_COL
import re


def get_background_length(patients: List[PatientData], vocabulary) -> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = get_background_tokens(vocabulary)
    example_concepts = patients[0].concepts
    background_length = len(set(example_concepts) & background_tokens)
    return background_length + 2  # +2 for [CLS] and [SEP] tokens


def get_background_length_pd(df: pd.DataFrame, vocabulary: dict) -> int:
    """
    Get the length of the background sentence for a pandas DataFrame, including first SEP token.

    Args:
        df: pandas DataFrame containing patient data. Assumes PID to be index
        vocabulary: Dictionary mapping tokens to IDs

    Returns:
        int: Length of background sequence including [CLS] and [SEP] tokens
    """
    background_tokens = get_background_tokens(vocabulary)
    # Get first index value
    if len(df) == 0:
        return 2
    first_idx = df.index[0]
    # Get data for first patient using index
    sub = df.loc[first_idx]
    example_concepts = set(sub[CONCEPT_COL].unique())
    background_length = len(example_concepts & background_tokens)
    return background_length + 2


def get_background_tokens(vocabulary: dict) -> set:
    return set([v for k, v in vocabulary.items() if k.startswith("BG_")])


def get_non_priority_tokens(vocabulary: dict, low_priority_prefixes: List[str]) -> set:
    """

    Get tokens that start with low_priority_prefixes.
    """
    return {
        v
        for k, v in vocabulary.items()
        if any(k.startswith(prefix) for prefix in low_priority_prefixes)
    }


def subset_patient_data(patient: PatientData, keep_indices: List[int]) -> PatientData:
    """
    Return a new PatientData containing only the rows at `keep_indices`
    for all list-type attributes. Non-list attributes remain unchanged.
    """
    # Convert the PatientData instance to a dictionary
    data = asdict(patient)

    # For each field in the dataclass, if the value is a list, subset it.
    # Otherwise, keep it as is.
    for f in fields(PatientData):
        val = data[f.name]
        if isinstance(val, list):
            data[f.name] = [val[i] for i in keep_indices]

    # Recreate a new PatientData from the updated dictionary
    return PatientData(**data)


def is_valid_regex(pattern):
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False
