from dataclasses import asdict, fields
from typing import List

from corebehrt.modules.preparation.dataset import PatientData


def get_background_length(patients: List[PatientData], vocabulary) -> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = set([v for k, v in vocabulary.items() if k.startswith("BG_")])
    example_concepts = patients[0].concepts
    background_length = len(set(example_concepts) & background_tokens)
    return background_length + 2  # +2 for [CLS] and [SEP] tokens


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
