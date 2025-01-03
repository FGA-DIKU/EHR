from collections import defaultdict
from dataclasses import fields, replace
from typing import List, Optional

from corebehrt.classes.dataset import PatientData
from corebehrt.functional.utils import subset_patient_data


def truncate_patient(
    patient: PatientData,
    background_length: int,
    max_len: int,
    sep_token: str,
    non_priority_tokens: Optional[set] = None,
) -> PatientData:
    """
    Truncate a patient's data to a maximum length while preserving important tokens.

    This function dispatches to one of two strategies:
      1. Standard truncation (front + back) if `non_priority_tokens` is None.
      2. Prioritized truncation, which drops non-priority tokens first,
         if `non_priority_tokens` is provided.

    Args:
        patient (PatientData): The patient data to truncate.
        background_length (int): Number of background tokens to preserve at the start.
        max_len (int): Maximum allowed length of the truncated data.
        sep_token (str): The separator token used in the sequence.
        non_priority_tokens (set, optional): Tokens that can be dropped first
            during truncation. Defaults to None.

    Returns:
        PatientData: A new (truncated) PatientData object.
    """
    if non_priority_tokens is None:
        return standard_truncate_patient(patient, background_length, max_len, sep_token)
    return prioritized_truncate_patient(
        patient, background_length, max_len, sep_token, non_priority_tokens
    )


def standard_truncate_patient(
    patient: PatientData,
    background_length: int,
    max_len: int,
    sep_token: str,
) -> PatientData:
    """
    Truncate a patient's data to `max_len`, preserving:
      - The first `background_length` items at the start.
      - The remainder from the end (tail) until reaching `max_len`.

    If the boundary item in the tail portion is a SEP token, reduce
    the tail length by 1 to avoid awkward splits.

    Args:
        patient (PatientData): The patient to truncate.
        background_length (int): Number of items to preserve from the start.
        max_len (int): Desired maximum length after truncation.
        sep_token (str): The separator token.

    Returns:
        PatientData: A new object reflecting the truncated data.
    """
    total_length = len(patient.concepts)
    if total_length <= max_len:
        return patient

    # Determine how many items from the end we can keep
    tail_length = max_len - background_length

    # If the boundary element is the SEP token, shift tail_length by 1
    if tail_length > 0 and patient.concepts[-tail_length] == sep_token:
        tail_length = max(tail_length - 1, 0)

    # Build truncated lists for each field
    truncated_values = {}
    for f in fields(patient):
        val = getattr(patient, f.name)
        if isinstance(val, list):
            truncated_values[f.name] = val[:background_length] + val[-tail_length:]
        else:
            truncated_values[f.name] = val

    return replace(patient, **truncated_values)


def prioritized_truncate_patient(
    patient: PatientData,
    background_length: int,
    max_len: int,
    sep_token: str,
    non_priority_tokens: set,
    unit: bool = False,
    low_priority_prefixes: Optional[List[str]] = None,
) -> PatientData:
    """
    Truncate a patient's data to `max_len`, preserving `background_length` items
    from the start, while deprioritizing (dropping) tokens in `non_priority_tokens` first.

    If `unit=True`, also ensure that all subunits corresponding to
    `low_priority_prefixes` remain together; otherwise, they're removed.

    Args:
        patient (PatientData): The patient to truncate.
        background_length (int): Number of items to preserve from the start.
        max_len (int): Desired maximum length after truncation.
        sep_token (str): The separator token.
        non_priority_tokens (set): The set of tokens that can be dropped first.
        unit (bool, optional): If True, remove positions that don't contain
            the full set of prefixes. Defaults to False.
        low_priority_prefixes (List[str], optional): The list of prefix strings
            that define a “unit.” Only relevant if `unit=True`.

    Returns:
        PatientData: A new (truncated) PatientData object.
    """
    if len(patient.concepts) <= max_len:
        return patient

    # 1) Compute how many tokens we can keep beyond background
    truncation_length = max_len - background_length

    # 2) Identify non-priority tokens
    non_priority_mask = _get_non_priority_mask(patient, non_priority_tokens)

    # 3) Drop non-priority tokens (beyond background) as needed
    patient = _drop_non_priority_tokens(
        patient, non_priority_mask, truncation_length, background_length
    )

    # 4) If unit=True, filter out incomplete sets of low-priority prefixes
    if unit and low_priority_prefixes:
        updated_mask = _get_non_priority_mask(patient, non_priority_tokens)
        patient = _filter_invalid_positions(
            patient, updated_mask, low_priority_prefixes
        )

    # 5) Finally, apply standard truncation if still longer than `max_len`
    return standard_truncate_patient(patient, background_length, max_len, sep_token)


def _drop_non_priority_tokens(
    patient: PatientData,
    non_priority_mask: List[bool],
    truncation_length: int,
    background_length: int,
) -> PatientData:
    """
    Drop non-priority tokens (beyond the background region) until the patient
    fits into the desired length (`background_length + truncation_length`).

    Steps:
      1. Compute the set of non-priority indices that lie beyond `background_length`.
      2. If dropping them all still exceeds our target length, drop them all.
      3. Otherwise, drop only the number required to reach the target length.

    Args:
        patient (PatientData): The patient's data.
        non_priority_mask (List[bool]): True for non-priority tokens, False otherwise.
        truncation_length (int): How many tokens can remain beyond background.
        background_length (int): How many tokens we must preserve at the start.

    Returns:
        PatientData: A new PatientData with selected tokens removed.
    """
    total_tokens = len(patient.concepts)
    target_total = background_length + truncation_length

    # Identify non-priority indices beyond the background region
    beyond_bg_indices = [
        i
        for i, is_np in enumerate(non_priority_mask)
        if is_np and i >= background_length
    ]
    n_non_priority_beyond_bg = len(beyond_bg_indices)

    # If dropping *all* of these tokens still results in more than `target_total`,
    # drop them all.
    if (total_tokens - n_non_priority_beyond_bg) > target_total:
        keep_indices = [i for i in range(total_tokens) if i not in beyond_bg_indices]
        return subset_patient_data(patient, keep_indices)

    # Otherwise, drop only as many as needed to reach the target length
    must_drop = total_tokens - target_total
    if must_drop <= 0:
        return patient  # No need to drop any

    # Drop the first `must_drop` from beyond_bg_indices (or choose another order).
    indices_to_drop = beyond_bg_indices[:must_drop]
    keep_indices = [i for i in range(total_tokens) if i not in indices_to_drop]
    return subset_patient_data(patient, keep_indices)


def _filter_invalid_positions(
    patient: PatientData,
    non_priority_mask: List[bool],
    low_priority_prefixes: List[str],
) -> PatientData:
    """
    Ensure that for any absolute position (`abspos`) containing one of the
    `low_priority_prefixes`, all subunits of that prefix are present.

    - If an `abspos` has any tokens from `low_priority_prefixes` but not all,
      remove those partial tokens at that position.

    Args:
        patient (PatientData): The patient's data.
        non_priority_mask (List[bool]): True for non-priority tokens, False otherwise.
        low_priority_prefixes (List[str]): The list of prefix strings that define a “unit.”

    Returns:
        PatientData: A new PatientData object with invalid positions dropped.
    """
    unit_len = len(low_priority_prefixes)
    pos_map = defaultdict(list)

    # Collect indices of non-priority tokens by abspos
    for i, is_np in enumerate(non_priority_mask):
        if is_np:
            pos_map[patient.abspos[i]].append(i)

    # Identify positions that do not have exactly `unit_len` non-priority tokens
    invalid_indices = []
    for pos, idx_list in pos_map.items():
        if len(idx_list) != unit_len:
            invalid_indices.extend(idx_list)

    if not invalid_indices:
        return patient

    keep_indices = [i for i in range(len(patient.concepts)) if i not in invalid_indices]
    return subset_patient_data(patient, keep_indices)


def _get_non_priority_mask(
    patient: PatientData,
    non_priority_tokens: set,
) -> List[bool]:
    """
    Construct a boolean mask indicating which tokens are 'non-priority'.

    Args:
        patient (PatientData): The patient whose tokens we inspect.
        non_priority_tokens (set): A set of token IDs that are considered non-priority.

    Returns:
        List[bool]: A list of booleans, one per token in `patient.concepts`.
    """
    return [token in non_priority_tokens for token in patient.concepts]
