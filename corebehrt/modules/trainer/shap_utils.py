from typing import Dict

import numpy as np
import torch


def insert_concept_shap_values(
    all_shap_values: np.ndarray, concepts: np.ndarray, shap_values: np.ndarray
) -> np.ndarray:
    """
    Insert shap values into all_shap_values
    We use the flattened concepts as indices to select the correct shap values for each concept.
    """
    ind = concepts.flatten()
    all_shap_values[ind] = (
        all_shap_values[ind] + shap_values.flatten()
    ) / 2  # running average
    return all_shap_values


def insert_shap_values(
    all_shap_values: Dict[str, torch.Tensor],
    concepts: torch.Tensor,
    shap_values: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Takes the shap values and inserts them into the all_shap_values dict.
    all_shap_values and shap_values:
        dict with keys 'concept', 'age', 'abspos', 'segment' and values of
        shape bs, seq_len, hidden_dim
    concepts is a tensor of shape (bs, seq_len)
    attention_mask is a tensor of shape (bs, seq_len)
    """
    for feature, shap_value in shap_values.items():
        if feature == "concept":
            all_shap_values["concept"] = insert_concept_shap_values(
                all_shap_values["concept"], concepts, np.mean(shap_value, axis=-1)
            )
        else:
            assert (
                shap_value.shape[0] == attention_mask.shape[0]
            ), "Shap value and attention mask must have the same batch size."
            all_shap_values[feature] += shap_value[attention_mask].mean()
    return all_shap_values


def split_batch_into_bg_and_fg(embeddings: Dict[str, torch.Tensor], n_bg_samples=10):
    """Split batch into two parts, one for background and one for the actual computation."""
    if n_bg_samples > len(embeddings["concept"]):
        raise ValueError(
            "Number of background samples must be smaller than the batch size."
        )
    bg_input_embeds = {k: e[:n_bg_samples] for k, e in embeddings.items()}
    fg_input_embeds = {k: e[n_bg_samples:] for k, e in embeddings.items()}
    return bg_input_embeds, fg_input_embeds