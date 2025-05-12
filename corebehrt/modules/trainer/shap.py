import numpy as np
import torch
from typing import Dict
from corebehrt.modules.model.model import CorebehrtForFineTuning
from corebehrt.constants.data import DEFAULT_VOCABULARY, MASK_TOKEN

class EHRMasker:
    """
    Masker for EHR data. Masks values in x with self.mask_value where mask is True.
    The __call__ method will be called by the explainer to mask the input data.
    """

    def __init__(self) -> None:
        self.mask_value = DEFAULT_VOCABULARY.get(MASK_TOKEN, None)

    def __call__(self, mask, x):
        """Mask values in x with self.mask_value where mask is True."""
        masked_x = np.where(mask, x, self.mask_value)
        return masked_x


class BEHRTWrapper(torch.nn.Module):
    """
    This wrapper is used to wrap the BEHRT model for SHAP explainer.
    The SHAP explainer will only mask the concept IDs, rest is passed unchanged to BEHRT.
    """

    def __init__(
        self, model: CorebehrtForFineTuning, batch: Dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch

    def __call__(self, concept: np.ndarray):
        """
        Compute the output of the model for the given concept IDs.
        To make compatible with SHAP, concepts are passed in the shape bs, 1, seq_len to shap explainer
        the explainer then passess n_permutations, seq_len to the model.
        We need to copy the other inputs to take the same shape as the concept.
        """
        batch_copy = self.batch.copy()  # don't modify the original batch#
        concept = torch.from_numpy(concept)  # shap explainer passes numpy array
        self.synchronise_shapes(batch_copy, concept)
        batch_copy["concept"] = concept
        output = self.model(batch=batch_copy).logits
        return output

    @staticmethod
    def synchronise_shapes(
        batch: Dict[str, torch.Tensor], concept: torch.Tensor
    ) -> None:
        """
        Synchronise the shape of the batch with the concept.
        """
        for (
            key
        ) in (
            batch
        ):  # copy all entries in batch to be the same shape along first dimension as concepts
            if (
                key != "concept"
            ):  # this will be taken from the explainer and is already in the correct shape
                batch[key] = batch[key].repeat(concept.shape[0], 1)


class DeepSHAP_BEHRTWrapper(torch.nn.Module):
    def __init__(self, model: CorebehrtForFineTuning) -> None:
        """
        Modelwrapper to use with DeepSHAP explainer.
        To be able to explain contributions of each input stream e.g. concept, age, etc.
        we need to embed each input stream separately.
        DeepExplainer will pass them as a list to the model where they can be summed and forwarded.
        """
        super().__init__()
        self.model = model
        self.f_embeddings = {
            "concept": model.embeddings.concept_embeddings,
            "age": model.embeddings.age_embeddings,
            "abspos": model.embeddings.abspos_embeddings,
            "segment": model.embeddings.segment_embeddings,
        }

    def __call__(self, *args):
        """
        Takes a list of embeddings and sums them to pass to BEHRTEncoder as inputs_embeds.
        """
        print("stacked input emebds", torch.stack(args).shape)
        # inputs_embeds = torch.stack(args[:-1]).sum(dim=0) # sum over the embeddings
        # outputs = self.model(inputs_embeds=inputs_embeds, batch={'attention_mask': args[-1]}) # last element is attention_mask
        inputs_embeds = torch.stack(args).sum(dim=0)  # sum over the embeddings
        print("inputs_embeds shape", inputs_embeds.shape)
        outputs = self.model(
            inputs_embeds=inputs_embeds
        )  # last element is attention_mask
        return outputs.logits

    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes a batch of inputs, where each feature is of shape bs, seq_len
        and returns the embeddings in the shape of bs, seq_len, hidden_dim for each feature.
        """
        embedded_batch = {
            k: self.f_embeddings[k](v)
            for k, v in batch.items()
            if k in self.f_embeddings
        }
        # embedded_batch['attention_mask'] = batch['attention_mask']
        return embedded_batch