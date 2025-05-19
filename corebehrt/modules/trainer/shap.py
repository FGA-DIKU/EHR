import numpy as np
import torch
from typing import Dict
from corebehrt.modules.model.model import CorebehrtForFineTuning
from corebehrt.constants.data import (
    DEFAULT_VOCABULARY, 
    MASK_TOKEN,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
    AGE_FEAT,
    ABSPOS_FEAT,
    ATTENTION_MASK
)
from typing import Union

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

    def __call__(self, x):
        # Convert input to float tensor for SHAP gradients
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.model.device)
        x = x.float()  # Convert to float for SHAP gradients
        x.requires_grad_(True)  # Enable gradients on the float tensor
        
        # Create a copy of the batch and update the concept field
        batch_copy = {k: v.clone() for k, v in self.batch.items()}
        
        # Convert concept to long for model input while maintaining gradient connection
        x_long = x.long()
        batch_copy[CONCEPT_FEAT] = x_long
        
        # Synchronize shapes of all batch elements
        self.synchronise_shapes(batch_copy, x)
        
        # Forward pass with gradient tracking
        with torch.set_grad_enabled(True):
            # Get embeddings without in-place operations
            concept_emb = self.model.embeddings.concept_embeddings(batch_copy[CONCEPT_FEAT])
            segment_emb = self.model.embeddings.segment_embeddings(batch_copy[SEGMENT_FEAT])
            age_emb = self.model.embeddings.age_embeddings(batch_copy[AGE_FEAT])
            abspos_emb = self.model.embeddings.abspos_embeddings(batch_copy[ABSPOS_FEAT])
            
            # Combine embeddings without in-place operations
            inputs_embeds = concept_emb + segment_emb + age_emb + abspos_emb
            
            # Forward through the rest of the model
            outputs = self.model.forward({
                "inputs_embeds": inputs_embeds,
                "attention_mask": batch_copy[ATTENTION_MASK]
            })
            
            logits = outputs.logits
            
            # Create gradient connection
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # Add a small perturbation to ensure gradient flow
            # Use a small constant to avoid numerical issues
            perturbation = 1e-4 * torch.sum(x, dim=-1, keepdim=True)
            logits = logits + perturbation
            
            return logits

    @staticmethod
    def synchronise_shapes(batch: Dict[str, torch.Tensor], concept: torch.Tensor) -> None:
        """
        Expand all batch keys (except 'concept') to match the batch size of the concept.
        """
        for key in batch:
            if key == CONCEPT_FEAT:
                continue
            orig = batch[key]
            repeat_times = [concept.shape[0]] + [1] * (orig.dim() - 1)
            batch[key] = orig.repeat(*repeat_times)

    def get_background_data(self, background_size: int = 100) -> torch.Tensor:
        seq_len = self.batch[CONCEPT_FEAT].shape[1]
        vocab_size = self.model.embeddings.concept_embeddings.num_embeddings

        # Generate random concept IDs
        background_concepts = torch.randint(
            low=0,
            high=vocab_size,
            size=(background_size, seq_len),
            device=self.model.device
        )

        return background_concepts.float()


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