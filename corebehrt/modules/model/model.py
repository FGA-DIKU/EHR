"""
Module: corebehrt_module

This module defines customized EHR-focused BERT models built on top of ModernBertModel:

- CorebehrtEncoder: replaces token embeddings with temporal EHR embeddings and causal encoder layers.
- CorebehrtForPretraining: extends the encoder for masked language model pretraining on EHR sequences.
- CorebehrtForFineTuning: extends the encoder for downstream classification/regression tasks on EHR data.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import ModernBertModel, GPT2Model
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    DEFAULT_VOCABULARY,
    PAD_TOKEN,
    SEGMENT_FEAT,
    TARGET,
    SEP_TOKEN,
)
from corebehrt.constants.model import (
    TIME2VEC_ABSPOS_SCALE,
    TIME2VEC_ABSPOS_SHIFT,
    TIME2VEC_AGE_SCALE,
    TIME2VEC_AGE_SHIFT,
)
from corebehrt.functional.modeling.attention import make_attention_causal
from corebehrt.modules.model.embeddings import EhrEmbeddings
from corebehrt.modules.model.heads import FineTuneHead

logger = logging.getLogger(__name__)


class CorebehrtEncoder(ModernBertModel):
    """
    Encoder backbone for EHR data using ModernBert.

    Attributes:
        embeddings (EhrEmbeddings): custom embeddings for concepts, segments, age, and absolute position.
        layers (nn.ModuleList): list of causal encoder layers replacing standard BERT layers.
    """

    def __init__(self, config):
        super().__init__(config)
        # config.is_decoder = True
        # config.add_cross_attention = False
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
            pad_token_id=config.pad_token_id,
            age_scale=getattr(config, "age_scale", TIME2VEC_AGE_SCALE),
            age_shift=getattr(config, "age_shift", TIME2VEC_AGE_SHIFT),
            abspos_scale=getattr(config, "abspos_scale", TIME2VEC_ABSPOS_SCALE),
            abspos_shift=getattr(config, "abspos_shift", TIME2VEC_ABSPOS_SHIFT),
        )
        self.is_causal = getattr(config, "is_causal", False)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass building embeddings and attention mask, then calling ModernBertModel.

        Args:
            batch (dict): must contain:
                - "concept": Tensor of token indices (B, L)
                - "segment": Tensor of segment IDs (B, L)
                - "age": Tensor of patient ages (B, L)
                - "abspos": Tensor of absolute position values (B, L)
            **kwargs: Additional arguments to pass to the ModernBertModel forward method

        Returns:
            BaseModelOutput: output of ModernBertModel with last_hidden_state, etc.
        """
        if ATTENTION_MASK in batch:
            attention_mask = batch[ATTENTION_MASK]
        else:
            attention_mask = (
                batch[CONCEPT_FEAT] != DEFAULT_VOCABULARY[PAD_TOKEN]
            ).float()

        inputs_embeds = self.embeddings(
            input_ids=batch[CONCEPT_FEAT],
            segments=batch[SEGMENT_FEAT].long(),
            age=batch[AGE_FEAT],
            abspos=batch[ABSPOS_FEAT],
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _update_attention_mask(
        self, attention_mask: torch.Tensor, output_attentions: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls super()._update_attention_mask and adds causal masking if self.is_causal=True.
        Returns:
            Tuple of:
            - Global attention mask
            - Sliding window mask for local attention
        """
        global_attention_mask, sliding_window_mask = super()._update_attention_mask(
            attention_mask, output_attentions
        )
        if self.is_causal:
            global_attention_mask = make_attention_causal(global_attention_mask)
            sliding_window_mask = make_attention_causal(sliding_window_mask)

        return global_attention_mask, sliding_window_mask


class CorebehrtForPretraining(CorebehrtEncoder):
    """
    Masked Language Model head for EHR pretraining.

    Adds a prediction head and linear decoder on top of CorebehrtEncoder.
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

    # Inspiration from ModernBertForMaskedLM
    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for masked language modeling.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos';
                          optional 'target' for labels (B, L).
            **kwargs: Additional arguments to pass to the encoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss/labels if targets provided.
        """
        outputs = super().forward(batch, **kwargs)
        last_hidden_state = outputs[0]

        labels = batch.get(TARGET)
        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.decoder(self.head(last_hidden_state))
        outputs.logits = logits

        if labels is not None:
            outputs.loss = self.get_loss(logits, labels)
            outputs.labels = labels

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class CorebehrtForFineTuning(CorebehrtEncoder):
    """
    Fine-tuning head for downstream classification on EHR sequences.

    Adds a binary classification head (BCEWithLogits) on top of sequence outputs.
    """

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "pos_weight", None):
            pos_weight = torch.tensor(config.pos_weight)
        else:
            pos_weight = None

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for fine-tuning.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos', 'attention_mask';
                          optional 'target' as labels.
            **kwargs: Additional arguments to pass to the encoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss if target provided.
        """
        outputs = super().forward(batch, **kwargs)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch[ATTENTION_MASK])
        outputs.logits = logits

        if batch.get(TARGET) is not None:
            outputs.loss = self.get_loss(logits, batch[TARGET])

        return outputs

    def get_loss(self, hidden_states, labels):
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))

class CorebehrtDecoder(GPT2Model):
    """
    Decoder backbone for EHR data using GPT2.

    Attributes:
        embeddings (EhrEmbeddings): custom embeddings for concepts, segments, age, and absolute position.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace the default GPT2 embeddings with our custom EHR embeddings
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
            pad_token_id=config.pad_token_id,
            age_scale=getattr(config, "age_scale", TIME2VEC_AGE_SCALE),
            age_shift=getattr(config, "age_shift", TIME2VEC_AGE_SHIFT),
            abspos_scale=getattr(config, "abspos_scale", TIME2VEC_ABSPOS_SCALE),
            abspos_shift=getattr(config, "abspos_shift", TIME2VEC_ABSPOS_SHIFT),
        )

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass building embeddings and attention mask, then calling GPT2Model.

        Args:
            batch (dict): must contain:
                - "concept": Tensor of token indices (B, L)
                - "segment": Tensor of segment IDs (B, L)
                - "age": Tensor of patient ages (B, L)
                - "abspos": Tensor of absolute position values (B, L)
            **kwargs: Additional arguments to pass to the GPT2Model forward method

        Returns:
            BaseModelOutput: output of GPT2Model with last_hidden_state, etc.
        """
        if ATTENTION_MASK in batch:
            attention_mask = batch[ATTENTION_MASK]
        else:
            attention_mask = (
                batch[CONCEPT_FEAT] != DEFAULT_VOCABULARY[PAD_TOKEN]
            ).float()

        inputs_embeds = self.embeddings(
            input_ids=batch[CONCEPT_FEAT],
            segments=batch[SEGMENT_FEAT].long(),
            age=batch[AGE_FEAT],
            abspos=batch[ABSPOS_FEAT],
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

class CorebehrtForLanguageModeling(CorebehrtDecoder):
    """
    Language modeling head for EHR generation using the decoder.

    Supports two modes:
    - Single prediction: Only predicts concept tokens (faster, simpler)
    - Multi-embedding prediction: Predicts concepts, segments, ages, and abspos (more complete)
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Configuration for prediction mode
        self.predict_all_embeddings = getattr(config, "predict_all_embeddings", False)
        
        # Language modeling head for concepts (always present)
        if hasattr(self, 'lm_head'):
            # Use the existing lm_head if available
            pass
        else:
            # Create a new language modeling head
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        
        # Additional prediction heads (only if predict_all_embeddings=True)
        if self.predict_all_embeddings:
            self.segment_head = nn.Linear(config.hidden_size, 1)  # Segment is continuous
            self.age_head = nn.Linear(config.hidden_size, 1)  # Age is continuous
            self.abspos_head = nn.Linear(config.hidden_size, 1)  # Abspos is continuous

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for language modeling prediction.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos';
                          optional 'target' for labels (B, L).
            **kwargs: Additional arguments to pass to the decoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss if targets provided.
        """
        outputs = super().forward(batch, **kwargs)
        last_hidden_state = outputs[0]

        # Always predict concepts
        concept_logits = self.lm_head(last_hidden_state)
        outputs.logits = concept_logits  # Keep for compatibility
        outputs.concept_logits = concept_logits

        # Predict other embeddings only if configured
        if self.predict_all_embeddings:
            segment_predictions = self.segment_head(last_hidden_state).squeeze(-1)
            age_predictions = self.age_head(last_hidden_state).squeeze(-1)
            abspos_predictions = self.abspos_head(last_hidden_state).squeeze(-1)
            
            # Store all predictions in outputs
            outputs.segment_predictions = segment_predictions
            outputs.age_predictions = age_predictions
            outputs.abspos_predictions = abspos_predictions

        # Calculate loss if labels are provided
        labels = batch.get(TARGET)
        if labels is not None:
            total_loss = 0
            
            # Concept loss (language modeling) - always present
            shift_concept_logits = concept_logits[..., :-1, :].contiguous()
            concept_loss = self.loss_fct(shift_concept_logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss += concept_loss
            
            # Additional losses only if predict_all_embeddings=True
            if self.predict_all_embeddings:
                # Segment loss (regression)
                if 'segment' in batch:
                    segment_labels = batch['segment'][..., 1:].contiguous()  # Shift segments
                    shift_segment_logits = segment_predictions[..., :-1].contiguous()
                    segment_loss = self.mse_loss(shift_segment_logits, segment_labels)
                    total_loss += segment_loss
                
                # Age loss (regression)
                if 'age' in batch:
                    age_labels = batch['age'][..., 1:].contiguous()  # Shift ages
                    shift_age_predictions = age_predictions[..., :-1].contiguous()
                    age_loss = self.mse_loss(shift_age_predictions, age_labels)
                    total_loss += age_loss
                
                # Abspos loss (regression)
                if 'abspos' in batch:
                    abspos_labels = batch['abspos'][..., 1:].contiguous()  # Shift abspos
                    shift_abspos_predictions = abspos_predictions[..., :-1].contiguous()
                    abspos_loss = self.mse_loss(shift_abspos_predictions, abspos_labels)
                    total_loss += abspos_loss
            
            outputs.loss = total_loss
            outputs.labels = labels

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for language modeling."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

    def _sample_token(self, logits, do_sample, temperature, top_k, top_p, repetition_penalty, generated_tokens):
        """
        Sample a token from logits using the same logic for all token types.
        
        Args:
            logits: Token logits
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            repetition_penalty: Repetition penalty
            generated_tokens: Previously generated tokens for repetition penalty
            
        Returns:
            Sampled tokens
        """
        batch_size = logits.size(0)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                for previous_token in generated_tokens[i]:
                    if previous_token.item() < next_token_logits.size(-1):
                        next_token_logits[i, previous_token.item()] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
            next_token_logits[~mask] = float('-inf')
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample or take argmax
        if do_sample:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        return next_tokens

    def generate(self, batch, max_length, do_sample, temperature, top_p, top_k, repetition_penalty, eos_token_id, predict_all_embeddings=False):
        """
        Generate sequences using the language model.
        
        Args:
            batch: Input batch with concept, segment, age, abspos, attention_mask
            max_length: Maximum length of generated sequences
            do_sample: Whether to use sampling or greedy decoding
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token ID
            predict_all_embeddings: Whether to predict all embeddings or just concepts
            
        Returns:
            Generated sequences with all embeddings
        """
        import random
        
        # Get initial input
        batch_size, seq_len = batch["concept"].shape
        
        # Initialize output with input
        generated_concepts = batch["concept"].clone()
        generated_segments = batch["segment"].clone()
        generated_ages = batch["age"].clone()
        generated_abspos = batch["abspos"].clone()
        
        # Generate tokens one by one
        for step in range(seq_len, max_length):
            # Prepare current batch for forward pass
            current_batch = {
                "concept": generated_concepts,
                "segment": generated_segments,
                "age": generated_ages,
                "abspos": generated_abspos,
                "attention_mask": torch.ones_like(generated_concepts)
            }
            
            # Get model outputs
            outputs = self.forward(current_batch)
            concept_logits = outputs.concept_logits
            
            # Get predictions for other embeddings (if available and requested)
            segment_predictions = outputs.segment_predictions if hasattr(outputs, 'segment_predictions') and predict_all_embeddings else None
            age_predictions = outputs.age_predictions if hasattr(outputs, 'age_predictions') and predict_all_embeddings else None
            abspos_predictions = outputs.abspos_predictions if hasattr(outputs, 'abspos_predictions') and predict_all_embeddings else None
            
            # For language modeling, we want to predict the next token
            next_concepts = self._sample_token(concept_logits, do_sample, temperature, top_k, top_p, repetition_penalty, generated_concepts)
            
            # Generate additional embeddings (model always needs them as input)
            if predict_all_embeddings:
                next_segments = segment_predictions[:, -1]  # Direct regression values
                next_ages = age_predictions[:, -1]  # Direct regression values
                next_abspos = abspos_predictions[:, -1]  # Direct regression values
            else:
                # Always use heuristics when not predicting all embeddings
                next_segments = torch.zeros_like(next_concepts, dtype=torch.float)
                next_ages = torch.zeros_like(next_concepts, dtype=torch.float)
                next_abspos = torch.zeros_like(next_concepts, dtype=torch.float)
                
                # Apply heuristics
                for i in range(len(next_concepts)):
                    # Segment logic - can stay same or increase by various amounts
                    if next_concepts[i].item() == DEFAULT_VOCABULARY[SEP_TOKEN]:  # SEP token
                        # SEP tokens often indicate new segments, but not always
                        if generated_segments.size(1) > 0:
                            # Allow same segment or increment by 1-3
                            increment = random.choice([0.0, 1.0, 2.0, 3.0])
                            next_segments[i] = generated_segments[i, -1] + increment
                        else:
                            next_segments[i] = 1.0
                    else:
                        # Non-SEP tokens usually stay in same segment
                        next_segments[i] = generated_segments[i, -1] if generated_segments.size(1) > 0 else 0.0
                    
                    # Age logic - can stay same or increase by small amounts
                    if generated_ages.size(1) > 0:
                        # Allow same age or small increments (0.1 to 0.5 hours)
                        increment = random.uniform(0.0, 0.5)
                        next_ages[i] = generated_ages[i, -1] + increment
                    else:
                        next_ages[i] = 0.1
                    
                    # Abspos logic - always increases by various amounts
                    if generated_abspos.size(1) > 0:
                        # Time always moves forward, but by varying amounts (0.5 to 3.0 hours)
                        increment = random.uniform(0.5, 3.0)
                        next_abspos[i] = generated_abspos[i, -1] + increment
                    else:
                        next_abspos[i] = 1.0
            
            # Append all embeddings to sequences (model always needs them)
            generated_concepts = torch.cat([generated_concepts, next_concepts.unsqueeze(-1)], dim=-1)
            generated_segments = torch.cat([generated_segments, next_segments.unsqueeze(-1)], dim=-1)
            generated_ages = torch.cat([generated_ages, next_ages.unsqueeze(-1)], dim=-1)
            generated_abspos = torch.cat([generated_abspos, next_abspos.unsqueeze(-1)], dim=-1)
            
            # Check for EOS tokens
            if eos_token_id in next_concepts:
                # Stop generation for sequences that hit EOS
                break
        
        # Return all generated embeddings
        return {
            'concepts': generated_concepts,
            'segments': generated_segments,
            'ages': generated_ages,
            'abspos': generated_abspos
        }

