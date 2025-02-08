from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertEmbeddings

from corebehrt.constants.model import (
    TIME2VEC_ABSPOS_MULTIPLIER,
    TIME2VEC_AGE_MULTIPLIER,
    TIME2VEC_MAX_CLIP,
    TIME2VEC_MIN_CLIP,
)


class EhrEmbeddings(ModernBertEmbeddings):
    """
    Extends ModernBertEmbeddings to also handle:
      - Time2Vec embeddings for age/abspos
      - segment (token_type) embeddings
      - use similar compilation technique as in ModernBertModel
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.age_embeddings = Time2Vec(
            config.hidden_size,
            init_scale=TIME2VEC_AGE_MULTIPLIER,
            clip_min=TIME2VEC_MIN_CLIP,
            clip_max=TIME2VEC_MAX_CLIP,
        )
        self.abspos_embeddings = Time2Vec(
            config.hidden_size,
            init_scale=TIME2VEC_ABSPOS_MULTIPLIER,
            clip_min=TIME2VEC_MIN_CLIP,
            clip_max=TIME2VEC_MAX_CLIP,
        )
        self.segment_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

    @torch.compile(dynamic=True)
    def compiled_embeddings(
        self,
        input_ids: torch.LongTensor,
        segment_ids: torch.LongTensor,
        age_float: torch.FloatTensor,
        abspos_float: torch.FloatTensor,
    ) -> torch.Tensor:
        concept_embeds = self.tok_embeddings(input_ids)
        segment_embeds = self.segment_embeddings(segment_ids)
        age_embeds = self.age_embeddings(age_float)
        abspos_embeds = self.abspos_embeddings(abspos_float)

        # Sum all embeddings before normalization
        hidden_states = concept_embeds + segment_embeds + age_embeds + abspos_embeds
        return self.drop(self.norm(hidden_states))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        age_float: Optional[torch.FloatTensor] = None,
        abspos_float: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if self.config.reference_compile:
                # Use compiled version that handles all embeddings at once
                hidden_states = self.compiled_embeddings(
                    input_ids, segment_ids, age_float, abspos_float
                )
            else:
                # Non-compiled path
                concept_embeds = self.tok_embeddings(input_ids)
                segment_embeds = self.segment_embeddings(segment_ids)
                age_embeds = self.age_embeddings(age_float)
                abspos_embeds = self.abspos_embeddings(abspos_float)

                # Sum all embeddings before normalization
                hidden_states = (
                    concept_embeds + segment_embeds + age_embeds + abspos_embeds
                )
                hidden_states = self.drop(self.norm(hidden_states))

        return hidden_states


class Time2Vec(torch.nn.Module):
    def __init__(
        self,
        output_dim: int = 768,
        function: callable = torch.cos,
        init_scale: float = 1,
        clip_min: float = None,
        clip_max: float = None,
    ):
        super().__init__()
        self.f = function
        self.clip_min = clip_min
        self.clip_max = clip_max
        # for i = 0
        self.w0 = torch.nn.Parameter(torch.randn(1, 1))
        self.phi0 = torch.nn.Parameter(torch.randn(1))
        # for 1 <= i <= k (output_dim)
        self.w = torch.nn.Parameter(torch.randn(1, output_dim - 1))
        self.phi = torch.nn.Parameter(torch.randn(output_dim - 1))

        self.init_scale = init_scale

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        if self.init_scale is not None:
            tau = tau * self.init_scale
        tau = tau.unsqueeze(2)

        linear_1 = torch.matmul(tau, self.w0) + self.phi0
        linear_2 = torch.matmul(tau, self.w)

        if self.clip_min is not None or self.clip_max is not None:
            linear_1 = torch.clamp(linear_1, self.clip_min, self.clip_max)

        periodic = self.f(linear_2 + self.phi)

        return torch.cat((linear_1, periodic), dim=-1)
