from typing import Dict

import torch
import torch.nn as nn

from corebehrt.constants.model import (
    TIME2VEC_ABSPOS_MULTIPLIER,
    TIME2VEC_AGE_MULTIPLIER,
    TIME2VEC_MAX_CLIP,
    TIME2VEC_MIN_CLIP,
)


class EhrEmbeddings(nn.Module):
    """
    Forward inputs:
        input_ids: torch.LongTensor             - (batch_size, sequence_length)
        token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
        position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)
            We abuse huggingface's standard position_ids to pass additional information (age, abspos)
            This makes BertModel's forward method compatible with our EhrEmbeddings

    Config:
        vocab_size: int                         - size of the vocabulary
        hidden_size: int                        - size of the hidden layer
        type_vocab_size: int                    - size of max segments
        layer_norm_eps: float                   - epsilon for layer normalization
        hidden_dropout_prob: float              - dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        type_vocab_size: int,
        embedding_dropout: float,
    ):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(embedding_dropout)

        # Initalize embeddings
        self.concept_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.age_embeddings = Time2Vec(
            hidden_size,
            init_scale=TIME2VEC_AGE_MULTIPLIER,
            clip_min=TIME2VEC_MIN_CLIP,
            clip_max=TIME2VEC_MAX_CLIP,
        )
        self.abspos_embeddings = Time2Vec(
            hidden_size,
            init_scale=TIME2VEC_ABSPOS_MULTIPLIER,
            clip_min=TIME2VEC_MIN_CLIP,
            clip_max=TIME2VEC_MAX_CLIP,
        )
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,  # concepts
        segments: torch.LongTensor,
        age: torch.Tensor,
        abspos: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.concept_embeddings(input_ids)

        embeddings += self.segment_embeddings(segments)
        embeddings += self.age_embeddings(age)
        embeddings += self.abspos_embeddings(abspos)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


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
