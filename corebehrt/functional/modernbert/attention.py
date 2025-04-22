"""
ModernBERT Attention Forward Functions, slightly modified to support causal attention

This module contains
1. Causal SDPA Attention: Uses PyTorch's built-in scaled dot product attention for efficient computation
2. Flash Attention: Leverages Flash Attention 2 for optimized attention computation when available
3. Eager Attention: A standard implementation with explicit attention weights computation

The module is designed to be flexible, allowing different attention mechanisms to be used based on hardware capabilities
and performance requirements. Flash Attention is used when available for optimal performance, falling back to SDPA or
eager attention when needed.

Original source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertAttention,
    ModernBertUnpaddedRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
else:
    flash_attn_varlen_qkvpacked_func = None


def sdpa_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    is_causal: bool = True,
    enable_gqa: bool = False,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]

    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    # We cannot use is_causal=True together with attention_mask in sdpa
    # Since we need to provide attention_mask to ignore padding tokens
    # we have to compute causal mask and combine it with attention_mask
    if is_causal:
        attention_mask = _make_attention_causal(attention_mask)
    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
            enable_gqa=enable_gqa,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


def flash_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    causal: bool = True,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    """
    Compared to original, we pass the additional causal flag.
    Since sequences are unpadded before passed to flash attention, no need to combine with causal mask manually.
    """
    # (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
            causal=causal,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
            causal=causal,
        )
    return (attn.view(bs, dim),)


def eager_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    is_causal: bool = True,
    **_kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    # Apply causal mask if requested
    if is_causal:
        attention_mask = _make_attention_causal(attention_mask)

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=module.attention_dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def _make_attention_causal(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Converts a standard attention mask into a causal attention mask.

    This function takes an attention mask and adds causality constraints to prevent
    tokens from attending to future positions. It handles both boolean and float masks.

    Args:
        attn_mask (torch.Tensor): Input attention mask of shape [B, H, L, L] or [..., L, L],
            where B is batch size, H is number of heads, and L is sequence length.
            For float masks: 0 for attended positions, -inf for masked positions.
            For bool masks: False for attended positions, True for masked positions.

    Returns:
        torch.Tensor: Causal attention mask with the same shape and dtype as input.
            The resulting mask combines the input mask with causal constraints:
            - Float masks: Positions above diagonal are set to -inf
            - Bool masks: Positions above diagonal are set to True

    Example:
        >>> # Input mask [1, 1, 3, 3] allowing all attention
        >>> mask = torch.zeros(1, 1, 3, 3)
        >>> print(mask[0, 0])
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
        >>>
        >>> # Output causal mask
        >>> causal = _make_attention_causal(mask)
        >>> print(causal[0, 0])
        tensor([[  0., -inf, -inf],
                [  0.,   0., -inf],
                [  0.,   0.,   0.]])
    """
    # Get sequence length from last dimension
    L = attn_mask.size(-1)
    device, dtype = attn_mask.device, attn_mask.dtype

    # Create causal mask: -inf (float) or True (bool) for future positions
    neg_inf = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((L, L), neg_inf, device=device, dtype=dtype), diagonal=1
    )
    # Expand dimensions to match input mask shape for broadcasting
    for _ in range(attn_mask.ndim - 2):
        causal = causal.unsqueeze(0)
    # Combine input mask with causal constraints
    return attn_mask + causal
