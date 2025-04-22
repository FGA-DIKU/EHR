"""
corebehrt.modules.model.attention

Custom causal attention implementations for ModernBert, allowing standard, SDPA, and FlashAttention variants to enforce causal masking.

Defines:
  - CausalModernBertAttention: subclass of ModernBertAttention that plugs in masked attention fns.
  - MODERNBERT_ATTENTION_FUNCTION: mapping from config._attn_implementation to forward routines.
"""

import torch
from transformers.models.modernbert.modeling_modernbert import ModernBertAttention

from corebehrt.functional.modernbert.attention import (
    eager_attention_forward,
    flash_attention_forward,
    sdpa_attention_forward,
)

MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class CausalModernBertAttention(ModernBertAttention):
    """
    Attention layer enforcing causal masking within ModernBert.

    Replaces the standard ModernBertAttention forward pass to use custom routines
    (`sdpa_attention_forward`, `flash_attention_forward`, `eager_attention_forward`) that
    inject a left-to-right mask and support sliding-window or full-context attention.

    Configuration:
        config._attn_implementation: selects 'flash_attention_2', 'eager', or 'sdpa'.
        config.is_causal (bool): whether to apply causal masking.
        config.local_attention: tuple (left, right) window sizes for local attention.
    """

    def __init__(self, config, layer_id=None):
        super().__init__(config, layer_id)
        # Set this flag to indicate we're using causal attention
        self.is_causal = config.to_dict().get("is_causal", False)

    def forward(
        self, hidden_states, attention_mask=None, output_attentions=False, **kwargs
    ):
        """
        This is a modified version of the ModernBertAttention class.
        We use our custom attention forward functions which use a causal attention mask.
        """
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        # Create sliding window mask if needed
        if self.local_attention != (-1, -1):
            seq_len = hidden_states.shape[1]
            sliding_window_mask = torch.ones(
                (bs, 1, seq_len, seq_len), device=hidden_states.device
            )
            for i in range(seq_len):
                start = max(0, i - self.local_attention[0])
                end = min(seq_len, i + self.local_attention[1] + 1)
                sliding_window_mask[:, :, i, start:end] = 1
            kwargs["sliding_window_mask"] = sliding_window_mask

        # Prepare attention function arguments
        attn_args = {
            "qkv": qkv,
            "attention_mask": attention_mask,
            "rotary_emb": self.rotary_emb,
            "local_attention": self.local_attention,
            "bs": bs,
            "dim": self.all_head_size,
            "output_attentions": output_attentions,
            "is_causal": self.is_causal,
        }
        attn_args.update(kwargs)

        # This below is the only change we make to the ModernBertAttention class
        # We access our custom attention forward functions directly instead of using what's inside modeling_modernbert
        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self, **attn_args
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]
