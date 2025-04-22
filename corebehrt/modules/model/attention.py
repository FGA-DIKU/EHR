"""
corebehrt.modules.model.attention

Custom causal attention implementations for ModernBert, allowing standard, SDPA, and FlashAttention variants to enforce causal masking.

Defines:
  - CausalModernBertAttention: subclass of ModernBertAttention that plugs in masked attention fns.
  - MODERNBERT_ATTENTION_FUNCTION: mapping from config._attn_implementation to forward routines.
"""

import torch
from typing import Optional
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
        self.is_causal = getattr(config, "is_causal", False)
        self.enable_gqa = getattr(config, "enable_gqa", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        # here is the only change we make to the ModernBertAttention forward pass:
        kwargs = self._update_kwargs(kwargs)

        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted

    def _update_kwargs(self, kwargs: dict) -> dict:
        """
        Update kwargs with additional parameters needed for attention computation.

        This method allows us to inject additional configuration parameters into the
        attention functions, such as causal masking and grouped query attention flags.

        Args:
            kwargs: Dictionary of existing keyword arguments

        Returns:
            Updated kwargs dictionary with additional attention parameters
        """
        kwargs["is_causal"] = self.is_causal
        kwargs["enable_gqa"] = self.enable_gqa
        return kwargs
