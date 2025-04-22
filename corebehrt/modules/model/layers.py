from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer

from corebehrt.modules.model.attention import CausalModernBertAttention


class CausalModernBertEncoderLayer(ModernBertEncoderLayer):
    """
    This class is a modified version of the ModernBertEncoderLayer class.
    It allows for a causal attention mechanism.
    """

    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.attn = CausalModernBertAttention(config=config, layer_id=layer_id)
