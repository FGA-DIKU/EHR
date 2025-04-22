import unittest

import torch
from transformers import ModernBertConfig

from corebehrt.modules.model.model import (
    CorebehrtEncoder,
    CorebehrtForFineTuning,
    CorebehrtForPretraining,
)
from corebehrt.constants.data import (
    ATTENTION_MASK,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
    AGE_FEAT,
    ABSPOS_FEAT,
    TARGET,
)


class TestCorebehrtModels(unittest.TestCase):
    def setUp(self):
        # Minimal config
        self.config = ModernBertConfig(
            vocab_size=20,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            type_vocab_size=2,
            pad_token_id=0,
            embedding_dropout=0.0,
            local_attention=2,
            global_attn_every_n_layers=1,
            max_position_embeddings=10,
            attention_dropout=0.0,
            deterministic_flash_attn=True,
        )
        # Force CPU-friendly path
        self.config._attn_implementation = "eager"
        self.config.is_causal = False
        self.config.enable_gqa = False

        B, L = 2, 5
        self.batch = {
            CONCEPT_FEAT: torch.randint(1, self.config.vocab_size, (B, L)),
            SEGMENT_FEAT: torch.zeros((B, L), dtype=torch.long),
            AGE_FEAT: torch.zeros((B, L)),
            ABSPOS_FEAT: torch.zeros((B, L)),
        }
        # mask for fine-tuning
        self.batch[ATTENTION_MASK] = (self.batch[CONCEPT_FEAT] != 0).long()

    def test_encoder_forward_shape(self):
        enc = CorebehrtEncoder(self.config)
        out = enc(self.batch)
        self.assertEqual(out.last_hidden_state.shape, (2, 5, self.config.hidden_size))

    def test_pretraining_logits_and_loss(self):
        model = CorebehrtForPretraining(self.config)
        batch = {
            **self.batch,
            TARGET: torch.randint(0, self.config.vocab_size, (2, 5), dtype=torch.long),
        }
        out = model(batch)
        # logits: (B, L, vocab_size)
        self.assertEqual(out.logits.shape, (2, 5, self.config.vocab_size))
        # loss is a scalar tensor
        self.assertTrue(torch.is_tensor(out.loss))
        self.assertEqual(out.loss.ndim, 0)

    def test_finetuning_logits_and_loss(self):
        model = CorebehrtForFineTuning(self.config)
        batch = {**self.batch, TARGET: torch.randint(0, 2, (2,), dtype=torch.float)}
        out = model(batch)
        # logits: (B, 1)
        self.assertEqual(out.logits.shape, (2, 1))
        # loss is a scalar tensor
        self.assertTrue(torch.is_tensor(out.loss))
        self.assertEqual(out.loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
