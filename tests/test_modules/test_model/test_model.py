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
    DEFAULT_VOCABULARY,
    PAD_TOKEN,
    IGNORE_LOSS_INDEX,
)


class TestCorebehrtModels(unittest.TestCase):
    def setUp(self):
        # Minimal config
        self.config = ModernBertConfig(
            vocab_size=20,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            type_vocab_size=10,
            pad_token_id=0,
            local_attention=2,
            global_attn_every_n_layers=1,
            max_position_embeddings=10,
            is_causal=False,
        )
        self.causal_config = ModernBertConfig(**self.config.to_dict())
        self.causal_config.is_causal = True
        # Force CPU-friendly path

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

    def test_pretraining_sparse_prediction(self):
        # Configure model for sparse prediction
        sparse_config = ModernBertConfig(**self.config.to_dict())
        sparse_config.sparse_prediction = True
        sparse_config.sparse_pred_ignore_index = IGNORE_LOSS_INDEX

        model = CorebehrtForPretraining(sparse_config)

        # Create a batch with some masked tokens
        batch = {**self.batch}
        # Use -100 as ignore index for some positions
        targets = torch.randint(0, sparse_config.vocab_size, (2, 5), dtype=torch.long)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, [1, 3]] = True  # Mask positions 1 and 3
        targets[~mask] = sparse_config.sparse_pred_ignore_index
        batch[TARGET] = targets

        out = model(batch)

        # Should only get predictions for the masked tokens
        expected_predictions = mask.sum().item()
        self.assertEqual(out.logits.shape[0], expected_predictions)
        self.assertEqual(out.logits.shape[1], sparse_config.vocab_size)

        # loss should be a scalar
        self.assertTrue(torch.is_tensor(out.loss))
        self.assertEqual(out.loss.ndim, 0)

    def test_causal_attention(self):
        # Test encoder with causal attention
        causal_enc = CorebehrtEncoder(self.causal_config)
        out = causal_enc(self.batch)
        self.assertEqual(out.last_hidden_state.shape, (2, 5, self.config.hidden_size))

    def test_causal_attention_mask(self):
        # Test that causal attention properly masks future tokens
        causal_enc = CorebehrtEncoder(self.causal_config)

        # Call forward to get the attention mask
        _ = causal_enc(self.batch)

        # Get the attention mask from the model's internal _update_attention_mask method
        attention_mask = (
            self.batch[CONCEPT_FEAT] != DEFAULT_VOCABULARY[PAD_TOKEN]
        ).float()
        global_mask, _ = causal_enc._update_attention_mask(
            attention_mask, output_attentions=False
        )

        # Check shape of masks (batch_size, num_heads, seq_len, seq_len)
        batch_size = self.batch[CONCEPT_FEAT].shape[0]
        seq_len = self.batch[CONCEPT_FEAT].shape[1]
        expected_shape = (batch_size, 1, seq_len, seq_len)
        self.assertEqual(global_mask.shape, expected_shape)

        # Verify causal pattern: upper triangle should be -inf
        # For each position i,j where j > i (upper triangle), value should be -inf
        neg_inf = torch.finfo(global_mask.dtype).min
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # future position
                    self.assertTrue(
                        torch.all(global_mask[..., i, j] == neg_inf),
                        f"Position ({i},{j}) should be masked (-inf) but got {global_mask[0, 0, i, j]}",
                    )
                else:  # current or past position
                    self.assertFalse(
                        torch.all(global_mask[..., i, j] == neg_inf),
                        f"Position ({i},{j}) should not be masked but got -inf",
                    )


if __name__ == "__main__":
    unittest.main()
