import unittest

import torch

from corebehrt.modules.model.embeddings import EhrEmbeddings, Time2Vec


class TestTime2Vec(unittest.TestCase):
    def test_shape_and_clipping(self):
        output_dim = 3
        tv = Time2Vec(
            output_dim=output_dim, shift=0.0, scale=1.0, clip_min=0.0, clip_max=0.0
        )
        # Force all weights to zero so we know the exact output
        tv.w0.data.fill_(0.0)
        tv.phi0.data.fill_(0.0)
        tv.w.data.fill_(0.0)
        tv.phi.data.fill_(0.0)

        tau = torch.zeros((2, 4))
        emb = tv(tau)
        # Expect shape (batch, seq_len, output_dim)
        self.assertEqual(emb.shape, (2, 4, output_dim))
        # First channel = linear_1 = 0 (clamped)
        self.assertTrue(torch.allclose(emb[..., 0], torch.zeros(2, 4)))
        # Remaining channels = cos(0 + 0) = 1
        self.assertTrue(torch.allclose(emb[..., 1:], torch.ones(2, 4, output_dim - 1)))


class TestEhrEmbeddings(unittest.TestCase):
    def setUp(self):
        self.ehr = EhrEmbeddings(
            vocab_size=10,
            hidden_size=8,
            type_vocab_size=4,
            embedding_dropout=0.0,
            pad_token_id=0,
        )

    def test_skip_inputs_embeds(self):
        dummy = torch.randn(2, 5, 8)
        out = self.ehr(
            input_ids=None, segments=None, age=None, abspos=None, inputs_embeds=dummy
        )
        # Should return the exact same tensor, no lookup
        self.assertIs(out, dummy)

    def test_input_validation(self):
        """Test input validation and error messages."""
        with self.assertRaises(ValueError):
            # Test missing both input_ids and inputs_embeds
            self.ehr(
                input_ids=None, segments=None, age=None, abspos=None, inputs_embeds=None
            )

    def test_shape_consistency(self):
        """Test output shapes with different input combinations."""
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 10, (batch_size, seq_len))
        segments = torch.randint(0, 4, (batch_size, seq_len))
        age = torch.rand(batch_size, seq_len)
        abspos = torch.arange(seq_len).expand(batch_size, -1)

        out = self.ehr(input_ids=input_ids, segments=segments, age=age, abspos=abspos)
        self.assertEqual(out.shape, (batch_size, seq_len, 8))  # 8 is h


if __name__ == "__main__":
    unittest.main()
