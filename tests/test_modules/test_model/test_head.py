import unittest
import torch
from corebehrt.modules.model.heads import BiGRU, FineTuneHead


class TestFineTuneHead(unittest.TestCase):
    def test_forward_modes(self):
        hidden_size = 6
        head = FineTuneHead(hidden_size)
        hidden_states = torch.randn(3, 7, hidden_size)
        attn_mask = torch.ones(3, 7, dtype=torch.long)

        emb = head(hidden_states, attn_mask, return_embedding=True)
        self.assertEqual(emb.shape, (3, hidden_size))

        logits = head(hidden_states, attn_mask, return_embedding=False)
        self.assertEqual(logits.shape, (3, 1))


class TestBiGRU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hidden_size = 4
        self.classifier_hidden_size = 2
        self.gru = BiGRU(self.hidden_size, self.classifier_hidden_size)
        self.batch = 2
        self.seq_len = 5
        self.hidden_states = torch.randn(self.batch, self.seq_len, self.hidden_size)
        self.attn_mask = torch.ones(self.batch, self.seq_len, dtype=torch.long)

    def test_return_embedding(self):
        emb = self.gru(self.hidden_states, self.attn_mask, return_embedding=True)
        self.assertEqual(emb.shape, (self.batch, self.hidden_size))

    def test_classification(self):
        logits = self.gru(self.hidden_states, self.attn_mask, return_embedding=False)
        self.assertEqual(logits.shape, (self.batch, 1))

    def test_gradient_flow(self):
        """Test if gradients flow properly through the network."""
        hidden = torch.randn(2, 3, self.hidden_size, requires_grad=True)
        mask = torch.ones(2, 3, dtype=torch.long)

        out = self.gru(hidden, mask)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertFalse(torch.allclose(hidden.grad, torch.zeros_like(hidden.grad)))

    def test_bidirectionality(self):
        """Test if the GRU is truly bidirectional."""
        # Create sequence that's different forward and backward
        seq = torch.tensor([[[1.0, 0, 0, 0]], [[0, 1.0, 0, 0]], [[0, 0, 1.0, 0]]])
        hidden = seq.expand(3, 2, 4)  # [seq_len, batch, hidden]
        mask = torch.ones(2, 3, dtype=torch.long)

        # Get embeddings for normal and reversed sequence
        emb1 = self.gru(hidden.transpose(0, 1), mask, return_embedding=True)
        emb2 = self.gru(hidden.flip(0).transpose(0, 1), mask, return_embedding=True)

        # Embeddings should be different due to bidirectionality
        self.assertFalse(torch.allclose(emb1, emb2, atol=1e-5))

    def test_empty_sequence(self):
        """Test handling of empty/minimal sequences."""
        hidden = torch.randn(2, 1, self.hidden_size)
        mask = torch.ones(2, 1, dtype=torch.long)

        out = self.gru(hidden, mask)
        self.assertEqual(out.shape, (2, 1))

    def test_mask_handling(self):
        """Test if attention mask properly affects the output."""
        hidden = torch.randn(2, 3, self.hidden_size)

        # Two different masks
        mask1 = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        mask2 = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)

        out1 = self.gru(hidden, mask1)
        out2 = self.gru(hidden, mask2)

        # Outputs should be different when using different masks
        self.assertFalse(torch.allclose(out1, out2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
