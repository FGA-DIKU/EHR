import unittest
import torch
from corebehrt.functional.modeling.attention import make_attention_causal


class TestMakeAttentionCausal(unittest.TestCase):
    def setUp(self):
        # precompute negative infinity for float32
        self.neg_inf32 = torch.finfo(torch.float32).min

    def test_2d_zero_mask(self):
        # Simple 2×2 mask of zeros → should get 0 on/below diag, -inf above
        mask = torch.zeros(2, 2, dtype=torch.float32)
        out = make_attention_causal(mask)

        expected = torch.tensor(
            [[0.0, self.neg_inf32], [0.0, 0.0]], dtype=torch.float32
        )

        self.assertTrue(
            torch.equal(out, expected), f"\nGot:\n{out}\nExpected:\n{expected}"
        )

    def test_3d_and_4d_broadcast(self):
        # Test that higher dims broadcast correctly
        B, H, L = 2, 3, 4
        base = torch.zeros(B, H, L, L, dtype=torch.float32)
        out = make_attention_causal(base)

        # build a single expected slice
        single = torch.triu(
            torch.full((L, L), self.neg_inf32, dtype=torch.float32), diagonal=1
        )
        single = single + torch.zeros((L, L))  # still zeros on/below diag

        # verify each batch/head
        for b in range(B):
            for h in range(H):
                self.assertTrue(
                    torch.equal(out[b, h], single), f"slice b={b},h={h} differs"
                )

    def test_preserves_dtype_and_device(self):
        # float64 on CPU
        mask64 = torch.zeros(3, 3, dtype=torch.float64)
        out64 = make_attention_causal(mask64)
        self.assertEqual(out64.dtype, mask64.dtype)
        self.assertEqual(out64.device, mask64.device)

        # float16 on CUDA (if available)
        if torch.cuda.is_available():
            mask16 = torch.zeros(2, 2, dtype=torch.float16, device="cuda")
            out16 = make_attention_causal(mask16)
            self.assertEqual(out16.dtype, mask16.dtype)
            self.assertEqual(out16.device, mask16.device)

    def test_combines_with_existing_mask(self):
        # existing nonzero values should be preserved below diag
        L = 3
        base = torch.zeros(L, L, dtype=torch.float32)
        base[1, 0] = -1.0
        base[2, 0] = -2.0
        out = make_attention_causal(base)

        expected = torch.tensor(
            [
                [0.0, self.neg_inf32, self.neg_inf32],
                [-1.0, 0.0, self.neg_inf32],
                [-2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(
            torch.allclose(out, expected, atol=0, rtol=0),
            f"\nGot:\n{out}\nExpected:\n{expected}",
        )

    def test_boolean_mask_raises(self):
        # currently boolean dtype isn't supported by this impl
        mask_bool = torch.zeros(2, 2, dtype=torch.bool)
        with self.assertRaises(TypeError):
            make_attention_causal(mask_bool)


if __name__ == "__main__":
    unittest.main()
