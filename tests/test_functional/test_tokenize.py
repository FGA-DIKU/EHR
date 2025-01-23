import unittest
import pandas as pd

# Import the updated functions and constants
from corebehrt.functional.tokenize import tokenize_partition
from corebehrt.functional.constants import (
    UNKNOWN_TOKEN,
)


class TestTokenizePartition(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(["C1", "C2", "C3", "C4", "C5"])
        self.vocabulary = {"C1": 1, "C2": 2, UNKNOWN_TOKEN: 0}

    def test_tokenize_with_known_vocab(self):
        """Test tokenization with a given vocabulary."""
        result = tokenize_partition(self.series, self.vocabulary)
        expected_tokens = [
            1,
            2,
            0,
            0,
            0,
        ]  # C1 -> 1, C2 -> 2, others -> UNKNOWN_TOKEN (0)
        self.assertListEqual(result.tolist(), expected_tokens)

    def test_tokenize_with_empty_vocab(self):
        """Test tokenization with an empty vocabulary."""
        vocabulary = {UNKNOWN_TOKEN: 0}
        result = tokenize_partition(self.series, vocabulary)
        expected_tokens = [0, 0, 0, 0, 0]
        self.assertListEqual(result.tolist(), expected_tokens)

    def test_tokenize_with_all_vocab(self):
        """Test tokenization when all concepts are in the vocabulary."""
        vocabulary = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, UNKNOWN_TOKEN: 0}
        result = tokenize_partition(self.series, vocabulary)
        expected_tokens = [1, 2, 3, 4, 5]
        self.assertListEqual(result.tolist(), expected_tokens)


if __name__ == "__main__":
    unittest.main()
