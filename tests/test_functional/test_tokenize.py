import unittest
import pandas as pd

# Import the updated functions and constants
from corebehrt.functional.tokenize import (
    limit_concept_length_partition,
    tokenize_partition,
)
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


class TestLimitConceptLengthPartition(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(["BG_GENDER_M", "C1", "BG_AGE_30", "C2", "BG_BP_HIGH"])
        self.cutoffs = {"BG_": 5}  # Limit concepts starting with 'BG_' to length 5

    def test_limit_concepts(self):
        """Test that concepts are correctly limited in length."""
        result = limit_concept_length_partition(self.series, self.cutoffs)
        expected_series = pd.Series(["BG_GE", "C1", "BG_AG", "C2", "BG_BP"])
        self.assertTrue(result.equals(expected_series))

    def test_no_cutoff(self):
        """Test that concepts remain unchanged when no cutoffs are provided."""
        result = limit_concept_length_partition(self.series, {})
        self.assertTrue(result.equals(self.series))

    def test_multiple_cutoffs(self):
        """Test that multiple cutoffs are applied correctly."""
        cutoffs = {"BG_": 5, "C": 2}
        result = limit_concept_length_partition(self.series, cutoffs)
        expected_series = pd.Series(["BG_GE", "C1", "BG_AG", "C2", "BG_BP"])
        self.assertTrue(result.equals(expected_series))

    def test_non_matching_prefix(self):
        """Test that concepts without matching prefixes are unchanged."""
        cutoffs = {"X_": 3}
        result = limit_concept_length_partition(self.series, cutoffs)
        self.assertTrue(result.equals(self.series))

    def test_empty_series(self):
        """Test that an empty series returns an empty series."""
        empty_series = pd.Series([], dtype=str)
        result = limit_concept_length_partition(empty_series, self.cutoffs)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
