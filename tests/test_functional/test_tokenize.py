import unittest
import pandas as pd

# Import the updated functions and constants
from corebehrt.functional.tokenize import (
    add_special_tokens_partition,
    limit_concept_length_partition,
    tokenize_partition,
)
from corebehrt.functional.constants import (
    CLS_TOKEN,
    SEP_TOKEN,
    UNKNOWN_TOKEN,
)


class TestAddSpecialTokensPartition(unittest.TestCase):
    def setUp(self):
        # Sample data for testing with 'PID' as the index
        self.data = pd.DataFrame(
            {
                "concept": ["C1", "C2", "C3", "C4", "C5", "C6"],
                "abspos": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
                "segment": [0, 0, 1, 0, 1, 0],
            },
            index=["P1", "P1", "P1", "P2", "P2", "P3"],
        )
        self.data.index.name = "PID"

    def test_add_cls_token(self):
        """Test that CLS_TOKEN is correctly added."""
        result = add_special_tokens_partition(self.data, add_sep=False, add_cls=True)
        cls_rows = result[result["concept"] == CLS_TOKEN]
        # There should be one CLS_TOKEN per PID
        self.assertEqual(cls_rows.index.nunique(), self.data.index.nunique())
        # CLS_TOKEN should have the smallest abspos for each PID
        for pid in self.data.index.unique():
            pid_rows = result[result.index == pid]
            cls_row = pid_rows[pid_rows["concept"] == CLS_TOKEN]
            earliest_abspos = pid_rows["abspos"].min()
            self.assertTrue((cls_row["abspos"] <= earliest_abspos).all())

    def test_add_sep_token(self):
        """Test that SEP_TOKEN is correctly added after segment changes."""
        result = add_special_tokens_partition(self.data, add_sep=True, add_cls=False)
        sep_rows = result[result["concept"] == SEP_TOKEN]
        # There should be a SEP_TOKEN for each segment change within a PID
        expected_sep_count = self.data.groupby("PID")["segment"].apply(
            lambda x: x.diff().fillna(0).ne(0).sum()
        )
        actual_sep_count = sep_rows.groupby(sep_rows.index)["concept"].count()
        for pid in expected_sep_count.index:
            self.assertEqual(expected_sep_count[pid], actual_sep_count.get(pid, 0))

    def test_add_both_special_tokens(self):
        """Test that both CLS_TOKEN and SEP_TOKEN are correctly added."""
        result = add_special_tokens_partition(self.data, add_sep=True, add_cls=True)
        # Verify CLS_TOKEN
        cls_rows = result[result["concept"] == CLS_TOKEN]
        self.assertEqual(cls_rows.index.nunique(), self.data.index.nunique())
        # Verify SEP_TOKEN
        sep_rows = result[result["concept"] == SEP_TOKEN]
        expected_sep_count = self.data.groupby("PID")["segment"].apply(
            lambda x: x.diff().fillna(0).ne(0).sum()
        )
        actual_sep_count = sep_rows.groupby(sep_rows.index)["concept"].count()
        for pid in expected_sep_count.index:
            self.assertEqual(expected_sep_count[pid], actual_sep_count.get(pid, 0))

    def test_no_special_tokens(self):
        """Test that no special tokens are added when flags are False."""
        result = add_special_tokens_partition(self.data, add_sep=False, add_cls=False)
        self.assertNotIn(CLS_TOKEN, result["concept"].values)
        self.assertNotIn(SEP_TOKEN, result["concept"].values)
        self.assertEqual(len(result), len(self.data))


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
