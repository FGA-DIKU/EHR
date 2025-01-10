import unittest
import pandas as pd
import dask.dataframe as dd

# Import the EHRTokenizer class and constants
from corebehrt.classes.tokenizer import EHRTokenizer
from corebehrt.functional.constants import (
    CLS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    UNKNOWN_TOKEN,
)


class TestEHRTokenizer(unittest.TestCase):
    def setUp(self):
        # Simplified example data
        data = {
            "PID": ["P1", "P1", "P1", "P2", "P2", "P3", "P3", "P4"],
            "concept": ["C1", "C2", "C3", "C1", "C4", "C5", "C6", "C7"],
            "age": [20, 21, 22, 23, 24, 25, 26, 27],
            "abspos": [1, 2, 3, 1, 2, 1, 2, 1],
            "segment": [0, 0, 1, 0, 1, 0, 1, 0],
        }
        # Create a Pandas DataFrame
        self.df = pd.DataFrame(data)
        # Set 'PID' as the index and sort by 'PID' and 'abspos'
        self.df = self.df.set_index("PID").sort_values(["PID", "abspos"])
        # Convert to Dask DataFrame with multiple partitions
        self.ddf = dd.from_pandas(self.df, npartitions=2)

    def test_tokenizer_with_new_vocab(self):
        """Test tokenization with a new vocabulary."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Check if vocabulary is updated with concepts
        self.assertIn("C1", tokenizer.vocabulary)
        self.assertIn("C2", tokenizer.vocabulary)

        # Ensure concepts are tokenized to integers
        self.assertTrue(pd.api.types.is_integer_dtype(result_df["concept"]))

    def test_tokenizer_with_existing_vocab(self):
        """Test tokenization with an existing vocabulary."""
        existing_vocab = {
            PAD_TOKEN: 0,
            CLS_TOKEN: 1,
            SEP_TOKEN: 2,
            UNKNOWN_TOKEN: 3,
            "C1": 4,
            "C2": 5,
        }
        tokenizer = EHRTokenizer(vocabulary=existing_vocab)
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Check if vocabulary is not updated
        self.assertEqual(tokenizer.vocabulary, existing_vocab)

        # Concepts not in vocabulary should be tokenized as UNKNOWN_TOKEN
        unk_token = tokenizer.vocabulary[UNKNOWN_TOKEN]
        unknown_concepts = ["C3", "C4", "C5", "C6", "C7"]
        for concept in unknown_concepts:
            self.assertNotIn(concept, tokenizer.vocabulary)
        self.assertTrue((result_df["concept"] == unk_token).any())

    def test_tokenizer_with_cutoffs(self):
        """Test tokenizer with cutoffs applied."""
        cutoffs = {"C2": 1}  # Limit concepts starting with 'C2' to length 1
        tokenizer = EHRTokenizer(cutoffs=cutoffs)
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Create inverse vocabulary to map tokens back to concepts
        inv_vocab = {v: k for k, v in tokenizer.vocabulary.items()}

        # Map the integer tokens back to concepts
        result_df["concept_str"] = result_df["concept"].map(inv_vocab)

        # Check that 'C2' is not in the vocabulary but 'C' is
        self.assertNotIn("C2", tokenizer.vocabulary)
        self.assertIn("C", tokenizer.vocabulary)

        # Get the original data before tokenization
        df_before_tokenization = self.ddf.compute()

        # Find indices where the original concept was 'C2'
        indices_c2 = df_before_tokenization[
            df_before_tokenization["concept"] == "C2"
        ].index

        # For these indices, check that the tokenized concept is 'C'
        for idx in indices_c2:
            tokenized_concept = result_df.loc[idx, "concept_str"]
            self.assertEqual(tokenized_concept, "C")

        # Ensure 'C2' does not appear in the tokenized concepts
        self.assertNotIn("C2", result_df["concept_str"].unique())

        # Check that all concepts starting with 'C2' are truncated in the vocabulary
        for concept in tokenizer.vocabulary.keys():
            if concept.startswith("C2"):
                self.fail(
                    f"Concept '{concept}' should have been truncated but is in vocabulary."
                )

        # Optionally, check that other concepts are unaffected
        unaffected_concepts = ["C1", "C3", "C4", "C5", "C6", "C7"]
        for concept in unaffected_concepts:
            self.assertIn(concept, tokenizer.vocabulary)

    def test_tokenizer_without_special_tokens(self):
        """Test tokenizer without adding special tokens."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Check that special tokens are not added
        self.assertNotIn(tokenizer.vocabulary[CLS_TOKEN], result_df["concept"].values)
        self.assertNotIn(tokenizer.vocabulary[SEP_TOKEN], result_df["concept"].values)

    def test_tokenizer_freeze_vocabulary(self):
        """Test tokenizer with frozen vocabulary."""
        tokenizer = EHRTokenizer()
        tokenizer.freeze_vocabulary()
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Since vocabulary is frozen and started empty, all concepts should be UNKNOWN_TOKEN
        unk_token = tokenizer.vocabulary[UNKNOWN_TOKEN]
        self.assertTrue((result_df["concept"] == unk_token).all())

    def test_tokenizer_multiple_pids(self):
        """Test tokenizer with multiple PIDs."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Check that all PIDs are present
        expected_pids = ["P1", "P2", "P3", "P4"]
        self.assertCountEqual(result_df.index.unique(), expected_pids)

    def test_tokenizer_abspos_order(self):
        """Test that abspos ordering is maintained after tokenization."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.ddf)
        result_df = result.compute()

        # Reset index to access 'PID' as a column
        result_df = result_df.reset_index()

        # Check that abspos within each PID is sorted correctly
        for pid in result_df["PID"].unique():
            pid_df = result_df[result_df["PID"] == pid]
            abspos_values = pid_df["abspos"].values
            self.assertTrue(
                all(x <= y for x, y in zip(abspos_values, abspos_values[1:]))
            )


if __name__ == "__main__":
    unittest.main()
