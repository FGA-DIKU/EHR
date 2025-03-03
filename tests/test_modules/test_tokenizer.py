import unittest
import pandas as pd

# Import the EHRTokenizer class and constants
from corebehrt.modules.features.tokenizer import EHRTokenizer
from corebehrt.constants.data import (
    CLS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    UNKNOWN_TOKEN,
    PID_COL,
    CONCEPT_COL,
)


class TestEHRTokenizer(unittest.TestCase):
    def setUp(self):
        # Simplified example data
        data = {
            PID_COL: [1, 1, 1, 2, 2, 3, 3, 4],
            CONCEPT_COL: ["C1", "C2", "C3", "C1", "C4", "C5", "C6", "C7"],
            "age": [20, 21, 22, 23, 24, 25, 26, 27],
            "abspos": [1, 2, 3, 1, 2, 1, 2, 1],
            "segment": [0, 0, 1, 0, 1, 0, 1, 0],
        }
        # Create a Pandas DataFrame
        self.df = pd.DataFrame(data)
        # Set 'PID' as the index and sort by 'PID' and 'abspos'
        self.df = self.df.set_index(PID_COL).sort_values([PID_COL, "abspos"])

    def test_tokenizer_with_new_vocab(self):
        """Test tokenization with a new vocabulary."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.df)

        # Check if special tokens are added
        self.assertIn(tokenizer.vocabulary[CLS_TOKEN], result[CONCEPT_COL].values)
        self.assertIn(tokenizer.vocabulary[SEP_TOKEN], result[CONCEPT_COL].values)

        # Check if vocabulary is updated with concepts
        self.assertIn("C1", tokenizer.vocabulary)
        self.assertIn("C2", tokenizer.vocabulary)

        # Ensure concepts are tokenized to integers
        self.assertTrue(pd.api.types.is_integer_dtype(result[CONCEPT_COL]))

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
        result = tokenizer(self.df)

        # Check if vocabulary is not updated
        self.assertEqual(tokenizer.vocabulary, existing_vocab)

        # Concepts not in vocabulary should be tokenized as UNKNOWN_TOKEN
        unk_token = tokenizer.vocabulary[UNKNOWN_TOKEN]
        unknown_concepts = ["C3", "C4", "C5", "C6", "C7"]
        for concept in unknown_concepts:
            self.assertNotIn(concept, tokenizer.vocabulary)
        self.assertTrue((result[CONCEPT_COL] == unk_token).any())

    def test_tokenizer_with_cutoffs(self):
        """Test tokenizer with cutoffs applied."""
        cutoffs = {"C2": 1}  # Limit concepts starting with 'C2' to length 1
        tokenizer = EHRTokenizer(cutoffs=cutoffs, cls_token=False, sep_tokens=False)
        result = tokenizer(self.df)

        # Create inverse vocabulary to map tokens back to concepts
        inv_vocab = {v: k for k, v in tokenizer.vocabulary.items()}

        # Map the integer tokens back to concepts
        result["concept_str"] = result[CONCEPT_COL].map(inv_vocab)

        # Check that 'C2' is not in the vocabulary but 'C' is
        self.assertNotIn("C2", tokenizer.vocabulary)
        self.assertIn("C", tokenizer.vocabulary)

        # Get the original data before tokenization
        df_before_tokenization = self.df

        # Find indices where the original concept was 'C2'
        indices_c2 = df_before_tokenization[
            df_before_tokenization[CONCEPT_COL] == "C2"
        ].index

        # For these indices, check that the tokenized concept is 'C'
        for idx in indices_c2:
            tokenized_concept = result.loc[idx, "concept_str"]
            self.assertEqual(tokenized_concept, "C")

        # Ensure 'C2' does not appear in the tokenized concepts
        self.assertNotIn("C2", result["concept_str"].unique())

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
        tokenizer = EHRTokenizer(sep_tokens=False, cls_token=False)
        result = tokenizer(self.df)

        # Check that special tokens are not added
        self.assertNotIn(tokenizer.vocabulary[CLS_TOKEN], result[CONCEPT_COL].values)
        self.assertNotIn(tokenizer.vocabulary[SEP_TOKEN], result[CONCEPT_COL].values)

    def test_tokenizer_freeze_vocabulary(self):
        """Test tokenizer with frozen vocabulary."""
        tokenizer = EHRTokenizer(sep_tokens=False, cls_token=False)
        tokenizer.freeze_vocabulary()
        result = tokenizer(self.df)

        # Since vocabulary is frozen and started empty, all concepts should be UNKNOWN_TOKEN
        unk_token = tokenizer.vocabulary[UNKNOWN_TOKEN]
        self.assertTrue((result[CONCEPT_COL] == unk_token).all())

    def test_tokenizer_multiple_pids(self):
        """Test tokenizer with multiple PIDs."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.df)

        # Check that all PIDs are present
        expected_pids = [1, 2, 3, 4]
        self.assertCountEqual(result.index.unique(), expected_pids)

    def test_tokenizer_abspos_order(self):
        """Test that abspos ordering is maintained after tokenization."""
        tokenizer = EHRTokenizer()
        result = tokenizer(self.df)

        # Reset index to access 'PID' as a column
        result = result.reset_index()

        # Check that abspos within each PID is sorted correctly
        for pid in result[PID_COL].unique():
            pid_df = result[result[PID_COL] == pid]
            abspos_values = pid_df["abspos"].values
            self.assertTrue(
                all(x <= y for x, y in zip(abspos_values, abspos_values[1:]))
            )

    def test_tokenizer_segment_changes(self):
        """Test that SEP_TOKEN tokens are added at segment changes."""
        tokenizer = EHRTokenizer(sep_tokens=True, cls_token=False)
        result = tokenizer(self.df)

        # Reset index to access 'PID' as a column
        result = result.reset_index()

        # Identify positions of SEP_TOKEN tokens
        sep_token_id = tokenizer.vocabulary[SEP_TOKEN]

        # For each PID, check that SEP_TOKEN tokens are correctly placed
        for pid in result[PID_COL].unique():
            pid_df = result[result[PID_COL] == pid].reset_index(drop=True)

            # Calculate segment changes as per the function
            segment_changes = pid_df.index[
                (pid_df["segment"] != pid_df["segment"].shift(-1))
                & (pid_df["segment"].shift(-1).notnull())
            ].tolist()

            # Get indices of SEP_TOKEN tokens for this PID
            sep_indices = pid_df.index[pid_df[CONCEPT_COL] == sep_token_id].tolist()

            # Check that number of SEP_TOKEN tokens matches number of segment changes
            self.assertEqual(len(segment_changes), len(sep_indices))

            # Check that SEP_TOKEN tokens appear at the correct positions
            for change_idx, sep_idx in zip(segment_changes, sep_indices):
                # The SEP_TOKEN should be at the same index as the segment change
                self.assertEqual(sep_idx, change_idx)

    def test_tokenizer_cls_token(self):
        """Test that CLS_TOKEN is added at the beginning of each PID sequence."""
        tokenizer = EHRTokenizer(cls_token=True, sep_tokens=False)
        result = tokenizer(self.df)

        # Reset index to access 'PID' as a column
        result = result.reset_index()

        cls_token_id = tokenizer.vocabulary[CLS_TOKEN]

        # Check that the first concept for each PID is CLS_TOKEN
        for pid in result[PID_COL].unique():
            pid_df = result[result[PID_COL] == pid]
            first_concept = pid_df[CONCEPT_COL].iloc[0]
            self.assertEqual(first_concept, cls_token_id)


if __name__ == "__main__":
    unittest.main()
