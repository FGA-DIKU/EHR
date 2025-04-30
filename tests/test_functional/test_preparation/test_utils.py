import re
import unittest

import pandas as pd

from corebehrt.constants.data import CONCEPT_COL, PID_COL
from corebehrt.functional.preparation.utils import (
    get_background_length,
    get_background_length_pd,
    get_background_tokens,
    get_concept_id_to_delay,
    get_non_priority_tokens,
    subset_patient_data,
)
from corebehrt.modules.preparation.dataset import PatientData


class TestBackgroundFunctions(unittest.TestCase):
    def setUp(self):
        # Create a small vocabulary. Keys are the "token strings", values are numeric IDs
        self.vocab = {
            "BG_TEST": 0,
            "BG_ANOTHER": 1,
            "REGULAR_TOKEN": 2,
            "BG_SOMETHING": 3,
            "REG2": 4,
            "LOW_A": 5,
            "LOW_B": 6,
        }

        # A second vocabulary for edge-case tests if needed
        self.empty_vocab = {}

        # A list of patient objects
        self.patients = [
            PatientData(
                pid=1,
                concepts=[0, 2, 3],  # tokens found in vocab
                abspos=[1, 2, 3],
                segments=[0, 0, 0],
                ages=[30, 31, 32],
                outcome=0,
            ),
            PatientData(
                pid=2,
                concepts=[1, 4],  # tokens found in vocab
                abspos=[4, 5],
                segments=[1, 1],
                ages=[40, 41],
                outcome=1,
            ),
        ]

    # ---------------------------------------------------------------------
    # 1) get_background_length
    # ---------------------------------------------------------------------
    def test_get_background_length_normal(self):
        # We have BG_TEST=0, BG_ANOTHER=1, BG_SOMETHING=3 in the vocab
        # The first patient's concepts are [0, 2, 3].
        # Intersection with background tokens = {0, 3} => length=2
        # +2 for [CLS] and [SEP] => total 4
        length = get_background_length(self.patients, self.vocab)
        self.assertEqual(length, 4)

    def test_get_background_length_empty_vocab(self):
        # No BG_ tokens at all
        length = get_background_length(self.patients, self.empty_vocab)
        # Intersection = 0, +2 => 2
        self.assertEqual(length, 2)

    # ---------------------------------------------------------------------
    # 2) get_background_length_dd
    # ---------------------------------------------------------------------
    def test_get_background_length_pd_normal(self):
        # Construct a pandas dataframe from a small pandas frame
        # This must have at least a "concept" column to match your usage
        pdf = pd.DataFrame(
            {
                PID_COL: [1] * 4,
                CONCEPT_COL: [0, 2, 3, 5],  # a few tokens
            }
        ).set_index(PID_COL)
        # Intersection with background tokens = {0, 3}
        # => 2 + 2 => 4
        length = get_background_length_pd(pdf, self.vocab)
        self.assertEqual(length, 4)

    def test_get_background_length_pd_empty_df(self):
        # If the DataFrame is empty, function should return 2 (CLS + SEP)
        pdf = pd.DataFrame(columns=[PID_COL, CONCEPT_COL]).set_index(PID_COL)

        length = get_background_length_pd(pdf, self.vocab)
        self.assertEqual(length, 2)

    # ---------------------------------------------------------------------
    # 3) get_background_tokens
    # ---------------------------------------------------------------------
    def test_get_background_tokens(self):
        # vocab has keys: BG_TEST=0, BG_ANOTHER=1, BG_SOMETHING=3 => background
        # The function returns these numeric values as a set => {0,1,3}
        bg_tokens = get_background_tokens(self.vocab)
        self.assertSetEqual(bg_tokens, {0, 1, 3})

    def test_get_background_tokens_empty_vocab(self):
        self.assertEqual(get_background_tokens(self.empty_vocab), set())

    # ---------------------------------------------------------------------
    # 4) get_non_priority_tokens
    # ---------------------------------------------------------------------
    def test_get_non_priority_tokens(self):
        # "low_priority_prefixes" = ["LOW_"]
        # Our vocab includes "LOW_A"=5, "LOW_B"=6 => we expect {5, 6}
        result = get_non_priority_tokens(self.vocab, ["LOW_"])
        self.assertSetEqual(result, {5, 6})

    def test_get_non_priority_tokens_multiple_prefixes(self):
        # Suppose we pass in prefixes = ["LOW_", "REG"]
        # Then we'd collect "LOW_A"=5, "LOW_B"=6, "REGULAR_TOKEN"=2, "REG2"=4
        result = get_non_priority_tokens(self.vocab, ["LOW_", "REG"])
        self.assertSetEqual(result, {2, 4, 5, 6})

    # ---------------------------------------------------------------------
    # 5) subset_patient_data
    # ---------------------------------------------------------------------
    def test_subset_patient_data_partial_indices(self):
        patient = self.patients[0]  # concepts=[0,2,3], abspos=[1,2,3] ...
        keep_indices = [0, 2]  # keep first and last

        new_patient = subset_patient_data(patient, keep_indices)
        # Check if list fields have length 2 now
        self.assertEqual(len(new_patient.concepts), 2)
        self.assertEqual(len(new_patient.abspos), 2)
        self.assertEqual(len(new_patient.segments), 2)
        self.assertEqual(len(new_patient.ages), 2)
        # Values should match
        self.assertEqual(new_patient.concepts, [0, 3])
        self.assertEqual(new_patient.abspos, [1, 3])
        self.assertEqual(new_patient.segments, [0, 0])
        self.assertEqual(new_patient.ages, [30, 32])
        # Non-list attribute remains
        self.assertEqual(new_patient.outcome, 0)
        self.assertEqual(new_patient.pid, 1)

    def test_subset_patient_data_full_indices(self):
        patient = self.patients[0]
        keep_indices = list(range(len(patient.concepts)))  # keep them all
        new_patient = subset_patient_data(patient, keep_indices)
        # Should be the same as original
        self.assertEqual(new_patient.concepts, patient.concepts)
        self.assertEqual(new_patient.abspos, patient.abspos)
        self.assertEqual(new_patient.segments, patient.segments)
        self.assertEqual(new_patient.ages, patient.ages)
        self.assertEqual(new_patient.outcome, patient.outcome)

    def test_subset_patient_data_empty_indices(self):
        patient = self.patients[0]
        new_patient = subset_patient_data(patient, [])
        # All list fields become empty
        self.assertEqual(new_patient.concepts, [])
        self.assertEqual(new_patient.abspos, [])
        self.assertEqual(new_patient.segments, [])
        self.assertEqual(new_patient.ages, [])
        # Non-list fields unchanged
        self.assertEqual(new_patient.outcome, 0)
        self.assertEqual(new_patient.pid, 1)


class TestConceptDelayMapping(unittest.TestCase):
    def setUp(self):
        # Create a test vocabulary with various concept patterns
        """
        Initializes a test vocabulary with concept keys for use in concept delay mapping tests.
        """
        self.vocab = {
            "D_12345": 101,
            "D_67890": 102,
            "LAB_HB": 201,
            "LAB_WBC": 202,
            "MED_123": 301,
            "BG_TEST": 401,
        }

    def test_basic_pattern_matching(self):
        # Test basic prefix matching
        """
        Tests that concept IDs are correctly mapped to delays based on prefix regex patterns.
        """
        concept_delays = {
            "^D_": 24,  # All D_ concepts get 24h delay
            "^LAB_": 48,  # All LAB_ concepts get 48h delay
        }

        result = get_concept_id_to_delay(concept_delays, self.vocab)

        expected = {
            101: 24,  # D_12345
            102: 24,  # D_67890
            201: 48,  # LAB_HB
            202: 48,  # LAB_WBC
        }
        self.assertEqual(result, expected)

    def test_overlapping_patterns(self):
        # Test when patterns might overlap
        """
        Tests that when multiple regex patterns match a concept, the delay from the last matching pattern is used.

        Verifies that `get_concept_id_to_delay` applies overlapping patterns in order, with later matches overriding earlier ones for the same concept ID.
        """
        concept_delays = {
            "^D_": 24,
            "5$": 72,  # Ends with 5
        }

        result = get_concept_id_to_delay(concept_delays, self.vocab)

        # D_12345 matches both patterns, last one wins
        expected = {
            101: 72,  # D_12345 (matches both patterns)
            102: 24,  # D_67890
        }
        self.assertEqual(result, expected)

    def test_no_matches(self):
        # Test when pattern matches nothing
        """
        Tests that get_concept_id_to_delay returns an empty dictionary when no vocab keys match the provided regex patterns.
        """
        concept_delays = {
            "^NONEXISTENT_": 24,
        }

        result = get_concept_id_to_delay(concept_delays, self.vocab)

        self.assertEqual(result, {})

    def test_complex_regex(self):
        # Test more complex regex patterns
        """
        Tests get_concept_id_to_delay with complex regex patterns for concept keys.

        Verifies that concept IDs are correctly mapped to delays when concept keys match
        patterns requiring specific digit counts or uppercase letter sequences.
        """
        concept_delays = {
            r"^D_\d{5}$": 24,  # Exactly D_ followed by 5 digits
            r"LAB_[A-Z]+$": 48,  # LAB_ followed by uppercase letters
        }

        result = get_concept_id_to_delay(concept_delays, self.vocab)

        expected = {
            101: 24,  # D_12345
            102: 24,  # D_67890
            201: 48,  # LAB_HB
            202: 48,  # LAB_WBC
        }
        self.assertEqual(result, expected)

    def test_invalid_regex(self):
        # Test handling of invalid regex patterns
        """
        Tests that get_concept_id_to_delay raises a re.error when given an invalid regex pattern.
        """
        concept_delays = {
            "[": 24,  # Invalid regex
        }

        with self.assertRaises(re.error):
            get_concept_id_to_delay(concept_delays, self.vocab)


if __name__ == "__main__":
    unittest.main()
