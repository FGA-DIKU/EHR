import random
import unittest

from corebehrt.classes.dataset import PatientData, PatientDataset
from corebehrt.functional.truncate import (
    _get_non_priority_mask,
    _drop_non_priority_tokens,
    _filter_invalid_positions,
    prioritized_truncate_patient,
    truncate_patient,
)


class TestTruncationFunctions(unittest.TestCase):
    def setUp(self):
        # -----------------------------------------------------------------
        # 1. Prepare vocabulary
        # -----------------------------------------------------------------
        self.vocabulary = {
            "[CLS]": 1,
            "[SEP]": 2,
            "D_A": 3,
            "D_B": 4,
            "D_C": 5,
            "LAB_A": 6,
            "VAL_60": 7,
            "VAL_70": 8,
            "VAL_80": 9,
            "VAL_90": 10,
            "VAL_100": 11,
            "BG_Gender": 12,
        }
        self.background_len = 3
        self.sep_token = self.vocabulary["[SEP]"]
        self.cls_token = self.vocabulary["[CLS]"]
        self.gender_token = self.vocabulary["BG_Gender"]

        # -----------------------------------------------------------------
        # 2. Create sample PatientData objects
        # -----------------------------------------------------------------
        # patient 1 (lots of tokens, forcing truncation)
        p1_concepts = (
            [self.cls_token, self.gender_token, self.sep_token]
            + [3] * 10
            + [self.sep_token]
            + [4] * 490
            + [self.sep_token]
            + [5] * 50
        )
        p1_abspos = [i for i in range(len(p1_concepts))]
        p1_segments = [0] * len(p1_concepts)
        p1_ages = [30 + 0.1 * i for i in range(len(p1_concepts))]
        self.patient1 = PatientData(
            pid="1",
            concepts=p1_concepts,
            abspos=p1_abspos,
            segments=p1_segments,
            ages=p1_ages,
        )

        # patient 2 (large but slightly different distribution)
        p2_concepts = (
            [self.cls_token, self.gender_token, self.sep_token]
            + [3] * 400
            + [self.sep_token]
            + [5] * 10
            + [self.sep_token]
            + [5] * 10
        )
        p2_abspos = [i for i in range(len(p2_concepts))]
        self.patient2 = PatientData(
            pid="2",
            concepts=p2_concepts,
            abspos=p2_abspos,
            segments=[0] * len(p2_concepts),
            ages=[40.0] * len(p2_concepts),
        )

        # patient 3 (shorter)
        p3_concepts = (
            [self.cls_token, self.gender_token, self.sep_token]
            + [3] * 10
            + [self.sep_token]
            + [4] * 58
        )
        p3_abspos = [i for i in range(len(p3_concepts))]
        self.patient3 = PatientData(
            pid="3",
            concepts=p3_concepts,
            abspos=p3_abspos,
            segments=[0] * len(p3_concepts),
            ages=[50.0] * len(p3_concepts),
        )

        # patient 4 (mixed LAB/VAL tokens)
        p4_concepts = (
            [self.cls_token, self.gender_token, self.sep_token]
            + [3] * 10
            + [self.sep_token]
            + [4] * 10
            + [3] * 10
            + [self.sep_token]
            + [5] * 20
            + [6 if i % 2 == 0 else random.randint(7, 11) for i in range(20)]
        )
        p4_abspos = [i for i in range(len(p4_concepts))]
        self.patient4 = PatientData(
            pid="4",
            concepts=p4_concepts,
            abspos=p4_abspos,
            segments=[0] * len(p4_concepts),
            ages=[60.0] * len(p4_concepts),
        )

        # patient 5 (unit testing)
        p5_concepts = [self.cls_token, self.gender_token, self.sep_token] + [
            6 if i % 2 == 0 else random.randint(7, 11) for i in range(20)
        ]
        # Example abspos array
        p5_abspos = [-100, -100, -100] + list(range(15, 25)) * 2
        self.patient5 = PatientData(
            pid="5",
            concepts=p5_concepts,
            abspos=p5_abspos,
            segments=[0] * len(p5_concepts),
            ages=[70.0] * len(p5_concepts),
        )

        # -----------------------------------------------------------------
        # 3. Create a PatientDataset with some patients
        # -----------------------------------------------------------------
        self.dataset = PatientDataset(
            patients=[self.patient1, self.patient2, self.patient3, self.patient4]
        )

    # ---------------------------------------------------------------------
    # Test 1: Single-patient, basic truncation
    # ---------------------------------------------------------------------
    def test_truncate_patient(self):
        max_len = 100
        truncated = truncate_patient(
            self.patient1, self.background_len, max_len, self.sep_token
        )
        self.assertTrue(
            len(truncated.concepts) <= max_len, "Truncated patient data exceeds max_len"
        )

    # ---------------------------------------------------------------------
    # Test 2: Single-patient, no truncation needed
    # ---------------------------------------------------------------------
    def test_truncate_patient_no_truncation_needed(self):
        max_len = 1000
        truncated = truncate_patient(
            self.patient3, self.background_len, max_len, self.sep_token
        )
        self.assertEqual(
            len(truncated.concepts),
            len(self.patient3.concepts),
            "Data should remain unchanged if already under max_len",
        )

    # ---------------------------------------------------------------------
    # Test 3: Check that background tokens (the first N) are kept
    # ---------------------------------------------------------------------
    def test_truncate_patient_keeps_background(self):
        max_len = 50
        truncated = truncate_patient(
            self.patient1, self.background_len, max_len, self.sep_token
        )
        # Check the first three tokens
        self.assertEqual(
            truncated.concepts[0], self.cls_token, "Missing [CLS] at index 0"
        )
        self.assertEqual(
            truncated.concepts[1], self.gender_token, "Missing BG token at index 1"
        )
        self.assertEqual(
            truncated.concepts[2], self.sep_token, "Missing [SEP] at index 2"
        )

    # ---------------------------------------------------------------------
    # Test 4: Parallel processing of multiple patients
    # ---------------------------------------------------------------------
    def test_truncate_dataset_parallel(self):
        max_len = 80
        # Process in parallel
        results = self.dataset.process_in_parallel(
            truncate_patient,
            background_length=self.background_len,
            max_len=max_len,
            sep_token=self.sep_token,
        )
        # 'results' is a list of truncated PatientData objects
        for patient_data in results:
            self.assertTrue(
                len(patient_data.concepts) <= max_len,
                f"PID {patient_data.pid} still exceeds max_len.",
            )

    # ---------------------------------------------------------------------
    # Test 5: Check prioritized truncation logic (example)
    # ---------------------------------------------------------------------
    def test_truncate_patient_w_priority_drop_all(self):
        max_len = 50
        # Build 'non_priority_tokens' for prefix = ['LAB', 'VAL']
        non_priority_tokens = {
            v
            for k, v in self.vocabulary.items()
            if any(k.startswith(pfx) for pfx in ["LAB", "VAL"])
        }
        truncated = truncate_patient(
            self.patient4,
            self.background_len,
            max_len,
            self.sep_token,
            non_priority_tokens,
        )
        # Example checks
        self.assertTrue(len(truncated.concepts) <= max_len)
        # Ensure all 'LAB'/'VAL' tokens are removed (in your real code, you might do partial removal)
        self.assertTrue(
            all(x not in non_priority_tokens for x in truncated.concepts),
            "Non-priority tokens still present",
        )

    # ---------------------------------------------------------------------
    # Test 6: Unit-based dropping
    # ---------------------------------------------------------------------
    def test_truncate_patient_w_priority_unit(self):
        max_len = 8
        non_priority_tokens = {
            v
            for k, v in self.vocabulary.items()
            if any(k.startswith(pfx) for pfx in ["LAB", "VAL"])
        }
        truncated = prioritized_truncate_patient(
            patient=self.patient5,
            background_length=self.background_len,
            max_len=max_len,
            sep_token=self.sep_token,
            non_priority_tokens=non_priority_tokens,
            unit=True,  # triggers the "unit" logic
            low_priority_prefixes=["LAB", "VAL"],
        )
        # You might expect a certain length or certain tokens removed:
        self.assertTrue(
            len(truncated.concepts) <= max_len, "Truncated data is still too long"
        )


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        # Minimal setup:
        # We'll define a few small patients with known positions, etc.

        # Example vocabulary for reference (optional in these tests)
        self.vocab = {
            "CLS": 1,
            "SEP": 2,
            "A": 3,
            "B": 4,
            "C": 5,
            "LAB_X": 6,
            "VAL_X": 7,
        }

        # Basic patient #1 with 10 tokens
        # Background region = first 3 tokens
        self.patient1 = PatientData(
            pid="P1",
            concepts=[1, 2, 3, 4, 4, 6, 7, 3, 4, 6],  # 10 tokens
            abspos=[0, 0, 1, 2, 3, 4, 4, 5, 6, 7],  # positions
            segments=[0] * 10,
            ages=[10] * 10,
            outcome=None,
        )

        # Another small patient #2
        self.patient2 = PatientData(
            pid="P2",
            concepts=[1, 6, 2, 3, 6, 7, 4],
            abspos=[0, 0, 0, 1, 1, 1, 2],
            segments=[0] * 7,
            ages=[20] * 7,
            outcome=None,
        )

    # -----------------------------------------------------------------
    # Test _get_non_priority_mask
    # -----------------------------------------------------------------
    def test_get_non_priority_mask(self):
        # Suppose tokens 6 and 7 (LAB_X, VAL_X) are non-priority
        non_priority = {6, 7}

        mask1 = _get_non_priority_mask(self.patient1, non_priority)
        self.assertEqual(len(mask1), len(self.patient1.concepts))
        # For patient1, tokens: [1,2,3,4,4,6,7,3,4,6]
        # non-priority => positions 5,6,9 => 6,7,6
        self.assertListEqual(
            mask1, [False, False, False, False, False, True, True, False, False, True]
        )

        mask2 = _get_non_priority_mask(self.patient2, non_priority)
        # For patient2, tokens: [1,6,2,3,6,7,4]
        # non-priority => positions 1,4,5 => 6,6,7
        self.assertListEqual(mask2, [False, True, False, False, True, True, False])

    # -----------------------------------------------------------------
    # Test _drop_non_priority_tokens
    # -----------------------------------------------------------------
    def test_drop_non_priority_tokens(self):
        # We'll drop non-priority tokens: 6,7 (like before)
        non_priority = {6, 7}
        mask1 = _get_non_priority_mask(self.patient1, non_priority)

        # We'll allow truncation_length=2 beyond background=3 => target=5
        # The patient has 10 tokens, so we want to reduce from 10 down to 5
        # by dropping non-priority tokens beyond index 2.
        truncated_p1 = _drop_non_priority_tokens(
            patient=self.patient1,
            non_priority_mask=mask1,
            truncation_length=2,
            background_length=3,
        )
        # Let's see what remains
        self.assertEqual(len(truncated_p1.concepts), 5)

        # Specifically, check that the first 3 tokens are untouched
        self.assertEqual(truncated_p1.concepts[0], 1)
        self.assertEqual(truncated_p1.concepts[1], 2)
        self.assertEqual(truncated_p1.concepts[2], 3)

        # Check patient2 with different settings
        mask2 = _get_non_priority_mask(self.patient2, non_priority)
        # Suppose background=2, trunc_len=3 => target=5
        # We have 7 tokens total, want to reduce to 5
        truncated_p2 = _drop_non_priority_tokens(
            patient=self.patient2,
            non_priority_mask=mask2,
            truncation_length=3,
            background_length=2,
        )
        self.assertEqual(len(truncated_p2.concepts), 5)
        # Check that indices < background=2 are always kept:
        # Original concepts: [1,6,2,3,6,7,4]
        # The first 2 = [1,6] => must remain
        self.assertEqual(truncated_p2.concepts[0], 1)
        self.assertEqual(truncated_p2.concepts[1], 6)

    # -----------------------------------------------------------------
    # Test _filter_invalid_positions
    # -----------------------------------------------------------------
    def test_filter_invalid_positions(self):
        # Let's say we pass non_priority= {6} only, so the mask focuses on 6
        non_priority = {6}
        mask1 = _get_non_priority_mask(self.patient1, non_priority)
        # Our "unit" is low_priority_prefixes=["LAB_X"], which we treat as length=1
        # So if a position has 6, it must have exactly 1 "6" for that position to remain.
        # patient1 abspos => [0,0,1,2,3,4,4,5,6,7]
        # concept =>       [1,2,3,4,4,6,7,3,4,6]
        # The "6" tokens are at indices 5 (pos=4) and 9 (pos=7).
        # At pos=4, we have concept 6 => that is exactly 1 => valid
        # At pos=7, we have concept 6 => that is exactly 1 => also valid
        # => So we expect no removal in this scenario
        filtered_p1 = _filter_invalid_positions(self.patient1, mask1, ["LAB_X"])
        self.assertEqual(len(filtered_p1.concepts), len(self.patient1.concepts))

        # Now let's test a scenario where one abspos has 2 "6" tokens => must remove them
        # We'll artificially tweak patient2's abspos so that 2 "6" tokens share the same pos
        p2_copy = PatientData(
            pid=self.patient2.pid,
            concepts=self.patient2.concepts[:],  # copy
            abspos=self.patient2.abspos[:],
            segments=self.patient2.segments[:],
            ages=self.patient2.ages[:],
            outcome=self.patient2.outcome,
        )
        # Force positions of concept=6 at idx=1 and idx=4 to be same
        # original: concepts=[1,6,2,3,6,7,4], abspos=[0,0,0,1,1,1,2]
        # new => let's set indices 1,4 => concept=6 => same position=100
        p2_copy.abspos[1] = 100
        p2_copy.abspos[4] = 100

        non_priority2 = {6}  # only 'LAB_X'
        mask2 = _get_non_priority_mask(p2_copy, non_priority2)
        filtered_p2 = _filter_invalid_positions(p2_copy, mask2, ["LAB_X"])
        # Because we have 2 "6" tokens at pos=100, that means we have 2 occurrences => not exactly 1 => remove them
        # So indices 1 and 4 get removed
        self.assertEqual(len(filtered_p2.concepts), 5)


if __name__ == "__main__":
    unittest.main()
