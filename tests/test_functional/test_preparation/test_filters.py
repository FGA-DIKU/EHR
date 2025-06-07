import unittest

from corebehrt.modules.preparation.dataset import PatientData
from corebehrt.functional.preparation.filter import (
    exclude_short_sequences,
    censor_patient,
    censor_patient_with_delays,
    filter_rows_by_regex,
)
import pandas as pd
from datetime import datetime


class TestExcludeShortSequences(unittest.TestCase):
    def test_exclude_short_sequences_basic(self):
        # Create some dummy PatientData objects
        p1 = PatientData(
            pid=1,
            concepts=[1, 2, 3],
            abspos=[10.0, 11.0, 12.0],
            segments=[0, 1, 1],
            ages=[40.0, 41.0, 42.0],
        )
        p2 = PatientData(
            pid=2,
            concepts=[1, 2],
            abspos=[5.0, 6.0],
            segments=[0, 0],
            ages=[30.0, 31.0],
        )
        patients = [p1, p2]

        # Only keep patients with concepts >= 3
        min_len = 3
        result = exclude_short_sequences(patients, min_len)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pid, 1)

    def test_exclude_short_sequences_empty(self):
        # No patients at all
        patients = []
        result = exclude_short_sequences(patients, min_len=2)
        self.assertEqual(len(result), 0)

    def test_exclude_short_sequences_edge_case(self):
        # Exactly matching the min_len
        p1 = PatientData(1, [1, 2], [10, 11], [0, 1], [40, 41])
        patients = [p1]
        result = exclude_short_sequences(patients, min_len=2)
        # p1 has concept length == min_len, so it should be included
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pid, 1)


class TestCensorPatient(unittest.TestCase):
    def test_censor_patient_basic(self):
        # Setup a single patient
        p1 = PatientData(
            pid=1,
            concepts=[101, 102, 103],
            abspos=[1.0, 2.5, 4.0],
            segments=[0, 0, 1],
            ages=[30.0, 31.2, 32.1],
            dob=2.5,  # 0 at censoring
        )
        censor_dates = {1: 2.5}

        censored = censor_patient(p1, censor_dates, predict_token_id=103)
        self.assertEqual(len(censored.concepts), 3)
        # We expect only abspos <= 2.5, so indexes 0 and 1 remain
        self.assertListEqual(censored.concepts, [101, 102, 103])
        self.assertListEqual(censored.abspos, [1.0, 2.5, 2.5])
        self.assertListEqual(censored.segments, [0, 0, 1])
        self.assertListEqual(censored.ages, [30.0, 31.2, 0])

    def test_censor_patient_no_events(self):
        # If everything is after the censor date, we get an empty list
        p1 = PatientData(
            pid=1,
            concepts=[101, 102],
            abspos=[5.0, 6.0],
            segments=[0, 1],
            ages=[30.0, 31.0],
            dob=2.0,  # 0 at censoring
        )
        censor_dates = {1: 2.0}

        censored = censor_patient(p1, censor_dates, predict_token_id=101)
        self.assertEqual(len(censored.concepts), 1)
        self.assertEqual(len(censored.abspos), 1)
        self.assertEqual(len(censored.segments), 1)
        self.assertEqual(len(censored.ages), 1)
        self.assertListEqual(censored.concepts, [101])
        self.assertListEqual(censored.abspos, [2.0])
        self.assertListEqual(censored.segments, [1])
        self.assertListEqual(censored.ages, [0])

    def test_censor_patient_all_included(self):
        # If censor_date is large, everything is included
        """
        Tests that all patient events are retained when the censor date is set beyond all event times.
        """
        p1 = PatientData(1, [101], [10.0], [1], [50.0], dob=999.0)
        censor_dates = {1: 999.0}
        censored = censor_patient(p1, censor_dates, predict_token_id=102)
        self.assertEqual(len(censored.concepts), 2)
        self.assertListEqual(censored.concepts, [101, 102])
        self.assertListEqual(censored.abspos, [10.0, 999.0])
        self.assertListEqual(censored.segments, [1, 1])
        self.assertListEqual(censored.ages, [50.0, 0])


class TestCensorPatientWithDelays(unittest.TestCase):
    def test_two_delay_groups_with_unmapped(self):
        # Setup patient with mixed concept types including unmapped concepts
        """
        Tests that censor_patient_with_delays correctly censors patient data with multiple delay groups and unmapped concepts.

        Verifies that only concepts with specified delays and occurring before their adjusted censor dates are retained, while unmapped concepts are excluded.
        """
        p1 = PatientData(
            pid=1,
            concepts=[101, 202, 999, 201, 998],  # 999, 998 are unmapped
            abspos=[1.0, 2.0, 2.5, 4.0, 5.0],
            segments=[0, 0, 1, 1, 1],
            ages=[30.0, 31.0, 32.0, 33.0, 34.0],
            dob=2.0,
        )

        # Setup delays: only specify delays for groups 1 and 2
        concept_delays = {
            101: 1,  # group 1
            201: 2,
            202: 2,  # group 2
        }

        censor_dates = {1: 2.0}  # base censor date

        censored = censor_patient_with_delays(
            p1, censor_dates, predict_token_id=103, concept_id_to_delay=concept_delays
        )

        self.assertListEqual(censored.concepts, [101, 202, 201, 103])
        self.assertListEqual(censored.abspos, [1.0, 2.0, 4.0, 2.0])
        self.assertListEqual(censored.segments, [0, 0, 1, 1])
        self.assertListEqual(censored.ages, [30.0, 31.0, 33.0, 0])

    def test_all_unmapped(self):
        # Test case where no concepts have specified delays
        """
        Tests that censor_patient_with_delays behaves like standard censoring when no concepts have delay mappings.

        Verifies that only events occurring at or before the censor date are retained when the concept delay mapping is empty.
        """
        p1 = PatientData(
            pid=1,
            concepts=[101, 102, 103],
            abspos=[1.0, 2.0, 3.0],
            segments=[0, 0, 1],
            ages=[30.0, 31.0, 32.0],
            dob=-24 * 365.25 * 2 + 2,  # should be exactly two years at censoring
        )

        concept_delays = {}  # empty delay mapping
        censor_dates = {1: 2.0}

        censored = censor_patient_with_delays(
            p1, censor_dates, predict_token_id=101, concept_id_to_delay=concept_delays
        )

        # Should behave like standard censoring
        self.assertListEqual(censored.concepts, [101, 102, 101])
        self.assertListEqual(censored.abspos, [1.0, 2.0, 2.0])
        self.assertListEqual(censored.segments, [0, 0, 1])
        self.assertAlmostEqual(censored.ages[0], 30.0)
        self.assertAlmostEqual(censored.ages[1], 31.0)
        self.assertAlmostEqual(censored.ages[2], 2.0)


class TestRegexFilter(unittest.TestCase):
    def setUp(self):
        """
        Initializes a sample DataFrame for use in test cases.

        Creates a DataFrame with subject IDs, codes, and timestamps to be used as test data in the test methods.
        """
        self.df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 3, 3, 3],
                "code": [
                    "DOB",
                    "DC521",
                    "MN001",
                    "PZZ3912",
                    "MA01",
                    "LAB_KOLESTEROL LDL;P",
                ],
                "time": [
                    datetime(1995, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                    datetime(2015, 7, 2),
                    datetime(2016, 8, 2),
                    datetime(2016, 8, 2),
                ],
            }
        )

    def test_positive_filter(self):
        md_regex = r"^(LAB_.*|P[A-Z].*)$"
        expected_md = ["DOB", "DC521", "MN001", "MA01"]
        md_df = filter_rows_by_regex(self.df, "code", md_regex)
        self.assertListEqual(md_df["code"].tolist(), expected_md)

    def test_negative_filter(self):
        non_md_regex = r"^(?!LAB_.*|P[A-Z].*).*$"
        expected_non_md = ["PZZ3912", "LAB_KOLESTEROL LDL;P"]
        non_md_df = filter_rows_by_regex(self.df, "code", non_md_regex)
        self.assertListEqual(non_md_df["code"].tolist(), expected_non_md)

    def test_empty_filter(self):
        all_regex = ".*"
        empty_df = filter_rows_by_regex(self.df, "code", all_regex)
        self.assertEqual(empty_df.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
