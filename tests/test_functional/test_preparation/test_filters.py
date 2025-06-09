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
        # Background token at DOB, then 2 more events
        p1 = PatientData(
            pid=1,
            concepts=[1, 2, 3],  # background, event1, event2
            abspos=[0.0, 100.0, 200.0],  # DOB, then events 100 and 200 hours later
            segments=[0, 1, 1],
            ages=[
                0.0,
                100.0 / (365.25 * 24),
                200.0 / (365.25 * 24),
            ],  # ages in years from DOB
        )
        # Background token only, then 1 event
        p2 = PatientData(
            pid=2,
            concepts=[1, 2],  # background, event1
            abspos=[0.0, 50.0],  # DOB, then event 50 hours later
            segments=[0, 0],
            ages=[0.0, 50.0 / (365.25 * 24)],
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
        p1 = PatientData(
            1,
            [1, 2],  # background + 1 event
            [0.0, 100.0],  # DOB + event 100 hours later
            [0, 1],
            [0.0, 100.0 / (365.25 * 24)],  # ages from DOB
        )
        patients = [p1]
        result = exclude_short_sequences(patients, min_len=2)
        # p1 has concept length == min_len, so it should be included
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pid, 1)


class TestCensorPatient(unittest.TestCase):
    def test_censor_patient_basic(self):
        # Setup a single patient - DOB at 0, events at 10, 30 hours later
        p1 = PatientData(
            pid=1,
            concepts=[101, 102, 103],  # background, event1, event2
            abspos=[0.0, 10.0, 30.0],  # DOB, then events at 10h and 30h
            segments=[0, 0, 1],
            ages=[0.0, 10.0 / (365.25 * 24), 30.0 / (365.25 * 24)],  # ages in years
        )
        censor_dates = pd.Series({1: 10.0})  # censor at 10 hours

        censored = censor_patient(p1, censor_dates, predict_token_id=999)
        self.assertEqual(
            len(censored.concepts), 3
        )  # background + 1 event + predict token
        # We expect only abspos <= censor_date, so indexes 0 and 1 remain, plus predict token
        self.assertListEqual(censored.concepts, [101, 102, 999])
        self.assertListEqual(censored.abspos, [0.0, 10.0, 10.0])
        self.assertListEqual(censored.segments, [0, 0, 0])
        # Ages: 0 at DOB, calculated at event, calculated at censor
        expected_censor_age = 10.0 / (365.25 * 24)
        self.assertAlmostEqual(censored.ages[2], expected_censor_age, places=6)

    def test_censor_patient_all_included(self):
        # If censor_date is large, everything is included
        p1 = PatientData(
            1,
            [101],
            [0.0],  # DOB at hour 0
            [0],
            [0.0],  # age 0 at DOB
        )
        censor_dates = pd.Series({1: 8760.0})  # censor at 8760 hours (1 year later)
        censored = censor_patient(p1, censor_dates, predict_token_id=102)
        self.assertEqual(len(censored.concepts), 2)
        self.assertListEqual(censored.concepts, [101, 102])
        self.assertListEqual(censored.abspos, [0.0, 8760.0])
        self.assertListEqual(censored.segments, [0, 0])
        # Age at censor should be 1 year
        self.assertAlmostEqual(censored.ages[1], 1.0, places=2)


class TestCensorPatientWithDelays(unittest.TestCase):
    def test_two_delay_groups_with_unmapped(self):
        # Setup patient with mixed concept types including unmapped concepts
        p1 = PatientData(
            pid=1,
            concepts=[
                100,
                101,
                202,
                999,
                201,
                998,
            ],  # background, events (999, 998 unmapped)
            abspos=[
                0.0,
                10.0,
                20.0,
                25.0,
                40.0,
                50.0,
            ],  # events at 10h, 20h, 25h, 40h, 50h
            segments=[0, 0, 0, 1, 1, 1],
            ages=[
                0.0,
                10.0 / (365.25 * 24),
                20.0 / (365.25 * 24),
                25.0 / (365.25 * 24),
                40.0 / (365.25 * 24),
                50.0 / (365.25 * 24),
            ],
        )

        # Setup delays: only specify delays for groups 1 and 2
        concept_delays = {
            101: 10,  # +10 hour delay
            201: 20,  # +20 hour delay
            202: 20,  # +20 hour delay
        }

        censor_dates = pd.Series({1: 20.0})  # base censor at 20 hours

        censored = censor_patient_with_delays(
            p1, censor_dates, predict_token_id=103, concept_id_to_delay=concept_delays
        )

        # Expected: background(0h), 101(10h with +10 delay, effective censor 30h), 202(20h with +20 delay, effective censor 40h), 201(40h with +20 delay, effective censor 40h), predict_token
        self.assertListEqual(censored.concepts, [100, 101, 202, 201, 103])
        self.assertListEqual(censored.abspos, [0.0, 10.0, 20.0, 40.0, 20.0])
        self.assertListEqual(censored.segments, [0, 0, 0, 1, 1])
        # Check ages
        self.assertAlmostEqual(censored.ages[0], 0.0)
        self.assertAlmostEqual(censored.ages[1], 10.0 / (365.25 * 24), places=6)
        self.assertAlmostEqual(censored.ages[2], 20.0 / (365.25 * 24), places=6)
        self.assertAlmostEqual(censored.ages[3], 40.0 / (365.25 * 24), places=6)
        self.assertAlmostEqual(
            censored.ages[4], 20.0 / (365.25 * 24), places=6
        )  # predict token age at censor

    def test_all_unmapped(self):
        # Test case where no concepts have specified delays
        p1 = PatientData(
            pid=1,
            concepts=[100, 101, 102, 103],  # background + 3 events
            abspos=[0.0, 10.0, 20.0, 30.0],  # events at 10h, 20h, 30h
            segments=[0, 0, 0, 1],
            ages=[
                0.0,
                10.0 / (365.25 * 24),
                20.0 / (365.25 * 24),
                30.0 / (365.25 * 24),
            ],
        )

        concept_delays = {}  # empty delay mapping
        censor_dates = pd.Series({1: 20.0})  # censor at 20 hours

        censored = censor_patient_with_delays(
            p1, censor_dates, predict_token_id=999, concept_id_to_delay=concept_delays
        )

        # Should behave like standard censoring - only events <= censor_date
        self.assertListEqual(censored.concepts, [100, 101, 102, 999])
        self.assertListEqual(censored.abspos, [0.0, 10.0, 20.0, 20.0])
        self.assertListEqual(censored.segments, [0, 0, 0, 0])
        self.assertAlmostEqual(censored.ages[0], 0.0)
        self.assertAlmostEqual(censored.ages[1], 10.0 / (365.25 * 24), places=6)
        self.assertAlmostEqual(censored.ages[2], 20.0 / (365.25 * 24), places=6)
        self.assertAlmostEqual(
            censored.ages[3], 20.0 / (365.25 * 24), places=6
        )  # predict token age


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
        md_regex = r"^(?:LAB_.*|P[A-Z].*)$"
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
