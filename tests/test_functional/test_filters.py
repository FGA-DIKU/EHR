import unittest
import operator
import pandas as pd

from corebehrt.modules.dataset import PatientData
from corebehrt.functional.filter import (
    exclude_short_sequences,
    censor_patient,
    filter_events_by_abspos,
)


class TestExcludeShortSequences(unittest.TestCase):
    def test_exclude_short_sequences_basic(self):
        # Create some dummy PatientData objects
        p1 = PatientData(
            pid="P1",
            concepts=[1, 2, 3],
            abspos=[10.0, 11.0, 12.0],
            segments=[0, 1, 1],
            ages=[40.0, 41.0, 42.0],
        )
        p2 = PatientData(
            pid="P2",
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
        self.assertEqual(result[0].pid, "P1")

    def test_exclude_short_sequences_empty(self):
        # No patients at all
        patients = []
        result = exclude_short_sequences(patients, min_len=2)
        self.assertEqual(len(result), 0)

    def test_exclude_short_sequences_edge_case(self):
        # Exactly matching the min_len
        p1 = PatientData("P1", [1, 2], [10, 11], [0, 1], [40, 41])
        patients = [p1]
        result = exclude_short_sequences(patients, min_len=2)
        # p1 has concept length == min_len, so it should be included
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pid, "P1")


class TestCensorPatient(unittest.TestCase):
    def test_censor_patient_basic(self):
        # Setup a single patient
        p1 = PatientData(
            pid="P1",
            concepts=[101, 102, 103],
            abspos=[1.0, 2.5, 4.0],
            segments=[0, 0, 1],
            ages=[30.0, 31.2, 32.1],
        )
        censor_dates = {"P1": 2.5}

        censored = censor_patient(p1, censor_dates)
        self.assertEqual(len(censored.concepts), 2)
        # We expect only abspos <= 2.5, so indexes 0 and 1 remain
        self.assertListEqual(censored.concepts, [101, 102])
        self.assertListEqual(censored.abspos, [1.0, 2.5])
        self.assertListEqual(censored.segments, [0, 0])
        self.assertListEqual(censored.ages, [30.0, 31.2])

    def test_censor_patient_no_events(self):
        # If everything is after the censor date, we get an empty list
        p1 = PatientData(
            pid="P1",
            concepts=[101, 102],
            abspos=[5.0, 6.0],
            segments=[0, 1],
            ages=[30.0, 31.0],
        )
        censor_dates = {"P1": 2.0}

        censored = censor_patient(p1, censor_dates)
        self.assertEqual(len(censored.concepts), 0)
        self.assertEqual(len(censored.abspos), 0)
        self.assertEqual(len(censored.segments), 0)
        self.assertEqual(len(censored.ages), 0)

    def test_censor_patient_all_included(self):
        # If censor_date is large, everything is included
        p1 = PatientData("P1", [101], [10.0], [1], [50.0])
        censor_dates = {"P1": 999.0}
        censored = censor_patient(p1, censor_dates)
        self.assertEqual(len(censored.concepts), 1)


class TestFilterEventsByAbspos(unittest.TestCase):
    def test_filter_events_by_abspos_le(self):
        # Suppose we have event data for multiple patients
        events = pd.DataFrame(
            {
                "PID": ["A", "A", "B", "B"],
                "abspos": [1.0, 3.0, 5.0, 2.0],
                "some_feature": [10, 20, 30, 40],
            }
        )
        # Reference abspos for each PID
        abspos_series = pd.Series([2.0, 3.0], index=["A", "B"])

        result = filter_events_by_abspos(events, abspos_series, operator.le)
        # For A: abspos <= 2.0 => only row 1.0 remains
        # For B: abspos <= 3.0 => only row 2.0 remains
        self.assertEqual(len(result), 2)
        self.assertListEqual(sorted(result["abspos"].tolist()), [1.0, 2.0])

    def test_filter_events_by_abspos_gt(self):
        # Another scenario
        events = pd.DataFrame(
            {
                "PID": ["A", "A", "B", "C"],
                "abspos": [2.0, 4.0, 3.0, 10.0],
                "info": [101, 102, 103, 999],
            }
        )
        abspos_series = pd.Series([3.0, 1.0, 5.0], index=["A", "B", "C"])
        # We'll do operator.gt: keep events where event abspos > reference abspos
        result = filter_events_by_abspos(events, abspos_series, operator.gt)

        # Explanation:
        # - For PID=A, reference=3.0 => keep events with abspos>3 => abspos=4.0
        # - For PID=B, reference=1.0 => keep events with abspos>1 => abspos=3.0
        # - For PID=C, reference=5.0 => keep events with abspos>5 => abspos=10.0
        self.assertEqual(len(result), 3)
        self.assertListEqual(sorted(result["abspos"].tolist()), [3.0, 4.0, 10.0])

    def test_filter_events_by_abspos_empty_series(self):
        # If the abspos_series is empty, we get no merges
        events = pd.DataFrame({"PID": ["A", "B"], "abspos": [2.0, 3.0]})
        abspos_series = pd.Series(dtype=float)  # empty

        result = filter_events_by_abspos(events, abspos_series, operator.le)
        self.assertEqual(len(result), 0)

    def test_filter_events_by_abspos_unsorted(self):
        # Test with unsorted abspos values
        events = pd.DataFrame(
            {
                "PID": ["A", "A", "A", "B"],
                "abspos": [3.0, 1.0, 2.0, 5.0],  # Not in increasing order
                "info": [101, 102, 103, 104],
            }
        )
        abspos_series = pd.Series([2.0, 4.0], index=["A", "B"])

        result = filter_events_by_abspos(events, abspos_series, operator.le)

        # Should still work correctly regardless of input order
        # For A: abspos <= 2.0 => keeps events with abspos 1.0 and 2.0
        # For B: abspos <= 4.0 => no events kept
        self.assertEqual(len(result), 2)
        self.assertListEqual(sorted(result["abspos"].tolist()), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
