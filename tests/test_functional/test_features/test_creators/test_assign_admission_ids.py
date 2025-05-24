import unittest
import uuid
from datetime import datetime, timedelta

import pandas as pd
from pandas import NaT

from corebehrt.functional.features.creators import (
    _assign_admission_ids,
    _assign_explicit_admission_ids,
    _assign_time_based_admission_ids,
)


class TestAssignAdmissionIds(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.base_time = datetime(2020, 1, 1, 10, 0, 0)

    def test_time_based_admission_simple(self):
        """Test basic time-based admission assignment"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),  # Same admission
                    self.base_time + timedelta(hours=50),  # New admission (>48h)
                    self.base_time + timedelta(hours=51),  # Same admission
                ],
                "code": ["A", "B", "C", "D"],
            }
        )

        result = _assign_admission_ids(concepts)

        # Should have 2 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 2)

        # First two events should have same admission ID
        self.assertEqual(result.iloc[0]["admission_id"], result.iloc[1]["admission_id"])

        # Last two events should have same admission ID (different from first two)
        self.assertEqual(result.iloc[2]["admission_id"], result.iloc[3]["admission_id"])
        self.assertNotEqual(
            result.iloc[0]["admission_id"], result.iloc[2]["admission_id"]
        )

    def test_explicit_admission_discharge(self):
        """Test explicit admission/discharge event handling"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=3),
                    self.base_time + timedelta(hours=50),
                    self.base_time + timedelta(hours=51),
                ],
                "code": ["A", "ADM_ADMISSION", "B", "ADM_DISCHARGE", "C", "D"],
            }
        )

        result = _assign_admission_ids(concepts)

        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 3)

        # Events during admission should have same ID
        admission_id = result.iloc[1]["admission_id"]  # ADM_ADMISSION event
        self.assertEqual(result.iloc[2]["admission_id"], admission_id)  # Event B
        self.assertEqual(result.iloc[3]["admission_id"], admission_id)  # ADM_DISCHARGE

        # Events outside admission should have different IDs
        self.assertNotEqual(result.iloc[0]["admission_id"], admission_id)  # Event A
        self.assertNotEqual(result.iloc[4]["admission_id"], admission_id)  # Event C
        self.assertNotEqual(result.iloc[5]["admission_id"], admission_id)  # Event D

    def test_multiple_patients(self):
        """Test handling multiple patients"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "2", "2", "2"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time,
                    self.base_time + timedelta(hours=50),
                    self.base_time + timedelta(hours=51),
                ],
                "code": ["A", "B", "C", "D", "E"],
            }
        )

        result = _assign_admission_ids(concepts)

        # Patient 1 should have 1 admission ID
        patient_1_admissions = result[result["subject_id"] == "1"][
            "admission_id"
        ].unique()
        self.assertEqual(len(patient_1_admissions), 1)

        # Patient 2 should have 2 admission IDs
        patient_2_admissions = result[result["subject_id"] == "2"][
            "admission_id"
        ].unique()
        self.assertEqual(len(patient_2_admissions), 2)

        # Admission IDs should be different between patients
        all_patient_1_ids = set(result[result["subject_id"] == "1"]["admission_id"])
        all_patient_2_ids = set(result[result["subject_id"] == "2"]["admission_id"])
        self.assertTrue(all_patient_1_ids.isdisjoint(all_patient_2_ids))

    def test_duplicate_indices(self):
        """Test handling of DataFrame with duplicate indices"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=50),
                    self.base_time + timedelta(hours=51),
                ],
                "code": ["A", "B", "C", "D"],
            }
        )
        # Create duplicate indices
        concepts.index = [0, 0, 1, 1]

        result = _assign_admission_ids(concepts)

        # Should still work correctly
        self.assertEqual(len(result), 4)
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 2)

        # Verify admission_id column was added
        self.assertIn("admission_id", result.columns)
        self.assertTrue(result["admission_id"].notna().all())

    def test_non_contiguous_indices(self):
        """Test handling of DataFrame with non-contiguous indices"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=50),
                    self.base_time + timedelta(hours=51),
                ],
                "code": ["A", "B", "C", "D"],
            }
        )
        # Create non-contiguous indices
        concepts.index = [0, 5, 10, 15]

        result = _assign_admission_ids(concepts)

        # Should work correctly
        self.assertEqual(len(result), 4)
        self.assertTrue(result["admission_id"].notna().all())

        # Check original indices are preserved
        self.assertEqual(list(result.index), [0, 5, 10, 15])

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        concepts = pd.DataFrame(columns=["subject_id", "time", "code"])
        result = _assign_admission_ids(concepts)

        self.assertEqual(len(result), 0)
        self.assertIn("admission_id", result.columns)

    def test_single_event(self):
        """Test handling of single event"""
        concepts = pd.DataFrame(
            {"subject_id": ["1"], "time": [self.base_time], "code": ["A"]}
        )

        result = _assign_admission_ids(concepts)

        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result.iloc[0]["admission_id"])

    def test_overlapping_admissions_warning(self):
        """Test behavior with overlapping admissions (should handle gracefully)"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=3),
                    self.base_time + timedelta(hours=4),
                ],
                "code": ["ADM_ADMISSION", "A", "ADM_ADMISSION", "B", "ADM_DISCHARGE"],
            }
        )

        result = _assign_admission_ids(concepts)

        # Should handle gracefully without crashing
        self.assertEqual(len(result), 5)
        self.assertTrue(result["admission_id"].notna().all())

    def test_discharge_without_admission(self):
        """Test discharge event without preceding admission"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["A", "ADM_DISCHARGE", "B"],
            }
        )

        result = _assign_admission_ids(concepts)

        # Should handle gracefully
        self.assertEqual(len(result), 3)
        self.assertTrue(result["admission_id"].notna().all())

    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1"],
                "time": [self.base_time, self.base_time + timedelta(hours=1)],
                "code": ["A", "B"],
            }
        )
        original_concepts = concepts.copy()

        result = _assign_admission_ids(concepts)

        # Original should be unchanged
        pd.testing.assert_frame_equal(concepts, original_concepts)

        # Result should have admission_id column
        self.assertIn("admission_id", result.columns)
        self.assertNotIn("admission_id", concepts.columns)

    def test_nat_timestamps(self):
        """Test handling of NaT (Not a Time) timestamps"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [self.base_time, NaT, self.base_time + timedelta(hours=1)],
                "code": ["A", "B", "C"],
            }
        )

        result = _assign_admission_ids(concepts)

        # Should handle gracefully
        self.assertEqual(len(result), 3)
        # All rows should have admission_id (even those with NaT)
        self.assertTrue(result["admission_id"].notna().all())

    def test_unsorted_timestamps(self):
        """Test that function handles unsorted timestamps correctly"""
        concepts = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time + timedelta(hours=2),  # Out of order
                    self.base_time,
                    self.base_time + timedelta(hours=50),
                    self.base_time + timedelta(hours=1),
                ],
                "code": ["A", "B", "C", "D"],
            }
        )

        result = _assign_admission_ids(concepts)

        # Should sort by timestamp internally and assign correctly
        self.assertEqual(len(result), 4)
        self.assertTrue(result["admission_id"].notna().all())


class TestAssignExplicitAdmissionIds(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.base_time = datetime(2020, 1, 1, 10, 0, 0)

        # Import the function we need to test directly

        def _get_adm_id():
            return str(uuid.uuid4())

        self._get_adm_id = _get_adm_id

    def test_basic_admission_discharge_flow(self):
        """Test basic admission and discharge flow"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=3),
                ],
                "code": ["ADM_ADMISSION", "A", "B", "ADM_DISCHARGE"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # All events should have the same admission ID
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

        # All admission IDs should be the same
        admission_id = result.iloc[0]["admission_id"]
        for _, row in result.iterrows():
            self.assertEqual(row["admission_id"], admission_id)

    def test_multiple_admissions_same_patient(self):
        """Test multiple separate admissions for the same patient"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=50),  # Outside admission, >48h gap
                    self.base_time + timedelta(hours=52),
                    self.base_time + timedelta(hours=53),
                    self.base_time + timedelta(hours=54),
                ],
                "code": [
                    "ADM_ADMISSION",
                    "A",
                    "ADM_DISCHARGE",
                    "B",
                    "ADM_ADMISSION",
                    "C",
                    "ADM_DISCHARGE",
                ],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should have 3 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 3)

        # First admission (indices 0, 1, 2)
        first_admission_id = result.iloc[0]["admission_id"]
        self.assertEqual(result.iloc[1]["admission_id"], first_admission_id)
        self.assertEqual(result.iloc[2]["admission_id"], first_admission_id)

        # Outside event (index 3) should have different ID
        outside_id = result.iloc[3]["admission_id"]
        self.assertNotEqual(outside_id, first_admission_id)

        # Second admission (indices 4, 5, 6)
        second_admission_id = result.iloc[4]["admission_id"]
        self.assertEqual(result.iloc[5]["admission_id"], second_admission_id)
        self.assertEqual(result.iloc[6]["admission_id"], second_admission_id)
        self.assertNotEqual(second_admission_id, first_admission_id)
        self.assertNotEqual(second_admission_id, outside_id)

    def test_discharge_without_prior_admission(self):
        """Test discharge event without preceding admission"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["A", "ADM_DISCHARGE", "B"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should have 3 unique admission IDs (each event gets its own)
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 3)

        # All admission IDs should be different
        ids = [result.iloc[i]["admission_id"] for i in range(3)]
        self.assertEqual(len(set(ids)), 3)

    def test_events_inside_active_admission(self):
        """Test that events during admission are properly grouped"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=3),
                    self.base_time + timedelta(hours=4),
                    self.base_time + timedelta(hours=5),
                ],
                "code": ["ADM_ADMISSION", "A", "B", "C", "D", "ADM_DISCHARGE"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # All events should have the same admission ID
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

        admission_id = result.iloc[0]["admission_id"]
        for _, row in result.iterrows():
            self.assertEqual(row["admission_id"], admission_id)

    def test_events_outside_admission_with_48_hour_grouping(self):
        """Test 48-hour grouping for events outside admission periods"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),  # Same group as first
                    self.base_time + timedelta(hours=50),  # New group (>48h gap)
                    self.base_time + timedelta(hours=51),  # Same group as third
                    self.base_time + timedelta(hours=100),  # New group (>48h gap)
                ],
                "code": ["A", "B", "C", "D", "E"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should have 3 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 3)

        # First two events should have same ID
        self.assertEqual(result.iloc[0]["admission_id"], result.iloc[1]["admission_id"])

        # Events 3 and 4 should have same ID (different from first group)
        self.assertEqual(result.iloc[2]["admission_id"], result.iloc[3]["admission_id"])
        self.assertNotEqual(
            result.iloc[0]["admission_id"], result.iloc[2]["admission_id"]
        )

        # Event 5 should have unique ID
        self.assertNotEqual(
            result.iloc[4]["admission_id"], result.iloc[0]["admission_id"]
        )
        self.assertNotEqual(
            result.iloc[4]["admission_id"], result.iloc[2]["admission_id"]
        )

    def test_nat_timestamps(self):
        """Test handling of NaT (Not a Time) timestamps"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    NaT,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["ADM_ADMISSION", "A", "B", "ADM_DISCHARGE"],
            }
        ).sort_values(by="time", na_position="first")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should handle gracefully without crashing
        self.assertEqual(len(result), 4)
        self.assertTrue(result["admission_id"].notna().all())

    def test_missing_codes(self):
        """Test handling of missing/NaN codes"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=3),
                ],
                "code": ["ADM_ADMISSION", None, "B", "ADM_DISCHARGE"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should handle gracefully, treating None as regular event
        self.assertEqual(len(result), 4)
        self.assertTrue(result["admission_id"].notna().all())

        # All events should have same admission ID (inside admission period)
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

    def test_overlapping_admissions(self):
        """Test behavior with overlapping admission events"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time
                    + timedelta(hours=2),  # Second admission before first discharge
                    self.base_time + timedelta(hours=3),
                    self.base_time + timedelta(hours=4),
                ],
                "code": ["ADM_ADMISSION", "A", "ADM_ADMISSION", "B", "ADM_DISCHARGE"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should handle gracefully without crashing
        self.assertEqual(len(result), 5)
        self.assertTrue(result["admission_id"].notna().all())

        # Second admission should start new admission period
        first_admission_id = result.iloc[0]["admission_id"]
        second_admission_id = result.iloc[2]["admission_id"]
        self.assertNotEqual(first_admission_id, second_admission_id)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        patient_data = pd.DataFrame(columns=["subject_id", "time", "code"])

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        self.assertEqual(len(result), 0)
        self.assertIn("admission_id", result.columns)
        self.assertEqual(result["admission_id"].dtype, object)

    def test_single_event(self):
        """Test handling of single event"""
        patient_data = pd.DataFrame(
            {"subject_id": ["1"], "time": [self.base_time], "code": ["A"]}
        )

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result.iloc[0]["admission_id"])

    def test_only_admission_events(self):
        """Test with only admission events (no discharge)"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["ADM_ADMISSION", "A", "B"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # All events should have same admission ID
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

    def test_only_discharge_events(self):
        """Test with only discharge events (no admission)"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["A", "ADM_DISCHARGE", "B"],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should have 3 different admission IDs (discharge without admission + 48h rule)
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 3)

    def test_consecutive_admissions_no_gap(self):
        """Test consecutive admissions with no time gap"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                    self.base_time + timedelta(hours=2),  # Same time as discharge
                    self.base_time + timedelta(hours=3),
                    self.base_time + timedelta(hours=4),
                ],
                "code": [
                    "ADM_ADMISSION",
                    "A",
                    "ADM_DISCHARGE",
                    "ADM_ADMISSION",
                    "B",
                    "ADM_DISCHARGE",
                ],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should have 2 different admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 2)

        # First admission group
        first_admission_id = result.iloc[0]["admission_id"]
        self.assertEqual(result.iloc[1]["admission_id"], first_admission_id)
        self.assertEqual(result.iloc[2]["admission_id"], first_admission_id)

        # Second admission group
        second_admission_id = result.iloc[3]["admission_id"]
        self.assertEqual(result.iloc[4]["admission_id"], second_admission_id)
        self.assertEqual(result.iloc[5]["admission_id"], second_admission_id)
        self.assertNotEqual(first_admission_id, second_admission_id)

    def test_admission_discharge_with_outside_events(self):
        """Test mix of admission periods and outside events"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,  # Outside event 0
                    self.base_time + timedelta(hours=2),  # Outside event (same group) 0
                    self.base_time
                    + timedelta(
                        hours=50, minutes=1
                    ),  # Outside event (new group, >48h) 1
                    self.base_time + timedelta(hours=52),  # Admission start 2
                    self.base_time + timedelta(hours=53),  # Inside admission 2
                    self.base_time + timedelta(hours=54),  # Discharge 2
                    self.base_time
                    + timedelta(hours=100),  # Outside event (new group, >48h) 3
                    self.base_time
                    + timedelta(hours=101),  # Outside event (same group) 3
                ],
                "code": [
                    "A",
                    "B",
                    "C",
                    "ADM_ADMISSION",
                    "D",
                    "ADM_DISCHARGE",
                    "E",
                    "F",
                ],
            }
        ).sort_values(by="time")

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should have 4 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 4)

        # First outside group (events 0, 1)
        first_outside_id = result.iloc[0]["admission_id"]
        self.assertEqual(result.iloc[1]["admission_id"], first_outside_id)

        # Second outside group (event 2)
        second_outside_id = result.iloc[2]["admission_id"]
        self.assertNotEqual(second_outside_id, first_outside_id)

        # Admission group (events 3, 4, 5)
        admission_id = result.iloc[3]["admission_id"]
        self.assertEqual(result.iloc[4]["admission_id"], admission_id)
        self.assertEqual(result.iloc[5]["admission_id"], admission_id)
        self.assertNotEqual(admission_id, first_outside_id)
        self.assertNotEqual(admission_id, second_outside_id)

        # Third outside group (events 6, 7)
        third_outside_id = result.iloc[6]["admission_id"]
        self.assertEqual(result.iloc[7]["admission_id"], third_outside_id)
        self.assertNotEqual(third_outside_id, first_outside_id)
        self.assertNotEqual(third_outside_id, second_outside_id)
        self.assertNotEqual(third_outside_id, admission_id)

    def test_preserves_dataframe_structure(self):
        """Test that the function preserves DataFrame structure and index"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["ADM_ADMISSION", "A", "ADM_DISCHARGE"],
                "extra_column": ["X", "Y", "Z"],
            }
        )
        # Set custom index
        patient_data.index = [10, 20, 30]

        result = _assign_explicit_admission_ids(patient_data, self._get_adm_id)

        # Should preserve original structure
        self.assertEqual(list(result.index), [10, 20, 30])
        self.assertIn("extra_column", result.columns)
        self.assertEqual(list(result["extra_column"]), ["X", "Y", "Z"])
        self.assertIn("admission_id", result.columns)


class TestAssignTimeBasedAdmissionIds(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.base_time = datetime(2020, 1, 1, 10, 0, 0)

        def _get_adm_id():
            return str(uuid.uuid4())

        self._get_adm_id = _get_adm_id

    def test_basic_time_based_grouping(self):
        """Test basic time-based admission grouping with 48-hour rule"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),  # Same group (< 48h)
                    self.base_time + timedelta(hours=50),  # New group (> 48h)
                    self.base_time + timedelta(hours=51),  # Same as previous (< 48h)
                ],
                "code": ["A", "B", "C", "D"],
            }
        ).sort_values(by="time")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should have 2 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 2)

        # First two events should have same admission ID
        self.assertEqual(result.iloc[0]["admission_id"], result.iloc[1]["admission_id"])

        # Last two events should have same admission ID (different from first group)
        self.assertEqual(result.iloc[2]["admission_id"], result.iloc[3]["admission_id"])
        self.assertNotEqual(
            result.iloc[0]["admission_id"], result.iloc[2]["admission_id"]
        )

    def test_single_event(self):
        """Test handling of single event"""
        patient_data = pd.DataFrame(
            {"subject_id": ["1"], "time": [self.base_time], "code": ["A"]}
        )

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result.iloc[0]["admission_id"])
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

    def test_all_events_within_48_hours(self):
        """Test multiple events all within 48-hour window"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=12),
                    self.base_time + timedelta(hours=24),
                    self.base_time + timedelta(hours=36),
                    self.base_time + timedelta(hours=47),  # Just under 48h from first
                ],
                "code": ["A", "B", "C", "D", "E"],
            }
        ).sort_values(by="time")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # All events should have same admission ID
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

        admission_id = result.iloc[0]["admission_id"]
        for _, row in result.iterrows():
            self.assertEqual(row["admission_id"], admission_id)

    def test_multiple_groups_with_large_gaps(self):
        """Test multiple groups separated by > 48 hour gaps"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,  # Group 1
                    self.base_time + timedelta(hours=1),  # Group 1
                    self.base_time + timedelta(hours=50),  # Group 2 (> 48h gap)
                    self.base_time + timedelta(hours=51),  # Group 2
                    self.base_time + timedelta(hours=100),  # Group 3 (> 48h gap)
                    self.base_time + timedelta(hours=101),  # Group 3
                ],
                "code": ["A", "B", "C", "D", "E", "F"],
            }
        ).sort_values(by="time")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should have 3 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 3)

        # Group 1: events 0, 1
        group1_id = result.iloc[0]["admission_id"]
        self.assertEqual(result.iloc[1]["admission_id"], group1_id)

        # Group 2: events 2, 3
        group2_id = result.iloc[2]["admission_id"]
        self.assertEqual(result.iloc[3]["admission_id"], group2_id)
        self.assertNotEqual(group1_id, group2_id)

        # Group 3: events 4, 5
        group3_id = result.iloc[4]["admission_id"]
        self.assertEqual(result.iloc[5]["admission_id"], group3_id)
        self.assertNotEqual(group1_id, group3_id)
        self.assertNotEqual(group2_id, group3_id)

    def test_exactly_48_hours_gap(self):
        """Test edge case: exactly 48 hours gap (should be same group)"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=48),  # Exactly 48h
                    self.base_time + timedelta(hours=49),  # 1h after the 48h event
                ],
                "code": ["A", "B", "C"],
            }
        ).sort_values(by="time")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should have 1 unique admission ID (48h exactly is not > 48h)
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 1)

        # All events should have same admission ID
        admission_id = result.iloc[0]["admission_id"]
        for _, row in result.iterrows():
            self.assertEqual(row["admission_id"], admission_id)

    def test_just_over_48_hours_gap(self):
        """Test edge case: just over 48 hours gap (should be different groups)"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=48, minutes=1),  # Just over 48h
                    self.base_time + timedelta(hours=49),
                ],
                "code": ["A", "B", "C"],
            }
        ).sort_values(by="time")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should have 2 unique admission IDs
        unique_admissions = result["admission_id"].unique()
        self.assertEqual(len(unique_admissions), 2)

        # First event in group 1, second and third in group 2
        group1_id = result.iloc[0]["admission_id"]
        group2_id = result.iloc[1]["admission_id"]
        self.assertNotEqual(group1_id, group2_id)
        self.assertEqual(result.iloc[2]["admission_id"], group2_id)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        patient_data = pd.DataFrame(columns=["subject_id", "time", "code"])

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        self.assertEqual(len(result), 0)
        self.assertIn("admission_id", result.columns)
        self.assertEqual(result["admission_id"].dtype, object)

    def test_nat_timestamps(self):
        """Test handling of NaT (Not a Time) timestamps"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    NaT,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=2),
                ],
                "code": ["A", "B", "C", "D"],
            }
        ).sort_values(by="time", na_position="first")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should handle gracefully without crashing
        self.assertEqual(len(result), 4)
        self.assertTrue(result["admission_id"].notna().all())

        # NaT should start a new group, and subsequent events should be grouped normally
        unique_admissions = result["admission_id"].unique()
        self.assertGreaterEqual(len(unique_admissions), 1)

    def test_all_nat_timestamps(self):
        """Test handling of all NaT timestamps"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [NaT, NaT, NaT],
                "code": ["A", "B", "C"],
            }
        )

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should handle gracefully
        self.assertEqual(len(result), 3)
        self.assertTrue(result["admission_id"].notna().all())

    def test_mixed_nat_and_valid_timestamps(self):
        """Test mix of NaT and valid timestamps"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    NaT,
                    self.base_time + timedelta(hours=50),
                    self.base_time + timedelta(hours=51),
                ],
                "code": ["A", "B", "C", "D", "E"],
            }
        ).sort_values(by="time", na_position="first")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should handle gracefully
        self.assertEqual(len(result), 5)
        self.assertTrue(result["admission_id"].notna().all())

    def test_preserves_dataframe_structure(self):
        """Test that the function preserves DataFrame structure and index"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=50),
                ],
                "code": ["A", "B", "C"],
                "extra_column": ["X", "Y", "Z"],
            }
        )
        # Set custom index
        patient_data.index = [10, 20, 30]

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should preserve original structure
        self.assertEqual(list(result.index), [10, 20, 30])
        self.assertIn("extra_column", result.columns)
        self.assertEqual(list(result["extra_column"]), ["X", "Y", "Z"])
        self.assertIn("admission_id", result.columns)

    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified"""
        patient_data = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    self.base_time,
                    self.base_time + timedelta(hours=1),
                    self.base_time + timedelta(hours=50),
                ],
                "code": ["A", "B", "C"],
            }
        )
        original_data = patient_data.copy()

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Original should be unchanged
        pd.testing.assert_frame_equal(patient_data, original_data)

        # Result should have admission_id column
        self.assertIn("admission_id", result.columns)
        self.assertNotIn("admission_id", patient_data.columns)

    def test_vectorized_operation_efficiency(self):
        """Test that function handles large datasets efficiently"""
        # Create a larger dataset to test vectorized operations
        times = []
        codes = []
        for i in range(1000):
            if i % 100 == 0:  # Every 100th event is > 48h gap
                times.append(self.base_time + timedelta(hours=i * 50))
            else:
                times.append(self.base_time + timedelta(hours=i))
            codes.append(f"EVENT_{i}")

        patient_data = pd.DataFrame(
            {"subject_id": ["1"] * 1000, "time": times, "code": codes}
        ).sort_values(by="time")

        result = _assign_time_based_admission_ids(patient_data, self._get_adm_id)

        # Should complete without error and have proper structure
        self.assertEqual(len(result), 1000)
        self.assertTrue(result["admission_id"].notna().all())
        self.assertIn("admission_id", result.columns)

        # Should have multiple groups due to 48h+ gaps
        unique_admissions = result["admission_id"].unique()
        self.assertGreater(len(unique_admissions), 1)


if __name__ == "__main__":
    unittest.main()
