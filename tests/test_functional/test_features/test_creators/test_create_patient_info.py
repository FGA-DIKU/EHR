import unittest

import pandas as pd
from datetime import datetime
from corebehrt.functional.features.creators import _create_patient_info


class TestCreatePatientInfo(unittest.TestCase):
    def setUp(self):
        """Set up test data for patient info testing"""
        # Concepts after processing by create_background (with BG_ prefixes and birthdate column)
        self.processed_concepts = pd.DataFrame(
            {
                "subject_id": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "2",
                    "2",
                    "2",
                    "2",
                    "3",
                    "3",
                    "3",
                    "4",
                    "4",
                    "4",
                    "4",
                ],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                    datetime(2022, 9, 3),
                    datetime(1995, 9, 3),
                    datetime(1995, 9, 3),
                    datetime(2015, 9, 10),
                    datetime(2018, 4, 5),
                    datetime(1962, 11, 5),
                    datetime(1962, 11, 5),
                    datetime(2020, 11, 6),
                    datetime(1980, 3, 15),
                    datetime(1980, 3, 15),
                    datetime(1980, 3, 15),
                    datetime(2010, 1, 1),
                ],
                "code": [
                    "DOB",
                    "BG_GENDER//F",
                    "BG_SMOKER//Y",
                    "A",
                    "DOD",
                    "DOB",
                    "BG_GENDER//M",
                    "A",
                    "D",
                    "DOB",
                    "BG_GENDER//M",
                    "D",
                    "DOB",
                    "BG_GENDER//F",
                    "BG_DIABETES//Y",
                    "B",
                ],
                "birthdate": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(1995, 9, 3),
                    datetime(1995, 9, 3),
                    datetime(1995, 9, 3),
                    datetime(1995, 9, 3),
                    datetime(1962, 11, 5),
                    datetime(1962, 11, 5),
                    datetime(1962, 11, 5),
                    datetime(1980, 3, 15),
                    datetime(1980, 3, 15),
                    datetime(1980, 3, 15),
                    datetime(1980, 3, 15),
                ],
            }
        )

    def test_basic_patient_info_creation(self):
        """Test basic patient info creation functionality"""
        patient_info = _create_patient_info(self.processed_concepts)

        # Check structure
        self.assertEqual(len(patient_info), 4)  # 4 unique patients
        self.assertIn("subject_id", patient_info.columns)
        self.assertIn("birthdate", patient_info.columns)
        self.assertIn("deathdate", patient_info.columns)

        # Check that we have the expected patients
        expected_patients = ["1", "2", "3", "4"]
        actual_patients = sorted(patient_info["subject_id"].tolist())
        self.assertEqual(actual_patients, expected_patients)

    def test_birthdate_extraction(self):
        """Test that birthdates are correctly extracted"""
        patient_info = _create_patient_info(self.processed_concepts)

        # Check specific birthdates
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        patient_2 = patient_info[patient_info["subject_id"] == "2"].iloc[0]
        patient_3 = patient_info[patient_info["subject_id"] == "3"].iloc[0]
        patient_4 = patient_info[patient_info["subject_id"] == "4"].iloc[0]

        self.assertEqual(patient_1["birthdate"], datetime(2000, 5, 1))
        self.assertEqual(patient_2["birthdate"], datetime(1995, 9, 3))
        self.assertEqual(patient_3["birthdate"], datetime(1962, 11, 5))
        self.assertEqual(patient_4["birthdate"], datetime(1980, 3, 15))

    def test_deathdate_extraction(self):
        """Test that death dates are correctly extracted"""
        patient_info = _create_patient_info(self.processed_concepts)

        # Patient 1 has DOD, others don't
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        patient_2 = patient_info[patient_info["subject_id"] == "2"].iloc[0]

        self.assertEqual(patient_1["deathdate"], datetime(2022, 9, 3))
        self.assertTrue(pd.isna(patient_2["deathdate"]))

    def test_background_variables_extraction(self):
        """Test that background variables are correctly extracted"""
        patient_info = _create_patient_info(self.processed_concepts)

        # Check that background columns exist
        self.assertIn("GENDER", patient_info.columns)
        self.assertIn("SMOKER", patient_info.columns)
        self.assertIn("DIABETES", patient_info.columns)

        # Check specific values
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        patient_2 = patient_info[patient_info["subject_id"] == "2"].iloc[0]
        patient_4 = patient_info[patient_info["subject_id"] == "4"].iloc[0]

        self.assertEqual(patient_1["GENDER"], "F")
        self.assertEqual(patient_1["SMOKER"], "Y")
        self.assertEqual(patient_2["GENDER"], "M")
        self.assertEqual(patient_4["DIABETES"], "Y")

        # Check that missing values are NaN
        self.assertTrue(pd.isna(patient_2["SMOKER"]))
        self.assertTrue(pd.isna(patient_1["DIABETES"]))

    def test_no_background_concepts(self):
        """Test patient info creation when there are no background concepts"""
        concepts_no_bg = pd.DataFrame(
            {
                "subject_id": ["1", "1", "2", "2"],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                    datetime(1995, 9, 3),
                    datetime(2015, 9, 10),
                ],
                "code": ["DOB", "A", "DOB", "B"],
                "birthdate": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(1995, 9, 3),
                    datetime(1995, 9, 3),
                ],
            }
        )

        patient_info = _create_patient_info(concepts_no_bg)

        # Should still have basic structure
        self.assertEqual(len(patient_info), 2)
        self.assertIn("subject_id", patient_info.columns)
        self.assertIn("birthdate", patient_info.columns)
        self.assertIn("deathdate", patient_info.columns)

        # Should have correct birthdates
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        patient_2 = patient_info[patient_info["subject_id"] == "2"].iloc[0]
        self.assertEqual(patient_1["birthdate"], datetime(2000, 5, 1))
        self.assertEqual(patient_2["birthdate"], datetime(1995, 9, 3))

    def test_empty_concepts(self):
        """Test patient info creation with empty concepts DataFrame"""
        empty_concepts = pd.DataFrame(
            columns=["subject_id", "time", "code", "birthdate"]
        )
        patient_info = _create_patient_info(empty_concepts)

        # Should return empty DataFrame with correct structure
        self.assertEqual(len(patient_info), 0)
        self.assertIn("subject_id", patient_info.columns)
        self.assertIn("birthdate", patient_info.columns)
        self.assertIn("deathdate", patient_info.columns)

        # Check that datetime columns have correct dtype
        self.assertEqual(patient_info["birthdate"].dtype, "datetime64[ns]")
        self.assertEqual(patient_info["deathdate"].dtype, "datetime64[ns]")

    def test_concepts_without_birthdate_column(self):
        """Test patient info creation when concepts don't have birthdate column (fallback to DOB codes)"""
        concepts_no_birthdate_col = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "2", "2", "2"],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                    datetime(1995, 9, 3),
                    datetime(1995, 9, 3),
                    datetime(2015, 9, 10),
                ],
                "code": ["DOB", "BG_GENDER//F", "A", "DOB", "BG_GENDER//M", "B"],
            }
        )

        patient_info = _create_patient_info(concepts_no_birthdate_col)

        # Should still extract birthdates correctly from DOB codes
        self.assertEqual(len(patient_info), 2)
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        patient_2 = patient_info[patient_info["subject_id"] == "2"].iloc[0]

        self.assertEqual(patient_1["birthdate"], datetime(2000, 5, 1))
        self.assertEqual(patient_2["birthdate"], datetime(1995, 9, 3))

    def test_duplicate_background_values(self):
        """Test handling of duplicate background values (should take first)"""
        concepts_with_duplicates = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                ],
                "code": [
                    "DOB",
                    "BG_GENDER//F",
                    "BG_GENDER//M",
                    "A",
                ],  # Duplicate GENDER values
                "birthdate": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                ],
            }
        )

        patient_info = _create_patient_info(concepts_with_duplicates)

        # Should take the first value (F)
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        self.assertEqual(patient_1["GENDER"], "F")

    def test_background_values_without_double_slash(self):
        """Test handling of background codes without '//' separator"""
        concepts_no_separator = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                ],
                "code": ["DOB", "BG_GENDER", "A"],  # No // separator
                "birthdate": [
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 5, 1),
                ],
            }
        )

        patient_info = _create_patient_info(concepts_no_separator)

        # Should handle gracefully (might result in NaN or empty string)
        self.assertEqual(len(patient_info), 1)
        # The exact behavior depends on how pandas handles the split, but it shouldn't crash


if __name__ == "__main__":
    unittest.main()
