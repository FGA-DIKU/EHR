import unittest
import pandas as pd
from datetime import datetime, timedelta
from pandas import NaT

from corebehrt.functional.features.creators import (
    create_age_in_years,
    create_abspos,
    create_segments,
    create_background,
    sort_features,
    _create_patient_info,
    _assign_admission_ids,
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.background_vars = ["GENDER"]

        self.concepts = pd.DataFrame(
            {
                "subject_id": [
                    "1",
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
                    "3",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                ],
                "time": [
                    datetime(2000, 5, 1),
                    NaT,
                    datetime(2015, 7, 1),
                    datetime(2015, 7, 2),
                    datetime(2016, 8, 2),
                    datetime(2022, 9, 3),
                    datetime(1995, 9, 3),
                    NaT,
                    datetime(2015, 9, 10),
                    datetime(2018, 4, 5),
                    datetime(1962, 11, 5),
                    NaT,
                    datetime(2020, 11, 6),
                    datetime(2022, 6, 6),
                    datetime(1975, 1, 6),
                    NaT,
                    datetime(2015, 12, 7),
                    datetime(2015, 12, 7),
                    datetime(2015, 12, 9),
                    datetime(2015, 12, 11),
                    datetime(2015, 12, 11),
                    datetime(2022, 10, 11),
                    datetime(2022, 10, 12),
                ],
                "code": [
                    "DOB",
                    "GENDER//F",
                    "A",
                    "B",
                    "C",
                    "DOD",
                    "DOB",
                    "GENDER//M",
                    "A",
                    "D",
                    "DOB",
                    "GENDER//M",
                    "D",
                    "D",
                    "DOB",
                    "GENDER//F",
                    "ADMISSION",
                    "E",
                    "A",
                    "B",
                    "DISCHARGE",
                    "X",
                    "Y",
                ],
            }
        )

        self.expected_segments = pd.Series(
            [0, 0, 1, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 1, 1, 1, 1, 2, 2],
            name="segment",
        )

        self.expected_patient_info = pd.DataFrame(
            {
                "subject_id": ["1", "2", "3", "4"],
                "birthdate": [
                    datetime(2000, 5, 1),
                    datetime(1995, 9, 3),
                    datetime(1962, 11, 5),
                    datetime(1975, 1, 6),
                ],
                "deathdate": [datetime(2022, 9, 3), NaT, NaT, NaT],
                "GENDER": ["F", "M", "M", "F"],
            }
        )

    def test_create_background(self):
        """
        Test the create_background function.
        """
        # Apply the function
        result, patient_info = create_background(self.concepts)

        # Expected number of rows: number of patients * number of background_vars
        expected_rows = len(self.concepts["code"])
        self.assertEqual(len(result), expected_rows)

        # Check that all background concepts are prefixed by 'BG_' and have birthdate as time
        self.assertEqual(result["code"].str.startswith("BG_").sum(), 4)
        bg_rows = result[result["code"].str.startswith("BG_")]
        dob_rows = result[result["code"] == "DOB"]
        self.assertTrue((bg_rows["time"] == dob_rows["time"].values).all())

        # Check patient_info
        self.assertTrue(
            (len(patient_info) == len(self.concepts["subject_id"].unique()))
        )
        self.assertTrue(
            all(
                patient_info.columns
                == ["subject_id", "birthdate", "deathdate", "GENDER"]
            )
        )
        pd.testing.assert_frame_equal(patient_info, self.expected_patient_info)

    def test_create_age_in_years(self):
        """
        Test the create_age_in_years function.
        """
        concepts_with_birthdate = self.concepts.merge(
            self.expected_patient_info[["subject_id", "birthdate"]],
            on="subject_id",
            how="left",
        )

        # Merge concepts with patients_info to get 'BIRTHDATE'
        result = create_age_in_years(concepts_with_birthdate)

        # Expected ages
        expected_ages = (
            self.concepts["time"]
            - self.expected_patient_info.set_index("subject_id")
            .loc[self.concepts["subject_id"]]["birthdate"]
            .values
        ).dt.days // 365.25

        # Assert the ages are as expected
        self.assertTrue((result["age"].dropna() == expected_ages.dropna()).all())

    def test_create_abspos(self):
        """
        Test the create_abspos function.
        """
        # Apply the function
        concepts_no_nan = self.concepts.dropna(
            subset=["time"]
        )  # Remove rows with NaT time (BG)
        result = create_abspos(concepts_no_nan)

        # Expected abspos
        expected_abspos = concepts_no_nan["time"].astype("int64") // 10**9 / 3600

        # Assert the abspos values are as expected
        self.assertTrue((result["abspos"] == expected_abspos).all())

    def test_create_segments(self):
        """
        Test the create_segments function.
        """
        # Prepare concepts DataFrame by adding 'abspos' (required for sorting) and 'BG' rows
        # concepts = self.concepts.rename(columns={"CONCEPT": "concept"})
        concepts_with_bg, _ = create_background(self.concepts)
        concepts_with_abspos = create_abspos(concepts_with_bg)

        # Apply the function
        sorted_concepts = sort_features(concepts_with_abspos)
        result = create_segments(sorted_concepts)

        # Assert the segments are as expected
        self.assertTrue((result["segment"] == self.expected_segments).all())


class TestCreateBackground(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.concepts = pd.DataFrame(
            {
                "subject_id": [
                    "1",
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
                    "3",
                ],
                "time": [
                    datetime(2000, 5, 1),
                    NaT,
                    datetime(2015, 7, 1),
                    datetime(2015, 7, 2),
                    datetime(2016, 8, 2),
                    datetime(2022, 9, 3),
                    datetime(1995, 9, 3),
                    NaT,
                    datetime(2015, 9, 10),
                    datetime(2018, 4, 5),
                    datetime(1962, 11, 5),
                    NaT,
                    datetime(2020, 11, 6),
                    datetime(2022, 6, 6),
                ],
                "code": [
                    "DOB",
                    "GENDER//F",
                    "A",
                    "B",
                    "C",
                    "DOD",
                    "DOB",
                    "GENDER//M",
                    "A",
                    "D",
                    "DOB",
                    "GENDER//M",
                    "D",
                    "D",
                ],
            }
        )

    def test_basic_functionality(self):
        """Test basic functionality of create_background"""
        result, patient_info = create_background(self.concepts)

        # Check that background concepts are properly prefixed
        bg_concepts = result[result["code"].str.startswith("BG_", na=False)]
        self.assertEqual(len(bg_concepts), 3)  # Should have 3 background concepts

        # Check that background concepts have birthdate as time
        for _, row in bg_concepts.iterrows():
            patient_id = row["subject_id"]
            expected_birthdate = patient_info[patient_info["subject_id"] == patient_id][
                "birthdate"
            ].iloc[0]
            self.assertEqual(row["time"], expected_birthdate)

    def test_duplicate_indices_bug(self):
        """Test the specific bug with duplicate indices"""
        # Create DataFrame with duplicate indices
        concepts_with_dup_indices = self.concepts.copy()
        concepts_with_dup_indices.index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

        result, patient_info = create_background(concepts_with_dup_indices)

        # Verify that only NaT rows were updated
        original_nat_count = self.concepts["time"].isna().sum()
        result_bg_count = result["code"].str.startswith("BG_", na=False).sum()
        self.assertEqual(original_nat_count, result_bg_count)

        # Verify that non-NaT rows were not affected
        non_nat_original = self.concepts[~self.concepts["time"].isna()]
        non_bg_result = result[~result["code"].str.startswith("BG_", na=False)]
        non_bg_result = non_bg_result[
            ~non_bg_result["code"].str.startswith("ADM_", na=False)
        ]

        # Check that timestamps of non-background concepts remain unchanged
        for idx in non_nat_original.index:
            if idx < len(non_bg_result):
                original_time = non_nat_original.loc[idx, "time"]
                if pd.notna(original_time):
                    # Find corresponding row in result
                    matching_rows = non_bg_result[
                        (
                            non_bg_result["subject_id"]
                            == non_nat_original.loc[idx, "subject_id"]
                        )
                        & (non_bg_result["code"] == non_nat_original.loc[idx, "code"])
                    ]
                    if not matching_rows.empty:
                        self.assertEqual(matching_rows.iloc[0]["time"], original_time)

    def test_non_contiguous_indices(self):
        """Test with non-contiguous indices"""
        concepts_non_contiguous = self.concepts.copy()
        concepts_non_contiguous.index = [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
        ]

        result, patient_info = create_background(concepts_non_contiguous)

        # Should still work correctly
        bg_concepts = result[result["code"].str.startswith("BG_", na=False)]
        self.assertEqual(len(bg_concepts), 3)

    def test_no_background_concepts(self):
        """Test with DataFrame containing no NaT values (no background concepts)"""
        concepts_no_bg = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1"],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                    datetime(2015, 7, 2),
                ],
                "code": ["DOB", "A", "B"],
            }
        )

        result, patient_info = create_background(concepts_no_bg)

        # Should have no background concepts
        bg_concepts = result[result["code"].str.startswith("BG_", na=False)]
        self.assertEqual(len(bg_concepts), 0)

    def test_admission_discharge_prefixing(self):
        """Test that admission and discharge codes are properly prefixed"""
        concepts_with_adm = pd.DataFrame(
            {
                "subject_id": ["1", "1", "1", "1"],
                "time": [
                    datetime(2000, 5, 1),
                    datetime(2015, 7, 1),
                    datetime(2015, 7, 2),
                    datetime(2015, 7, 3),
                ],
                "code": ["DOB", "ADMISSION", "DISCHARGE", "A"],
            }
        )

        result, patient_info = create_background(concepts_with_adm)

        # Check that ADMISSION and DISCHARGE are prefixed with ADM_
        adm_concepts = result[result["code"].str.startswith("ADM_", na=False)]
        self.assertEqual(len(adm_concepts), 2)
        self.assertIn("ADM_ADMISSION", result["code"].values)
        self.assertIn("ADM_DISCHARGE", result["code"].values)

    def test_patient_info_creation(self):
        """Test patient info DataFrame creation"""
        result, patient_info = create_background(self.concepts)

        # Check structure
        self.assertEqual(len(patient_info), 3)  # 3 unique patients
        self.assertIn("subject_id", patient_info.columns)
        self.assertIn("birthdate", patient_info.columns)
        self.assertIn("deathdate", patient_info.columns)
        self.assertIn("GENDER", patient_info.columns)

        # Check specific values
        patient_1 = patient_info[patient_info["subject_id"] == "1"].iloc[0]
        self.assertEqual(patient_1["GENDER"], "F")
        self.assertEqual(patient_1["birthdate"], datetime(2000, 5, 1))
        self.assertEqual(patient_1["deathdate"], datetime(2022, 9, 3))

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame(columns=["subject_id", "time", "code"])
        result, patient_info = create_background(empty_df)
        print("===========")
        print("patient info")
        print(patient_info)
        self.assertEqual(len(result), 0)
        self.assertEqual(len(patient_info), 0)

    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified"""
        original_concepts = self.concepts.copy()
        result, patient_info = create_background(self.concepts)

        # Original should be unchanged
        pd.testing.assert_frame_equal(self.concepts, original_concepts)


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

        # Should have 3 unique admission IDs
        print("==========")
        print(result)
        print("==========")
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
        print("==========")
        print(result)
        print("==========")
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


if __name__ == "__main__":
    unittest.main()
