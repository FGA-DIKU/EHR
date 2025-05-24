import unittest
import pandas as pd
from datetime import datetime
from pandas import NaT

from corebehrt.functional.features.creators import (
    create_background,
)


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

        result, _ = create_background(concepts_with_dup_indices)

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

        result, _ = create_background(concepts_no_bg)

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
        self.assertEqual(len(result), 0)
        self.assertEqual(len(patient_info), 0)

    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified"""
        original_concepts = self.concepts.copy()
        _, _ = create_background(self.concepts)

        # Original should be unchanged
        pd.testing.assert_frame_equal(self.concepts, original_concepts)


if __name__ == "__main__":
    unittest.main()
