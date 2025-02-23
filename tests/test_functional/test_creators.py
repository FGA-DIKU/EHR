import unittest
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
import numpy as np

from corebehrt.functional.features.creators import (
    create_age_in_years,
    create_abspos,
    create_segments,
    create_background,
    sort_features,
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.origin_point = datetime(2020, 1, 26)
        self.background_vars = ["GENDER"]

        # Create sample data as pandas DataFrames
        self.concepts = pd.DataFrame(
            {
                "subject_id": ["1", "2", "3", "1"],
                "code": ["DA1", "DA2", "MA1", "DA2"],
                "time": pd.to_datetime(
                    ["2020-01-02", "2021-03-20", "2022-05-08", "2023-01-02"]
                ),
                "numeric_value": [np.nan, '0.2', '0.98', np.nan],
            }
        )
        self.patients_info = pd.DataFrame(
            {
                "subject_id": ["1", "2", "3"],
                "birthdate": pd.to_datetime(["2000-01-02", "2000-03-20", "2000-05-08"]),
                "GENDER": ["Male", "Female", "Male"],
            }
        )

        self.expected_segments = pd.Series([0, 1, 0, 0], name="segment")

    def test_create_age_in_years(self):
        """
        Test the create_age_in_years function.
        """
        # Merge concepts with patients_info to get 'BIRTHDATE'
        concepts_with_birthdate = self.concepts.merge(
            self.patients_info[["subject_id", "birthdate"]], on="subject_id", how="left"
        )

        # Apply the function
        result = create_age_in_years(concepts_with_birthdate)

        # Expected ages
        expected_ages = (
            self.concepts["time"]
            - self.patients_info.set_index("subject_id")
            .loc[self.concepts["subject_id"]]["birthdate"]
            .values
        ).dt.days // 365.25

        # Assert the ages are as expected
        self.assertTrue((result["age"] == expected_ages).all())

    def test_create_abspos(self):
        """
        Test the create_abspos function.
        """
        # Apply the function
        result = create_abspos(self.concepts, self.origin_point)

        # Expected abspos
        expected_abspos = (
            self.concepts["time"] - self.origin_point
        ).dt.total_seconds() / 3600

        # Assert the abspos values are as expected
        self.assertTrue((result["abspos"] == expected_abspos).all())

    def test_create_segments(self):
        """
        Test the create_segments function.
        """
        # Prepare concepts DataFrame by adding 'abspos' (required for sorting)
        # concepts = self.concepts.rename(columns={"CONCEPT": "concept"})
        concepts_with_abspos = create_abspos(self.concepts, self.origin_point)

        # Apply the function
        sorted_concepts = sort_features(concepts_with_abspos)
        result = create_segments(sorted_concepts)

        # Assert the segments are as expected
        self.assertTrue((result["segment"] == self.expected_segments).all())

    # def test_create_background(self):
    #     """
    #     Test the create_background function.
    #     """
    #     # Apply the function
    #     result = create_background(self.patients_info, self.background_vars)

    #     # Expected number of rows: number of patients * number of background_vars
    #     expected_rows = len(self.patients_info) * len(self.background_vars)
    #     self.assertEqual(len(result), expected_rows)

    #     # Check that all added concepts are prefixed by 'BG_'
    #     self.assertTrue(result["code"].str.startswith("BG_").all())

    #     # Check that each patient has one entry for each background_var
    #     group_counts = result.groupby("subject_id").size()
    #     self.assertTrue((group_counts == len(self.background_vars)).all())

    #     # Verify that the 'concept' column contains the correct values
    #     for pid, group in result.groupby("subject_id"):
    #         patient_info = self.patients_info[self.patients_info["subject_id"] == pid]
    #         for var in self.background_vars:
    #             expected_concept = f"BG_{var}_{patient_info[var].values[0]}"
    #             self.assertIn(expected_concept, group["code"].values)

    # def test_create_death(self):
    #     """
    #     Test the create_death function.
    #     """
    #     # Add 'DEATHDATE' to patients_info
    #     patients_info_with_death = self.patients_info_pd.copy()
    #     patients_info_with_death["deathdate"] = pd.to_datetime(
    #         ["2025-01-01", pd.NaT, "2022-12-31"]
    #     )

    #     # Convert to Dask DataFrame
    #     patients_info_dd = dd.from_pandas(patients_info_with_death, npartitions=1)

    #     # Apply the function
    #     result = create_death(patients_info_dd)

    #     # Compute the result
    #     result_df = result.compute()

    #     # Expected result should have rows for patients with non-null DEATHDATE
    #     expected_pids = patients_info_with_death[
    #         ~patients_info_with_death["deathdate"].isna()
    #     ]["subject_id"].values

    #     # Assert that the result contains the correct subject_ids
    #     self.assertTrue(set(result_df["subject_id"]) == set(expected_pids))

    #     # Assert that 'concept' column is 'Death'
    #     self.assertTrue((result_df["code"] == "DOD").all())

    #     # Assert that 'TIMESTAMP' matches 'DEATHDATE' in patients_info
    #     for pid in expected_pids:
    #         death_timestamp = result_df[result_df["subject_id"] == pid]["time"].values[0]
    #         expected_timestamp = patients_info_with_death[
    #             patients_info_with_death["subject_id"] == pid
    #         ]["deathdate"].values[0]
    #         self.assertEqual(death_timestamp, expected_timestamp)


if __name__ == "__main__":
    unittest.main()
