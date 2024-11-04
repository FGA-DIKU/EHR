import unittest
import pandas as pd
import dask.dataframe as dd
from datetime import datetime

from corebehrt.functional.creators import (
    create_age_in_years,
    create_abspos,
    create_segments,
    create_background,
    create_death,
    sort_features
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.origin_point = datetime(2020, 1, 26)
        self.background_vars = ["GENDER"]

        # Create sample data as pandas DataFrames
        self.concepts_pd = pd.DataFrame(
            {
                "PID": ["1", "2", "3", "1"],
                "CONCEPT": ["DA1", "DA2", "MA1", "DA2"],
                "TIMESTAMP": pd.to_datetime(
                    ["2020-01-02", "2021-03-20", "2022-05-08", "2023-01-02"]
                ),
                "ADMISSION_ID": ["A", "B", "C", "D"],
            }
        )
        self.patients_info_pd = pd.DataFrame(
            {
                "PID": ["1", "2", "3"],
                "BIRTHDATE": pd.to_datetime(["2000-01-02", "2000-03-20", "2000-05-08"]),
                "GENDER": ["Male", "Female", "Male"],
            }
        )

        # Convert pandas DataFrames to Dask DataFrames
        self.concepts = dd.from_pandas(self.concepts_pd, npartitions=1)
        self.patients_info = dd.from_pandas(self.patients_info_pd, npartitions=1)
        self.expected_segments = pd.Series([0, 1, 0, 0], name="segment")

    def test_create_age_in_years(self):
        """
        Test the create_age_in_years function.
        """
        # Merge concepts with patients_info to get 'BIRTHDATE'
        concepts_with_birthdate = self.concepts.merge(
            self.patients_info[["PID", "BIRTHDATE"]], on="PID", how="left"
        )

        # Apply the function
        result = create_age_in_years(concepts_with_birthdate)

        # Compute the result
        result_df = result.compute()

        # Expected ages
        expected_ages = (
            self.concepts_pd["TIMESTAMP"]
            - self.patients_info_pd.set_index("PID")
            .loc[self.concepts_pd["PID"]]["BIRTHDATE"]
            .values
        ).dt.days // 365.25

        # Assert the ages are as expected
        self.assertTrue((result_df["age"] == expected_ages).all())

    def test_create_abspos(self):
        """
        Test the create_abspos function.
        """
        # Apply the function
        result = create_abspos(self.concepts, self.origin_point)

        # Compute the result
        result_df = result.compute()

        # Expected abspos
        expected_abspos = (
            self.concepts_pd["TIMESTAMP"] - self.origin_point
        ).dt.total_seconds() / 3600

        # Assert the abspos values are as expected
        self.assertTrue((result_df["abspos"] == expected_abspos).all())

    def test_create_segments(self):
        """
        Test the create_segments function.
        """
        # Prepare concepts DataFrame by adding 'abspos' (required for sorting)
        concepts = self.concepts.rename(columns={"CONCEPT": "concept"})
        concepts_with_abspos = create_abspos(concepts, self.origin_point)

        # Apply the function
        sorted_concepts = sort_features(concepts_with_abspos)
        result = create_segments(sorted_concepts)

        # Compute the result
        result_df = result.compute()

        # Assert the segments are as expected
        self.assertTrue((result_df["segment"] == self.expected_segments).all())

    def test_create_background(self):
        """
        Test the create_background function.
        """
        # Apply the function
        result = create_background(self.patients_info, self.background_vars)

        # Compute the result
        result_df = result.compute()

        # Expected number of rows: number of patients * number of background_vars
        expected_rows = len(self.patients_info_pd) * len(self.background_vars)
        self.assertEqual(len(result_df), expected_rows)

        # Check that all added concepts are prefixed by 'BG_'
        self.assertTrue(result_df["concept"].str.startswith("BG_").all())

        # Check that each patient has one entry for each background_var
        group_counts = result_df.groupby("PID").size()
        self.assertTrue((group_counts == len(self.background_vars)).all())

        # Verify that the 'concept' column contains the correct values
        for pid, group in result_df.groupby("PID"):
            patient_info = self.patients_info_pd[self.patients_info_pd["PID"] == pid]
            for var in self.background_vars:
                expected_concept = f"BG_{var}_{patient_info[var].values[0]}"
                self.assertIn(expected_concept, group["concept"].values)

    def test_create_death(self):
        """
        Test the create_death function.
        """
        # Add 'DEATHDATE' to patients_info
        patients_info_with_death = self.patients_info_pd.copy()
        patients_info_with_death["DEATHDATE"] = pd.to_datetime(
            ["2025-01-01", pd.NaT, "2022-12-31"]
        )

        # Convert to Dask DataFrame
        patients_info_dd = dd.from_pandas(patients_info_with_death, npartitions=1)

        # Apply the function
        result = create_death(patients_info_dd)

        # Compute the result
        result_df = result.compute()

        # Expected result should have rows for patients with non-null DEATHDATE
        expected_pids = patients_info_with_death[
            ~patients_info_with_death["DEATHDATE"].isna()
        ]["PID"].values

        # Assert that the result contains the correct PIDs
        self.assertTrue(set(result_df["PID"]) == set(expected_pids))

        # Assert that 'concept' column is 'Death'
        self.assertTrue((result_df["concept"] == "Death").all())

        # Assert that 'TIMESTAMP' matches 'DEATHDATE' in patients_info
        for pid in expected_pids:
            death_timestamp = result_df[result_df["PID"] == pid]["TIMESTAMP"].values[0]
            expected_timestamp = patients_info_with_death[
                patients_info_with_death["PID"] == pid
            ]["DEATHDATE"].values[0]
            self.assertEqual(death_timestamp, expected_timestamp)


if __name__ == "__main__":
    unittest.main()
