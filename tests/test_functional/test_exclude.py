import unittest
import pandas as pd
import dask.dataframe as dd
from corebehrt.functional.exclude import filter_table_by_exclude_pids, filter_patients_by_gender

class TestExcludeFunctions(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {"PID": [1, 2, 3, 4, 5], "concept": [1, 2, 3, 2, 1]}
        )
        self.data_dd = dd.from_pandas(self.data, npartitions=1)

    def test_filter_table_by_exclude_pids_without_path(self):
        # Without a path, data should be returned as is
        filtered_data = filter_table_by_exclude_pids(self.data_dd, None)
        filtered_data_pd = filtered_data.compute()

        # Check that the data remains unchanged
        self.assertEqual(len(filtered_data_pd), len(self.data))

    def test_filter_patients_by_gender_with_matching_gender(self):
        # Sample vocabulary dictionary
        vocabulary = {
            "BG_GENDER_Male": 1,
            "BG_GENDER_Female": 2,
        }
        
        # Filtering by gender "Male"
        filtered_data = filter_patients_by_gender(self.data_dd, vocabulary, "Male")
        filtered_data_pd = filtered_data.compute()

        # Only PIDs with the gender token 10 ("Male") should be included
        self.assertTrue(set(filtered_data_pd["PID"]).issubset({1, 5}))
        self.assertEqual(len(filtered_data_pd), 2)

    def test_filter_patients_by_gender_with_no_matching_gender(self):
        # Sample vocabulary dictionary
        vocabulary = {
            "BG_GENDER_Male": 1,
            "BG_GENDER_Female": 2,
        }
        
        # Filtering by a gender not in the data
        with self.assertRaises(ValueError):
            filter_patients_by_gender(self.data_dd, vocabulary, "Other")


    def test_filter_patients_by_gender_without_gender(self):
        # Without specifying gender, the data should be returned as is
        vocabulary = {
            "BG_GENDER_Male": 1,
            "BG_GENDER_Female": 2,
        }

        filtered_data = filter_patients_by_gender(self.data_dd, vocabulary)
        filtered_data_pd = filtered_data.compute()

        # Check that the data remains unchanged
        self.assertEqual(len(filtered_data_pd), len(self.data))

if __name__ == "__main__":
    unittest.main()