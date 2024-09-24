import unittest

import dask.dataframe as dd
import pandas as pd

from corebehrt.functional.exclude import filter_patients_by_gender

import dask.dataframe as dd


class TestExcludeFunctions(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({"PID": [1, 2, 3, 4, 5], "concept": [1, 2, 3, 2, 1]})
        self.data_dd = dd.from_pandas(self.data, npartitions=1)

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
        with self.assertRaises(KeyError):
            filter_patients_by_gender(self.data_dd, vocabulary, "Other")

<<<<<<<< HEAD:tests/test_classes/test_exclude.py
    def test_exclude_short_sequences_dd(self):
        df = pd.DataFrame(
            {
                "PID": [1, 1, 1, 1, 2, 2, 3],
                "concept": ["A", "B", "C", "D", "E", "F", "G"],
            }
        )
        ddf = dd.from_pandas(df, npartitions=1)

        result = self.excluder.exclude_short_sequences(ddf)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.compute()["PID"].nunique(), 1)

    def test_normalize_segments(self):
        df = pd.DataFrame(
            {
                "PID": [1, 1, 1, 2, 2],
                "segment": ["a", "b", "c", "x", "y"],
                "age": [10, 10, -2, 130, 20],
            }
        )
        result = self.excluder.exclude_incorrect_events(df)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["segment"].tolist(), [0, 1, 0])
========
    def test_filter_patients_by_gender_without_gender(self):
        # Without specifying gender, the data should be returned as is
        vocabulary = {
            "BG_GENDER_Male": 1,
            "BG_GENDER_Female": 2,
        }
>>>>>>>> main:tests/test_functional/test_exclude.py

        filtered_data = filter_patients_by_gender(self.data_dd, vocabulary)
        filtered_data_pd = filtered_data.compute()

        # Check that the data remains unchanged
        self.assertEqual(len(filtered_data_pd), len(self.data))

    def test_exclude_pids(self):
        df = pd.DataFrame(
            {
                "PID": [1, 1, 1, 2, 2, 3],
                "concept": ["A", "B", "C", "D", "E", "F"],
                "age": [-1, 50, 120, 30, 80, np.nan],
                "segment": [0, 1, 2, 0, 1, 0],
                "abspos": [-20, 20, 60, -200, 203, 204],
            }
        )

        ddf = dd.from_pandas(df, npartitions=1)
        exclude_path = "tests/data/prepped/pids/exclude_pids.pt"
        result = self.excluder.exclude_pids(ddf, exclude_path).compute()
        self.assertEqual(len(result), 3)
        self.assertEqual(result["PID"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
