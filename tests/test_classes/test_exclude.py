import unittest

import numpy as np
import pandas as pd

from corebehrt.classes.excluder import Excluder

import dask.dataframe as dd


class TestExcluder(unittest.TestCase):
    def setUp(self):
        self.excluder = Excluder(min_age=0, max_age=100, min_len=3, background_length=1)

        # Sample data for testing
        self.sample_df = pd.DataFrame(
            {
                "PID": [1, 1, 1, 2, 2, 3],
                "age": [-1, 50, 120, 30, 80, np.nan],
                "concept": ["A", "B", "C", "D", "E", "F"],
                "segment": [0, 1, 2, 0, 1, 0],
            }
        )

        self.sample_list = [["A", "B", "C", "D"], ["E"], ["F", "G", "H", "I", "J"]]

        self.sample_dict = {
            "concept": [["A", "B", "C", "D"], ["E"], ["F", "G", "H", "I", "J"]],
            "value": [[1, 2, 3, 4], [5], [6, 7, 8, 9, 10]],
        }

    def test_exclude_incorrect_events(self):
        result = self.excluder.exclude_incorrect_events(self.sample_df)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result["age"].between(0, 100)))
        self.assertFalse(result["age"].isnull().any())

    def test_exclude_short_sequences_df(self):
        df = pd.DataFrame(
            {
                "PID": [1, 1, 1, 1, 2, 2, 3],
                "concept": ["A", "B", "C", "D", "E", "F", "G"],
            }
        )
        result, kept_indices = self.excluder.exclude_short_sequences(df)
        self.assertEqual(len(result), 4)
        self.assertEqual(result["PID"].nunique(), 1)
        self.assertEqual(len(kept_indices), 4)

    def test_exclude_short_sequences_list(self):
        result, kept_indices = self.excluder.exclude_short_sequences(self.sample_list)
        self.assertEqual(len(result), 2)
        self.assertTrue(
            all(
                len(seq) >= self.excluder.min_len + self.excluder.background_length
                for seq in result
            )
        )
        self.assertEqual(len(kept_indices), 2)

    def test_exclude_short_sequences_dict(self):
        result, kept_indices = self.excluder.exclude_short_sequences(self.sample_dict)
        self.assertEqual(len(result["concept"]), 2)
        self.assertTrue(
            all(
                len(seq) >= self.excluder.min_len + self.excluder.background_length
                for seq in result["concept"]
            )
        )
        self.assertEqual(len(kept_indices), 2)

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

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.excluder.exclude_short_sequences("invalid input")

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
