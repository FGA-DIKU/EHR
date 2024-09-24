import unittest
from unittest.mock import patch
import pandas as pd
import dask.dataframe as dd
from corebehrt.functional.utils import (
    filter_table_by_pids,
    select_random_subset,
    normalize_segments_dask,
)
from corebehrt.functional.exclude import filter_table_by_exclude_pids
import random
from pandas.testing import assert_frame_equal


class TestPrepDataUtilsFunctions(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        data = pd.DataFrame(
            {"PID": [1, 2, 3, 4, 5], "Value": ["A", "B", "C", "D", "E"]}
        )
        self.data_dd = dd.from_pandas(data, npartitions=1)

    def test_select_data_by_pids(self):
        selected_data = filter_table_by_pids(self.data_dd, [1, 2])
        self.assertEqual(len(selected_data), 2)
        self.assertTrue(set(selected_data.compute()["PID"]).issubset({1, 2}))

    def test_select_random_subset(self):
        random.seed(42)  # Ensure reproducibility
        subset_data = select_random_subset(self.data_dd, 3)
        self.assertEqual(len(subset_data), 3)
        # Check if the subset contains unique PIDs
        self.assertEqual(len(subset_data.compute()["PID"].unique()), 3)

    def test_select_random_subset_with_n_greater_than_data_length(self):
        subset_data = select_random_subset(self.data_dd, 10)
        self.assertEqual(
            len(subset_data), 5
        )  # Should return all data because n > len(data)

    @patch("corebehrt.functional.exclude.load_pids")
    def test_filter_table_by_exclude_pids_with_path(self, mock_load_pids):
        # Mock load_pids to return a list of PIDs to exclude
        loaded_pids = [1, 2]
        filtered_data = filter_table_by_exclude_pids(self.data_dd, loaded_pids)
        filtered_data_pd = filtered_data.compute()

        self.assertFalse(set([1, 2]).intersection(set(filtered_data_pd["PID"])))
        self.assertEqual(len(filtered_data_pd), 3)

    def test_normalize_segments_dask(self):
        sample_data = pd.DataFrame(
            {"PID": [1, 1, 1, 2, 2, 3], "segment": [10, 20, 30, 5, 7, 8]}
        )
        sample_dd = dd.from_pandas(sample_data, npartitions=1)
        normalized_dd = normalize_segments_dask(sample_dd)
        normalized_pd = normalized_dd.compute()

        expected_data = {"PID": [1, 1, 1, 2, 2, 3], "segment": [0, 1, 2, 0, 1, 0]}
        expected_df = dd.from_pandas(
            pd.DataFrame(expected_data), npartitions=2
        ).compute()

        assert_frame_equal(normalized_pd, expected_df, check_like=True)


if __name__ == "__main__":
    unittest.main()
