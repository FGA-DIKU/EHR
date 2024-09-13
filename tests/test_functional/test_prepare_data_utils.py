import unittest
from unittest.mock import patch
import pandas as pd
import dask.dataframe as dd
from corebehrt.functional.utils import (
    filter_table_by_pids,
    select_random_subset,
)
from corebehrt.functional.exclude import filter_table_by_exclude_pids
import random


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
        mock_load_pids.return_value = [1, 2]
        filtered_data = filter_table_by_exclude_pids(self.data_dd, 'dummy_path')
        filtered_data_pd = filtered_data.compute()

        self.assertFalse(set([1, 2]).intersection(set(filtered_data_pd['PID'])))
        self.assertEqual(len(filtered_data_pd), 3)

    def test_filter_table_by_exclude_pids_without_path(self):
        filtered_data = filter_table_by_exclude_pids(self.data_dd, None)
        filtered_data_pd = filtered_data.compute()
        self.assertEqual(len(filtered_data_pd), 5)

    


if __name__ == "__main__":
    unittest.main()
