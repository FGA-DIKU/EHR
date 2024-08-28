import unittest
from unittest.mock import patch
import pandas as pd
import dask.dataframe as dd
from corebehrt.functional.utils import (
    select_data_by_pids,
    exclude_pids,
    select_random_subset,
)
import random


class TestPrepDataUtilsFunctions(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        data = pd.DataFrame(
            {"PID": [1, 2, 3, 4, 5], "Value": ["A", "B", "C", "D", "E"]}
        )
        self.data_dd = dd.from_pandas(data, npartitions=1)

    def test_select_data_by_pids(self):
        selected_data = select_data_by_pids(self.data_dd, [1, 2])
        self.assertEqual(len(selected_data), 2)
        self.assertTrue(set(selected_data.compute()["PID"]).issubset({1, 2}))

    @patch("corebehrt.functional.utils.load_pids")
    def test_exclude_pids(self, mock_load_pids):
        mock_load_pids.return_value = [1, 2]
        excluded_data = exclude_pids(self.data_dd, "dummy_path")
        self.assertEqual(len(excluded_data), 3)
        self.assertTrue(set(excluded_data.compute()["PID"]).issubset({3, 4, 5}))

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


if __name__ == "__main__":
    unittest.main()
