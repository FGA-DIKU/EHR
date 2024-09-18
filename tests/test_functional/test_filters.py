import unittest

import dask.dataframe as dd
import pandas as pd
from pandas.testing import assert_frame_equal

from corebehrt.functional.filter import censor_data


class TestFilterModule(unittest.TestCase):

    def test_censor_data(self):
        # Create sample data
        data = pd.DataFrame({
            'PID': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3', 'P4'],
            'abspos': [100, 200, 300, 150, 250, 400, 500],
            'concept': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        })
        data_dd = dd.from_pandas(data, npartitions=1)

        # Create sample censor_dates
        censor_dates = pd.Series({
            'P1': 250,
            'P2': 200,
            'P3': 450,
            # P4 is not in censor_dates
        })

        # Call the function
        result_dd = censor_data(data_dd, censor_dates)

        # Collect the result
        result = result_dd.compute()

        # Expected output
        expected = pd.DataFrame({
            'PID': ['P1', 'P1', 'P2', 'P3'],
            'abspos': [100, 200, 150, 400],
            'concept': ['A', 'B', 'D', 'F']
        })

        # Sort and reset index for comparison
        result_sorted = result.sort_values(['PID', 'abspos']).reset_index(drop=True)
        expected_sorted = expected.sort_values(['PID', 'abspos']).reset_index(drop=True)

        # Compare the result with the expected output
        assert_frame_equal(result_sorted, expected_sorted)

    def test_censor_data_empty_censor_dates(self):
        # Test with empty censor_dates Series
        data = pd.DataFrame({
            'PID': ['P1', 'P1'],
            'abspos': [100, 200],
            'concept': ['A', 'B']
        })
        data_dd = dd.from_pandas(data, npartitions=1)

        censor_dates = pd.Series(dtype='int')

        result_dd = censor_data(data_dd, censor_dates)
        result = result_dd.compute()
        self.assertTrue(result.empty)

    def test_censor_data_pid_not_in_censor_dates(self):
        # Test with PIDs in data not in censor_dates
        data = pd.DataFrame({
            'PID': ['P1', 'P5'],
            'abspos': [100, 200],
            'concept': ['A', 'B']
        })
        data_dd = dd.from_pandas(data, npartitions=1)

        censor_dates = pd.Series({
            'P1': 150,
            # P5 is not in censor_dates
        })

        result_dd = censor_data(data_dd, censor_dates)
        result = result_dd.compute()

        expected = pd.DataFrame({
            'PID': ['P1'],
            'abspos': [100],
            'concept': ['A']
        })

        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

if __name__ == "__main__":
    unittest.main()
