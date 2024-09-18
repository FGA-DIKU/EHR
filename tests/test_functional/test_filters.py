import unittest
import pyarrow
import dask.dataframe as dd
import pandas as pd
from pandas.testing import assert_frame_equal

from corebehrt.functional.filter import (censor_data,
                                         filter_patients_by_age_at_last_event)


class TestCensorModule(unittest.TestCase):

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
        result['PID'] = result['PID'].astype('str')
        result['concept'] = result['concept'].astype('str')
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

        censor_dates = pd.Series(dtype='int', index=[])

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
        result['PID'] = result['PID'].astype('str')
        result['concept'] = result['concept'].astype('str')
        expected = pd.DataFrame({
            'PID': ['P1'],
            'abspos': [100],
            'concept': ['A']
        })

        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


class TestFilterPatientsByAge(unittest.TestCase):

    def test_filter_patients_by_age_at_last_event(self):
        # Create sample data
        data = pd.DataFrame({
            'PID': ['P1', 'P1', 'P2', 'P2', 'P3', 'P4'],
            'age': [30, 31, 45, 46, 60, 65],
            'concept': ['A', 'B', 'C', 'D', 'E', 'F']
        })
        data_dd = dd.from_pandas(data, npartitions=1)

        # Define age range
        min_age = 31
        max_age = 60

        # Call the function
        result_dd = filter_patients_by_age_at_last_event(data_dd, min_age, max_age)
        result = result_dd.compute()

        # Expected output
        expected = pd.DataFrame({
            'PID': pd.Series(['P1', 'P1', 'P2', 'P2', 'P3']),
            'age': [30, 31, 45, 46, 60],
            'concept': pd.Series(['A', 'B', 'C', 'D', 'E'])
        })
        # Sort and reset index for comparison
        result['PID'] = result['PID'].astype('str')
        result['concept'] = result['concept'].astype('str')
        result_sorted = result.sort_values(['PID', 'age']).reset_index(drop=True)
        expected_sorted = expected.sort_values(['PID', 'age']).reset_index(drop=True)

        # Compare the result with the expected output
        assert_frame_equal(result_sorted, expected_sorted)

    def test_filter_patients_by_age_at_last_event_no_patients_in_range(self):
        # Create sample data
        data = pd.DataFrame({
            'PID': ['P1', 'P2', 'P3'],
            'age': [25, 26, 27],
            'concept': ['A', 'B', 'C']
        })
        data_dd = dd.from_pandas(data, npartitions=1)

        # Define age range where no patients fall into
        min_age = 30
        max_age = 40

        # Call the function
        result_dd = filter_patients_by_age_at_last_event(data_dd, min_age, max_age)
        result = result_dd.compute()
        
        # check that result is empty
        self.assertTrue(result.empty)
        

    def test_filter_patients_by_age_at_last_event_all_patients_in_range(self):
        # Create sample data
        data = pd.DataFrame({
            'PID': ['P1', 'P1', 'P2', 'P2'],
            'age': [40, 41, 42, 43],
            'concept': ['A', 'B', 'C', 'D']
        })
        data_dd = dd.from_pandas(data, npartitions=1)

        # Define age range that includes all patients
        min_age = 40
        max_age = 50

        # Call the function
        result_dd = filter_patients_by_age_at_last_event(data_dd, min_age, max_age)
        result = result_dd.compute()

        # Expected output is the same as the input data
        expected = data
        result['PID'] = result['PID'].astype('str')
        result['concept'] = result['concept'].astype('str')
        # Sort and reset index for comparison
        result_sorted = result.sort_values(['PID', 'age']).reset_index(drop=True)
        expected_sorted = expected.sort_values(['PID', 'age']).reset_index(drop=True)

        # Compare the result with the expected output
        assert_frame_equal(result_sorted, expected_sorted)


if __name__ == "__main__":
    unittest.main()
