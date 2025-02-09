# tests/test_functional/test_filters.py

import unittest
import time
import dask.dataframe as dd
import pandas as pd
import numpy as np

from corebehrt.functional.preparation.filter import (
    filter_table_by_pids,
    filter_table_by_pids_reshuffle
)

class TestFilterPerformance(unittest.TestCase):
    def setUp(self):
        # Create a large test dataset
        n_rows = 1_000_000
        n_pids = 10_000
        
        data = {
            'PID': np.random.choice(range(n_pids), n_rows),
            'Value': np.random.randn(n_rows)
        }
        self.df = dd.from_pandas(pd.DataFrame(data), npartitions=4)
        self.pids_to_filter = list(range(0, n_pids, 2))  # Select half of PIDs
        
    def test_filter_performance(self):
        # Time the optimized version
        start = time.time()
        filtered_df = filter_table_by_pids(self.df, self.pids_to_filter)
        filtered_df.compute()  # Force computation
        optimized_time = time.time() - start
        
        # Time the original version (using direct dask filtering)
        start = time.time()
        filtered_df = self.df[self.df['PID'].isin(self.pids_to_filter)]
        filtered_df.compute()  # Force computation
        original_time = time.time() - start
        
        # Assert the optimized version is faster
        self.assertLess(optimized_time, original_time)

if __name__ == '__main__':
    unittest.main()