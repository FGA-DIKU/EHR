import unittest

import dask.dataframe as dd
import pandas as pd

from corebehrt.functional.preparation.filter import filter_table_by_pids


class TestPrepDataUtilsFunctions(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        data = pd.DataFrame(
            {"subject_id": [1, 2, 3, 4, 5], "Value": ["A", "B", "C", "D", "E"]}
        )
        self.data_dd = dd.from_pandas(data, npartitions=1)

    def test_select_data_by_pids(self):
        selected_data = filter_table_by_pids(self.data_dd, [1, 2])
        self.assertEqual(len(selected_data), 2)
        self.assertTrue(set(selected_data.compute()["subject_id"]).issubset({1, 2}))


if __name__ == "__main__":
    unittest.main()
