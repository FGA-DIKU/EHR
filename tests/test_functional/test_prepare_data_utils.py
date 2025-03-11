import unittest

import pandas as pd

from corebehrt.functional.preparation.filter import filter_table_by_pids
from corebehrt.constants.data import PID_COL, VALUE_COL


class TestPrepDataUtilsFunctions(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {PID_COL: [1, 2, 3, 4, 5], VALUE_COL: ["A", "B", "C", "D", "E"]}
        )

    def test_select_data_by_pids(self):
        selected_data = filter_table_by_pids(self.data, [1, 2])
        self.assertEqual(len(selected_data), 2)
        self.assertTrue(set(selected_data[PID_COL]).issubset({1, 2}))


if __name__ == "__main__":
    unittest.main()
