import unittest

import numpy as np
import pandas as pd

from corebehrt.modules.features.excluder import Excluder


class TestExcluder(unittest.TestCase):
    def setUp(self):
        self.excluder = Excluder(min_age=0, max_age=100)

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


if __name__ == "__main__":
    unittest.main()
