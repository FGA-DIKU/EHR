import unittest
import pandas as pd

from corebehrt.functional.preparation.utils import get_background_length_pd
from corebehrt.constants.data import CONCEPT_COL, PID_COL


class TestGetBackgroundLengthPD(unittest.TestCase):
    def setUp(self):
        # Test data setup that will be used by multiple tests
        self.data = {
            PID_COL: [1, 1, 1, 2, 2, 2],
            CONCEPT_COL: [1, 2, 3, 1, 2, 3],
        }

        self.vocabulary = {
            "BG_1": 1,
            "BG_2": 2,
            "normal_concept": 3,
            "[CLS]": 4,
            "[SEP]": 5,
        }

    def test_normal_case(self):
        # Create Pandas DataFrame from test data
        df = pd.DataFrame(self.data).set_index(PID_COL)
        # Test the function
        result = get_background_length_pd(df, self.vocabulary)

        # We expect 4: 2 background concepts + 2 special tokens ([CLS] and [SEP])
        self.assertEqual(result, 4)

    def test_empty_dataframe(self):
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=[PID_COL, CONCEPT_COL]).set_index(PID_COL)

        result = get_background_length_pd(empty_df, self.vocabulary)

        # Should return 2 for empty DataFrame (just [CLS] and [SEP] tokens)
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
