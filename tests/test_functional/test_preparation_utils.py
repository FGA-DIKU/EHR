import unittest
import dask.dataframe as dd
import pandas as pd

from corebehrt.functional.preparation.utils import get_background_length_dd


class TestGetBackgroundLengthDD(unittest.TestCase):
    def setUp(self):
        # Test data setup that will be used by multiple tests
        self.data = {
            "PID": [
                "patient1",
                "patient1",
                "patient1",
                "patient2",
                "patient2",
                "patient2",
            ],
            "concept": [1, 2, 3, 1, 2, 3],
        }

        self.vocabulary = {
            "BG_1": 1,
            "BG_2": 2,
            "normal_concept": 3,
            "[CLS]": 4,
            "[SEP]": 5,
        }

    def test_normal_case(self):
        # Create Dask DataFrame from test data
        df = pd.DataFrame(self.data).set_index("PID")
        ddf = dd.from_pandas(df, npartitions=2)
        # Test the function
        result = get_background_length_dd(ddf, self.vocabulary)

        # We expect 4: 2 background concepts + 2 special tokens ([CLS] and [SEP])
        self.assertEqual(result, 4)

    def test_empty_dataframe(self):
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=["PID", "concept"]).set_index("PID")
        empty_ddf = dd.from_pandas(empty_df, npartitions=1)

        result = get_background_length_dd(empty_ddf, self.vocabulary)

        # Should return 2 for empty DataFrame (just [CLS] and [SEP] tokens)
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
