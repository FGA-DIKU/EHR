import unittest
import pandas as pd
from corebehrt.constants.data import CONCEPT_COL, TIMESTAMP_COL, PID_COL, VALUE_COL
from corebehrt.main.helper.create_data import handle_aggregations
import numpy as np

class TestHandleAggregations(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = {
            PID_COL: [1, 1, 1, 2, 2, 2],
            TIMESTAMP_COL: pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-02 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:05:00",
                    "2024-01-01 00:10:00",
                ]
            ),
            CONCEPT_COL: ["A", "A", "A", "B", "B", "B"],
            VALUE_COL: [10, 20, 30, 40, 50, 60],
        }

        self.data_w_prefix = {
            PID_COL: [1, 1, 1, 1, 1, 1],
            TIMESTAMP_COL: pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00",
                ]
            ),
            CONCEPT_COL: ["LAB_A", "LAB_A", "LAB_A", "D_B", "D_B", "D_B"],
            VALUE_COL: [10, 20, 30, np.nan, np.nan, np.nan],
        }

        self.df = pd.DataFrame(self.data)
        self.df_w_prefix = pd.DataFrame(self.data_w_prefix)

    def test_aggregation_without_window(self):
        result = handle_aggregations(self.df, agg_type="first")
        expected_data = {
            PID_COL: [1, 1, 2, 2, 2],
            TIMESTAMP_COL: pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-02 00:00:00",
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:05:00",
                    "2024-01-01 00:10:00",
                ]
            ),
            CONCEPT_COL: ["A", "A", "B", "B", "B"],
            VALUE_COL: [10, 30, 40, 50, 60],
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_aggregation_with_window(self):
        result = handle_aggregations(self.df, agg_type="first", agg_window=25)
        print(result)
        expected_data = {
            PID_COL: [1, 2, 2],
            CONCEPT_COL: ["A", "B", "B"],
            TIMESTAMP_COL: pd.to_datetime(
                ["2023-01-01 00:00:00", "2023-01-01 00:00:00", "2024-01-01 00:10:00"]
            ),
            VALUE_COL: [10, 40, 60],
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_no_aggregation(self):
        result = handle_aggregations(self.df)
        pd.testing.assert_frame_equal(result, self.df)

    def test_aggregation_with_prefix(self):
        result_w_regex = handle_aggregations(
            self.df_w_prefix, agg_type="first", regex="^LAB.*"
        )
        print(result_w_regex)
        result_wo_regex = handle_aggregations(
            self.df_w_prefix, agg_type="first", regex=".*"
        )
        self.assertEqual(len(result_w_regex), 4)
        self.assertEqual(len(result_wo_regex), 2)


if __name__ == "__main__":
    unittest.main()
