import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from corebehrt.functional.preparation.truncate import truncate_patient_df
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL, CONCEPT_COL


class TestTruncatePatientDf(unittest.TestCase):
    def setUp(self):
        """
        We'll create two dataframes:
          1. self.test_data: no SEP tokens, purely "token_i".
          2. self.test_data_with_sep: includes a SEP token exactly
             at position #6 (so that it's at the boundary if we do
             max_len=6 with background_length=2, etc.).
        """
        self.test_data = pd.DataFrame(
            {
                "abspos": range(10),
                CONCEPT_COL: [f"token_{i}" for i in range(10)],
                "other_col": [f"value_{i}" for i in range(10)],
            }
        )

        # Place "SEP" precisely at index 6 so that with tail_length=4,
        # the boundary row is index=6 => boundary_token=='SEP'.
        self.test_data_with_sep = pd.DataFrame(
            {
                "abspos": range(10),
                CONCEPT_COL: [
                    "token_0",
                    "token_1",
                    "token_2",
                    "token_3",
                    "token_4",
                    "token_5",
                    "SEP",
                    "token_7",
                    "token_8",
                    "token_9",
                ],
                "other_col": [f"value_{i}" for i in range(10)],
            }
        )

    def test_no_truncation_needed(self):
        """If the total length <= max_len, return the entire DataFrame."""
        result = truncate_patient_df(
            self.test_data.copy(), max_len=15, background_length=3, sep_token="SEP"
        )
        assert_frame_equal(
            result.reset_index(drop=True), self.test_data.reset_index(drop=True)
        )

    def test_basic_truncation(self):
        """
        Test standard truncation: keep 'background_length' rows from the front,
        and fill up the remainder of 'max_len' from the tail.
        No boundary SEP triggers here because 'sep_token' doesn't appear.
        """
        result = truncate_patient_df(
            self.test_data.copy(), max_len=6, background_length=2, sep_token="SEP"
        )

        # Expect 2 from the front, then 4 from the tail = total 6.
        expected = pd.concat(
            [
                self.test_data.iloc[:2],
                self.test_data.iloc[-4:],
            ]
        )
        # Compare with index reset, to ensure equality ignoring row labels
        assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )
        self.assertEqual(len(result), 6)

    def test_truncation_with_sep_token(self):
        """
        If the boundary item is SEP, reduce tail_length by 1.
        With max_len=6 and background_length=2,
          - tail_length starts as 4
          - boundary_idx = 10 - 4 = 6
          - concept at index=6 => 'SEP' => tail_length => 3
          - final result => 2 front + 3 tail = 5 total
        """
        result = truncate_patient_df(
            self.test_data_with_sep.copy(),
            max_len=6,
            background_length=2,
            sep_token="SEP",
        )
        # 2 front => indices [0,1], tail => last 3 => indices [7,8,9]
        expected = pd.concat(
            [self.test_data_with_sep.iloc[:2], self.test_data_with_sep.iloc[-3:]]
        )

        self.assertEqual(
            len(result), 5, "Should have truncated from 6 to 5 due to SEP boundary"
        )
        assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_all_background_no_tail(self):
        """
        If max_len == background_length, we only keep the front portion;
        there's no tail portion left.
        """
        result = truncate_patient_df(
            self.test_data.copy(), max_len=3, background_length=3, sep_token="SEP"
        )
        expected = self.test_data.iloc[:3]
        assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
