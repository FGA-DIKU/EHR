import unittest
import dask.dataframe as dd
import pandas as pd
from datetime import datetime

from corebehrt.functional.tokenize import (
    add_separator_token,
    add_cls_token,
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.features = dd.from_pandas(
            pd.DataFrame(
                {
                    "PID": map(str, [1, 1, 1, 2, 2, 3, 3, 3, 3]),
                    "concept": [1, 2, 2, 2, 3, 4, 2, 2, 5],
                    "age": [33.1, 33.2, 33.3, 21.9, 22.0, 36.1, 36.7, 38.1, 38.2],
                    "segment": [0, 1, 1, 0, 1, 0, 1, 2, 2],
                    "abspos": map(float, [1, 2, 3, 1, 2, 1, 2, 3, 4]),
                }
            ).set_index("PID"),
        ).reset_index()

    def test_add_sep(self):
        expected_concept = [
            1,
            "[SEP]",
            2,
            2,
            "[SEP]",
            2,
            "[SEP]",
            3,
            "[SEP]",
            4,
            "[SEP]",
            2,
            "[SEP]",
            2,
            5,
            "[SEP]",
        ]
        expected_segment = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2]

        result = add_separator_token(self.features).compute()
        self.assertEqual(result.concept.tolist(), expected_concept)
        self.assertEqual(result.segment.tolist(), expected_segment)

    def test_add_cls(self):
        expected_concept = [
            "[CLS]",
            1,
            2,
            2,
            "[CLS]",
            2,
            3,
            "[CLS]",
            4,
            2,
            2,
            5,
        ]
        expected_segment = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 2, 2]

        result = add_cls_token(self.features).compute()
        self.assertEqual(result.concept.tolist(), expected_concept)
        self.assertEqual(result.segment.tolist(), expected_segment)