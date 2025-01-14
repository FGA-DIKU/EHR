import unittest
import pandas as pd
from corebehrt.classes.normalizer import ValuesNormalizer
import dask.dataframe as dd


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        # Create larger sample data as pandas DataFrames
        self.concepts_pd = pd.DataFrame(
            {
                "CONCEPT": ["A", "A", "A", "B", "B", "C", "C"],
                "RESULT": list(range(6)) + [None],
            }
        )
        self.concepts_pd["RESULT"] = self.concepts_pd["RESULT"].astype(str)
        # Convert pandas DataFrames to Dask DataFrames with npartitions=2
        self.concepts = dd.from_pandas(self.concepts_pd, npartitions=2)

        self.expected_concepts = pd.DataFrame(
            {
                "CONCEPT": ["A", "A", "A", "B", "B", "C", "C"],
                "RESULT": [
                    "VAL_0",
                    "VAL_50",
                    "VAL_100",
                    "VAL_0",
                    "VAL_100",
                    "VAL_UNIQUE",
                    None,
                ],
            }
        )
        self.expected_concepts = dd.from_pandas(
            self.expected_concepts, npartitions=2
        ).compute()

    def test_normalizer(self):
        concepts = (
            ValuesNormalizer.min_max_normalize_results(self.concepts)
            .compute()
            .sort_index()
        )

        # Check if all but the NaN value are as expected
        pd.testing.assert_frame_equal(
            concepts.iloc[:-1], self.expected_concepts.iloc[:-1], check_dtype=False
        )

        # Check that NaN value is kept NaN
        self.assertEqual(
            concepts.iloc[-1]["CONCEPT"], self.expected_concepts.iloc[-1]["CONCEPT"]
        )
        self.assertTrue(pd.isna(concepts.iloc[-1]["RESULT"]))
