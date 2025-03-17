import unittest
from datetime import datetime

import pandas as pd
from pandas import NaT

from corebehrt.constants.data import CONCEPT_COL, PID_COL, TIMESTAMP_COL
from corebehrt.modules.features.features import FeatureCreator


class TestFeatureCreator(unittest.TestCase):
    def setUp(self):
        # Create larger sample data as pandas DataFrames
        self.concepts = pd.DataFrame(
            {
                PID_COL: [
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                ],
                TIMESTAMP_COL: [
                    datetime(1998, 1, 1),
                    NaT,
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2021, 1, 2),
                    datetime(1960, 1, 3),
                    NaT,
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 10),
                    datetime(2002, 1, 5),
                    NaT,
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 6),
                    datetime(1970, 1, 6),
                    NaT,
                    datetime(2020, 1, 7),
                    datetime(2020, 2, 8),
                ],
                CONCEPT_COL: [
                    "DOB",
                    "GENDER//M",
                    "A",
                    "B",
                    "DOD",
                    "DOB",
                    "GENDER//F",
                    "C",
                    "D",
                    "DOB",
                    "GENDER//F",
                    "E",
                    "F",
                    "DOB",
                    "GENDER//M",
                    "G",
                    "H",
                ],
            }
        )

        self.feature_creator = FeatureCreator()
        self.expected_segments = pd.Series(
            [0, 0, 1, 1, 2, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 2],
            name="segment",  # bg + death
        )

    def test_create_ages(self):
        result, _ = self.feature_creator(self.concepts)
        self.assertIn("age", result.columns)
        self.assertTrue(all(result["age"] >= 0))

    def test_create_segments(self):
        result, _ = self.feature_creator(self.concepts)
        self.assertIn("segment", result.columns)

        # Sort result_df to ensure consistent ordering
        result = result.sort_values([PID_COL, "abspos"]).reset_index(drop=True)

        # Ensure the lengths match
        self.assertEqual(len(result["segment"]), len(self.expected_segments))

    def test_create_background(self):
        result, _ = self.feature_creator(self.concepts)
        self.assertTrue(any(result[CONCEPT_COL].str.startswith("BG_GENDER")))
        # Compare the segment values
        self.assertTrue(
            (result["segment"].values == self.expected_segments.values).all()
        )

    def test_create_abspos(self):
        result, _ = self.feature_creator(self.concepts)
        self.assertIn("abspos", result.columns)
        # Additional testing can be done here, e.g., checking specific values

    def test_death(self):
        result, _ = self.feature_creator(self.concepts)
        self.assertTrue(any(result[CONCEPT_COL] == "DOD"))
        self.assertEqual(
            sum(result[CONCEPT_COL] == "DOD"), 1
        )  # Only one patient has death info

    def test_all_features_created(self):
        result, _ = self.feature_creator(self.concepts)
        expected_columns = {PID_COL, "age", "segment", "abspos", CONCEPT_COL}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

    def test_missing_required_columns(self):
        concepts_missing_column = self.concepts.drop(CONCEPT_COL, axis=1)
        with self.assertRaises(ValueError):
            self.feature_creator(concepts_missing_column)


if __name__ == "__main__":
    unittest.main()
