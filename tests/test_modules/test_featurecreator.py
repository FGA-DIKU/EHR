import unittest
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
from corebehrt.modules.features.features import FeatureCreator
import numpy as np
from pandas import NaT


class TestFeatureCreator(unittest.TestCase):
    def setUp(self):
        # Create larger sample data as pandas DataFrames
        self.concepts = pd.DataFrame(
            {
                "subject_id": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "2",
                    "2",
                    "2",
                    "2",
                    "3",
                    "3",
                    "3",
                    "3",
                    "4",
                    "4",
                    "4",
                    "4",
                ],
                "time": [
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
                "code": [
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

        self.feature_creator = FeatureCreator(
            origin_point=datetime(2020, 1, 1),
        )
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
        result = result.sort_values(["subject_id", "abspos"]).reset_index(drop=True)

        # Ensure the lengths match
        self.assertEqual(len(result["segment"]), len(self.expected_segments))

    def test_create_background(self):
        result, _ = self.feature_creator(self.concepts)
        self.assertTrue(any(result["code"].str.startswith("BG_GENDER")))
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
        self.assertTrue(any(result["code"] == "DOD"))
        self.assertEqual(
            sum(result["code"] == "DOD"), 1
        )  # Only one patient has death info

    def test_all_features_created(self):
        result, _ = self.feature_creator(self.concepts)
        expected_columns = {"subject_id", "age", "segment", "abspos", "code"}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

    def test_origin_point_as_dict(self):
        feature_creator = FeatureCreator(
            origin_point={"year": 2020, "month": 1, "day": 1}
        )
        result, _ = feature_creator(self.concepts)
        self.assertIn("abspos", result.columns)

    def test_missing_required_columns(self):
        concepts_missing_column = self.concepts.drop("code", axis=1)
        with self.assertRaises(ValueError):
            self.feature_creator(concepts_missing_column)


if __name__ == "__main__":
    unittest.main()
