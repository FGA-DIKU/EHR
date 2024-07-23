import unittest
import pandas as pd
from datetime import datetime
from corebehrt.classes.features import (
    FeatureCreator,
)  # Replace 'your_module' with the actual module name


class TestFeatureCreator(unittest.TestCase):
    def setUp(self):
        self.concepts = pd.DataFrame(
            {
                "PID": ["1", "1", "2", "2"],
                "TIMESTAMP": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 4),
                ],
                "CONCEPT": ["A", "B", "C", "D"],
                "ADMISSION_ID": [1, 1, 2, 2],
            }
        )

        self.patients_info = pd.DataFrame(
            {
                "PID": ["1", "2"],
                "BIRTHDATE": [datetime(1990, 1, 1), datetime(1995, 1, 1)],
                "DEATHDATE": [datetime(2021, 1, 1), pd.NaT],
                "GENDER": ["M", "F"],
            }
        )

        self.feature_creator = FeatureCreator(
            # ages=True,
            # segment=True,
            origin_point=datetime(2020, 1, 1),
            background_vars=["GENDER"],
        )

    def test_create_ages(self):
        result = self.feature_creator(self.concepts, self.patients_info)
        self.assertIn("age", result.columns)
        self.assertTrue(all(result["age"] >= 0))

    def test_create_segments(self):
        result = self.feature_creator(self.concepts, self.patients_info)
        self.assertIn("segment", result.columns)
        self.assertEqual(set(result["segment"]), {0, 1})

    def test_create_abspos(self):
        result = self.feature_creator(self.concepts, self.patients_info)
        self.assertIn("abspos", result.columns)
        # additional testing required?

    def test_create_background(self):
        result = self.feature_creator(self.concepts, self.patients_info)
        self.assertTrue(any(result["concept"].str.startswith("BG_GENDER")))

    def test_create_death(self):
        result = self.feature_creator(self.concepts, self.patients_info)
        self.assertTrue(any(result["concept"] == "Death"))
        self.assertEqual(
            sum(result["concept"] == "Death"), 1
        )  # Only one patient has death info

    def test_all_features_created(self):
        result = self.feature_creator(self.concepts, self.patients_info)
        expected_columns = {"PID", "TIMESTAMP", "concept", "age", "segment", "abspos"}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

    def test_no_background_vars(self):
        feature_creator = FeatureCreator(background_vars=[])
        result = feature_creator(self.concepts, self.patients_info)
        self.assertFalse(any(result["concept"].str.startswith("BG_")))

    def test_origin_point_as_dict(self):
        feature_creator = FeatureCreator(
            origin_point={"year": 2020, "month": 1, "day": 1}
        )
        result = feature_creator(self.concepts, self.patients_info)
        self.assertIn("abspos", result.columns)

    def test_invalid_background_var(self):
        feature_creator = FeatureCreator(background_vars=["INVALID_COLUMN"])
        with self.assertRaises(KeyError):
            feature_creator(self.concepts, self.patients_info)

    def test_missing_required_columns(self):
        concepts_missing_column = self.concepts.drop("ADMISSION_ID", axis=1)
        with self.assertRaises(KeyError):
            self.feature_creator(concepts_missing_column, self.patients_info)


if __name__ == "__main__":
    unittest.main()
