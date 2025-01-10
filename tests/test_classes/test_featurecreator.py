import unittest
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
from corebehrt.classes.features import FeatureCreator
from corebehrt.functional.constants import CLS_TOKEN, SEP_TOKEN


class TestFeatureCreator(unittest.TestCase):
    def setUp(self):
        # Create larger sample data as pandas DataFrames
        self.concepts_pd = pd.DataFrame(
            {
                "PID": ["1", "1", "2", "2", "3", "3", "4", "4"],
                "TIMESTAMP": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 4),
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 6),
                    datetime(2020, 1, 7),
                    datetime(2020, 1, 8),
                ],
                "CONCEPT": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "ADMISSION_ID": [1, 1, 2, 2, 3, 3, 4, 4],
            }
        )

        self.patients_info_pd = pd.DataFrame(
            {
                "PID": ["1", "2", "3", "4"],
                "BIRTHDATE": [
                    datetime(1990, 1, 1),
                    datetime(1995, 1, 1),
                    datetime(2000, 1, 1),
                    datetime(2005, 1, 1),
                ],
                "DEATHDATE": [datetime(2021, 1, 1), pd.NaT, pd.NaT, pd.NaT],
                "GENDER": ["M", "F", "M", "F"],
            }
        )

        # Convert pandas DataFrames to Dask DataFrames with npartitions=2
        self.concepts = dd.from_pandas(self.concepts_pd, npartitions=2).astype({"PID": "string[pyarrow]", "ADMISSION_ID": "string[pyarrow]", "CONCEPT": "string[pyarrow]"})
        self.patients_info = dd.from_pandas(self.patients_info_pd, npartitions=2).astype({"PID": "string[pyarrow]"})

        self.feature_creator = FeatureCreator(
            origin_point=datetime(2020, 1, 1),
            background_vars=["GENDER"],
            cls_token=True,
            sep_token=True,
        )
        self.expected_segments = pd.Series(
            [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], name="segment"  # bg + death
        )

    def test_create_ages(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertIn("age", result_df.columns)
        self.assertTrue(all(result_df["age"] >= 0))

    def test_create_segments(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertIn("segment", result_df.columns)

        # Sort result_df to ensure consistent ordering
        result_df = result_df.sort_values(["PID", "abspos"]).reset_index(drop=True)

        # Ensure the lengths match
        self.assertEqual(len(result_df["segment"]), len(self.expected_segments))

        # Compare the segment values
        self.assertTrue(
            (result_df["segment"].values == self.expected_segments.values).all()
        )

    def test_create_abspos(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertIn("abspos", result_df.columns)
        # Additional testing can be done here, e.g., checking specific values

    def test_create_background(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertTrue(any(result_df["concept"].str.startswith("BG_GENDER")))
        self.assertEqual((result_df["concept"] == CLS_TOKEN).sum(), len(self.patients_info))

    def test_no_cls(self):
        feature_creator = FeatureCreator(
            origin_point=datetime(2020, 1, 1),
            background_vars=["GENDER"],
            cls_token=False,
            sep_token=True,
        )
        result = feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertFalse(any(result_df["concept"] == CLS_TOKEN))

    def test_no_sep(self):
        feature_creator = FeatureCreator(
            origin_point=datetime(2020, 1, 1),
            background_vars=["GENDER"],
            cls_token=True,
            sep_token=False,
        )
        result = feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertFalse(any(result_df["concept"] == SEP_TOKEN))

    def test_sep_token(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()

        self.assertTrue((result_df.groupby(["PID", "segment"])["concept"].tail(1) == SEP_TOKEN).all())

    def test_cls_token(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()

        self.assertTrue((result_df.groupby("PID")["concept"].head(1) == CLS_TOKEN).all())

    def test_create_death(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertTrue(any(result_df["concept"] == "Death"))
        self.assertEqual(
            sum(result_df["concept"] == "Death"), 1
        )  # Only one patient has death info

    def test_all_features_created(self):
        result = self.feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        expected_columns = {"PID", "concept", "age", "segment", "abspos"}
        self.assertTrue(expected_columns.issubset(set(result_df.columns)))

    def test_no_background_vars(self):
        feature_creator = FeatureCreator(background_vars=[])
        result = feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertFalse(any(result_df["concept"].str.startswith("BG_")))

    def test_origin_point_as_dict(self):
        feature_creator = FeatureCreator(
            origin_point={"year": 2020, "month": 1, "day": 1}
        )
        result = feature_creator(self.patients_info, self.concepts)
        result_df = result.compute()
        self.assertIn("abspos", result_df.columns)

    def test_invalid_background_var(self):
        feature_creator = FeatureCreator(background_vars=["INVALID_COLUMN"])
        with self.assertRaises(ValueError):
            feature_creator(self.patients_info, self.concepts).compute()

    def test_missing_required_columns(self):
        concepts_missing_column = self.concepts.drop("ADMISSION_ID", axis=1)
        with self.assertRaises(ValueError):
            self.feature_creator(self.patients_info, concepts_missing_column).compute()


if __name__ == "__main__":
    unittest.main()
