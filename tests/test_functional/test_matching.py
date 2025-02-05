import unittest
import pandas as pd
from corebehrt.functional.cohort_handling.matching import (
    get_col_booleans,
    startswith_match,
    contains_match,
)


class TestMatchingFunctions(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame(
            {
                "CONCEPT": [
                    "COVID",
                    "COVID_TEST",
                    "POSITIVE",
                    "NEGATIVE",
                    "TEST_COVID",
                    "OTHER",
                ],
                "VALUE": [
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                    "POSITIVE",
                    "NEGATIVE",
                ],
            }
        )

    def test_startswith_match_case_sensitive(self):
        # Test startswith_match with case-sensitive matching
        result = startswith_match(
            self.df, "CONCEPT", ["COVID", "TEST"], case_sensitive=True
        )
        expected = pd.Series([True, True, False, False, True, False], name="CONCEPT")
        pd.testing.assert_series_equal(result, expected)

    def test_startswith_match_case_insensitive(self):
        # Test startswith_match with case-insensitive matching
        result = startswith_match(
            self.df, "CONCEPT", ["covid", "test"], case_sensitive=False
        )
        expected = pd.Series([True, True, False, False, True, False], name="CONCEPT")
        pd.testing.assert_series_equal(result, expected)

    def test_contains_match_case_sensitive(self):
        # Test contains_match with case-sensitive matching
        result = contains_match(
            self.df, "CONCEPT", ["COVID", "TEST"], case_sensitive=True
        )
        expected = pd.Series([True, True, False, False, True, False], name="CONCEPT")
        pd.testing.assert_series_equal(result, expected)

    def test_contains_match_case_insensitive(self):
        # Test contains_match with case-insensitive matching
        result = contains_match(
            self.df, "CONCEPT", ["covid", "test"], case_sensitive=False
        )
        expected = pd.Series([True, True, False, False, True, False], name="CONCEPT")
        pd.testing.assert_series_equal(result, expected)

    def test_get_col_booleans_startswith(self):
        # Test get_col_booleans with startswith matching
        result = get_col_booleans(
            self.df,
            ["CONCEPT", "VALUE"],
            [["COVID"], ["POSITIVE"]],
            match_how="startswith",
        )

        expected = [
            pd.Series([True, True, False, False, False, False], name="CONCEPT"),
            pd.Series([True, False, True, False, True, False], name="VALUE"),
        ]
        for res, exp in zip(result, expected):
            pd.testing.assert_series_equal(res, exp)

    def test_get_col_booleans_contains(self):
        # Test get_col_booleans with contains matching
        result = get_col_booleans(
            self.df,
            ["CONCEPT", "VALUE"],
            [["OVID"], ["POSITIVE"]],
            match_how="contains",
        )
        expected = [
            pd.Series([True, True, False, False, True, False], name="CONCEPT"),
            pd.Series([True, False, True, False, True, False], name="VALUE"),
        ]
        for res, exp in zip(result, expected):
            pd.testing.assert_series_equal(res, exp)

    def test_get_col_booleans_invalid_match_how(self):
        # Test get_col_booleans with an invalid match_how argument
        with self.assertRaises(ValueError):
            get_col_booleans(self.df, ["CONCEPT"], [["COVID"]], match_how="invalid")


if __name__ == "__main__":
    unittest.main()
