import unittest
import pandas as pd
from corebehrt.functional.normalize import min_max_normalize


class TestNormalize(unittest.TestCase):
    def test_min_max_normalize(self):
        # Create Series
        x = pd.Series([1, 2, 3, 4, 5, 6])

        normalized = min_max_normalize(x)

        expected = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        pd.testing.assert_series_equal(normalized, expected)

    def test_min_max_normalize_single_value(self):
        # Create Series with a single value
        x = pd.Series([1, 1, 1, 1, 1, 1])

        normalized = min_max_normalize(x)

        expected = pd.Series(["UNIQUE"] * 6)
        pd.testing.assert_series_equal(normalized, expected)

    def test_min_max_normalize_nan(self):
        # Create Series with NaN values
        x = pd.Series([1, 2, 3, 4, 5, 6, None])

        normalized = min_max_normalize(x)

        expected = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        pd.testing.assert_series_equal(normalized, expected)