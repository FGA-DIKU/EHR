import unittest
from datetime import datetime, timezone

import pandas as pd

from corebehrt.functional.utils.time import get_hours_since_epoch


class TestGetHoursSinceEpoch(unittest.TestCase):
    def setUp(self):
        # Known reference points
        self.epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.y2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)
        self.y2020 = datetime(2020, 1, 1, tzinfo=timezone.utc)

        # Expected hours since epoch for reference points
        self.hours_2000 = 262968  # (30 years * 365.25 days * 24 hours)
        self.hours_2020 = 438288  # (50 years * 365.25 days * 24 hours)

    def test_single_datetime(self):
        """Test with a single datetime object"""
        result = get_hours_since_epoch(self.y2000)
        self.assertAlmostEqual(result, self.hours_2000, delta=1)

    def test_pandas_series(self):
        """Test with a pandas Series of timestamps"""
        dates = pd.Series([self.epoch, self.y2000, self.y2020])
        result = get_hours_since_epoch(dates)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result.iloc[0], 0, delta=1)
        self.assertAlmostEqual(result.iloc[1], self.hours_2000, delta=1)
        self.assertAlmostEqual(result.iloc[2], self.hours_2020, delta=1)

    def test_empty_series(self):
        """Test with an empty pandas Series"""
        empty_series = pd.Series([], dtype="datetime64[ns]")
        result = get_hours_since_epoch(empty_series)
        self.assertTrue(result.empty)

    def test_timezone_handling(self):
        """Test timezone handling"""
        # Create same timestamp in different timezone representations
        naive_dt = datetime(2000, 1, 1)
        utc_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)

        # Both should give same result after conversion
        naive_result = get_hours_since_epoch(pd.Series([naive_dt]))
        utc_result = get_hours_since_epoch(pd.Series([utc_dt]))

        self.assertAlmostEqual(naive_result.iloc[0], utc_result.iloc[0], delta=1)

    def test_series_string_dates(self):
        """Test with string dates that need to be parsed"""
        date_strings = pd.Series(["1970-01-01", "2000-01-01", "2020-01-01"])
        result = get_hours_since_epoch(date_strings)

        self.assertAlmostEqual(result.iloc[0], 0, delta=1)
        self.assertAlmostEqual(result.iloc[1], self.hours_2000, delta=1)
        self.assertAlmostEqual(result.iloc[2], self.hours_2020, delta=1)

    def test_invalid_dates(self):
        """Test handling of invalid dates"""
        with self.assertRaises(ValueError):
            get_hours_since_epoch(pd.Series(["invalid_date"]))

    def test_future_dates(self):
        """Test dates far in the future"""
        future_date = datetime(2100, 1, 1, tzinfo=timezone.utc)
        result = get_hours_since_epoch(pd.Series([future_date]))
        expected_hours = (2100 - 1970) * 365.25 * 24
        self.assertAlmostEqual(result.iloc[0], expected_hours, delta=24)

    def test_pre_epoch_dates(self):
        """Test dates before epoch (should return negative hours)"""
        # One day before epoch
        pre_epoch_date = datetime(1969, 12, 31, tzinfo=timezone.utc)
        result = get_hours_since_epoch(pd.Series([pre_epoch_date]))
        self.assertAlmostEqual(result.iloc[0], -24, delta=1)

        # One year before epoch
        pre_epoch_year = datetime(1969, 1, 1, tzinfo=timezone.utc)
        result = get_hours_since_epoch(pd.Series([pre_epoch_year]))
        self.assertAlmostEqual(result.iloc[0], -365.25 * 24, delta=24)

    def test_mixed_timezone_series(self):
        """Test series with mixed timezone-aware and naive datetimes"""
        mixed_dates = pd.Series(
            [
                datetime(2000, 1, 1),  # naive
                datetime(2000, 1, 1, tzinfo=timezone.utc),  # UTC
            ]
        )
        result = get_hours_since_epoch(mixed_dates)

        # Both should give same result
        self.assertAlmostEqual(result.iloc[0], result.iloc[1], delta=1)

    def test_precision(self):
        """Test precision of hour calculations"""
        # Test with a specific time
        test_dt = datetime(2000, 1, 1, 12, 30, 0, tzinfo=timezone.utc)  # 12:30:00
        result = get_hours_since_epoch(pd.Series([test_dt]))

        # Should be 12.5 hours more than start of 2000
        self.assertAlmostEqual(result.iloc[0], self.hours_2000 + 12.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
