import unittest
from datetime import datetime

import pandas as pd

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.modules.cohort_handling.index_dates import IndexDateHandler


class TestIndexDateHandler(unittest.TestCase):
    def setUp(self):
        """Set up sample data for testing IndexDateHandler."""
        self.pids = {"p1", "p2", "p3"}

        # For get_index_timestamps_for_exposed tests:
        # Each patient has at most ONE exposure row:
        self.exposures = pd.DataFrame(
            {
                PID_COL: ["p1", "p2", "p4"],  # p4 not in self.pids
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2021-01-01 08:00:00",  # p1
                        "2021-01-02 10:00:00",  # p2
                        "2021-01-03 00:00:00",  # p4 (irrelevant to self.pids)
                    ]
                ),
            }
        )

        # For draw_index_dates_for_unexposed tests:
        self.censoring_timestamps = pd.Series(
            {
                "p1": pd.Timestamp("2021-05-01 00:00:00"),
                "p2": pd.Timestamp("2021-05-02 00:00:00"),
                # p3 missing, p4 missing
            },
            name=TIMESTAMP_COL,
        )
        self.censoring_timestamps.index.name = PID_COL

        self.all_pids = ["p1", "p2", "p3", "p4"]

        # For determine_index_dates tests:
        self.patients_info = pd.DataFrame({PID_COL: ["p1", "p2", "p3"]})
        # We'll reuse self.exposures in the 'relative' mode tests

    # --------------------------------------------------------
    # create_timestamp_series
    # --------------------------------------------------------
    def test_create_timestamp_series(self):
        """Check that create_timestamp_series returns a Series with the same length as pids, all the same datetime."""
        ts_series = IndexDateHandler.create_timestamp_series(
            self.pids, datetime(2020, 6, 15)
        )

        self.assertEqual(len(ts_series), len(self.pids))
        # Each PID should map to the same datetime(2020,6,15).
        for pid in self.pids:
            self.assertIn(pid, ts_series.index)
            self.assertEqual(ts_series[pid], datetime(2020, 6, 15))

        self.assertEqual(ts_series.index.name, PID_COL, "Index name should be PID_COL.")
        self.assertEqual(
            ts_series.name, TIMESTAMP_COL, "Series name should be TIMESTAMP_COL."
        )

    # --------------------------------------------------------
    # get_index_timestamps_for_exposed
    # --------------------------------------------------------
    def test_get_index_timestamps_for_exposed_basic(self):
        """
        Since each patient has at most one exposure,
        confirm that the returned Series has an entry for each matched PID, offset by hours.
        """
        # pids = {p1, p2, p3}; exposures has p1, p2, p4
        # => p1, p2 are valid; p3 not in exposures => no row
        offset_hours = 5
        result = IndexDateHandler.get_index_timestamps_for_exposed(
            self.pids, offset_hours, self.exposures
        )
        # We expect a Series indexed by p1, p2 only, each plus 5 hours
        self.assertEqual(len(result), 2, "p1 and p2 => 2 rows total.")
        self.assertIn("p1", result.index)
        self.assertIn("p2", result.index)
        # p3 => no entry => not in index
        self.assertNotIn("p3", result.index)
        # p4 => not in self.pids => filtered out

        # Check offset
        p1_expected = pd.Timestamp("2021-01-01 08:00:00") + pd.Timedelta(hours=5)
        self.assertEqual(result["p1"], p1_expected)
        p2_expected = pd.Timestamp("2021-01-02 10:00:00") + pd.Timedelta(hours=5)
        self.assertEqual(result["p2"], p2_expected)

        # Validate metadata
        self.assertEqual(result.index.name, PID_COL)
        self.assertEqual(result.name, TIMESTAMP_COL)

    def test_get_index_timestamps_for_exposed_no_matches(self):
        """If pids don't overlap with exposures at all, result is empty."""
        pids_no_overlap = {"x1", "x2"}
        result = IndexDateHandler.get_index_timestamps_for_exposed(
            pids_no_overlap, 2, self.exposures
        )
        self.assertTrue(result.empty, "No matching PIDs => empty result.")
        self.assertEqual(result.index.name, PID_COL)

    # --------------------------------------------------------
    # draw_index_dates_for_unexposed
    # --------------------------------------------------------
    def test_draw_index_dates_for_unexposed_missing(self):
        """Missing PIDs get random choice from existing timestamps."""
        # p3, p4 are missing in self.censoring_timestamps => must be assigned random picks
        combined = IndexDateHandler.draw_index_dates_for_unexposed(
            data_pids=self.all_pids, censoring_timestamps=self.censoring_timestamps
        )
        self.assertEqual(len(combined), 4)
        self.assertIn("p3", combined.index)
        self.assertIn("p4", combined.index)

        # The newly added p3/p4 should be randomly chosen from [p1, p2]'s values
        possible = [
            pd.Timestamp("2021-05-01 00:00:00"),
            pd.Timestamp("2021-05-02 00:00:00"),
        ]
        self.assertIn(combined["p3"], possible)
        self.assertIn(combined["p4"], possible)

        # Check index name
        self.assertEqual(combined.index.name, PID_COL)

    def test_draw_index_dates_for_unexposed_no_missing(self):
        """If no PIDs are missing, we just return the original censoring_timestamps unchanged."""
        all_in_series = pd.Series(
            {
                "p1": pd.Timestamp("2021-05-01"),
                "p2": pd.Timestamp("2021-05-02"),
                "p3": pd.Timestamp("2021-05-03"),
            }
        )
        all_in_series.index.name = PID_COL

        combined = IndexDateHandler.draw_index_dates_for_unexposed(
            censoring_timestamps=all_in_series, data_pids=["p1", "p2", "p3"]
        )

        # Compare
        pd.testing.assert_series_equal(
            combined.sort_index(), all_in_series.sort_index(), check_names=True
        )

    # --------------------------------------------------------
    # determine_index_dates
    # --------------------------------------------------------
    def test_determine_index_dates_absolute(self):
        """Check absolute mode sets the same datetime for all pids."""
        # Using a Config-like dict; you can pass a normal dict if your code permits
        result = IndexDateHandler.determine_index_dates(
            patients_info=self.patients_info,
            index_date_mode="absolute",
            absolute_timestamp={"year": 2022, "month": 1, "day": 15},
        )
        # We have p1, p2, p3 => each should have 2022-01-15
        self.assertEqual(len(result), 3)
        for pid in ["p1", "p2", "p3"]:
            self.assertEqual(result[pid], pd.Timestamp("2022-01-15"))
        self.assertEqual(result.index.name, PID_COL)

    def test_determine_index_dates_relative(self):
        """
        Check relative mode:
          - We get an offset for exposures.
          - Then we fill in missing pids with random picks.
        """
        # p1 => exposure=2021-01-01 08:00 => +5 => 2021-01-01 13:00
        # p2 => exposure=2021-01-02 10:00 => +5 => 2021-01-02 15:00
        # p3 => no exposure => random pick from p1,p2 => either 2021-01-01 13:00 or 2021-01-02 15:00
        result = IndexDateHandler.determine_index_dates(
            patients_info=self.patients_info,
            index_date_mode="relative",
            n_hours_from_exposure=5,
            exposures=self.exposures,
        )

        self.assertIn("p1", result.index)
        self.assertIn("p2", result.index)
        self.assertIn("p3", result.index)
        self.assertEqual(len(result), 3, "p1, p2, p3 => total 3")

        self.assertEqual(result["p1"], pd.Timestamp("2021-01-01 13:00:00"))
        self.assertEqual(result["p2"], pd.Timestamp("2021-01-02 15:00:00"))

        # p3 should be randomly chosen from the existing [p1, p2] values
        possible = [
            pd.Timestamp("2021-01-01 13:00:00"),
            pd.Timestamp("2021-01-02 15:00:00"),
        ]
        self.assertIn(result["p3"], possible)

    def test_determine_index_dates_unsupported_mode(self):
        """Passing an invalid index_date_mode should raise ValueError."""
        with self.assertRaises(ValueError):
            IndexDateHandler.determine_index_dates(
                patients_info=self.patients_info,
                index_date_mode="invalid_mode",
            )


if __name__ == "__main__":
    unittest.main()
