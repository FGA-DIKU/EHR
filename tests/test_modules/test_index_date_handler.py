import unittest
from datetime import datetime

import pandas as pd

from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
from corebehrt.modules.cohort_handling.index_dates import IndexDateHandler


class TestIndexDateHandler(unittest.TestCase):
    def setUp(self):
        """Set up sample data for testing IndexDateHandler."""
        self.pids = {1, 2, 3}

        # For get_index_timestamps_for_exposed tests:
        # Each patient has at most ONE exposure row:
        self.exposures = pd.DataFrame(
            {
                PID_COL: [1, 2, 4],  # p4 not in self.pids
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
                1: pd.Timestamp("2021-05-01 00:00:00"),
                2: pd.Timestamp("2021-05-02 00:00:00"),
                # p3 missing, p4 missing
            },
            name=TIMESTAMP_COL,
        )
        self.censoring_timestamps.index.name = PID_COL

        self.all_pids = [1, 2, 3, 4]

        # For determine_index_dates tests:
        self.patients_info = pd.DataFrame({PID_COL: [1, 2, 3]})
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
        self.assertIn(1, result.index)
        self.assertIn(2, result.index)
        # p3 => no entry => not in index
        self.assertNotIn(3, result.index)
        # p4 => not in self.pids => filtered out

        # Check offset
        p1_expected = pd.Timestamp("2021-01-01 08:00:00") + pd.Timedelta(hours=5)
        self.assertEqual(result[1], p1_expected)
        p2_expected = pd.Timestamp("2021-01-02 10:00:00") + pd.Timedelta(hours=5)
        self.assertEqual(result[2], p2_expected)

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
        self.assertIn(3, combined.index)
        self.assertIn(4, combined.index)

        # The newly added p3/p4 should be randomly chosen from [p1, p2]'s values
        possible = [
            pd.Timestamp("2021-05-01 00:00:00"),
            pd.Timestamp("2021-05-02 00:00:00"),
        ]
        self.assertIn(combined[3], possible)
        self.assertIn(combined[4], possible)

        # Check index name
        self.assertEqual(combined.index.name, PID_COL)

    def test_draw_index_dates_for_unexposed_no_missing(self):
        """If no PIDs are missing, we just return the original censoring_timestamps unchanged."""
        all_in_series = pd.Series(
            {
                1: pd.Timestamp("2021-05-01"),
                2: pd.Timestamp("2021-05-02"),
                3: pd.Timestamp("2021-05-03"),
            }
        )
        all_in_series.index.name = PID_COL

        combined = IndexDateHandler.draw_index_dates_for_unexposed(
            censoring_timestamps=all_in_series, data_pids=[1, 2, 3]
        )

        # Compare
        pd.testing.assert_series_equal(
            combined.sort_index(), all_in_series.sort_index(), check_names=True
        )

    def test_draw_index_dates_for_unexposed_with_maximum_dates(self):
        """Test that maximum_index_dates constraint is properly applied."""
        # Create test data with some dates that exceed maximum
        censoring_timestamps = pd.Series(
            {
                1: pd.Timestamp("2021-05-01"),
                2: pd.Timestamp("2021-05-02"),
            }
        )
        censoring_timestamps.index.name = PID_COL

        # Create maximum index dates that some existing timestamps exceed
        maximum_index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                TIMESTAMP_COL: [
                    pd.Timestamp("2021-05-01 12:00"),  # p1's timestamp is before this
                    pd.Timestamp("2021-05-01 12:00"),  # p2's timestamp is after this
                    pd.Timestamp("2021-05-01 12:00"),  # p3 will be drawn
                    pd.Timestamp("2021-05-01 12:00"),  # p4 will be drawn
                ],
            }
        )

        # p1 should be redrawn (original is before max), p2 should stay (within range)
        # p3, p4 should be drawn from valid range
        result = IndexDateHandler.draw_index_dates_for_unexposed(
            data_pids=[1, 2, 3, 4],
            censoring_timestamps=censoring_timestamps,
            maximum_index_dates=maximum_index_dates,
        )

        self.assertEqual(len(result), 4)

        # p1 should be redrawn to be <= max date
        self.assertLessEqual(result[1], pd.Timestamp("2021-05-01 12:00"))

        # p2 should remain unchanged (was already within range)
        self.assertEqual(result[2], pd.Timestamp("2021-05-02"))

        # p3, p4 should be drawn from valid timestamps (<= max date)
        for pid in [3, 4]:
            self.assertLessEqual(result[pid], pd.Timestamp("2021-05-01 12:00"))
            self.assertIn(
                result[pid], [pd.Timestamp("2021-05-01"), pd.Timestamp("2021-05-02")]
            )

    def test_draw_index_dates_for_unexposed_with_minimum_and_maximum_dates(self):
        """Test that both minimum and maximum_index_dates constraints work together."""
        censoring_timestamps = pd.Series(
            {
                1: pd.Timestamp("2021-05-01 12:00"),
                2: pd.Timestamp("2021-05-02 12:00"),
            }
        )
        censoring_timestamps.index.name = PID_COL

        minimum_index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                TIMESTAMP_COL: [
                    pd.Timestamp("2021-05-01 10:00"),  # p1's timestamp is after this
                    pd.Timestamp("2021-05-03 00:00"),  # p2's timestamp is before this
                    pd.Timestamp("2021-05-01 10:00"),  # p3 will be drawn
                    pd.Timestamp("2021-05-01 10:00"),  # p4 will be drawn
                ],
            }
        )

        maximum_index_dates = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                TIMESTAMP_COL: [
                    pd.Timestamp("2021-05-01 14:00"),  # p1's timestamp is before this
                    pd.Timestamp("2021-05-01 14:00"),  # p2's timestamp is after this
                    pd.Timestamp("2021-05-01 14:00"),  # p3 will be drawn
                    pd.Timestamp("2021-05-01 14:00"),  # p4 will be drawn
                ],
            }
        )

        # p1 should stay (within both min and max)
        # p2 should be redrawn (exceeds max)
        # p3, p4 should be drawn from valid range
        result = IndexDateHandler.draw_index_dates_for_unexposed(
            data_pids=[1, 2, 3, 4],
            censoring_timestamps=censoring_timestamps,
            minimum_index_dates=minimum_index_dates,
            maximum_index_dates=maximum_index_dates,
        )

        self.assertEqual(len(result), 4)

        # p1 should remain unchanged (within both constraints)
        self.assertEqual(result[1], pd.Timestamp("2021-05-01 12:00"))

        # p2 should be redrawn to be within both min and max
        self.assertGreaterEqual(result[2], pd.Timestamp("2021-05-03 00:00"))
        self.assertLessEqual(result[2], pd.Timestamp("2021-05-01 14:00"))

        # p3, p4 should be drawn from valid range
        for pid in [3, 4]:
            self.assertGreaterEqual(result[pid], pd.Timestamp("2021-05-01 10:00"))
            self.assertLessEqual(result[pid], pd.Timestamp("2021-05-01 14:00"))

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
        for pid in [1, 2, 3]:
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

        self.assertIn(1, result.index)
        self.assertIn(2, result.index)
        self.assertIn(3, result.index)
        self.assertEqual(len(result), 3, "p1, p2, p3 => total 3")

        self.assertEqual(result[1], pd.Timestamp("2021-01-01 13:00:00"))
        self.assertEqual(result[2], pd.Timestamp("2021-01-02 15:00:00"))

        # p3 should be randomly chosen from the existing [p1, p2] values
        possible = [
            pd.Timestamp("2021-01-01 13:00:00"),
            pd.Timestamp("2021-01-02 15:00:00"),
        ]
        self.assertIn(result[3], possible)

    def test_determine_index_dates_unsupported_mode(self):
        """Passing an invalid index_date_mode should raise ValueError."""
        with self.assertRaises(ValueError):
            IndexDateHandler.determine_index_dates(
                patients_info=self.patients_info,
                index_date_mode="invalid_mode",
            )


if __name__ == "__main__":
    unittest.main()
