import unittest
import pandas as pd

from corebehrt.constants.data import (
    BIRTHDATE_COL,
    DEATHDATE_COL,
    PID_COL,
    TIMESTAMP_COL,
)
from corebehrt.modules.cohort_handling.patient_filter import (
    filter_by_age,
    filter_by_categories,
    filter_df_by_pids,
    exclude_pids_from_df,
    filter_by_death,
    filter_by_prior_outcomes,
)


class TestFilterFunctions(unittest.TestCase):
    """
    Unit tests for patient filtering functions.

    - `patients_info` DataFrame has:
        PID_COL, BIRTHDATE_COL, DEATHDATE_COL, TIMESTAMP_COL (the index date),
        plus any category columns for filter_by_categories.

    - filter_by_age() expects BIRTHDATE_COL and TIMESTAMP_COL to be datetime64[ns].
    - filter_by_death() expects DEATHDATE_COL (datetime64[ns]) to exist.
    - filter_by_prior_outcomes() expects a second DataFrame with PID_COL and TIMESTAMP_COL.
    """

    def setUp(self):
        """Prepare test data in the required format."""
        self.patients_info = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5],
                BIRTHDATE_COL: pd.to_datetime(
                    [
                        "1980-01-01",  # p1
                        "1990-01-01",  # p2
                        "2000-01-01",  # p3
                        "1970-01-01",  # p4
                        "1995-01-01",  # p5
                    ]
                ),
                DEATHDATE_COL: pd.to_datetime(
                    [
                        None,  # p1 => alive
                        "2020-01-01",  # p2 => same day as index minus 1 day
                        None,  # p3 => alive
                        "1985-01-01",  # p4 => died 5 years after 1980
                        "2025-01-01",  # p5 => died borderline after index
                    ]
                ),
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2020-01-01",  # p1 => ~40 years after 1980
                        "2020-01-02",  # p2 => ~30 years after 1990
                        "2010-01-01",  # p3 => ~10 years after 2000
                        "1980-01-01",  # p4 => ~10 years after 1970
                        "2025-01-02",  # p5 => ~30 years after 1995
                    ]
                ),
                "category": ["A", "B", "B", "A", "C"],
            }
        )

        # Outcomes: must have PID_COL and TIMESTAMP_COL
        self.outcomes = pd.DataFrame(
            {
                PID_COL: [2, 5],
                TIMESTAMP_COL: pd.to_datetime(["2018-01-01", "2024-01-01"]),
            }
        )

    # ----------------------------------------------------------------------
    # filter_df_by_pids
    # ----------------------------------------------------------------------
    def test_filter_df_by_pids_empty(self):
        """
        If the list/set of PIDs is empty, expect an empty DataFrame.
        """
        result = filter_df_by_pids(self.patients_info, [])
        self.assertTrue(result.empty)

    def test_filter_df_by_pids_subset(self):
        """
        Filtering by a specific subset of PIDs should return only those rows.
        """
        pids_to_keep = {1, 3}
        result = filter_df_by_pids(self.patients_info, pids_to_keep)
        self.assertEqual(len(result), 2)
        self.assertSetEqual(set(result[PID_COL]), pids_to_keep)

    def test_filter_df_by_pids_nonexistent(self):
        """
        If the PIDs to keep don't exist in the DataFrame,
        the result should be empty.
        """
        pids_to_keep = {"xyz", "abc"}
        result = filter_df_by_pids(self.patients_info, pids_to_keep)
        self.assertTrue(result.empty)

    # ----------------------------------------------------------------------
    # exclude_pids_from_df
    # ----------------------------------------------------------------------
    def test_exclude_pids_from_df_empty(self):
        """
        If pids_to_exclude is empty, we keep all rows.
        """
        result = exclude_pids_from_df(self.patients_info, [])
        pd.testing.assert_frame_equal(result, self.patients_info)

    def test_exclude_pids_from_df_subset(self):
        """
        Exclude a subset of PIDs; others remain.
        """
        pids_to_exclude = {1, 3}
        result = exclude_pids_from_df(self.patients_info, pids_to_exclude)
        self.assertEqual(len(result), 3)
        self.assertSetEqual(set(result[PID_COL]), {2, 4, 5})

    def test_exclude_pids_from_df_nonexistent(self):
        """
        Excluding nonexistent PIDs leaves the DataFrame unchanged.
        """
        pids_to_exclude = {"xyz", "abc"}
        result = exclude_pids_from_df(self.patients_info, pids_to_exclude)
        pd.testing.assert_frame_equal(result, self.patients_info)

    def test_exclude_pids_from_df_all(self):
        """
        Excluding all PIDs results in an empty DataFrame.
        """
        all_pids = set(self.patients_info[PID_COL])
        result = exclude_pids_from_df(self.patients_info, all_pids)
        self.assertTrue(result.empty)

    # ----------------------------------------------------------------------
    # filter_by_categories
    # ----------------------------------------------------------------------
    def test_filter_by_categories_no_filters(self):
        """
        No category_filters => returns the original DataFrame.
        """
        result = filter_by_categories(self.patients_info)
        self.assertEqual(len(result), len(self.patients_info))

    def test_filter_by_categories_include(self):
        """
        Keep only rows where category in ['B'].
        """
        category_filters = {"category": {"include": ["B"]}}
        result = filter_by_categories(self.patients_info, category_filters)
        self.assertSetEqual(set(result[PID_COL]), {2, 3})

    def test_filter_by_categories_exclude(self):
        """
        Exclude rows where category in ['A'].
        """
        category_filters = {"category": {"exclude": ["A"]}}
        result = filter_by_categories(self.patients_info, category_filters)
        self.assertSetEqual(set(result[PID_COL]), {2, 3, 5})

    def test_filter_by_categories_multiple(self):
        """
        Apply multiple include/exclude filters simultaneously.
        """
        df = self.patients_info.copy()
        df["region"] = ["N", "X", "Y", "X", "N"]  # Additional column for testing

        category_filters = {
            "category": {"include": ["B"]},  # keep only 'B'
            "region": {"exclude": ["X"]},  # exclude region='X'
        }
        filtered = filter_by_categories(df, category_filters)
        # p2 => category=B, region=X => gets excluded due to region
        # p3 => category=B, region=Y => included
        # Others => category != B => excluded anyway
        self.assertSetEqual(set(filtered[PID_COL]), {3})

    # ----------------------------------------------------------------------
    # filter_by_age
    # ----------------------------------------------------------------------
    def test_filter_by_age_no_limits(self):
        """
        No min_age or max_age => all patients remain.
        """
        result = filter_by_age(self.patients_info)
        self.assertEqual(len(result), 5)

    def test_filter_by_age_min_only(self):
        """
        min_age=30 => exclude patients younger than 30.
        """
        result = filter_by_age(self.patients_info, min_age=30)
        self.assertSetEqual(set(result[PID_COL]), {1, 2, 5})

    def test_filter_by_age_max_only(self):
        """
        max_age=31 => exclude patients older than 31.
        """
        result = filter_by_age(self.patients_info, max_age=31)
        self.assertSetEqual(set(result[PID_COL]), {2, 3, 4, 5})

    def test_filter_by_age_min_and_max(self):
        """
        min_age=20, max_age=40 => keep 20 <= age <= 40.
        """
        result = filter_by_age(self.patients_info, min_age=20, max_age=40)
        self.assertSetEqual(set(result[PID_COL]), {1, 2, 5})

    def test_filter_by_age_negative_min(self):
        """
        Negative min_age should practically keep everyone.
        """
        result = filter_by_age(self.patients_info, min_age=-5)
        self.assertEqual(len(result), 5)

    def test_filter_by_age_same_day_birthday(self):
        """
        If BIRTHDATE_COL == TIMESTAMP_COL, age = 0.
        Ensure that 0 fits min_age=0 but fails min_age=1.
        """
        df = self.patients_info.copy()
        new_row = pd.DataFrame(
            [
                {
                    PID_COL: 6,
                    BIRTHDATE_COL: pd.to_datetime("2020-01-01"),
                    DEATHDATE_COL: pd.NaT,
                    TIMESTAMP_COL: pd.to_datetime("2020-01-01"),
                    "category": "X",
                }
            ]
        )
        df = pd.concat([df, new_row], ignore_index=True)

        # min_age=0 => p6 should remain
        res_min0 = filter_by_age(df, min_age=0)
        self.assertIn(6, res_min0[PID_COL].values)

        # min_age=1 => p6 should be excluded
        res_min1 = filter_by_age(df, min_age=1)
        self.assertNotIn(6, res_min1[PID_COL].values)

    # ----------------------------------------------------------------------
    # filter_by_death
    # ----------------------------------------------------------------------
    def test_filter_by_death_basic(self):
        """
        Exclude patients who died before or on their index date.
        Keep those alive or died after index.
        """
        result = filter_by_death(self.patients_info)
        # Surviving:
        # p1 (no death date),
        # p3 (no death date),
        # p5 died after index? => index=2025-01-02, death=2025-01-01 (actually 1 day before?)
        #   * In your data: p5 has deathdate=2025-01-01, index=2025-01-02 => died before index
        #   * So p5 should be excluded if strictly "died before or on index".
        #   * Double-check if you want strict '>' or '>=': The code uses `>` => p5 is excluded
        self.assertSetEqual(set(result[PID_COL]), {1, 3, 4})

    def test_filter_by_death_all_alive(self):
        """
        If everyone is alive (DEATHDATE_COL = NaT),
        we expect no exclusions.
        """
        df = self.patients_info.copy()
        df[DEATHDATE_COL] = pd.NaT
        result = filter_by_death(df)
        self.assertEqual(len(result), len(df))

    def test_filter_by_death_all_dead_before(self):
        """
        If everyone died well before the index date,
        all are excluded => empty result.
        """
        df = self.patients_info.copy()
        df[DEATHDATE_COL] = pd.to_datetime("1979-12-31")
        result = filter_by_death(df)
        self.assertTrue(result.empty)

    # ----------------------------------------------------------------------
    # filter_by_prior_outcomes
    # ----------------------------------------------------------------------
    def test_filter_by_prior_outcomes_basic(self):
        """
        Exclude patients whose earliest outcome date is before their index date.
        """
        result = filter_by_prior_outcomes(self.patients_info, self.outcomes)
        # outcomes: p2 has outcome=2018, p2's index=2020 => outcome before index => exclude p2
        #           p5 has outcome=2024, p5's index=2025 => outcome before index => exclude p5
        # Remaining: p1, p3, p4
        self.assertSetEqual(set(result[PID_COL]), {1, 3, 4})

    def test_filter_by_prior_outcomes_empty_outcomes(self):
        """
        If the outcomes DataFrame is empty, nothing is excluded.
        """
        empty_outcomes = pd.DataFrame(columns=[PID_COL, TIMESTAMP_COL])
        result = filter_by_prior_outcomes(self.patients_info, empty_outcomes)
        pd.testing.assert_frame_equal(result, self.patients_info)

    def test_filter_by_prior_outcomes_no_prior_outcomes(self):
        """
        If all outcomes are after index dates, no one is excluded.
        Example: outcomes date is after each patient's index date.
        """
        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 2],
                TIMESTAMP_COL: pd.to_datetime(["2021-01-01", "2021-01-01"]),
            }
        )
        result = filter_by_prior_outcomes(self.patients_info, outcomes)
        pd.testing.assert_frame_equal(result, self.patients_info)

    def test_filter_by_prior_outcomes_all_prior(self):
        """
        If every patient has an outcome before their index date,
        all are excluded => empty DataFrame.
        """
        outcomes = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5],
                TIMESTAMP_COL: pd.to_datetime("1979-01-01"),
            }
        )
        result = filter_by_prior_outcomes(self.patients_info, outcomes)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
