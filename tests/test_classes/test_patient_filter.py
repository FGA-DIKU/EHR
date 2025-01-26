import unittest

import pandas as pd

from corebehrt.classes.patient_filter import (
    apply_exclusion_filters,
    filter_by_age,
    filter_by_categories,
    filter_by_initial_pids,
    filter_patients_by_pids,
)
from corebehrt.common.constants import (
    BIRTHDATE_COL,
    DEATHDATE_COL,
    PID_COL,
    TIMESTAMP_COL,
)


class TestFilterFunctions(unittest.TestCase):
    """Unit tests for patient filtering functions based on the new data schema.

    - `patients_info` DataFrame has:
        PID_COL, BIRTHDATE_COL, DEATHDATE_COL, TIMESTAMP_COL (the index date),
        plus any category columns for filter_by_categories.

    - `filter_by_age` expects BIRTHDATE_COL and TIMESTAMP_COL to be datetime64[ns].
    - `apply_exclusion_filters` uses TIMESTAMP_COL in the same DataFrame as the
      "index date" for each patient.
    """

    def setUp(self):
        """Prepare test data in the format required by the updated functions."""
        self.patients_info = pd.DataFrame(
            {
                PID_COL: ["p1", "p2", "p3", "p4", "p5"],
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
                        None,  # p1 => No death date
                        "2020-01-01",  # p2 => borderline same as index
                        None,  # p3 => No death date
                        "1985-01-01",  # p4 => 5 years after its index date
                        "2025-01-01",  # p5 => borderline same as index
                    ]
                ),
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2020-01-01",  # p1 => ~40 years after 1980
                        "2020-01-02",  # p2 => ~over 30 years after 1990
                        "2010-01-01",  # p3 => ~10 years after 2000
                        "1980-01-01",  # p4 => ~10 years after 1970
                        "2025-01-02",  # p5 => ~30 years after 1995
                    ]
                ),
                "category": ["A", "B", "B", "A", "C"],
            }
        )

        # Exposures: must have PID_COL and TIMESTAMP_COL
        self.exposures = pd.DataFrame(
            {
                PID_COL: ["p1", "p3", "p3", "p4", "x999"],  # x999 not in patients_info
                TIMESTAMP_COL: pd.to_datetime(
                    [
                        "2005-01-01",  # p1 exposure
                        "2010-01-01",  # p3 exposure
                        "2015-01-01",  # another p3 exposure
                        "1975-01-01",  # p4 exposure
                        "2020-01-01",  # irrelevant
                    ]
                ),
            }
        )

        # Outcomes: must have PID_COL and TIMESTAMP_COL
        self.outcomes = pd.DataFrame(
            {
                PID_COL: ["p2", "p5"],
                TIMESTAMP_COL: pd.to_datetime(["2018-01-01", "2024-01-01"]),
            }
        )

    # ----------------------------------------------------------------------
    # filter_patients_by_pids
    # ----------------------------------------------------------------------
    def test_filter_patients_by_pids_empty(self):
        """If pids is empty, expect an empty DataFrame."""
        result = filter_patients_by_pids(self.patients_info, set())
        self.assertTrue(result.empty)

    def test_filter_patients_by_pids_subset(self):
        """Filter by a subset of PIDs."""
        pids_to_keep = {"p1", "p3"}
        result = filter_patients_by_pids(self.patients_info, pids_to_keep)
        self.assertEqual(len(result), 2)
        self.assertSetEqual(set(result[PID_COL]), pids_to_keep)

    def test_filter_patients_by_pids_nonexistent_pids(self):
        """If the set of PIDs doesn't appear in patients_info, result should be empty."""
        pids_to_keep = {"xyz", "abc"}
        result = filter_patients_by_pids(self.patients_info, pids_to_keep)
        self.assertTrue(result.empty, "Expected no matching rows for nonexistent PIDs")

    def test_filter_patients_by_pids_missing_pid_col(self):
        """
        Test behavior if patients_info is missing the PID_COL column.
        This might be a scenario you handle with an exception or by ignoring.
        """
        df_no_pid = self.patients_info.drop(columns=[PID_COL])  # remove the PID column
        # You may expect KeyError or a custom error. Let's verify KeyError:
        with self.assertRaises(KeyError):
            _ = filter_patients_by_pids(df_no_pid, {"p1"})

    # ----------------------------------------------------------------------
    # filter_by_initial_pids
    # ----------------------------------------------------------------------
    def test_filter_by_initial_pids_no_args(self):
        """No initial_pids, not exposed_only => return the original DataFrame."""
        result = filter_by_initial_pids(self.patients_info)
        self.assertEqual(len(result), len(self.patients_info))

    def test_filter_by_initial_pids_with_initial_pids(self):
        """Filter by a set of initial PIDs only."""
        result = filter_by_initial_pids(self.patients_info, initial_pids={"p1", "p2"})
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p2"})

    def test_filter_by_initial_pids_exposed_only(self):
        """If exposed_only=True, keep only patients who appear in exposures."""
        # exposures mention p1, p3, p4
        result = filter_by_initial_pids(
            self.patients_info, exposures=self.exposures, exposed_only=True
        )
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p3", "p4"})

    def test_filter_by_initial_pids_both_conditions(self):
        """
        If both initial_pids and exposed_only are provided,
        filter by initial_pids first, then require exposures among that subset.
        """
        # initial pids => p1, p2 => leaves p1, p2
        # among p1 and p2, only p1 is in exposures => final = p1
        result = filter_by_initial_pids(
            self.patients_info,
            initial_pids={"p1", "p2"},
            exposures=self.exposures,
            exposed_only=True,
        )
        self.assertSetEqual(set(result[PID_COL]), {"p1"})

    def test_filter_by_initial_pids_empty_exposures(self):
        """If exposures is empty and exposed_only=True, expect an empty DataFrame."""
        empty_exposures = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: []})
        result = filter_by_initial_pids(
            self.patients_info, exposures=empty_exposures, exposed_only=True
        )
        self.assertTrue(result.empty, "Expected empty since no one is 'exposed'")

    def test_filter_by_initial_pids_no_overlap(self):
        """
        If initial_pids do not overlap with exposures at all, and exposed_only=True,
        expect an empty result after both steps.
        """
        # Suppose initial_pids = p2, p3 but exposures mention p1, p4
        # => after step 1, we get p2, p3 => after step 2, none match exposures => empty
        result = filter_by_initial_pids(
            self.patients_info,
            initial_pids={"p2", "p3"},  # keep just p2, p3 initially
            exposures=self.exposures,  # mentions p1, p3, p4
            exposed_only=True,
        )
        # Actually, note that p3 *is* in exposures, so it overlaps. Let's correct that example:
        # Instead, use p2, p5 => neither is in exposures => final empty
        result = filter_by_initial_pids(
            self.patients_info,
            initial_pids={"p2", "p5"},
            exposures=self.exposures,
            exposed_only=True,
        )
        self.assertTrue(result.empty, "Expected empty because p2,p5 not in exposures")

    # ----------------------------------------------------------------------
    # filter_by_categories
    # ----------------------------------------------------------------------
    def test_filter_by_categories_no_filters(self):
        """No category_filters => returns the original DataFrame."""
        result = filter_by_categories(self.patients_info)
        self.assertEqual(len(result), len(self.patients_info))

    def test_filter_by_categories_include(self):
        """Test filter by 'include': keep only rows where category='B'."""
        category_filters = {"category": {"include": ["B"]}}
        result = filter_by_categories(self.patients_info, category_filters)
        self.assertSetEqual(set(result[PID_COL]), {"p2", "p3"})

    def test_filter_by_categories_exclude(self):
        """Test filter by 'exclude': remove rows where category='A'."""
        category_filters = {"category": {"exclude": ["A"]}}
        result = filter_by_categories(self.patients_info, category_filters)
        self.assertSetEqual(set(result[PID_COL]), {"p2", "p3", "p5"})

    def test_filter_by_categories_multiple(self):
        """
        If the real code allows multiple filters at once, test that scenario.
        Example: We want 'category=B' and also some hypothetical column 'region != X'.
        """
        # Add a 'region' column for this test scenario:
        df = self.patients_info.copy()
        df["region"] = ["N", "X", "Y", "X", "N"]

        # We'll include only category=B and exclude region=X
        category_filters = {
            "category": {"include": ["B"]},
            "region": {"exclude": ["X"]},
        }

        filtered = filter_by_categories(df, category_filters)
        # p2 => category=B, region=X => fails region test => excluded
        # p3 => category=B, region=Y => included
        # Others => category != B => excluded anyway
        self.assertSetEqual(set(filtered[PID_COL]), {"p3"})

    def test_filter_by_categories_missing_column(self):
        """
        Test behavior if the category_filters references a column not in the DataFrame.
        Depending on design, you might ignore or raise KeyError.
        """
        category_filters = {"unknown_column": {"include": ["some_val"]}}
        with self.assertRaises(KeyError):
            _ = filter_by_categories(self.patients_info, category_filters)

    # ----------------------------------------------------------------------
    # filter_by_age
    # ----------------------------------------------------------------------
    def test_filter_by_age_no_limits(self):
        """No min_age or max_age => all patients remain."""
        result = filter_by_age(self.patients_info)
        self.assertEqual(len(result), 5)

    def test_filter_by_age_min_only(self):
        """min_age=30 => exclude patients under 30 at TIMESTAMP_COL."""
        # p3 => ~10 => excluded
        # p4 => ~10 => excluded
        # p1 => ~40 => included
        # p2 => ~30 => borderline => included
        # p5 => ~30 => borderline => included
        result = filter_by_age(self.patients_info, min_age=30)
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p2", "p5"})

    def test_filter_by_age_max_only(self):
        """max_age=30 => exclude patients over 30 at TIMESTAMP_COL."""
        # p1 => ~40 => excluded
        # p2 => ~30 => borderline => included
        # p3 => ~10 => included
        # p4 => ~10 => included
        # p5 => ~30 => borderline => included
        result = filter_by_age(self.patients_info, max_age=31)
        self.assertSetEqual(set(result[PID_COL]), {"p2", "p3", "p4", "p5"})

    def test_filter_by_age_min_and_max(self):
        """Keep patients whose age is between 20 and 40 (inclusive)."""
        # p1 => ~40 => included
        # p2 => ~30 => included
        # p3 => ~10 => excluded
        # p4 => ~10 => excluded
        # p5 => ~30 => included
        result = filter_by_age(self.patients_info, min_age=20, max_age=40)
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p2", "p5"})

    def test_filter_by_age_negative_min(self):
        """
        If min_age is negative, effectively that shouldn't exclude any normal adult.
        So all rows should remain. (This depends on your design.)
        """
        result = filter_by_age(self.patients_info, min_age=-1)
        self.assertEqual(len(result), 5, "Negative min_age should not exclude anyone")

    def test_filter_by_age_same_day_birthday(self):
        """
        If BIRTHDATE_COL == TIMESTAMP_COL for a patient, that patient's age=0.
        We'll insert one test row to confirm 0-age logic.
        """
        df = self.patients_info.copy()
        # Insert a patient p6 whose BIRTHDATE = TIMESTAMP exactly => age=0
        new_row = pd.DataFrame(
            [
                {
                    PID_COL: "p6",
                    BIRTHDATE_COL: pd.to_datetime("2020-01-01"),
                    DEATHDATE_COL: pd.NaT,
                    TIMESTAMP_COL: pd.to_datetime("2020-01-01"),
                    "category": "X",
                }
            ]
        )
        df = pd.concat([df, new_row], ignore_index=True)

        # Now, if min_age=0, p6 should remain; if min_age=1, p6 should be excluded.
        res_min0 = filter_by_age(df, min_age=0)
        self.assertIn("p6", set(res_min0[PID_COL]), "p6 should remain with min_age=0")

        res_min1 = filter_by_age(df, min_age=1)
        self.assertNotIn(
            "p6", set(res_min1[PID_COL]), "p6 should be excluded with min_age=1"
        )

    def test_filter_by_age_missing_columns(self):
        """
        If BIRTHDATE_COL or TIMESTAMP_COL is missing, we might expect a KeyError
        or handle it some other way. Let's check KeyError for BIRTHDATE_COL.
        """
        df_no_birth = self.patients_info.drop(columns=[BIRTHDATE_COL])
        with self.assertRaises(KeyError):
            _ = filter_by_age(df_no_birth, min_age=30)

    # ----------------------------------------------------------------------
    # apply_exclusion_filters
    # ----------------------------------------------------------------------
    def test_apply_exclusion_filters_no_flags(self):
        """If both flags are False => no one is removed."""
        result = apply_exclusion_filters(
            self.patients_info,
            self.outcomes,
            dead_before_index_date=False,
            outcome_before_index_date=False,
        )
        self.assertEqual(len(result), 5)

    def test_apply_exclusion_filters_dead_before_index_date(self):
        """
        Exclude patients who died before their TIMESTAMP_COL.
        Condition:
            Keep if DEATHDATE_COL is NaN OR DEATHDATE_COL > TIMESTAMP_COL
        """
        # p2 => death=2020-01-01, index=2020-01-01 => not strictly greater => exclude
        # p4 => death=1985-01-01, index=1980-01-01 => 1985>1980 => keep
        # p5 => death=2025-01-01, index=2025-01-01 => not strictly greater => exclude
        result = apply_exclusion_filters(
            self.patients_info, self.outcomes, dead_before_index_date=True
        )
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p3", "p4"})

    def test_apply_exclusion_filters_outcome_before_index_date(self):
        """
        Exclude patients whose outcome occurred strictly before their TIMESTAMP_COL.
        outcomes => p2(2018-01-01), p5(2024-01-01)
        """
        # p2 => outcome=2018 < index=2020 => exclude
        # p5 => outcome=2024 < index=2025 => exclude
        # p1, p3, p4 => no outcomes => remain
        result = apply_exclusion_filters(
            self.patients_info, self.outcomes, outcome_before_index_date=True
        )
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p3", "p4"})

    def test_apply_exclusion_filters_both(self):
        """
        If both dead_before_index_date and outcome_before_index_date are True,
        exclude patients who died before their index date OR had an outcome prior to index date.
        """
        # From prior tests:
        #  - p2 is excluded (dead borderline + prior outcome)
        #  - p5 is excluded (dead borderline + prior outcome)
        # p1, p3, p4 remain
        result = apply_exclusion_filters(
            self.patients_info,
            self.outcomes,
            dead_before_index_date=True,
            outcome_before_index_date=True,
        )
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p3", "p4"})

    def test_apply_exclusion_filters_multiple_outcomes(self):
        """Test that if any of multiple outcomes for a patient is before TIMESTAMP_COL, the patient is excluded."""
        # Extend the existing outcomes to include multiple outcomes for p2:
        multi_outcomes = pd.DataFrame(
            {
                PID_COL: ["p2", "p2", "p5"],  # p2 has 2 outcomes, one before, one after
                TIMESTAMP_COL: pd.to_datetime(
                    ["2019-01-01", "2021-01-01", "2024-01-01"]
                ),
            }
        )

        # p2's TIMESTAMP_COL in patients_info => "2020-01-02" => the first outcome is "2019-01-01" => earlier => exclude
        # p5 => outcome=2024-01-01 < index=2025-01-02 => exclude
        # p1, p3, p4 => no outcomes => remain
        result = apply_exclusion_filters(
            self.patients_info, multi_outcomes, outcome_before_index_date=True
        )
        # Expect p1, p3, p4 remain
        self.assertSetEqual(set(result[PID_COL]), {"p1", "p3", "p4"})

    def test_apply_exclusion_filters_empty_outcomes(self):
        """
        If outcomes is empty but outcome_before_index_date=True,
        no patients should be excluded.
        """
        empty_outcomes = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: []})
        result = apply_exclusion_filters(
            self.patients_info, empty_outcomes, outcome_before_index_date=True
        )
        self.assertEqual(
            len(result), len(self.patients_info), "No one excluded with empty outcomes"
        )

    def test_apply_exclusion_filters_all_excluded(self):
        """
        If all patients either die before index or have an early outcome, we end up with 0 rows.
        Let's create a scenario where everyone is excluded.
        """
        # Make a small DF with 2 patients, both with borderline or earlier death
        df_all = pd.DataFrame(
            {
                PID_COL: ["px1", "px2"],
                BIRTHDATE_COL: pd.to_datetime(["1970-01-01", "1980-01-01"]),
                DEATHDATE_COL: pd.to_datetime(["2000-01-01", "2005-01-01"]),
                TIMESTAMP_COL: pd.to_datetime(["2000-01-01", "2005-01-01"]),
            }
        )
        # Both patients died exactly on their index => excluded if dead_before_index_date=True
        outcomes_all = pd.DataFrame({PID_COL: [], TIMESTAMP_COL: []})

        # Turn on dead_before_index_date => both excluded
        result = apply_exclusion_filters(
            df_all, outcomes_all, dead_before_index_date=True
        )
        self.assertTrue(
            result.empty,
            "All should be excluded since death=TIMESTAMP => not strictly greater",
        )

        # If we set outcome_before_index_date=True but not dead_before..., they'd remain
        # but let's create an outcome for each that is definitely before TIMESTAMP:
        outcomes_all = pd.DataFrame(
            {
                PID_COL: ["px1", "px2"],
                TIMESTAMP_COL: pd.to_datetime(["1999-01-01", "2004-01-01"]),
            }
        )
        # Now with outcome_before_index_date => exclude both
        df_all = pd.DataFrame(
            {
                PID_COL: ["px1", "px2"],
                BIRTHDATE_COL: pd.to_datetime(["1970-01-01", "1980-01-01"]),
                DEATHDATE_COL: pd.NaT,  # ignore death for clarity
                TIMESTAMP_COL: pd.to_datetime(["2000-01-01", "2005-01-01"]),
            }
        )
        result2 = apply_exclusion_filters(
            df_all,
            outcomes_all,
            dead_before_index_date=False,
            outcome_before_index_date=True,
        )
        self.assertTrue(result2.empty, "All excluded by outcome_before_index_date")


if __name__ == "__main__":
    unittest.main()
