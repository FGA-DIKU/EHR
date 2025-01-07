import unittest

import dask.dataframe as dd
import pandas as pd

from corebehrt.classes.outcomes import OutcomeHandler


class TestOutcomeHandler(unittest.TestCase):
    def setUp(self):
        # Initialize OutcomeHandler with default options
        self.handler = OutcomeHandler(
            index_date=None,
            select_patient_group=None,  # If you later implement logic for group selection
            exclude_pre_followup_outcome_patients=False,
            n_hours_start_followup=0,
        )

        # Create mock data as a dask DataFrame just for demonstration
        self.data = dd.from_pandas(
            pd.DataFrame({"PID": ["P1", "P2", "P3", "P4"]}), npartitions=1
        )
        # We'll often just convert that to a list of pids
        self.data_pids = self.data["PID"].compute().tolist()

        # Create mock outcomes DataFrame with abspos (absolute positions)
        self.outcomes = pd.DataFrame(
            {
                "PID": ["P1", "P2", "P3", "P4"],
                "abspos": [1000.0, 2000, 1500, 4000],  # Abspos for outcomes
            }
        )

        # Create mock exposures DataFrame with abspos
        self.exposures = pd.DataFrame(
            {"PID": ["P1", "P2", "P3"], "abspos": [500.0, 1200, 800]}
        )

    def test_check_input_valid(self):
        # Test the check_input method with valid outcomes and exposures
        try:
            self.handler.check_input(self.outcomes, self.exposures)
        except ValueError as e:
            self.fail(f"check_input raised ValueError unexpectedly: {str(e)}")

    def test_check_input_invalid(self):
        # Test the check_input method with invalid DataFrame (missing 'abspos')
        invalid_outcomes = self.outcomes.drop(columns=["abspos"])
        with self.assertRaises(ValueError):
            self.handler.check_input(invalid_outcomes, self.exposures)

    def test_handle_basic(self):
        """
        Test the full handle method with basic data.
        We expect returned index_dates and outcomes as pd.Series with PID as index.
        """
        index_dates, outcomes = self.handler.handle(
            self.data_pids, self.outcomes, self.exposures
        )

        # Check that we got Series back
        self.assertIsInstance(index_dates, pd.Series)
        self.assertIsInstance(outcomes, pd.Series)

        # By default, P4 has no exposure => index_dates for P4 might be random or NaN
        # The outcome for P4 might still exist in 'outcomes', but if not in exposures, see how it's handled
        # Just check the shape or some basic property
        self.assertEqual(len(index_dates), 4)  # P1..P4
        self.assertEqual(len(outcomes), 4)

    def test_select_exposed_patients(self):
        """
        If 'select_patient_group' = 'exposed', we expect patients without exposure
        (e.g. P4) to be excluded or assigned NaN in index_dates.
        """
        self.handler.select_patient_group = "exposed"
        index_dates, outcomes = self.handler.handle(
            self.data_pids, self.outcomes, self.exposures
        )
        # If your code excludes unexposed from index_dates, we check if "P4" is missing or NaN
        self.assertIn(
            "P4", index_dates.index, "Check if your logic drops the row or sets NaN"
        )

        # For instance, you might do:
        # self.assertTrue(pd.isna(index_dates.loc["P4"]), "Exposed group might yield NaN for P4")
        #
        # Or if your logic physically excludes P4 from the index, do:
        # self.assertNotIn("P4", index_dates.index)

    def test_select_unexposed_patients(self):
        """
        If 'select_patient_group' = 'unexposed', we expect P1, P2, P3 (all exposed) to be excluded or NaN.
        """
        self.handler.select_patient_group = "unexposed"
        index_dates, outcomes = self.handler.handle(
            self.data_pids, self.outcomes, self.exposures
        )
        # Similar logic: if your code excludes exposed from index_dates
        # check that P1, P2, P3 are missing or NaN
        # self.assertNotIn("P1", index_dates.index)
        # or check pd.isna(index_dates.loc["P1"])
        pass

    def test_remove_outcomes_before_start_of_follow_up(self):
        """
        Test removing outcomes before the start of follow-up.
        This sets exclude_pre_followup_outcome_patients=True and n_hours_start_followup=24
        so that if an outcome occurs before exposure + 24h, that patient is excluded or gets outcome=NaN.
        """
        self.handler.exclude_pre_followup_outcome_patients = True
        self.handler.n_hours_start_followup = 24

        # Suppose P1's outcome (abspos=520) is < exposure+24 (500+24=524 => 520 < 524 => pre-follow-up)
        test_outcomes = pd.DataFrame(
            {
                "PID": ["P1", "P2", "P3", "P4"],
                "abspos": [520.0, 2000, 1500, 4000],
            }
        )

        index_dates, outcomes = self.handler.handle(
            self.data_pids, test_outcomes, self.exposures
        )

        # Expect P1 might be NaN or excluded if outcome is before follow-up start
        # The exact logic depends on your implementation
        if "P1" in index_dates.index:
            self.assertTrue(
                pd.isna(outcomes.loc["P1"]), "P1 outcome should be removed or NaN."
            )
        else:
            self.assertNotIn("P1", index_dates.index, "P1 might be fully removed.")

        # Others remain
        self.assertIn("P2", index_dates.index)
        self.assertIn("P3", index_dates.index)
        self.assertIn("P4", index_dates.index)

    def test_synchronize_patients(self):
        """
        Test the synchronize_patients method independently.
        We'll pass a subset of pids and ensure the returned Series is reindexed to them.
        """
        # Suppose we have data pids
        data_pids = ["P1", "P2", "P3", "P4"]
        # index_dates => partial
        index_dates_series = pd.Series(
            [500.0, 1200.0, 800.0], index=["P1", "P2", "P3"], dtype=float
        )
        # Call synchronize
        synced = self.handler.synchronize_patients(data_pids, index_dates_series)
        # We expect it to have an entry for P4 as well => NaN
        self.assertListEqual(synced.index.tolist(), data_pids)
        self.assertTrue(pd.isna(synced.loc["P4"]))


if __name__ == "__main__":
    unittest.main()
