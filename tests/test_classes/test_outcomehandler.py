import unittest
import pandas as pd
import dask.dataframe as dd

# Import your OutcomeHandler class accordingly
from corebehrt.classes.outcomes import OutcomeHandler

# Mock logger
import logging

logger = logging.getLogger(__name__)


class TestOutcomeHandler(unittest.TestCase):
    def setUp(self):
        # Initialize OutcomeHandler with default options
        self.handler = OutcomeHandler(
            index_date=None,
            select_patient_group=None,
            exclude_pre_followup_outcome_patients=False,
            n_hours_start_followup=0,
        )

        # Create mock data as Dask DataFrame
        self.data = dd.from_pandas(
            pd.DataFrame({"PID": ["P1", "P2", "P3", "P4"]}), npartitions=1
        )

        # Create mock outcomes DataFrame with abspos (absolute positions)
        self.outcomes = pd.DataFrame(
            {
                "PID": ["P1", "P2", "P3", "P4"],
                "abspos": [
                    1000.0,
                    2000,
                    1500,
                    4000,
                ],  # Abspos values instead of timestamps
            }
        )

        # Create mock exposures DataFrame with abspos
        self.exposures = pd.DataFrame(
            {"PID": ["P1", "P2", "P3"], "abspos": [500.0, 1200, 800]}  # Abspos values
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

    def test_handle(self):
        # Test the full handle method
        updated_data, index_dates, outcomes = self.handler.handle(
            self.data, self.outcomes, self.exposures
        )

        # Check that index_dates and outcomes are not None
        self.assertIsNotNone(index_dates)
        self.assertIsNotNone(outcomes)

    def test_select_exposed_patients(self):
        # Test selecting only exposed patients
        self.handler.select_patient_group = "exposed"
        updated_data, index_dates, outcomes = self.handler.handle(
            self.data, self.outcomes, self.exposures
        )

        # Get the PIDs from updated_data
        updated_pids = updated_data["PID"].compute().tolist()

        # Check that the unexposed patients were excluded (P4 is unexposed)
        self.assertNotIn("P4", updated_pids)

    def test_select_unexposed_patients(self):
        # Test selecting only unexposed patients
        self.handler.select_patient_group = "unexposed"
        updated_data, index_dates, outcomes = self.handler.handle(
            self.data, self.outcomes, self.exposures
        )

        # Get the PIDs from updated_data
        updated_pids = updated_data["PID"].compute().tolist()

        # Check that the exposed patients were excluded (P1, P2, P3 are exposed)
        self.assertNotIn("P1", updated_pids)
        self.assertNotIn("P2", updated_pids)
        self.assertNotIn("P3", updated_pids)
        self.assertIn("P4", updated_pids)

    def test_remove_outcomes_before_start_of_follow_up(self):
        # Test removing outcomes before the start of follow-up
        self.handler.exclude_pre_followup_outcome_patients = True
        self.handler.n_hours_start_followup = (
            24  # Follow-up starts 24 hours after exposure
        )

        # Modify outcomes to have an outcome before follow-up start for P1
        test_outcomes = pd.DataFrame(
            {
                "PID": ["P1", "P2", "P3", "P4"],
                "abspos": [520.0, 2000, 1500, 4000],  # P1's outcome is before follow-up
            }
        )

        # Exclude patients with outcomes before follow-up start
        updated_data, index_dates, outcomes = self.handler.handle(
            self.data, test_outcomes, self.exposures
        )

        # Get updated PIDs
        updated_pids = updated_data["PID"].compute().tolist()

        # P1 should be removed because outcome is before follow-up start (500 + 24 = 524)
        self.assertNotIn("P1", updated_pids)
        # P2, P3, P4 should be in updated_data
        self.assertIn("P2", updated_pids)
        self.assertIn("P3", updated_pids)
        self.assertIn("P4", updated_pids)

    def test_synchronize_patients(self):
        # Test the synchronize_patients method
        # Mock data as Dask DataFrame
        data = dd.from_pandas(
            pd.DataFrame({"PID": ["P1", "P2", "P3", "P4"]}), npartitions=1
        )

        index_dates = pd.Series(
            {
                "P1": 500.0,  # Abspos values
                "P2": 1200,
                "P3": 800,
            }
        )

        outcomes = pd.Series(
            {
                "P1": 1000.0,  # Abspos values
                "P2": 2000,
                "P3": 1500,
            }
        )
        import numpy as np

        expected_index_dates = pd.Series(
            [500.0, 1200.0, 800.0, np.nan],
            index=["P1", "P2", "P3", "P4"],
            dtype=pd.Float64Dtype(),
        )
        expected_outcomes = pd.Series(
            [1000.0, 2000, 1500, pd.NA],
            index=["P1", "P2", "P3", "P4"],
            dtype=pd.Float64Dtype(),
        )
        index_dates = self.handler.synchronize_patients(data, index_dates)
        outcomes = self.handler.synchronize_patients(data, outcomes)
        # Check that index_dates and outcomes are as expected
        pd.testing.assert_series_equal(index_dates, expected_index_dates)
        pd.testing.assert_series_equal(outcomes, expected_outcomes)


if __name__ == "__main__":
    unittest.main()
