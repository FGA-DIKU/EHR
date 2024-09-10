import unittest

import pandas as pd

from corebehrt.classes.outcomes import \
    OutcomeHandler  # Import your OutcomeHandler class accordingly
from corebehrt.common.utils import \
    Data  # Assuming Data is imported from the right module


class TestOutcomeHandler(unittest.TestCase):

    def setUp(self):
        # Initialize OutcomeHandler with default options
        self.handler = OutcomeHandler(
            index_date=None,
            select_patient_group=None,
            exclude_pre_followup_outcome_patients=False,
            n_hours_start_followup=0
        )

        # Mock data for Data class
        self.data = Data()
        self.data.pids = ['P1', 'P2', 'P3', 'P4']

        # Create mock outcomes DataFrame with abspos (absolute positions)
        self.outcomes = pd.DataFrame({
            'PID': ['P1', 'P2', 'P3', 'P4'],
            'TIMESTAMP': [1000, 2000, 1500, 4000]  # Abspos values instead of timestamps
        })

        # Create mock exposures DataFrame with abspos
        self.exposures = pd.DataFrame({
            'PID': ['P1', 'P2', 'P3'],
            'TIMESTAMP': [500, 1200, 800]  # Abspos values
        })

    def test_check_input_valid(self):
        # Test the check_input method with valid outcomes and exposures
        try:
            self.handler.check_input(self.outcomes, self.exposures)
        except ValueError as e:
            self.fail(f"check_input raised ValueError unexpectedly: {str(e)}")

    def test_check_input_invalid(self):
        # Test the check_input method with invalid DataFrame (missing 'TIMESTAMP')
        invalid_outcomes = self.outcomes.drop(columns=['TIMESTAMP'])
        with self.assertRaises(ValueError):
            self.handler.check_input(invalid_outcomes, self.exposures)

    def test_handle(self):
        # Test the full handle method
        updated_data = self.handler.handle(self.data, self.outcomes, self.exposures)

        # Check that the outcomes and index dates are added to the data
        self.assertIsNotNone(updated_data.outcomes)
        self.assertIsNotNone(updated_data.censor_outcomes)

    def test_select_exposed_patients(self):
        # Test selecting only exposed patients
        self.handler.select_patient_group = 'exposed'
        updated_data = self.handler.handle(self.data, self.outcomes, self.exposures)

        # Check that the unexposed patients were excluded (P4 is unexposed)
        self.assertNotIn('P4', updated_data.pids)

    def test_select_unexposed_patients(self):
        # Test selecting only unexposed patients
        self.handler.select_patient_group = 'unexposed'
        updated_data = self.handler.handle(self.data, self.outcomes, self.exposures)

        # Check that the exposed patients were excluded (P1, P2, P3 are exposed)
        self.assertNotIn('P1', updated_data.pids)
        self.assertNotIn('P2', updated_data.pids)
        self.assertNotIn('P3', updated_data.pids)
        self.assertIn('P4', updated_data.pids)

    def test_remove_outcomes_before_start_of_follow_up(self):
        # Test removing outcomes before the start of follow-up
        self.handler.exclude_pre_followup_outcome_patients = True
        self.handler.n_hours_start_followup = 24  # Example: follow-up starts 24 hours after exposure (in abspos units)

        # Exclude patients with outcomes before follow-up start
        updated_data = self.handler.handle(self.data, self.outcomes, self.exposures)

        # Check that patients with outcomes before follow-up are removed
        # For instance, if follow-up starts at abspos + 24, we can assert accordingly
        self.assertIn('P1', updated_data.pids)
        # Adjust assertions based on your follow-up logic and expected behavior

    def test_assign_exposures_and_outcomes_to_data(self):
        # Test the assignment of exposures and outcomes to the Data object
                # Mock data for Data class
        data = Data()
        data.pids = ['P1', 'P2', 'P3', 'P4']

        index_dates = pd.Series({
            'P1': 500,  # Abspos values
            'P2': 1200,
            'P3': 800,
        })

        outcomes = pd.Series({
            'P1': 1000,  # Abspos values
            'P2': 2000,
            'P3': 1500,
        })

        expected_outcomes = pd.Series({
            'P1': 1000,
            'P2': 2000,
            'P3': 1500,
            'P4': pd.NA
        }, dtype=pd.Int64Dtype())
        expected_censor_outcomes = pd.Series({
            'P1': 500,
            'P2': 1200,
            'P3': 800,
            'P4': pd.NA
        }, dtype=pd.Int64Dtype())
        updated_data = self.handler.assign_exposures_and_outcomes_to_data(data, index_dates, outcomes)
        # Check that outcomes and censor outcomes were properly assigned
        # assert that values are close
        self.assert_outcomes_equal(updated_data.outcomes, list(expected_outcomes.values))
        self.assert_outcomes_equal(updated_data.censor_outcomes, list(expected_censor_outcomes.values))

    def assert_outcomes_equal(self, actual, expected):
        """Helper function to compare lists of outcomes, including pd.NA and None."""
        for act, exp in zip(actual, expected):
            if pd.isna(act) and pd.isna(exp):
                continue  # Both are missing values
            else:
                self.assertEqual(act, exp, f"Expected {exp} but got {act}")

if __name__ == '__main__':
    unittest.main()
