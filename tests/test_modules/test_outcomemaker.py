import unittest

import pandas as pd

from corebehrt.constants.data import CONCEPT_COL, PID_COL
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.outcomes import OutcomeMaker


class TestOutcomeMaker(unittest.TestCase):
    def setUp(self):
        # Create a mock outcomes configuration for testing
        self.outcomes = {
            "TEST_OUTCOME": {
                "type": ["code"],
                "match": [["2"]],
                "exclude": ["D23"],
                "match_how": "contains",
                "case_sensitive": True,
            },
            "TEST_CENSOR": {
                "type": ["code"],
                "match": [["D1"]],
                "match_how": "startswith",
                "case_sensitive": False,
            },
        }

        # Create a mock concepts_plus DataFrame
        self.concepts_plus = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                CONCEPT_COL: ["D13", "D2", "D23", "D2"],
                "time": [
                    pd.Timestamp("2020-01-10"),
                    pd.Timestamp("2020-01-12"),
                    pd.Timestamp("2020-01-12"),
                    pd.Timestamp("2020-01-15"),
                ],
                "numeric_value": [1, 2, 3, 4],
            }
        )

        # Create a mock patients_info DataFrame
        self.patients_info = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4],
                "info1": [1, 2, 3, 4],
                "time": [
                    pd.Timestamp("2020-01-10"),
                    pd.Timestamp("2020-01-12"),
                    pd.NaT,
                    pd.Timestamp("2020-01-15"),
                ],
            }
        )

        # Patient set
        self.patient_set = [1, 2, 3]

        # OutcomeMaker instance
        self.outcome_maker = OutcomeMaker(self.outcomes)

    def test_outcome_maker(self):
        # Call OutcomeMaker with the mock data
        result = self.outcome_maker(
            self.concepts_plus, self.patients_info, self.patient_set
        )
        # Expected outcome for TEST_OUTCOME
        expected_outcome = pd.DataFrame(
            {
                PID_COL: [2],
                "time": [
                    pd.Timestamp("2020-01-12"),
                ],
            },
            index=[1],
        )
        expected_outcome["abspos"] = get_hours_since_epoch(expected_outcome["time"])
        expected_outcome["abspos"] = expected_outcome["abspos"].astype(int)
        # Check that the outcome table matches the expected result
        pd.testing.assert_frame_equal(
            result["TEST_OUTCOME"].astype("int64"),
            expected_outcome.astype("int64"),
            check_index_type=False,
        )

        # Expected outcome for TEST_CENSOR
        expected_censor = pd.DataFrame(
            {PID_COL: [1], "time": [pd.Timestamp("2020-01-10")]}, index=[0]
        )
        expected_censor["abspos"] = get_hours_since_epoch(expected_censor["time"])
        expected_censor["abspos"] = expected_censor["abspos"].astype(int)
        # Check that the censor table matches the expected result
        pd.testing.assert_frame_equal(
            result["TEST_CENSOR"].astype("int64"),
            expected_censor.astype("int64"),
            check_index_type=False,
        )


if __name__ == "__main__":
    unittest.main()
