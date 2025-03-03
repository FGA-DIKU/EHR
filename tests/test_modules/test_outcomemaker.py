import unittest
from datetime import datetime

import pandas as pd

from corebehrt.functional.utils.time import get_abspos_from_origin_point
from corebehrt.modules.cohort_handling.outcomes import OutcomeMaker
from corebehrt.constants.data import (
    PID_COL,
    CONCEPT_COL,
)
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

        # Mock origin point
        self.origin_point = datetime(**{"year": 2020, "month": 1, "day": 26})

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
        self.outcome_maker = OutcomeMaker(self.outcomes, self.origin_point)

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
        expected_outcome["abspos"] = get_abspos_from_origin_point(
            expected_outcome["time"], self.origin_point
        )
        expected_outcome["abspos"] = expected_outcome["abspos"].astype(int)
        # Check that the outcome table matches the expected result
        pd.testing.assert_frame_equal(
            result["TEST_OUTCOME"], expected_outcome, check_index_type=False
        )

        # Expected outcome for TEST_CENSOR
        expected_censor = pd.DataFrame(
            {PID_COL: [1], "time": [pd.Timestamp("2020-01-10")]}, index=[0]
        )
        expected_censor["abspos"] = get_abspos_from_origin_point(
            expected_censor["time"], self.origin_point
        )
        expected_censor["abspos"] = expected_censor["abspos"].astype(int)
        # Check that the censor table matches the expected result
        pd.testing.assert_frame_equal(
            result["TEST_CENSOR"], expected_censor, check_index_type=False
        )


if __name__ == "__main__":
    unittest.main()
