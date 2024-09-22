import unittest

import pandas as pd

from corebehrt.classes.outcomes import OutcomeMaker
from corebehrt.functional.utils import get_abspos_from_origin_point
from datetime import datetime


class TestOutcomeMaker(unittest.TestCase):
    def setUp(self):
        # Create a mock outcomes configuration for testing
        self.outcomes = {
            "TEST_OUTCOME": {
                "type": ["CONCEPT"],
                "match": [["2"]],
                "exclude": ["D23"],
                "match_how": "contains",
                "case_sensitive": True,
            },
            "TEST_CENSOR": {
                "type": ["CONCEPT"],
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
                "PID": ["P1", "P2", "P3", "P4"],
                "CONCEPT": ["D13", "D2", "D23", "D2"],
                "TIMESTAMP": [
                    pd.Timestamp("2020-01-10"),
                    pd.Timestamp("2020-01-12"),
                    pd.Timestamp("2020-01-12"),
                    pd.Timestamp("2020-01-15"),
                ],
                "ADMISSION_ID": [1, 2, 3, 4],
            }
        )

        # Create a mock patients_info DataFrame
        self.patients_info = pd.DataFrame(
            {
                "PID": ["P1", "P2", "P3", "P4"],
                "info1": [1, 2, 3, 4],
                "TIMESTAMP": [
                    pd.Timestamp("2020-01-10"),
                    pd.Timestamp("2020-01-12"),
                    pd.NaT,
                    pd.Timestamp("2020-01-15"),
                ],
            }
        )

        # Patient set
        self.patient_set = ["P1", "P2", "P3"]

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
                "PID": ["P2"],
                "TIMESTAMP": [
                    pd.Timestamp("2020-01-12"),
                ],
            },
            index=[1],
        )
        expected_outcome["abspos"] = get_abspos_from_origin_point(
            expected_outcome["TIMESTAMP"], self.origin_point
        )
        expected_outcome["abspos"] = expected_outcome["abspos"].astype(int)
        # Check that the outcome table matches the expected result
        pd.testing.assert_frame_equal(
            result["TEST_OUTCOME"], expected_outcome, check_index_type=False
        )

        # Expected outcome for TEST_CENSOR
        expected_censor = pd.DataFrame(
            {"PID": ["P1"], "TIMESTAMP": [pd.Timestamp("2020-01-10")]}, index=[0]
        )
        expected_censor["abspos"] = get_abspos_from_origin_point(
            expected_censor["TIMESTAMP"], self.origin_point
        )
        expected_censor["abspos"] = expected_censor["abspos"].astype(int)
        # Check that the censor table matches the expected result
        pd.testing.assert_frame_equal(
            result["TEST_CENSOR"], expected_censor, check_index_type=False
        )


if __name__ == "__main__":
    unittest.main()
