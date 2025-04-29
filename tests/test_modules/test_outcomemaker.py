import datetime
import unittest

import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    CONCEPT_COL,
    PID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
)
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.outcomes import (
    OutcomeMaker,
)  # Update with your actual import path


class TestOutcomeMaker(unittest.TestCase):
    """Test outcome maker functionality"""

    def setUp(self):
        """Setup test data"""
        # Create test concepts data
        self.concepts_plus = pd.DataFrame(
            {
                PID_COL: [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                CONCEPT_COL: [
                    "D10.1",
                    "M112",
                    "D02.3",
                    "M112",
                    "D10.2",
                    "D10.5",
                    "M112",
                    "DOD",
                    "DI20",
                    "DOD",
                ],
                VALUE_COL: [
                    "pos",
                    "neg",
                    "pos",
                    "pos",
                    "neg",
                    "pos",
                    "neg",
                    "yes",
                    "yes",
                    "yes",
                ],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 1, 1, 10, 0),
                    datetime.datetime(2020, 1, 2, 10, 0),
                    datetime.datetime(2020, 1, 3, 10, 0),
                    datetime.datetime(2020, 1, 4, 10, 0),
                    datetime.datetime(2020, 1, 5, 10, 0),
                    datetime.datetime(2020, 1, 6, 10, 0),
                    datetime.datetime(2020, 1, 7, 10, 0),
                    datetime.datetime(2020, 1, 10, 10, 0),  # Patient 4: Death
                    datetime.datetime(
                        2020, 1, 9, 15, 0
                    ),  # Patient 4: Heart infarct (19 hours before death)
                    datetime.datetime(
                        2020, 1, 15, 10, 0
                    ),  # Patient 5: Death (with no prior heart infarct)
                ],
            }
        )

        # Add a patient with multiple events for testing more complex combinations
        combination_data = pd.DataFrame(
            {
                PID_COL: [6, 6, 6, 6, 7, 7],
                CONCEPT_COL: ["I63", "B01", "I63", "B01", "I63", "B01"],
                VALUE_COL: ["diag", "med", "diag", "med", "diag", "med"],
                TIMESTAMP_COL: [
                    datetime.datetime(2020, 2, 1, 10, 0),  # Patient 6: Stroke 1
                    datetime.datetime(
                        2020, 2, 1, 20, 0
                    ),  # Patient 6: Anticoagulant 1 (10 hours after stroke 1)
                    datetime.datetime(2020, 2, 10, 10, 0),  # Patient 6: Stroke 2
                    datetime.datetime(
                        2020, 2, 9, 10, 0
                    ),  # Patient 6: Anticoagulant 2 (24 hours before stroke 2)
                    datetime.datetime(2020, 2, 15, 10, 0),  # Patient 7: Stroke
                    datetime.datetime(
                        2020, 2, 20, 10, 0
                    ),  # Patient 7: Anticoagulant (120 hours after - outside window)
                ],
            }
        )

        self.concepts_plus = pd.concat(
            [self.concepts_plus, combination_data], ignore_index=True
        )

        # Create patient info data
        self.patients_info = pd.DataFrame(
            {
                PID_COL: [1, 2, 3, 4, 5, 6, 7],
                "gender": ["M", "F", "M", "F", "M", "F", "M"],
            }
        )

        # Patient set for testing
        self.patient_set = [1, 2, 3, 4, 5, 6, 7]

    def test_basic_outcome_creation(self):
        """Test basic outcome creation"""
        # Define test outcomes
        outcomes = {
            "TEST_OUTCOME": {
                "type": ["code"],
                "match": [["D10"]],
                "match_how": "startswith",
                "case_sensitive": True,
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result
        self.assertIn("TEST_OUTCOME", result)
        test_outcome = result["TEST_OUTCOME"]

        # Should contain 3 rows (Patient 1, 2, and 3 have D10.x codes)
        self.assertEqual(len(test_outcome), 3)
        self.assertTrue(all(pid in test_outcome[PID_COL].values for pid in [1, 2, 3]))

        # Check ABSPOS calculation
        self.assertTrue(ABSPOS_COL in test_outcome.columns)
        # Verify that ABSPOS is calculated correctly for one entry
        first_timestamp = test_outcome.iloc[0][TIMESTAMP_COL]
        expected_hours = get_hours_since_epoch(first_timestamp)
        self.assertEqual(test_outcome.iloc[0][ABSPOS_COL], int(expected_hours))

    def test_exclude_and_case_sensitivity(self):
        """Test exclude feature and case sensitivity"""
        # Define test outcomes with exclude and case insensitivity
        outcomes = {
            "TEST_EXCLUDE": {
                "type": ["code"],
                "match": [["D1"]],
                "exclude": ["D10.5"],  # Exclude D10.5 specifically
                "match_how": "startswith",
                "case_sensitive": True,
            },
            "TEST_CASE_INSENSITIVE": {
                "type": ["code"],
                "match": [
                    ["m112"]
                ],  # lowercase, should match with case_sensitive=False
                "match_how": "startswith",
                "case_sensitive": False,
            },
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check exclude result
        self.assertIn("TEST_EXCLUDE", result)
        exclude_outcome = result["TEST_EXCLUDE"]
        # Should contain 2 rows (Patient 1, 2 have D10.x codes, but not D10.5)
        self.assertEqual(len(exclude_outcome), 2)
        self.assertTrue(all(pid in exclude_outcome[PID_COL].values for pid in [1, 2]))

        # Check case insensitive result
        self.assertIn("TEST_CASE_INSENSITIVE", result)
        case_outcome = result["TEST_CASE_INSENSITIVE"]
        # Should contain 3 rows (Patient 1, 2, 3 have M112 codes)
        self.assertEqual(len(case_outcome), 3)
        self.assertTrue(all(pid in case_outcome[PID_COL].values for pid in [1, 2, 3]))

    def test_negation(self):
        """Test negation feature"""
        # Define test outcomes with negation
        outcomes = {
            "NOT_M112": {
                "type": ["code"],
                "match": [["M112"]],
                "match_how": "startswith",
                "case_sensitive": True,
                "negation": True,
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result
        self.assertIn("NOT_M112", result)
        not_outcome = result["NOT_M112"]

        # Should contain rows for DOD, DI20, I63, and B01 codes (all non-M112 codes)
        expected_concepts = [
            "DOD",
            "DI20",
            "I63",
            "B01",
            "D10.1",
            "D02.3",
            "D10.2",
            "D10.5",
        ]
        # Count how many of these we expect
        expected_count = len(
            self.concepts_plus[self.concepts_plus[CONCEPT_COL].isin(expected_concepts)]
        )
        self.assertEqual(len(not_outcome), expected_count)

    def test_death_from_mi_combination(self):
        """Test combination outcome for death from myocardial infarction"""
        # Define outcome for death with MI
        outcomes = {
            "DEATH_FROM_MI": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "startswith",
                    },
                    "window_hours": 24,  # Look for heart infarct within 24 hours of death
                    "direction": "before",  # Heart infarct should come before death
                    "timestamp_source": "primary",  # Use death timestamp
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result
        self.assertIn("DEATH_FROM_MI", result)
        death_mi_outcome = result["DEATH_FROM_MI"]

        # Should only include patient 4 who had DI20 before DOD within the window
        self.assertEqual(len(death_mi_outcome), 1)
        self.assertEqual(death_mi_outcome.iloc[0][PID_COL], 4)

        # Timestamp should be from the primary event (DOD)
        expected_timestamp = datetime.datetime(
            2020, 1, 10, 10, 0
        )  # DOD timestamp for patient 4
        self.assertEqual(death_mi_outcome.iloc[0][TIMESTAMP_COL], expected_timestamp)

    def test_stroke_with_anticoagulant_combination(self):
        """Test combination outcome for stroke with anticoagulant therapy"""
        # Define outcome for stroke with anticoagulant
        outcomes = {
            "STROKE_WITH_ANTICOAGULANT": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["I63"]],  # Stroke
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["B01"]],  # Anticoagulant
                        "match_how": "startswith",
                    },
                    "window_hours": 48,  # Look for anticoagulant within 48 hours
                    "direction": "any",  # Anticoagulant can be before or after stroke
                    "timestamp_source": "primary",  # Use stroke timestamp
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Add debug prints

        # Check result
        self.assertIn("STROKE_WITH_ANTICOAGULANT", result)
        stroke_outcome = result["STROKE_WITH_ANTICOAGULANT"]

        # Should include both stroke events for patient 6
        # Patient 7's stroke and anticoagulant are outside the window
        self.assertEqual(len(stroke_outcome), 2)
        self.assertTrue(all(pid == 6 for pid in stroke_outcome[PID_COL].values))

        # Timestamps should match the two stroke events for patient 6
        expected_timestamps = [
            datetime.datetime(2020, 2, 1, 10, 0),  # Stroke 1
            datetime.datetime(2020, 2, 10, 10, 0),  # Stroke 2
        ]
        actual_timestamps = pd.to_datetime(stroke_outcome[TIMESTAMP_COL].values)
        self.assertTrue(all(ts in actual_timestamps for ts in expected_timestamps))

    def test_secondary_timestamp_source(self):
        """Test combination with secondary timestamp source"""
        # Define outcome using the secondary timestamp
        outcomes = {
            "MI_BEFORE_DEATH": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "startswith",
                    },
                    "window_hours": 24,
                    "direction": "before",
                    "timestamp_source": "secondary",  # Use heart infarct timestamp instead
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result
        self.assertIn("MI_BEFORE_DEATH", result)
        mi_outcome = result["MI_BEFORE_DEATH"]

        # Should only include patient 4
        self.assertEqual(len(mi_outcome), 1)
        self.assertEqual(mi_outcome.iloc[0][PID_COL], 4)

        # Timestamp should be from the secondary event (DI20)
        expected_timestamp = datetime.datetime(
            2020, 1, 9, 15, 0
        )  # DI20 timestamp for patient 4
        self.assertEqual(mi_outcome.iloc[0][TIMESTAMP_COL], expected_timestamp)

    def test_direction_after(self):
        """Test combination with 'after' direction"""
        # Define outcome with direction 'after'
        outcomes = {
            "ANTICOAGULANT_AFTER_STROKE": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["I63"]],  # Stroke
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["B01"]],  # Anticoagulant
                        "match_how": "startswith",
                    },
                    "window_hours": 48,
                    "direction": "after",  # Anticoagulant must be after stroke
                    "timestamp_source": "primary",
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result
        self.assertIn("ANTICOAGULANT_AFTER_STROKE", result)
        after_outcome = result["ANTICOAGULANT_AFTER_STROKE"]

        # Should only include the first stroke for patient 6 (which has anticoagulant after it)
        self.assertEqual(len(after_outcome), 1)
        self.assertEqual(after_outcome.iloc[0][PID_COL], 6)

        # Timestamp should match the first stroke
        expected_timestamp = datetime.datetime(2020, 2, 1, 10, 0)
        self.assertEqual(after_outcome.iloc[0][TIMESTAMP_COL], expected_timestamp)

    def test_empty_result(self):
        """Test handling of combinations that yield no results"""
        # Define outcome with impossible criteria
        outcomes = {
            "IMPOSSIBLE_COMBINATION": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["NONEXISTENT_CODE"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["ANOTHER_NONEXISTENT"]],
                        "match_how": "startswith",
                    },
                    "window_hours": 24,
                }
            }
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result
        self.assertIn("IMPOSSIBLE_COMBINATION", result)
        empty_outcome = result["IMPOSSIBLE_COMBINATION"]

        # Should be empty but have the proper columns
        self.assertEqual(len(empty_outcome), 0)
        self.assertTrue(
            all(
                col in empty_outcome.columns
                for col in [PID_COL, TIMESTAMP_COL, ABSPOS_COL]
            )
        )

    def test_multiple_outcomes_together(self):
        """Test handling multiple outcome types in the same call"""
        # Define multiple outcome types
        outcomes = {
            "BASIC_OUTCOME": {
                "type": ["code"],
                "match": [["D10"]],
                "match_how": "startswith",
            },
            "COMBINATION_OUTCOME": {
                "combinations": {
                    "primary": {
                        "type": ["code"],
                        "match": [["DOD"]],
                        "match_how": "startswith",
                    },
                    "secondary": {
                        "type": ["code"],
                        "match": [["DI20"]],
                        "match_how": "startswith",
                    },
                    "window_hours": 24,
                }
            },
        }

        # Create outcome maker
        outcome_maker = OutcomeMaker(outcomes)

        # Get outcomes
        result = outcome_maker(self.concepts_plus, self.patients_info, self.patient_set)

        # Check result - all outcome types should be present
        self.assertEqual(len(result), 2)
        self.assertIn("BASIC_OUTCOME", result)
        self.assertIn("COMBINATION_OUTCOME", result)


if __name__ == "__main__":
    unittest.main()
