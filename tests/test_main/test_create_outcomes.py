from os.path import exists, join

import pandas as pd

from corebehrt.constants.paths import DATA_CFG, OUTCOMES_CFG
from corebehrt.main.create_outcomes import main_data

from tests.test_main.base import TestMainScript


class TestCreateOutcomes(TestMainScript):
    def setUp(self):
        """
        Sets up the test environment and configuration for outcome generation tests.
        
        Initializes a temporary outcomes directory and writes a configuration file specifying
        paths and outcome definitions for use in subsequent tests.
        """
        super().setUp()

        # Create config file
        self.outcomes_dir = join(self.tmp_dir, "outcomes")

        self.set_config(
            {
                "paths": {
                    "data": "./tests/data/raw_with_values",
                    "features": "./tests/data/features",
                    "outcomes": self.outcomes_dir,
                },
                "outcomes": {
                    "TEST_OUTCOME": {
                        "type": ["code"],
                        "match": [["DE11"]],
                        "exclude": ["D437"],
                        "match_how": "startswith",
                        "case_sensitive": True,
                    },
                    "TEST_CENSOR": {
                        "type": ["code"],
                        "match": [["M112"]],
                        "match_how": "startswith",
                        "case_sensitive": False,
                    },
                    "TEST_COMBINED": {
                        "combinations": {
                            "primary": {
                                "type": ["code"],
                                "match": [["M8"]],
                                "match_how": "startswith",
                            },
                            "secondary": {
                                "type": ["code"],
                                "match": [["LAB", "D5"]],
                                "match_how": "startswith",
                            },
                            "window_hours_min": -100,
                            "window_hours_max": 100,
                            "timestamp_source": "primary",
                        }
                    },
                },
            }
        )

    def test_create_outcomes(self):
        ### Call create outcomes script

        """
        Tests that the main_data function generates correct outcome files and configuration copies.
        
        This test verifies that outcome CSV files and configuration files are created in the specified
        output directory, and that the generated outcome files match expected results in both structure
        and content.
        """
        main_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of data and outcomes configuration files should be created
        #    in the outcomes dir
        self.check_config(
            join(self.outcomes_dir, DATA_CFG), f"./tests/data/features/{DATA_CFG}"
        )
        self.check_config(join(self.outcomes_dir, OUTCOMES_CFG))

        # 2: Check the required outcome/exposure files are created as expected
        for file_name in ["TEST_CENSOR", "TEST_OUTCOME", "TEST_COMBINED"]:
            path = join(self.outcomes_dir, f"{file_name}.csv")

            # Exists:
            self.assertTrue(exists(path))
            output = pd.read_csv(path)

            # Columns?
            self.assertEqual(output.columns.to_list(), ["subject_id", "time", "abspos"])

            # Compare to expected
            expected_output = pd.read_csv(
                join(f"./tests/data/outcomes/{file_name}.csv")
            )
            for idx, ((_, row), (_, expected_row)) in enumerate(
                zip(output.iterrows(), expected_output.iterrows())
            ):
                for column in output.columns:
                    self.assertEqual(
                        row[column],
                        expected_row[column],
                        f"Unexpected value at row {idx}, column {column}",
                    )
