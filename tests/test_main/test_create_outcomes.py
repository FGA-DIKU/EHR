from os import makedirs
from os.path import join, exists
import yaml
import unittest
import pandas as pd
import torch
import numpy as np
import random
import shutil
import logging

from corebehrt.main.create_outcomes import main_data
from corebehrt.common.setup import DATA_CFG, OUTCOMES_CFG

from .base import TestMainScript


class TestCreateOutcomes(TestMainScript):
    def setUp(self):
        super().setUp()

        # Create config file
        self.outcomes_dir = join(self.tmp_dir, "outcomes")

        self.set_config(
            {
                "paths": {
                    "data": "./tests/data/raw",
                    "features": "./tests/data/features",
                    "outcomes": self.outcomes_dir,
                },
                "loader": {
                    "concepts": ["diagnose"],
                    "batchsize": 50,
                    "chunksize": 300,
                },
                "outcomes": {
                    "TEST_OUTCOME": {
                        "type": ["CONCEPT"],
                        "match": [["157"]],
                        "exclude": ["157141000119108"],
                        "match_how": "contains",
                        "case_sensitive": True,
                    },
                    "TEST_CENSOR": {
                        "type": ["CONCEPT"],
                        "match": [["10509"]],
                        "match_how": "startswith",
                        "case_sensitive": False,
                    },
                },
            }
        )

    def test_create_outcomes(self):
        ### Call create outcomes script

        main_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of data and outcomes configuration files should be created
        #    in the outcomes dir
        self.check_config(
            join(self.outcomes_dir, DATA_CFG), f"./tests/data/features/{DATA_CFG}"
        )
        self.check_config(join(self.outcomes_dir, OUTCOMES_CFG))

        # 2: Check the required outcome/exposure files are created as expected
        for file_name in ["TEST_CENSOR", "TEST_OUTCOME"]:
            path = join(self.outcomes_dir, f"{file_name}.csv")

            # Exists:
            self.assertTrue(exists(path))
            output = pd.read_csv(path)

            # Columns?
            self.assertEqual(output.columns.to_list(), ["TIMESTAMP", "PID", "abspos"])

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
