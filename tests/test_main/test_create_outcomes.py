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


class TestCreateOutcomes(unittest.TestCase):
    def setUp(self):
        # Create tmp directory to use for output
        self.root_dir = "./.test_create_outcomes"
        makedirs(self.root_dir, exist_ok=True)

        # Create config file
        self.output_dir = join(self.root_dir, "outputs")
        self.config_path = join(self.root_dir, "create_outcomes.yaml")
        self.outcomes_dir = join(self.output_dir, "outcomes")

        self.config = {
            "logging": {"level": logging.INFO, "path": join(self.root_dir, "logs")},
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

        with open(self.config_path, "w") as config_file:
            yaml.dump(self.config, config_file)

        # Set seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def tearDown(self):
        # Remove all outputs
        shutil.rmtree(self.root_dir)

    def test_create_outcomes(self):
        ### Call create data script

        main_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of data and outcomes configuration files should be created
        #    in the outcomes dir

        # 1.1: Data config
        self.assertTrue(exists(join(self.outcomes_dir, DATA_CFG)))

        # 1.2: Outcomes config (also check contents)
        cfg_file = join(self.outcomes_dir, OUTCOMES_CFG)
        self.assertTrue(exists(cfg_file))
        with open(cfg_file) as f:
            config = yaml.safe_load(f)
        self.assertEqual(config, self.config)

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
