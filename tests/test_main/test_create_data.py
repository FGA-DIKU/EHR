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
from corebehrt.main.create_data import main_data


class TestCreateData(unittest.TestCase):
    def setUp(self):
        # Create tmp directory to use for output
        self.root_dir = "./.test_create_data"
        makedirs(self.root_dir, exist_ok=True)

        # Create config file
        self.output_dir = join(self.root_dir, "outputs")
        self.config_path = join(self.root_dir, "create_data.yaml")
        self.tokenized_dir = join(self.output_dir, "tokenized")

        self.config = {
            "env": "local",
            "output_dir": self.output_dir,
            "tokenized_dir_name": "tokenized",
            "paths": {},
            "loader": {
                "data_dir": "./tests/data/raw",
                "concepts": ["diagnosis", "medication"],
            },
            "features": {
                "origin_point": {"year": 2020, "month": 1, "day": 26},
                "background_vars": ["GENDER"],
            },
            "tokenizer": {"sep_tokens": True, "cls_token": True},
            "excluder": {"min_len": 2, "min_age": -1, "max_age": 120},
            "split_ratios": {"pretrain": 0.72, "finetune": 0.18, "test": 0.1},
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
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()
        shutil.rmtree(self.root_dir)

    def test_create_data(self):
        ### Call create data script

        main_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of configuration file should be created in the output dir.
        self.assertTrue(exists(join(self.output_dir, "data_config.yaml")))
        with open(join(self.output_dir, "data_config.yaml")) as f:
            config = yaml.safe_load(f)
        self.assertEqual(config, self.config)

        # 2: Check that the features file is created as expected
        path = join(self.output_dir, "features", "features.csv")
        self.assertTrue(exists(path))
        features = pd.read_csv(path)
        self.assertEqual(
            features.columns.to_list(), ["PID", "concept", "age", "segment", "abspos"]
        )

        expected_features = pd.read_csv("./tests/data/prepped/features/features.csv")
        for idx, ((_, row), (_, expected_row)) in enumerate(
            zip(features.iterrows(), expected_features.iterrows())
        ):
            for column in features.columns:
                self.assertEqual(
                    row[column],
                    expected_row[column],
                    f"Unexpected value at row {idx}, column {column}",
                )

        # 3: Check vocabulary
        vocab_path = join(self.tokenized_dir, "vocabulary.pt")
        self.assertTrue(exists(vocab_path))
        vocab = torch.load(vocab_path)
        expected_vocab = torch.load("./tests/data/prepped/tokenized/vocabulary.pt")
        self.assertEqual(len(vocab), len(expected_vocab))

        # 4: Check the tokenized files
        for suffix in ["pretrain", "finetune", "test"]:
            pids_path = join(self.tokenized_dir, f"pids_{suffix}.pt")
            self.assertTrue(exists(pids_path))
            pids = torch.load(pids_path)
            expected_pids = torch.load(
                f"./tests/data/prepped/tokenized/pids_{suffix}.pt"
            )
            self.assertEqual(pids, expected_pids)
