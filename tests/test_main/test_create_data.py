import logging
import random
import shutil
import unittest
from os import makedirs
from os.path import exists, join

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
import yaml
from tests.helpers import compute_column_checksum

from corebehrt.main.create_data import main_data

DATA_CFG = "data_config.yaml"


class TestCreateData(unittest.TestCase):
    def setUp(self):
        # Create tmp directory to use for output
        self.root_dir = "./.test_create_data"
        makedirs(self.root_dir, exist_ok=True)

        # Create config file
        self.output_dir = join(self.root_dir, "outputs")
        self.config_path = join(self.root_dir, "create_data.yaml")
        self.features_dir = join(self.output_dir, "features")
        self.tokenized_dir = join(self.output_dir, "tokenized")

        self.config = {
            "logging": {"level": logging.INFO, "path": join(self.root_dir, "logs")},
            "paths": {
                "data": "./tests/data/raw",
                "features": self.features_dir,
                "tokenized": self.tokenized_dir,
            },
            "loader": {
                "concept_types": ["diagnose", "medication"],
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

        # 1: Copy of configuration file should be created in the features and tokenized dirs.
        for _dir in [self.features_dir, self.tokenized_dir]:
            cfg_file = join(_dir, DATA_CFG)
            self.assertTrue(exists(cfg_file))
            with open(cfg_file) as f:
                config = yaml.safe_load(f)
            self.assertEqual(config, self.config)

        # 2: Check that the features file is created as expected
        self.assertTrue(exists(self.features_dir))
        features = dd.read_csv(join(self.features_dir, "*.csv")).compute()
        self.assertEqual(
            features.columns.to_list(), ["PID", "concept", "age", "abspos", "segment"]
        )

        expected_features = dd.read_csv("./tests/data/prepped/features/*.csv").compute()

        # 2.1: check patients
        self.assertListEqual(
            features["PID"].tolist(),
            expected_features["PID"].tolist(),
            "PID lists do not match.",
        )

        # 2.2: check number of entries per patient
        features_group = features.groupby("PID").size()
        expected_group = expected_features.groupby("PID").size()
        pd.testing.assert_series_equal(
            features_group,
            expected_group,
            check_names=False,
            obj="Event counts per PID do not match.",
        )

        # 2.3: checksum
        for col in features.columns:
            if col != "segment":
                checksum = compute_column_checksum(features, col)
                expected_checksum = compute_column_checksum(expected_features, col)
                self.assertEqual(
                    checksum, expected_checksum, f"Checksum for {col} does not match."
                )
            else:  # compare sets for every patient
                for pid in features["PID"].unique():
                    segment = set(features[features["PID"] == pid]["segment"].values)
                    expected_segment = set(
                        expected_features[expected_features["PID"] == pid][
                            "segment"
                        ].values
                    )
                    self.assertEqual(
                        segment,
                        expected_segment,
                        f"Segments for PID {pid} do not match.",
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
