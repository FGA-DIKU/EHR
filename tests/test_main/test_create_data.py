from os.path import exists, join

import dask.dataframe as dd
import pandas as pd
import torch

from corebehrt.constants.paths import DATA_CFG
from corebehrt.main.create_data import main_data
from tests.helpers import compute_column_checksum

from .base import TestMainScript


class TestCreateData(TestMainScript):
    def setUp(self):
        super().setUp()

        # Paths
        self.features_dir = join(self.tmp_dir, "features")
        self.tokenized_dir = join(self.tmp_dir, "tokenized")

        self.set_config(
            {
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
                "excluder": {"min_age": -1, "max_age": 120},
                "split_ratios": {"pretrain": 0.72, "finetune": 0.18, "test": 0.1},
            }
        )

    def test_create_data(self):
        ### Call create data script

        main_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of configuration file should be created in the features and tokenized dirs.
        self.check_config(join(self.features_dir, DATA_CFG))
        self.check_config(join(self.tokenized_dir, DATA_CFG))

        # 2: Check that the features file is created as expected
        self.assertTrue(exists(self.features_dir))
        features = dd.read_parquet(self.features_dir).compute()
        self.assertEqual(
            features.columns.to_list(), ["PID", "concept", "age", "abspos", "segment"]
        )

        expected_features = dd.read_parquet("./tests/data/features").compute()

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
        vocab = torch.load(vocab_path, weights_only=True)
        expected_vocab = torch.load(
            "./tests/data/tokenized/vocabulary.pt", weights_only=True
        )
        self.assertEqual(len(vocab), len(expected_vocab))

        # 4: Check the tokenized files
        for suffix in ["pretrain", "finetune", "test"]:
            pids_path = join(self.tokenized_dir, f"pids_{suffix}.pt")
            self.assertTrue(exists(pids_path))
            pids = torch.load(pids_path, weights_only=True)
            expected_pids = torch.load(
                f"./tests/data/tokenized/pids_{suffix}.pt", weights_only=True
            )
            self.assertEqual(pids, expected_pids)
