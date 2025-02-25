from os.path import exists, join

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
                "features": {
                    "origin_point": {"year": 2020, "month": 1, "day": 26},
                },
                "tokenizer": {"sep_tokens": True, "cls_token": True},
                "excluder": {"min_age": -1, "max_age": 120},
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
        features_train = pd.read_parquet(join(self.features_dir, "train"))
        features_tuning = pd.read_parquet(join(self.features_dir, "tuning"))
        features_held_out = pd.read_parquet(join(self.features_dir, "held_out"))
        features = pd.concat([features_train, features_tuning, features_held_out])
        self.assertEqual(
            features.columns.to_list(),
            ["subject_id", "age", "abspos", "segment", "code"],
        )

        expected_features_train = pd.read_parquet(
            join("./tests/data/features", "train")
        )
        expected_features_tuning = pd.read_parquet(
            join("./tests/data/features", "tuning")
        )
        expected_features_held_out = pd.read_parquet(
            join("./tests/data/features", "held_out")
        )
        expected_features = pd.concat(
            [
                expected_features_train,
                expected_features_tuning,
                expected_features_held_out,
            ]
        )
        expected_features["subject_id"] = expected_features["subject_id"].astype(str)

        # 2.1: check patients
        self.assertListEqual(
            features["subject_id"].tolist(),
            expected_features["subject_id"].tolist(),
            "subject_id lists do not match.",
        )

        # 2.2: check number of entries per patient
        features_group = features.groupby("subject_id").size()
        expected_group = expected_features.groupby("subject_id").size()
        pd.testing.assert_series_equal(
            features_group,
            expected_group,
            check_names=False,
            obj="Event counts per subject_id do not match.",
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
                for subject_id in features["subject_id"].unique():
                    segment = set(
                        features[features["subject_id"] == subject_id]["segment"].values
                    )
                    expected_segment = set(
                        expected_features[
                            expected_features["subject_id"] == subject_id
                        ]["segment"].values
                    )
                    self.assertEqual(
                        segment,
                        expected_segment,
                        f"Segments for subject_id {subject_id} do not match.",
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
        for suffix in ["train", "tuning", "held_out"]:
            pids_path = join(self.tokenized_dir, f"pids_{suffix}.pt")
            self.assertTrue(exists(pids_path))
            pids = torch.load(pids_path, weights_only=False)
            expected_pids = torch.load(
                f"./tests/data/tokenized/pids_{suffix}.pt", weights_only=False
            )
            self.assertEqual(pids, expected_pids)
