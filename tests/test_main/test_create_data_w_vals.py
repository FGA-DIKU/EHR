from os.path import exists, join

import dask.dataframe as dd
from tests.helpers import compute_column_checksum
import torch
import pandas as pd

from corebehrt.main.create_data import main_data
from corebehrt.common.setup import DATA_CFG

from .base import TestMainScript
from corebehrt.functional.convert import convert_to_sequences

class TestCreateData(TestMainScript):
    def setUp(self):
        super().setUp()

        # Paths
        self.features_dir = join(self.tmp_dir, "features")
        self.tokenized_dir = join(self.tmp_dir, "tokenized")
        self.data_path = "./tests/data/raw_with_values"
        self.set_config(
            {
                "paths": {
                    "data": self.data_path,
                    "features": self.features_dir,
                    "tokenized": self.tokenized_dir,
                },
                "loader": {
                    "concept_types": ["diagnose", "medication", "labtest"],
                    "include_values": ["labtest"],
                },
                "features": {
                "origin_point": {"year": 2020, "month": 1, "day": 26},
                "background_vars": ["GENDER"],
                "values": {
                    "value_type": "binned",
                    "value_type_kwargs": {"multiplication_factor": 100},
                    "normalize_args": {
                        "func": "corebehrt.classes.normalizer.ValuesNormalizer.min_max_normalize_results",
                        "kwargs": {"min_count": 3},
                    },
                },
            },
                "tokenizer": {"sep_tokens": True, "cls_token": True},
                "excluder": {"min_len": 2, "min_age": -1, "max_age": 120},
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
        features = dd.read_csv(join(self.features_dir, "*.csv")).compute()
        self.assertEqual(
            features.columns.to_list(), ["PID", "concept", "age", "abspos", "segment"]
        )

        # 3: Check patients
        expected_pids = pd.read_csv(f"{self.data_path}/patients_info.csv")[
            "PID"
        ].tolist()
        self.assertEqual(
            sorted(features["PID"].unique().tolist()), sorted(expected_pids)
        )

        # 4: Check vocabulary
        vocab_path = join(self.tokenized_dir, "vocabulary.pt")
        self.assertTrue(exists(vocab_path))
        vocab = torch.load(vocab_path)
        bg_tokens = [v for k, v in vocab.items() if k.startswith("BG")]
        self.assertEqual(len(bg_tokens), 2)
        val_tokens = [v for k, v in vocab.items() if k.startswith("VAL")]

        # 5. Check tokenisation
        for mode in ["pretrain", "finetune", "test"]:
            tokenised_features = dd.read_csv(
                join(
                    self.tokenized_dir,
                    f"features_{mode}",
                    "*.csv",
                )
            )
            sequences, _ = convert_to_sequences(tokenised_features)
            for cons, positions in zip(sequences["concept"], sequences["abspos"]):
                self.assertTrue(cons[0] == vocab["[CLS]"])
                self.assertTrue((cons[1] in bg_tokens))

                index_vals = [i for i, x in enumerate(cons) if x in val_tokens]
                for i in range(len(index_vals) - 1):
                    self.assertNotEqual(
                        index_vals[i] + 1,
                        index_vals[i + 1],
                        f"Indices {index_vals[i]} and {index_vals[i + 1]} are next to each other",
                    )
