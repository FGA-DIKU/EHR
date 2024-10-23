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

from corebehrt.main.create_data import main_data
from corebehrt.functional.convert import convert_to_sequences

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
            "paths": {
                "save_features_dir_name": "features",
            },
            "loader": {
                "data_dir": "./tests/data/raw_w_labs",
                "concept_types": ["diagnose", "medication"], #, "labtest"],
                "include_values": ["labtest"],
                "value_type": "binned_value"
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
        path = join(self.output_dir, "features")
        self.assertTrue(exists(path))
        features = dd.read_csv(join(path, "*.csv")).compute()
        self.assertEqual(
            features.columns.to_list(), ["PID", "concept", "age", "abspos", "segment"]
        )

        # 3: Check patients
        expected_pids = pd.read_csv("./tests/data/raw_w_labs/patients_info.csv")["PID"].tolist()
        self.assertEqual(sorted(features["PID"].unique().tolist()), sorted(expected_pids))
       
        # 3: Check vocabulary
        vocab_path = join(self.tokenized_dir, "vocabulary.pt")
        self.assertTrue(exists(vocab_path))
        vocab = torch.load(vocab_path)
        bg_tokens = [v for k, v in vocab.items() if k.startswith("BG")]
        self.assertEqual(len(bg_tokens), 2)
        val_tokens = [v for k, v in vocab.items() if k.startswith("VAL")]
        
        inv_vocab = {v: k for k, v in vocab.items()}
        # Check tokenisation
        for mode in ["pretrain", "finetune", "test"]:
            tokenised_features  = dd.read_csv(
                    join(
                        self.tokenized_dir,
                        f"features_{mode}",
                        "*.csv",
                    )
                )
            sequences, _ = convert_to_sequences(tokenised_features)
            for cons, positions in zip(sequences["concept"], sequences["abspos"]):
                self.assertTrue(cons[0] == vocab['[CLS]'])
                self.assertTrue((cons[1] in bg_tokens))
               
                index_vals = [i for i, x in enumerate(cons) if x in val_tokens]
                for i in range(len(index_vals) - 1):
                    self.assertNotEqual(index_vals[i] + 1, index_vals[i + 1], 
                                        f"Indices {index_vals[i]} and {index_vals[i + 1]} are next to each other")
