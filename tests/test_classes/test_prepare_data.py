import unittest
import shutil
import numpy as np
import yaml
import torch
import random
from os import makedirs
from os.path import join
from corebehrt.classes.prepare_data import DatasetPreparer
from corebehrt.classes.dataset import MLMDataset
from corebehrt.common.config import load_config


class TestPrepareData(unittest.TestCase):
    def setUp(self):
        # Create tmp directory to use for output
        self.root_dir = "./.test_tmp"
        makedirs(self.root_dir, exist_ok=True)

        self.config_path = join(self.root_dir, "config.yaml")

        # Create config file
        self.config = {
            "env": "local",
            "paths": {
                "data_path": "./tests/data/prepped",
                "output_path": join(self.root_dir, "pretrain"),
                "run_name": "test_run",
            },
            "data": {
                "dataset": {
                    "select_ratio": 1.0,
                    "masking_ratio": 0.8,
                    "replace_ratio": 0.1,
                    "ignore_special_tokens": True,
                },
                "truncation_len": 20,
                "num_train_patients": 100,
                "num_val_patients": 20,
                "val_ratio": 0.2,
                "min_len": 2,
            },
            "model": {
                "linear": True,
                "hidden_size": 96,
                "num_hidden_layers": 3,
                "num_attention_heads": 3,
                "intermediate_size": 64,
                "type_vocab_size": 240,
                "plusplus": True,
                "abspos_embeddings": False,
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

    def test_prepare_mlm(self):
        cfg = load_config(self.config_path)

        train_dataset, val_dataset = DatasetPreparer(cfg).prepare_mlm_dataset()

        # Test types
        self.assertIsInstance(train_dataset, MLMDataset)
        self.assertIsInstance(val_dataset, MLMDataset)

        # Test lengths
        self.assertEqual(len(train_dataset), 5)
        self.assertEqual(len(val_dataset), 2)
