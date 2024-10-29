import logging
import random
import shutil
import unittest
from os import makedirs
from os.path import exists, join

import numpy as np
import torch
import yaml


class TestMainScript(unittest.TestCase):
    def setUp(self):
        # Create tmp directory to use for output
        self.tmp_dir = "./.test_tmp"
        makedirs(self.tmp_dir, exist_ok=True)

        # Path to config file
        self.config_path = join(self.tmp_dir, "test_config.yaml")

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
        shutil.rmtree(self.tmp_dir)

    def set_config(self, config: dict):
        self.config = {
            "logging": {"level": logging.INFO, "path": join(self.tmp_dir, "logs")},
        } | config

        with open(self.config_path, "w") as config_file:
            yaml.dump(self.config, config_file)

    def check_config(self, cfg_path: str, compare_cfg_path: str = None) -> None:
        # Check exist
        self.assertTrue(exists(cfg_path))

        # Load
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Compare to...
        if compare_cfg_path is None:
            # ... current config
            compare_cfg = self.config
        else:
            # ... config @ compare_cfg_path
            with open(compare_cfg_path) as f:
                compare_cfg = yaml.safe_load(f)

        self.assertEqual(cfg.keys(), compare_cfg.keys())
        for key in cfg.keys():
            self.assertEqual(cfg[key], compare_cfg[key])
