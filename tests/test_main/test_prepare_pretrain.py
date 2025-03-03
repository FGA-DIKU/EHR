from os.path import exists, join

import yaml

from corebehrt.constants.paths import DATA_CFG, PREPARE_PRETRAIN_CFG
from corebehrt.main.prepare_training_data import main_prepare_data

from .base import TestMainScript


class TestCreateOutcomes(TestMainScript):
    def setUp(self):
        super().setUp()

        self.processed_dir = join(self.tmp_dir, "pretrain/processed_data")

        # Load config from pretrain config file
        prepare_pretrain_config_path = (
            f"./tests/data/pretrain/processed_data/{PREPARE_PRETRAIN_CFG}"
        )
        with open(prepare_pretrain_config_path) as f:
            config = yaml.safe_load(f)

        # Update paths in the loaded config
        config["paths"] = {
            "features": "./tests/data/tokenized",
            "tokenized": "./tests/data/tokenized",
            "prepared_data": self.processed_dir,
        }

        self.set_config(config)

    def test_prepare_pretrain(self):
        ### Call pretrain script

        main_prepare_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of data and pretrain configuration files should be created
        #    in the outcomes dir
        self.check_config(
            join(self.processed_dir, DATA_CFG), f"./tests/data/features/{DATA_CFG}"
        )

        # 2: pids
        for file_name in [
            "pids_train",
            "pids_val",
            "patients_train",
            "patients_val",
            "patients",
        ]:
            self.assertTrue(exists(join(self.processed_dir, f"{file_name}.pt")))
