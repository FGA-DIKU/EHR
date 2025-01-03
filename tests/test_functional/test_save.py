import unittest
import dask.dataframe as dd
import torch
import os
import tempfile
import shutil
from corebehrt.functional.save import save_pids_splits
import pandas as pd


class TestSaveFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_pids_splits(self):
        mock_train_df = dd.from_pandas(
            pd.DataFrame({"PID": ["pid1", "pid2"]}), npartitions=1
        )
        mock_val_df = dd.from_pandas(
            pd.DataFrame({"PID": ["pid3", "pid4"]}), npartitions=1
        )

        save_pids_splits(mock_train_df, mock_val_df, self.test_dir)

        # Load and verify train PIDs
        train_pids = torch.load(
            os.path.join(self.test_dir, "pids_train.pt"), weights_only=True
        )
        self.assertEqual(train_pids, ["pid1", "pid2"])
        # Load and verify val PIDs
        val_pids = torch.load(
            os.path.join(self.test_dir, "pids_val.pt"), weights_only=True
        )
        self.assertEqual(val_pids, ["pid3", "pid4"])


if __name__ == "__main__":
    unittest.main()
