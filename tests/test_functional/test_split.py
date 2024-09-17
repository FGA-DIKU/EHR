import unittest
from unittest.mock import patch, MagicMock
from corebehrt.functional.split import split_pids_into_pt_ft_test, split_pids_into_train_val, load_train_val_split
import dask.dataframe as dd
import pandas as pd
import tempfile
import os 
import torch

class TestSplitFunctions(unittest.TestCase):
    def setUp(self):
        # Setup temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create temporary .pt files
        self.single_file = os.path.join(self.temp_dir.name, "single.pt")
        torch.save({1, 2, 3}, self.single_file)

        self.multi_files = [
            os.path.join(self.temp_dir.name, f"multi_{i}.pt") for i in range(2)
        ]
        for i, file in enumerate(self.multi_files):
            torch.save({i * 3 + 1, i * 3 + 2, i * 3 + 3}, file)

        self.split_path = self.temp_dir.name
        self.mode_files = [
            os.path.join(self.split_path, f"pids_train.pt"),
            os.path.join(self.split_path, f"pids_val.pt"),
            os.path.join(self.split_path, f"pids_val.pt"),
        ]
        torch.save({1, 2, 3}, self.mode_files[0])
        torch.save({4, 5, 6, 8}, self.mode_files[1])

        self.test_data = dd.from_pandas(pd.DataFrame({
            'PID': [1, 1, 1, 2, 2, 2, 3, 4],
            'Value': [5, 6, 7, 8, 9, 10, 11, 12]
        }), npartitions=1)

        self.large_test_data = dd.from_pandas(pd.DataFrame({
            'PID': [1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
            'Value': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 300]
        }), npartitions=1)

    def tearDown(self):
        # Cleanup temporary directory
        self.temp_dir.cleanup()

    def test_split_pids_into_pt_ft_test_valid_input(self):
        pids = ['pid1', 'pid2', 'pid3', 'pid4']
        pretrain, finetune, test = 0.5, 0.25, 0.25
        expected = (['pid1', 'pid2'], ['pid3'], ['pid4'])
        result = split_pids_into_pt_ft_test(pids, pretrain, finetune, test)
        self.assertEqual(result, expected)

    def test_split_pids_into_pt_ft_test_invalid_sum(self):
        pids = ['pid1', 'pid2', 'pid3']
        with self.assertRaises(ValueError):
            split_pids_into_pt_ft_test(pids, 0.3, 0.3, 0.3)

    def test_split_pids_into_train_val(self):
        train_data, val_data = split_pids_into_train_val(self.test_data, 0.75)
        self.assertEqual(train_data['PID'].nunique().compute(), 3)
        self.assertEqual(val_data['PID'].nunique().compute(), 1)


    def test_load_train_val_split(self):
        train_data, val_data = load_train_val_split(self.large_test_data, self.split_path)
        self.assertEqual(train_data['PID'].nunique().compute(), 3)
        self.assertEqual(val_data['PID'].nunique().compute(), 4)
        self.assertEqual(len(train_data.compute()), 7)
        self.assertEqual(len(val_data.compute()), 4)

if __name__ == '__main__':
    unittest.main()