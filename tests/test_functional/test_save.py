import unittest
import dask.dataframe as dd
import torch
import os
import tempfile
import shutil
from corebehrt.functional.save import save_sequence_lengths, save_data, save_pids_splits
import pandas as pd
from os.path import join

class TestSaveFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_sequence_lengths(self):
        # Create a mock DataFrame with expected structure
        df = pd.DataFrame({'col1': list(range(10)) + list(range(5)), 'PID': [1]*10 + [2]*5})
        mock_df = dd.from_pandas(df, npartitions=1)
        desc = '_test'
        
        # Call the function under test
        save_sequence_lengths(mock_df, self.test_dir, desc)
        
        # Load the saved file and verify its contents
        saved_lengths = torch.load(os.path.join(self.test_dir, f'sequences_lengths{desc}.pt'))
        self.assertListEqual(saved_lengths, [10, 5])

    def test_save_data(self):
        mock_df = dd.from_pandas(pd.DataFrame({'col1': range(6)}), npartitions=1)
        mock_df['PID'] = 1
        vocabulary = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
        desc = '_test'
        
        save_data(mock_df, vocabulary, self.test_dir, desc)
        
        # Verify the CSV file exists and check a few properties
        data_path = os.path.join(self.test_dir, f'data{desc}')
        self.assertTrue(os.path.exists(data_path))
        loaded_data = dd.read_csv(join(data_path,"*.csv",)) 
        self.assertEqual(loaded_data['col1'].compute().tolist(), list(range(6)))

        # Load and verify vocabulary
        loaded_vocabulary = torch.load(os.path.join(self.test_dir, f'vocabulary.pt'))
        self.assertEqual(vocabulary, loaded_vocabulary)

    def test_save_pids_splits(self):
        mock_train_df = dd.from_pandas(pd.DataFrame({'PID': ['pid1', 'pid2']}), npartitions=1)
        mock_val_df = dd.from_pandas(pd.DataFrame({'PID': ['pid3', 'pid4']}), npartitions=1)
        
        save_pids_splits(mock_train_df, mock_val_df, self.test_dir)
        
        # Load and verify train PIDs
        train_pids = torch.load(os.path.join(self.test_dir, 'pids_train.pt'))
        self.assertEqual(train_pids, ['pid1', 'pid2'])
        # Load and verify val PIDs
        val_pids = torch.load(os.path.join(self.test_dir, 'pids_val.pt'))
        self.assertEqual(val_pids, ['pid3', 'pid4'])

if __name__ == '__main__':
    unittest.main()