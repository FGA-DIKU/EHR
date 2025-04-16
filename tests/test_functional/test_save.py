import os
import shutil
import tempfile
import unittest

import torch

from corebehrt.modules.preparation.dataset import PatientData, PatientDataset
from corebehrt.functional.io_operations.save import save_pids_splits


class TestSaveFunctions(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_pids_splits(self):
        # Create mock PatientDataset instances with PatientData objects
        mock_train_dataset = PatientDataset(
            patients=[
                PatientData(pid=1, concepts=[], abspos=[], segments=[], ages=[]),
                PatientData(pid=2, concepts=[], abspos=[], segments=[], ages=[]),
            ],
        )
        mock_val_dataset = PatientDataset(
            patients=[
                PatientData(pid=3, concepts=[], abspos=[], segments=[], ages=[]),
                PatientData(pid=4, concepts=[], abspos=[], segments=[], ages=[]),
            ],
        )

        save_pids_splits(mock_train_dataset, mock_val_dataset, self.test_dir)

        # Load and verify train PIDs
        train_pids = torch.load(
            os.path.join(self.test_dir, "pids_train.pt"), weights_only=True
        )
        self.assertEqual(train_pids, [1, 2])
        # Load and verify val PIDs
        val_pids = torch.load(
            os.path.join(self.test_dir, "pids_val.pt"), weights_only=True
        )
        self.assertEqual(val_pids, [3, 4])


if __name__ == "__main__":
    unittest.main()
