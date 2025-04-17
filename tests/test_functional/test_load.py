import os
import unittest
import tempfile
import torch
from corebehrt.functional.io_operations.load import load_pids


class TestLoadPIDs(unittest.TestCase):
    def setUp(self):
        # Setup temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create temporary .pt files
        self.single_file = os.path.join(self.temp_dir.name, "single.pt")
        torch.save([1, 2, 3], self.single_file)

        self.multi_files = [
            os.path.join(self.temp_dir.name, f"multi_{i}.pt") for i in range(2)
        ]
        for i, file in enumerate(self.multi_files):
            torch.save([i * 3 + 1, i * 3 + 2, i * 3 + 3], file)

        self.split_path = self.temp_dir.name
        self.mode_files = [
            os.path.join(self.split_path, "pids_train.pt"),
            os.path.join(self.split_path, "pids_test.pt"),
        ]
        torch.save([1, 2, 3], self.mode_files[0])
        torch.save([4, 2, 3, 100], self.mode_files[1])

    def tearDown(self):
        # Cleanup temporary directory
        self.temp_dir.cleanup()

    # Test load_pids
    def test_load_pids_single_file(self):
        pids = load_pids(self.single_file)
        self.assertEqual(pids, {1, 2, 3})

    def test_load_pids_multiple_files(self):
        pids = load_pids(self.multi_files)
        self.assertEqual(pids, {1, 2, 3, 4, 5, 6})


if __name__ == "__main__":
    unittest.main()
