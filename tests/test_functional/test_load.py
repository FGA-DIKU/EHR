import os
import unittest
import tempfile
import torch
from corebehrt.functional.load import load_pids, load_predefined_pids


class TestLoadPIDs(unittest.TestCase):
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
            os.path.join(self.split_path, f"pids_test.pt"),
        ]
        torch.save({1, 2, 3}, self.mode_files[0])
        torch.save({4, 2, 3, 100}, self.mode_files[1])

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

    # Test load_predefined_pids
    def test_load_predefined_pids_with_pids_pt(self):
        predefined_pids_file = os.path.join(self.split_path, "pids.pt")
        torch.save({7, 8, 9}, predefined_pids_file)
        pids = load_predefined_pids(self.split_path)
        self.assertEqual(pids, {7, 8, 9})

    def test_load_predefined_pids_with_pt_path(self):
        pids = load_predefined_pids(self.single_file)
        self.assertEqual(pids, {1, 2, 3})

    def test_load_predefined_pids_list(self):
        pids = load_predefined_pids(self.multi_files)
        self.assertEqual(pids, {1, 2, 3, 4, 5, 6})

    def test_load_predefined_pids_with_single_mode(self):
        pids = load_predefined_pids(self.split_path, mode="train")
        self.assertEqual(pids, {1, 2, 3})

    def test_load_predefined_pids_with_multiple_modes(self):
        pids = load_predefined_pids(self.split_path, mode=["train", "test"])
        self.assertEqual(pids, {1, 2, 3, 4, 100})


if __name__ == "__main__":
    unittest.main()