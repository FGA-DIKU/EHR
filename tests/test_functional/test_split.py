import os
import random
import tempfile
import unittest

import numpy as np
import torch

from corebehrt.modules.dataset import PatientData, PatientDataset
from corebehrt.functional.split import (
    get_n_splits_cv_pids,
    load_train_val_split,
    split_into_test_and_train_val_pids,
    split_pids_into_pt_ft_test,
    split_pids_into_train_val,
)


class TestSplitFunctions(unittest.TestCase):
    def setUp(self):
        # Create a small list of PatientData
        # Just minimal fields needed: pid, concepts, abspos, segments, ages, outcome if needed.
        self.mock_patients = [
            PatientData(
                pid="pid1", concepts=[], abspos=[], segments=[], ages=[], outcome=0
            ),
            PatientData(
                pid="pid2", concepts=[], abspos=[], segments=[], ages=[], outcome=0
            ),
            PatientData(
                pid="pid3", concepts=[], abspos=[], segments=[], ages=[], outcome=0
            ),
            PatientData(
                pid="pid4", concepts=[], abspos=[], segments=[], ages=[], outcome=0
            ),
            PatientData(
                pid="pid5", concepts=[], abspos=[], segments=[], ages=[], outcome=0
            ),
        ]

        # Create a PatientDataset
        self.dataset = PatientDataset(self.mock_patients)
        self.all_pids = self.dataset.get_pids()

        # Temporary dir for load_train_val_split testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.split_path = self.temp_dir.name

        # Make train_pids.pt / val_pids.pt
        train_pids = ["pid1", "pid2"]
        val_pids = ["pid3", "pid4", "pid5"]
        torch.save(train_pids, os.path.join(self.split_path, "pids_train.pt"))
        torch.save(val_pids, os.path.join(self.split_path, "pids_val.pt"))

    def tearDown(self):
        self.temp_dir.cleanup()

    # ------------------------------------------------------------------------------
    # 1) split_pids_into_pt_ft_test
    # ------------------------------------------------------------------------------
    def test_split_pids_into_pt_ft_test_valid_input(self):
        # Sort so we have a stable ordering
        sorted_pids = sorted(
            self.all_pids
        )  # e.g. ["pid1", "pid2", "pid3", "pid4", "pid5"]
        pretrain, finetune, test = 0.4, 0.4, 0.2
        pt_list, ft_list, test_list = split_pids_into_pt_ft_test(
            sorted_pids, pretrain, finetune, test
        )

        # Check lengths
        total = len(sorted_pids)
        self.assertEqual(len(pt_list), int(total * pretrain))
        self.assertEqual(len(ft_list), int(total * finetune))
        self.assertEqual(len(test_list), total - len(pt_list) - len(ft_list))

        # Combined, they should be the original set with no overlaps
        combined = pt_list + ft_list + test_list
        self.assertCountEqual(combined, sorted_pids)

    def test_split_pids_into_pt_ft_test_invalid_sum(self):
        with self.assertRaises(ValueError):
            split_pids_into_pt_ft_test(self.all_pids, 0.5, 0.3, 0.3)

    # ------------------------------------------------------------------------------
    # 2) split_pids_into_train_val
    # ------------------------------------------------------------------------------
    def test_split_pids_into_train_val(self):
        # 80-20 split
        random.seed(42)  # for reproducibility
        train_ds, val_ds = split_pids_into_train_val(self.dataset, val_split=0.2)

        # Expect 4 train, 1 val (since we have 5 total)
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(val_ds), 1)

        all_split_pids = train_ds.get_pids() + val_ds.get_pids()
        self.assertCountEqual(all_split_pids, self.all_pids)

    # ------------------------------------------------------------------------------
    # 3) load_train_val_split
    # ------------------------------------------------------------------------------
    def test_load_train_val_split(self):
        # We have train_pids.pt = ["pid1", "pid2"], val_pids.pt = ["pid3", "pid4", "pid5"]
        train_ds, val_ds = load_train_val_split(self.dataset, self.split_path)
        # Check train
        self.assertListEqual(sorted(train_ds.get_pids()), ["pid1", "pid2"])
        # Check val
        self.assertListEqual(sorted(val_ds.get_pids()), ["pid3", "pid4", "pid5"])

    # ------------------------------------------------------------------------------
    # 4) get_n_splits_cv_pids
    # ------------------------------------------------------------------------------
    def test_get_n_splits_cv_pids(self):
        n_splits = 2
        # Convert dataset pids to a list
        pids = self.dataset.get_pids()  # e.g. 5 pids
        folds = list(get_n_splits_cv_pids(n_splits, pids))

        # We expect 2 folds
        self.assertEqual(len(folds), 2)
        # For each fold, we have (train_pids, val_pids)
        # Check that train + val covers all pids with no overlap
        for train_p, val_p in folds:
            self.assertCountEqual(train_p + val_p, pids)
            self.assertTrue(set(train_p).isdisjoint(set(val_p)))

    # ------------------------------------------------------------------------------
    # 5) split_into_test_and_train_val_pids
    # ------------------------------------------------------------------------------
    def test_split_into_test_and_train_val_pids(self):
        test_split = 0.4
        pids = self.dataset.get_pids()
        np.random.seed(42)  # ensure reproducibility in the random choice

        test_pids, train_val_pids = split_into_test_and_train_val_pids(pids, test_split)
        # 5 total, 40% => 2 test, 3 train_val
        self.assertEqual(len(test_pids), 2)
        self.assertEqual(len(train_val_pids), 3)
        # No overlap, combined = all
        self.assertTrue(set(test_pids).isdisjoint(train_val_pids))
        self.assertCountEqual(list(test_pids) + list(train_val_pids), pids)


if __name__ == "__main__":
    unittest.main()
