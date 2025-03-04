# import os
# import random
# import tempfile
# import unittest

# import numpy as np
# import torch

# from corebehrt.modules.preparation.dataset import PatientData, PatientDataset
# from corebehrt.functional.features.split import (
#     create_folds,
#     get_n_splits_cv_pids,
#     split_into_test_and_train_val_pids,
#     split_pids_into_pt_ft_test,
#     split_pids_into_train_val,
#     split_test,
# )


# class TestSplitFunctions(unittest.TestCase):
#     def setUp(self):
#         # Create a small list of PatientData
#         # Just minimal fields needed: pid, concepts, abspos, segments, ages, outcome if needed.
#         self.mock_patients = [
#             PatientData(
#                 pid="pid1", concepts=[], abspos=[], segments=[], ages=[], outcome=0
#             ),
#             PatientData(
#                 pid="pid2", concepts=[], abspos=[], segments=[], ages=[], outcome=0
#             ),
#             PatientData(
#                 pid="pid3", concepts=[], abspos=[], segments=[], ages=[], outcome=0
#             ),
#             PatientData(
#                 pid="pid4", concepts=[], abspos=[], segments=[], ages=[], outcome=0
#             ),
#             PatientData(
#                 pid="pid5", concepts=[], abspos=[], segments=[], ages=[], outcome=0
#             ),
#         ]

#         # Create a PatientDataset
#         self.dataset = PatientDataset(self.mock_patients)
#         self.all_pids = self.dataset.get_pids()

#         # Temporary dir for load_train_val_split testing
#         self.temp_dir = tempfile.TemporaryDirectory()
#         self.split_path = self.temp_dir.name

#         # Make train_pids.pt / val_pids.pt
#         train_pids = ["pid1", "pid2"]
#         val_pids = ["pid3", "pid4", "pid5"]
#         torch.save(train_pids, os.path.join(self.split_path, "pids_train.pt"))
#         torch.save(val_pids, os.path.join(self.split_path, "pids_val.pt"))

#     def tearDown(self):
#         self.temp_dir.cleanup()

#     # -------------------------------------------------------------------------
#     # 1) split_pids_into_pt_ft_test
#     # -------------------------------------------------------------------------
#     def test_split_pids_into_pt_ft_test_valid_input(self):
#         # Sort so we have a stable ordering
#         sorted_pids = sorted(
#             self.all_pids
#         )  # e.g. ["pid1", "pid2", "pid3", "pid4", "pid5"]
#         pretrain, finetune, test = 0.4, 0.4, 0.2
#         pt_list, ft_list, test_list = split_pids_into_pt_ft_test(
#             sorted_pids, pretrain, finetune, test
#         )

#         # Check lengths
#         total = len(sorted_pids)
#         self.assertEqual(len(pt_list), int(total * pretrain))
#         self.assertEqual(len(ft_list), int(total * finetune))
#         self.assertEqual(len(test_list), total - len(pt_list) - len(ft_list))

#         # Combined, they should be the original set with no overlaps
#         combined = pt_list + ft_list + test_list
#         self.assertCountEqual(combined, sorted_pids)

#     def test_split_pids_into_pt_ft_test_invalid_sum(self):
#         with self.assertRaises(ValueError):
#             split_pids_into_pt_ft_test(self.all_pids, 0.5, 0.3, 0.3)

#     # -------------------------------------------------------------------------
#     # 2) split_test
#     # -------------------------------------------------------------------------
#     def test_split_test_normal_ratio(self):
#         pids = sorted(self.all_pids)
#         test_ratio = 0.4
#         train_pids, test_pids = split_test(pids, test_ratio=test_ratio)
#         self.assertEqual(len(test_pids), int(len(pids) * test_ratio))
#         self.assertEqual(len(train_pids), len(pids) - len(test_pids))
#         self.assertCountEqual(pids, train_pids + test_pids)
#         # Ensure disjoint
#         self.assertTrue(set(train_pids).isdisjoint(test_pids))

#     def test_split_test_zero_ratio(self):
#         """If test_ratio=0, all go to train, none to test."""
#         pids = sorted(self.all_pids)
#         train_pids, test_pids = split_test(pids, test_ratio=0.0)
#         self.assertEqual(len(test_pids), 0)
#         self.assertEqual(len(train_pids), len(pids))
#         self.assertCountEqual(train_pids, pids)

#     def test_split_test_full_ratio(self):
#         """If test_ratio=1, all go to test, none to train."""
#         pids = sorted(self.all_pids)
#         train_pids, test_pids = split_test(pids, test_ratio=1.0)
#         self.assertEqual(len(train_pids), 0)
#         self.assertEqual(len(test_pids), len(pids))
#         self.assertCountEqual(test_pids, pids)

#     # -------------------------------------------------------------------------
#     # 3) split_pids_into_train_val
#     # -------------------------------------------------------------------------
#     def test_split_pids_into_train_val(self):
#         # 80-20 split
#         random.seed(42)  # for reproducibility
#         train_ds, val_ds = split_pids_into_train_val(self.dataset, val_split=0.2)

#         # Expect 4 train, 1 val (since we have 5 total)
#         self.assertEqual(len(train_ds), 4)
#         self.assertEqual(len(val_ds), 1)

#         all_split_pids = train_ds.get_pids() + val_ds.get_pids()
#         self.assertCountEqual(all_split_pids, self.all_pids)

#     def test_split_pids_into_train_val_invalid_split(self):
#         """Test that invalid val_split raises AssertionError."""
#         # val_split must be between 0 and 1, exclusively
#         with self.assertRaises(AssertionError):
#             split_pids_into_train_val(self.dataset, val_split=0)
#         with self.assertRaises(AssertionError):
#             split_pids_into_train_val(self.dataset, val_split=1)
#         with self.assertRaises(AssertionError):
#             split_pids_into_train_val(self.dataset, val_split=-0.5)
#         with self.assertRaises(AssertionError):
#             split_pids_into_train_val(self.dataset, val_split=1.5)

#     # -------------------------------------------------------------------------
#     # 4) get_n_splits_cv_pids
#     # -------------------------------------------------------------------------
#     def test_get_n_splits_cv_pids(self):
#         n_splits = 2
#         # Convert dataset pids to a list
#         pids = self.dataset.get_pids()  # e.g. 5 pids
#         folds = list(get_n_splits_cv_pids(n_splits, pids))

#         # We expect 2 folds
#         self.assertEqual(len(folds), 2)
#         # For each fold, we have (train_pids, val_pids)
#         for train_p, val_p in folds:
#             self.assertCountEqual(train_p + val_p, pids)
#             self.assertTrue(set(train_p).isdisjoint(set(val_p)))

#     def test_get_n_splits_cv_pids_single_fold(self):
#         """Test the branch n_splits=1, which should yield a single 80-20 split."""
#         pids = sorted(self.all_pids)
#         folds = list(get_n_splits_cv_pids(1, pids, val_split=0.2, seed=42))
#         # Should yield exactly one (train_pids, val_pids) tuple
#         self.assertEqual(len(folds), 1)
#         train_p, val_p = folds[0]
#         self.assertEqual(len(train_p), int(len(pids) * 0.8))
#         self.assertEqual(len(val_p), len(pids) - len(train_p))
#         self.assertCountEqual(train_p + val_p, pids)

#     def test_get_n_splits_cv_pids_invalid_n_splits(self):
#         """Test that n_splits < 1 raises ValueError."""
#         pids = self.dataset.get_pids()
#         with self.assertRaises(ValueError):
#             list(get_n_splits_cv_pids(0, pids))

#     # -------------------------------------------------------------------------
#     # 5) split_into_test_and_train_val_pids
#     # -------------------------------------------------------------------------
#     def test_split_into_test_and_train_val_pids(self):
#         test_split = 0.4
#         pids = self.dataset.get_pids()
#         np.random.seed(42)  # ensure reproducibility in the random choice

#         test_pids, train_val_pids = split_into_test_and_train_val_pids(pids, test_split)
#         # 5 total, 40% => 2 test, 3 train_val
#         self.assertEqual(len(test_pids), 2)
#         self.assertEqual(len(train_val_pids), 3)
#         # No overlap, combined = all
#         self.assertTrue(set(test_pids).isdisjoint(train_val_pids))
#         self.assertCountEqual(list(test_pids) + list(train_val_pids), pids)

#     # -------------------------------------------------------------------------
#     # 6) create_folds
#     # -------------------------------------------------------------------------
#     def test_create_folds(self):
#         """Test that create_folds generates the correct number of folds
#         and that each fold has disjoint train/val sets whose union is the
#         entire set of pids."""
#         pids = sorted(self.all_pids)
#         num_folds = 3
#         folds = create_folds(pids, num_folds=num_folds, seed=42)

#         # We expect a list of 3 dicts: [{'train': [...], 'val': [...]}, ...]
#         self.assertEqual(len(folds), num_folds)
#         for fold in folds:
#             self.assertIn("train", fold)
#             self.assertIn("val", fold)
#             train_p, val_p = fold["train"], fold["val"]
#             # Check disjoint
#             self.assertTrue(set(train_p).isdisjoint(val_p))
#             # Combined => subset of original pids
#             combined = train_p + val_p
#             self.assertTrue(all(pid in pids for pid in combined))

#         # Optionally, check coverage of each PID across folds if you wish.


# if __name__ == "__main__":
#     unittest.main()
