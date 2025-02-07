import unittest
from corebehrt.main.helper.finetune_cv import get_n_folds, TRAIN_KEY, VAL_KEY


class TestGetNFolds(unittest.TestCase):
    def setUp(self):
        # Create test data that will be used across test methods
        self.test_pids = [f"patient_{i}" for i in range(100)]

    def test_two_folds(self):
        # Test with 2 folds
        folds = get_n_folds(n_folds=2, train_val_pids=self.test_pids, val_split=0.2)

        # Check basic structure
        self.assertIsInstance(folds, list)
        self.assertEqual(len(folds), 2)

        for fold in folds:
            # Check dictionary structure
            self.assertIsInstance(fold, dict)
            self.assertEqual(set(fold.keys()), {TRAIN_KEY, VAL_KEY})

            # Check that train and val sets are disjoint
            self.assertEqual(
                len(set(fold[TRAIN_KEY]).intersection(set(fold[VAL_KEY]))), 0
            )

            # Check that all PIDs are used
            self.assertEqual(
                set(fold[TRAIN_KEY]).union(set(fold[VAL_KEY])), set(self.test_pids)
            )

            # Check that validation set is approximately 1/n_folds of the data
            val_ratio = len(fold[VAL_KEY]) / len(self.test_pids)
            self.assertLessEqual(
                abs(val_ratio - 0.5), 0.1
            )  # For 2 folds, expect ~50% split

    def test_single_fold(self):
        # Test with 1 fold
        folds = get_n_folds(n_folds=1, train_val_pids=self.test_pids, val_split=0.2)

        self.assertEqual(len(folds), 1)
        fold = folds[0]

        self.assertIsInstance(fold, dict)
        self.assertEqual(set(fold.keys()), {TRAIN_KEY, VAL_KEY})
        self.assertEqual(len(set(fold[TRAIN_KEY]).intersection(set(fold[VAL_KEY]))), 0)
        self.assertEqual(
            set(fold[TRAIN_KEY]).union(set(fold[VAL_KEY])), set(self.test_pids)
        )

    def test_invalid_inputs(self):
        # Test zero folds
        with self.assertRaises(ValueError):
            get_n_folds(n_folds=0, train_val_pids=self.test_pids)

        # Test empty patient list
        with self.assertRaises(ValueError):
            get_n_folds(n_folds=2, train_val_pids=[])


if __name__ == "__main__":
    unittest.main()
