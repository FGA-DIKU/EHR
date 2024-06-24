import unittest
from corebehrt.data.split import get_n_splits_cv
class TestDataObject:
    def __init__(self, pids):
        self.pids = pids

class TestSplit(unittest.TestCase):
    def setUp(self):
        self.data = TestDataObject(list(range(10)))

    def test_get_n_splits_cv(self):
        n_splits = 5
        splits = list(get_n_splits_cv(self.data, n_splits))
        self.assertEqual(len(splits), n_splits)
        for train_indices, val_indices in splits:
            self.assertEqual(len(train_indices), len(self.data.pids) - len(self.data.pids) // n_splits)
            self.assertEqual(len(val_indices), len(self.data.pids) // n_splits)

    def test_reproduceability(self):
        n_splits = 5
        splits1 = list(get_n_splits_cv(self.data, n_splits))
        splits2 = list(get_n_splits_cv(self.data, n_splits))
        self.assertEqual(splits1, splits2)

if __name__ == '__main__':
    unittest.main()