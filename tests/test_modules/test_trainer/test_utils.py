import unittest

import numpy as np
from torch.utils.data import WeightedRandomSampler

from corebehrt.modules.setup.config import Config
from corebehrt.modules.trainer.utils import (
    PositiveWeight,
    Sampling,
    compute_labels,
    get_loss_weight,
    get_sampler,
)


class TestComputeLabels(unittest.TestCase):
    def test_empty_list(self):
        outcomes = []
        with self.assertRaises(ValueError) as context:
            _ = compute_labels(outcomes)

    def test_balanced_classes(self):
        outcomes = [0, 1, 0, 1, 0, 1]
        result = compute_labels(outcomes)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], 3)

    def test_imbalanced_classes(self):
        outcomes = [0, 0, 0, 0, 1]
        result = compute_labels(outcomes)
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], 1)

    def test_multiple_classes(self):
        outcomes = [0, 1, 2, 0, 1, 2, 3]
        result = compute_labels(outcomes)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 2)
        self.assertEqual(result[2], 2)
        self.assertEqual(result[3], 1)


class TestSampling(unittest.TestCase):
    def test_inverse_sqrt_balanced(self):
        outcomes = [0, 1, 0, 1, 0, 1]  # 3 of each class
        weights = Sampling.inverse_sqrt(outcomes)

        # Should return a weight for each sample
        self.assertEqual(len(weights), len(outcomes))

        expected_weight = 1 / np.sqrt(3)  # Both classes have count of 3
        # Check weights match their class
        for outcome, weight in zip(outcomes, weights):
            self.assertAlmostEqual(weight, expected_weight)

    def test_inverse_sqrt_imbalanced(self):
        outcomes = [0, 0, 0, 0, 1]  # 4 zeros, 1 one
        weights = Sampling.inverse_sqrt(outcomes)

        # Should return a weight for each sample
        self.assertEqual(len(weights), len(outcomes))

        # Check each sample has correct weight based on its class
        for outcome, weight in zip(outcomes, weights):
            if outcome == 0:
                self.assertAlmostEqual(weight, 1 / np.sqrt(4))  # majority class
            else:
                self.assertAlmostEqual(weight, 1 / np.sqrt(1))  # minority class

    def test_effective_n_samples_balanced(self):
        outcomes = [0, 1, 0, 1, 0, 1]  # 3 of each class
        weights = Sampling.effective_n_samples(outcomes)

        # Should return a weight for each sample
        self.assertEqual(len(weights), len(outcomes))

        # For balanced data, all weights should be equal
        first_weight = weights[0]
        for weight in weights:
            self.assertAlmostEqual(weight, first_weight, places=5)

    def test_effective_n_samples_imbalanced(self):
        outcomes = [0, 0, 0, 0, 1]  # 4 zeros, 1 one
        weights = Sampling.effective_n_samples(outcomes)

        # Should return a weight for each sample
        self.assertEqual(len(weights), len(outcomes))

        # Group weights by class
        class0_weights = [w for o, w in zip(outcomes, weights) if o == 0]
        class1_weights = [w for o, w in zip(outcomes, weights) if o == 1]

        # All weights within same class should be equal
        self.assertTrue(all(np.isclose(w, class0_weights[0]) for w in class0_weights))
        self.assertTrue(all(np.isclose(w, class1_weights[0]) for w in class1_weights))

        # Minority class (1) should have higher weight than majority class (0)
        self.assertTrue(class1_weights[0] > class0_weights[0])

    def test_single_class_error(self):
        outcomes = [0, 0, 0]  # Only one class
        with self.assertRaises(ValueError) as context:
            Sampling.inverse_sqrt(outcomes)
        self.assertTrue("requires at least 2 classes" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            Sampling.effective_n_samples(outcomes)
        self.assertTrue("requires at least 2 classes" in str(context.exception))


class TestPositiveWeight(unittest.TestCase):
    def test_sqrt_balanced(self):
        outcomes = [0, 1, 0, 1, 0, 1]
        weight = PositiveWeight.sqrt(outcomes)

        # For balanced data, sqrt(neg/pos) should be 1
        self.assertAlmostEqual(weight, 1.0)

    def test_sqrt_imbalanced(self):
        outcomes = [0, 0, 0, 0, 1]
        weight = PositiveWeight.sqrt(outcomes)

        # For this imbalanced data, sqrt(4/1) = 2
        self.assertAlmostEqual(weight, 2.0)

    def test_sqrt_no_positives(self):
        outcomes = [0, 0, 0, 0]

        # Should raise ValueError when no positive samples
        with self.assertRaises(ValueError):
            PositiveWeight.sqrt(outcomes)

    def test_effective_n_samples_balanced(self):
        outcomes = [0, 1, 0, 1, 0, 1]
        weight = PositiveWeight.effective_n_samples(outcomes)

        # For balanced classes, the effective weight should be close to 1
        self.assertAlmostEqual(weight, 1.0, places=5)

    def test_effective_n_samples_imbalanced(self):
        outcomes = [0, 0, 0, 0, 1]
        weight = PositiveWeight.effective_n_samples(outcomes)

        # For imbalanced data, pos_weight should be > 1 when pos samples are fewer
        self.assertTrue(weight > 1.0)

    def test_effective_n_samples_missing_class(self):
        # Test with missing negative class
        outcomes_no_neg = [1, 1, 1]

        with self.assertRaises(ValueError):
            PositiveWeight.effective_n_samples(outcomes_no_neg)

        # Test with missing positive class
        outcomes_no_pos = [0, 0, 0]

        with self.assertRaises(ValueError):
            PositiveWeight.effective_n_samples(outcomes_no_pos)


class TestGetSampler(unittest.TestCase):
    def test_no_sampler_function(self):
        cfg = Config({"trainer_args": {}})
        outcomes = [0, 1, 0, 1]
        sampler = get_sampler(cfg, outcomes)
        self.assertIsNone(sampler)

    def test_with_sampler_function(self):
        cfg = Config(
            {
                "trainer_args": {
                    "sampler_function": {
                        "_target_": "corebehrt.modules.trainer.utils.Sampling.inverse_sqrt"
                    }
                }
            }
        )
        outcomes = [0, 1, 0, 1]  # balanced data
        sampler = get_sampler(cfg, outcomes)

        # Verify sampler is created correctly
        self.assertIsInstance(sampler, WeightedRandomSampler)
        self.assertEqual(sampler.num_samples, len(outcomes))
        # For balanced data, all weights should be equal
        expected_weight = 1 / np.sqrt(2)  # 2 samples per class
        np.testing.assert_allclose(sampler.weights.numpy(), [expected_weight] * 4)


class TestGetLossWeight(unittest.TestCase):
    def test_no_weight_function(self):
        cfg = Config({"trainer_args": {}})
        outcomes = [0, 1, 0, 1]
        weight = get_loss_weight(cfg, outcomes)
        self.assertIsNone(weight)

    def test_empty_outcomes(self):
        cfg = Config(
            {
                "trainer_args": {
                    "loss_weight_function": {
                        "_target_": "corebehrt.modules.trainer.utils.PositiveWeight.sqrt"
                    }
                }
            }
        )
        outcomes = []
        weight = get_loss_weight(cfg, outcomes)
        self.assertIsNone(weight)

    def test_with_weight_function(self):
        cfg = Config(
            {
                "trainer_args": {
                    "loss_weight_function": {
                        "_target_": "corebehrt.modules.trainer.utils.PositiveWeight.sqrt"
                    }
                }
            }
        )
        outcomes = [0, 0, 0, 0, 1]  # 4:1 ratio

        weight = get_loss_weight(cfg, outcomes)
        self.assertAlmostEqual(weight, 2.0)  # sqrt(4/1) = 2


if __name__ == "__main__":
    unittest.main()
