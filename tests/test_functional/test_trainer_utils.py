import unittest
from corebehrt.common.config import Config
from corebehrt.functional.trainer_utils import replace_steps_with_epochs, convert_epochs_to_steps

class TestTrainerUtils(unittest.TestCase):

    def test_convert_epochs_to_steps(self):
        # Test case 1: Basic conversion
        self.assertEqual(convert_epochs_to_steps(num_epochs=1, num_patients=100, batch_size=10), 10)

        # Test case 2: Multiple epochs
        self.assertEqual(convert_epochs_to_steps(num_epochs=5, num_patients=1000, batch_size=20), 250)

        # Test case 3: Fractional result (should be rounded down)
        self.assertEqual(convert_epochs_to_steps(num_epochs=3, num_patients=50, batch_size=8), 18)

        # Test case 4: Edge case with batch size equal to num_patients
        self.assertEqual(convert_epochs_to_steps(num_epochs=2, num_patients=100, batch_size=100), 2)

    def test_replace_steps_with_epochs(self):
        # Test case 1: Basic replacement
        scheduler_cfg = Config({
            "warmup_epochs": 2,
            "max_epochs": 10
        })
        result = replace_steps_with_epochs(scheduler_cfg, batch_size=5, num_patients=100)
        self.assertIn("warmup_steps", result)
        self.assertIn("max_steps", result)
        self.assertEqual(result["warmup_steps"], 40)
        self.assertEqual(result["max_steps"], 200)

        # Test case 2: Mixed keys
        scheduler_cfg = Config({
            "warmup_epochs": 1,
            "max_steps": 1000,
            "cooldown_epochs": 3
        })
        result = replace_steps_with_epochs(scheduler_cfg, batch_size=10, num_patients=200)
        self.assertIn("warmup_steps", result)
        self.assertIn("max_steps", result)
        self.assertIn("cooldown_steps", result)
        self.assertEqual(result["warmup_steps"], 20)
        self.assertEqual(result["max_steps"], 1000)  # Unchanged
        self.assertEqual(result["cooldown_steps"], 60)

        # Test case 3: No epoch keys
        scheduler_cfg = Config({
            "warmup_steps": 100,
            "max_steps": 1000
        })
        result = replace_steps_with_epochs(scheduler_cfg, batch_size=5, num_patients=50)
        self.assertEqual(result, scheduler_cfg)  # Should be unchanged

        # Test case 4: Empty config
        scheduler_cfg = Config({})
        result = replace_steps_with_epochs(scheduler_cfg, batch_size=5, num_patients=100)
        self.assertEqual(result, scheduler_cfg)  # Should be unchanged

    def test_convert_epochs_to_steps_multiple(self):
        test_cases = [
            (1, 100, 10, 10),
            (5, 1000, 20, 250),
            (3, 50, 8, 18),
            (2, 100, 100, 2),
        ]
        for num_epochs, num_patients, batch_size, expected in test_cases:
            with self.subTest(num_epochs=num_epochs, num_patients=num_patients, batch_size=batch_size):
                self.assertEqual(convert_epochs_to_steps(num_epochs, num_patients, batch_size), expected)

if __name__ == '__main__':
    unittest.main()
