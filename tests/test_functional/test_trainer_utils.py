import unittest
from corebehrt.modules.setup.config import Config
from corebehrt.functional.trainer_utils import (
    replace_steps_with_epochs,
    convert_epochs_to_steps,
    dynamic_padding,
)
import torch


class TestTrainerUtils(unittest.TestCase):

    def test_convert_epochs_to_steps(self):
        # Test case 1: Basic conversion
        self.assertEqual(
            convert_epochs_to_steps(num_epochs=1, num_patients=100, batch_size=10), 10
        )

        # Test case 2: Multiple epochs
        self.assertEqual(
            convert_epochs_to_steps(num_epochs=5, num_patients=1000, batch_size=20), 250
        )

        # Test case 3: Fractional result (should be rounded down)
        self.assertEqual(
            convert_epochs_to_steps(num_epochs=3, num_patients=50, batch_size=8), 18
        )

        # Test case 4: Edge case with batch size equal to num_patients
        self.assertEqual(
            convert_epochs_to_steps(num_epochs=2, num_patients=100, batch_size=100), 2
        )

    def test_replace_steps_with_epochs(self):
        # Test case 1: Basic replacement
        scheduler_cfg = Config({"warmup_epochs": 2, "max_epochs": 10})
        result = replace_steps_with_epochs(
            scheduler_cfg, batch_size=5, num_patients=100
        )
        self.assertIn("warmup_steps", result)
        self.assertIn("max_steps", result)
        self.assertEqual(result["warmup_steps"], 40)
        self.assertEqual(result["max_steps"], 200)

        # Test case 2: Mixed keys
        scheduler_cfg = Config(
            {"warmup_epochs": 1, "max_steps": 1000, "cooldown_epochs": 3}
        )
        result = replace_steps_with_epochs(
            scheduler_cfg, batch_size=10, num_patients=200
        )
        self.assertIn("warmup_steps", result)
        self.assertIn("max_steps", result)
        self.assertIn("cooldown_steps", result)
        self.assertEqual(result["warmup_steps"], 20)
        self.assertEqual(result["max_steps"], 1000)  # Unchanged
        self.assertEqual(result["cooldown_steps"], 60)

        # Test case 3: No epoch keys
        scheduler_cfg = Config({"warmup_steps": 100, "max_steps": 1000})
        result = replace_steps_with_epochs(scheduler_cfg, batch_size=5, num_patients=50)
        self.assertEqual(result, scheduler_cfg)  # Should be unchanged

        # Test case 4: Empty config
        scheduler_cfg = Config({})
        result = replace_steps_with_epochs(
            scheduler_cfg, batch_size=5, num_patients=100
        )
        self.assertEqual(result, scheduler_cfg)  # Should be unchanged

    def test_convert_epochs_to_steps_multiple(self):
        test_cases = [
            (1, 100, 10, 10),
            (5, 1000, 20, 250),
            (3, 50, 8, 18),
            (2, 100, 100, 2),
        ]
        for num_epochs, num_patients, batch_size, expected in test_cases:
            with self.subTest(
                num_epochs=num_epochs, num_patients=num_patients, batch_size=batch_size
            ):
                self.assertEqual(
                    convert_epochs_to_steps(num_epochs, num_patients, batch_size),
                    expected,
                )

    def test_dynamic_padding(self):
        # Test case 1: Padding sequences of different lengths
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "target": torch.tensor([1]),
            },
            {
                "concept": torch.tensor([1, 2, 3, 4, 5]),
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "target": torch.tensor([0]),
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 5))
        self.assertEqual(padded_batch["input_ids"].shape, (2, 5))
        self.assertEqual(padded_batch["attention_mask"].shape, (2, 5))
        self.assertEqual(padded_batch["target"].shape, (2, 1))

        # Check padding values
        self.assertTrue(
            torch.equal(padded_batch["input_ids"][0], torch.tensor([1, 2, 3, 0, 0]))
        )
        self.assertTrue(
            torch.equal(
                padded_batch["attention_mask"][0], torch.tensor([1, 1, 1, 0, 0])
            )
        )

        # Test case 2: Non-sequence tensors should remain unchanged
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                "static_feature": torch.tensor([1.0, 2.0, 3.0]),
                "target": torch.tensor([1]),
            },
            {
                "concept": torch.tensor([1, 2, 3]),
                "static_feature": torch.tensor([4.0, 5.0, 6.0]),
                "target": torch.tensor([0]),
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 3))
        self.assertEqual(padded_batch["static_feature"].shape, (2, 3))
        self.assertEqual(padded_batch["target"].shape, (2, 1))

        # Check values
        self.assertTrue(
            torch.equal(
                padded_batch["static_feature"],
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            )
        )

        self.assertTrue(torch.equal(padded_batch["target"], torch.tensor([[1], [0]])))

    def test_dynamic_padding_mlm_target(self):
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "target": torch.tensor([1, 0, 1]),  # Same length as input sequence
            },
            {
                "concept": torch.tensor([1, 2, 3, 4, 5]),
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "target": torch.tensor(
                    [0, 1, 1, 0, 1]
                ),  # Same length as input sequence
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 5))
        self.assertEqual(padded_batch["input_ids"].shape, (2, 5))
        self.assertEqual(padded_batch["attention_mask"].shape, (2, 5))
        self.assertEqual(
            padded_batch["target"].shape, (2, 5)
        )  # Target should be padded to max length

        # Check padding values
        self.assertTrue(
            torch.equal(
                padded_batch["target"][0],
                torch.tensor([1, 0, 1, -100, -100]),  # Padded with zeros
            )
        )
        self.assertTrue(
            torch.equal(
                padded_batch["target"][1],
                torch.tensor([0, 1, 1, 0, 1]),  # Original values unchanged
            )
        )


if __name__ == "__main__":
    unittest.main()
