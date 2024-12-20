import os
import re
import logging
from corebehrt.common.utils import Data
from typing import Dict

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = "PID"
BG_GENDER_KEYS = {
    "male": ["M", "Mand", "male", "Male", "man", "MAN", "1"],
    "female": ["W", "Kvinde", "F", "female", "Female", "woman", "WOMAN", "0"],
}
MIN_POSITIVES = {"finetune": 1, None: 1}
CHECKPOINT_FOLDER = "checkpoints"
ORIGIN_POINT = {
    "year": 2020,
    "month": 1,
    "day": 26,
    "hour": 0,
    "minute": 0,
    "second": 0,
}


class Utilities:
    @classmethod
    def process_data(cls, data: Data, func: callable, args_for_func: Dict={})->Dict:
        """Apply a function to all datasets in a dictionary"""
        data = func(data, **args_for_func)

        return data

    @staticmethod
    def get_last_checkpoint_epoch(checkpoint_folder: str) -> int:
        """Returns the epoch of the last checkpoint."""
        # Regular expression to match the pattern retry_XXX
        pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
        max_epoch = None
        for filename in os.listdir(checkpoint_folder):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch
        # Return the folder with the maximum retry number
        if max_epoch is None:
            raise ValueError(
                "No checkpoint found in folder {}".format(checkpoint_folder)
            )

        return max_epoch
