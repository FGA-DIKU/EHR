import os
import re
import logging

logger = logging.getLogger(__name__)  # Get the logger for this module

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
