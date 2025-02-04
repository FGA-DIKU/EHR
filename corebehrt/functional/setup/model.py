import re
import os
from os.path import join
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.config import Config


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
        raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))

    return max_epoch


def load_model_cfg_from_checkpoint(model_dir: str, config_name: str) -> Config:
    """
    Load the model sub section from the configuration file given in the
    model dir.
    """
    return load_config(join(model_dir, f"{config_name}.yaml")).model
