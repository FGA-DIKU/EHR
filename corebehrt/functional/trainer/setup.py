import torch

from corebehrt.modules.setup.config import Config


def replace_steps_with_epochs(
    scheduler_cfg: Config, batch_size: int, num_patients: int
) -> Config:
    """Replace steps with epochs in scheduler config"""
    new_cfg = Config({})
    for key, value in scheduler_cfg.items():
        if key.endswith("_epochs"):
            new_key = key.replace("_epochs", "_steps")
            new_cfg[new_key] = convert_epochs_to_steps(
                num_epochs=value, num_patients=num_patients, batch_size=batch_size
            )
        else:
            new_cfg[key] = value
    return new_cfg


def convert_epochs_to_steps(num_epochs: int, num_patients: int, batch_size: int) -> int:
    """Convert number of epochs to number of steps based on number of patients and batch size"""
    return int(num_patients / batch_size * num_epochs)
