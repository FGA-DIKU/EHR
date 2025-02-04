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


def dynamic_padding(batch: list) -> dict:
    """
    Collate function that handles both:
      - Binary classification with a 0D scalar 'target'
      - MLM with a 1D sequence 'target' that matches 'concept'

    Steps:
      1) Determine max sequence length from the 'concept' field.
      2) For each sample in the batch:
         - For each key, if the tensor is 1D and matches the sequence length, pad to 'max_len'.
           * If key == 'target' and it's 1D, pad with -100 (MLM style).
           * Else pad with 0.
         - If the tensor is 0D (a scalar), skip padding.
      3) Stack along dim=0 to produce a batch dict.
    """

    # 1) Find maximum sequence length from 'concept'
    max_len = max(sample["concept"].shape[0] for sample in batch)

    # 2) Pad each field if needed
    for sample in batch:
        seq_len = sample["concept"].shape[0]
        diff = max_len - seq_len

        for key, tensor_field in sample.items():
            # If it's a 0D scalar (like a single float for binary classification), skip
            if tensor_field.dim() == 0:
                continue

            # If it's 1D and matches seq_len, we pad it to 'max_len'
            if tensor_field.dim() == 1 and tensor_field.shape[0] == seq_len:
                # For MLM 'target' we typically fill with -100
                if key == "target":
                    # Only do this if target is indeed a sequence (MLM).
                    # If it's binary classification, 'target' will be 0D so we won't enter here.
                    filler = torch.full((diff,), -100, dtype=tensor_field.dtype)
                else:
                    # For other sequence fields, pad with 0
                    filler = torch.zeros(diff, dtype=tensor_field.dtype)

                # Concatenate
                sample[key] = torch.cat([tensor_field, filler], dim=0)

    # 3) Stack into a dict of batch tensors
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)

    return collated
