import torch

from corebehrt.common.config import Config
from corebehrt.common.constants import DEFAULT_BUCKETS

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


def _common_dynamic_padding(batch: list, use_buckets: bool = False) -> dict:
    """Pad sequences in a batch to a common length with dynamic or bucketed padding.
    
    This collate function handles two types of targets:
    1. Binary classification: where 'target' is a 0D scalar value
    2. Masked Language Modeling (MLM): where 'target' is a 1D sequence matching 'concept' length
    
    Args:
        batch: List of dictionaries, each containing tensors with fields like:
            - concept: Base tensor that determines sequence length (required)
            - target: Either a 0D scalar (classification) or 1D sequence (MLM)
            - Other fields: Must match concept sequence length if 1D
        use_buckets: If True, pad to nearest bucket size instead of max length
    
    Returns:
        dict: Collated batch with all sequences padded to same length where:
            - target sequences are padded with -100 (MLM convention)
            - other sequences are padded with 0
            - 0D scalars are left unchanged
    """

    # 1) Find maximum sequence length from 'concept'
    max_len = max(sample["concept"].shape[0] for sample in batch)
    if use_buckets:
        max_len = get_bucket_length(max_len)
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

def dynamic_padding(batch: list) -> dict:
    """Pad sequences in a batch to the maximum sequence length in that batch.
    
    Args:
        batch: List of dictionaries containing tensors (see _common_dynamic_padding)
    
    Returns:
        dict: Collated batch with all sequences padded to max length
    """
    return _common_dynamic_padding(batch, use_buckets=False)

def bucketed_dynamic_padding(batch: list) -> dict:
    """Pad sequences in a batch to the nearest predefined bucket size instead of padding to arbitrary lengths. 
    This helps for compilation of the model, since only a few different kernels are needed.
    
    Args:
        batch: List of dictionaries containing tensors with fields:
            - concept: Base tensor determining sequence length (required)
            - target: Either a 0D scalar or 1D sequence for predictions
            - Other fields: Must match concept sequence length if 1D
    
    Returns:
        dict: Collated batch with all sequences padded to nearest bucket size
    """
    return _common_dynamic_padding(batch, use_buckets=True)

def get_bucket_length(length: int, buckets: list = None) -> int:
    """Find the smallest predefined bucket size that can fit a given sequence length.
    
    Args:
        length: Sequence length to find a bucket for
        buckets: Optional list of bucket sizes in ascending order. 
                Uses DEFAULT_BUCKETS if None.
    
    Returns:
        int: Smallest bucket size that can contain the sequence.
             If sequence is longer than all buckets, returns the largest bucket size.
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]  # Use largest bucket if sequence is longer
