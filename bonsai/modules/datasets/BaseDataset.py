from typing import Dict, List

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Store patient sequences in packed tensors instead of Python dictionaries.

    All per-token fields share the same offsets. A cloned patient slice is
    returned so task-specific transforms can modify it without changing the
    stored dataset.
    """

    def __init__(self, subjects: List[Dict]):
        if not subjects:
            raise ValueError("Cannot create a dataset from an empty subject list")

        feature_keys = [key for key in subjects[0] if key != "subject_id"]
        required = {"code", "abspos", "age", "segment"}
        if not required.issubset(feature_keys):
            raise ValueError(
                f"Missing required subject features: {required - set(feature_keys)}"
            )

        expected_keys = set(subjects[0])
        lengths = []
        for i, subject in enumerate(subjects):
            if set(subject) != expected_keys:
                raise ValueError(f"Inconsistent feature keys for subject at index {i}")
            length = len(subject["code"])
            for key in feature_keys:
                value = subject[key]
                if not isinstance(value, torch.Tensor) or value.ndim != 1:
                    raise ValueError(
                        f"Feature {key!r} at index {i} must be a 1-D tensor"
                    )
                if len(value) != length:
                    raise ValueError(
                        f"Misaligned feature {key!r} for subject at index {i}"
                    )
            lengths.append(length)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        self.offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), lengths_tensor.cumsum(0))
        )
        self.subject_ids = torch.tensor(
            [subject["subject_id"] for subject in subjects], dtype=torch.long
        )
        self.features = {
            key: torch.cat([subject[key] for subject in subjects], dim=0)
            for key in feature_keys
        }

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        subject = {"subject_id": int(self.subject_ids[index])}
        for key, values in self.features.items():
            subject[key] = values[start:end].clone()
        return subject
