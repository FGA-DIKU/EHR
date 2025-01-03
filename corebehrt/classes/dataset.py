import os
from dataclasses import dataclass
from os.path import join
from typing import List

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from corebehrt.data.mask import ConceptMasker


@dataclass
class PatientData:
    pid: str
    concepts: List[int]  # or List[str], depending on your use
    abspos: List[float]  # or int, depends on your data
    segments: List[int]
    ages: List[float]  # e.g. age at each concept
    outcome: int = None


class PatientDataset:
    """A dataset class for managing patient data and vocabulary.

    This class provides functionality to store and process patient data along with their
    associated vocabulary. It supports parallel processing of patient data and saving/loading
    functionality.

    Attributes:
        patients (List[PatientData]): List of patient data objects containing medical concepts,
            positions, segments and ages.
        vocabulary (dict): Dictionary mapping tokens to indices for the medical concepts.
    """

    def __init__(self, patients: List[PatientData], vocabulary: dict):
        """Initialize the PatientDataset.

        Args:
            patients (List[PatientData]): List of patient data objects.
            vocabulary (dict): Dictionary mapping tokens to indices.
        """
        self.patients = patients
        self.vocabulary = vocabulary

    def __len__(self):
        """Get the number of patients in the dataset."""
        return len(self.patients)

    def __getitem__(self, idx: int):
        """Get a patient by index.

        Args:
            idx (int): Index of the patient to retrieve.

        Returns:
            PatientData: The patient data at the given index.
        """
        return self.patients[idx]

    def process_in_parallel(self, func, n_jobs=-1, **kwargs):
        """Process all patients in parallel using the given function.

        Args:
            func: Function to apply to each patient.
            n_jobs (int): Number of parallel jobs. -1 means using all processors.
            **kwargs: Additional keyword arguments passed to the function.

        Returns:
            list: Results of applying the function to each patient.
        """
        loop = tqdm(self.patients, desc=f"{func.__name__}", mininterval=10)
        return Parallel(n_jobs=n_jobs)(delayed(func)(p, **kwargs) for p in loop)

    def save(self, save_dir: str, suffix: str = ""):
        """Save patient data and vocabulary to disk.

        Args:
            save_dir (str): Directory path to save the files.
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.patients, join(save_dir, f"patients{suffix}.pt"))
        if not os.path.exists(join(save_dir, "vocabulary.pt")):
            torch.save(self.vocabulary, join(save_dir, "vocabulary.pt"))

    def filter_by_pids(self, pids: List[str]) -> "PatientDataset":
        pids_set = set(pids)
        return PatientDataset(
            [p for p in self.patients if p.pid in pids_set], self.vocabulary
        )

    def get_pids(self) -> List[str]:
        return [p.pid for p in self.patients]

    def get_outcomes(self) -> List[int]:
        return [p.outcome for p in self.patients]

    def assign_outcomes(self, outcomes: pd.Series):
        """Assigns binary outcomes to each patient in the dataset.

        Takes a pandas Series mapping patient IDs to outcomes absolute positions and assigns a binary outcome
        to each patient in the dataset. The outcome is 1.0 if the patient has a non-null
        outcome in the Series, and 0.0 otherwise.

        Args:
            outcomes (pd.Series): Series with patient IDs as index and outcomes as values.
                The actual outcome values are not used, only whether they are null or not.

        Returns:
            PatientDataset: Returns self for method chaining.
        """
        outcomes = pd.notna(outcomes).astype(int)
        for p in self.patients:
            p.outcome = outcomes[p.pid]

        return self


class MLMDataset(Dataset):
    def __init__(
        self,
        patients: List[PatientData],
        vocabulary: dict,
        select_ratio: float,
        masking_ratio: float = 0.8,
        replace_ratio: float = 0.1,
        ignore_special_tokens: bool = True,
    ):
        self.patients = patients
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(
            vocabulary,
            select_ratio,
            masking_ratio,
            replace_ratio,
            ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        """
        1. Retrieve the PatientData.
        2. Mask the 'concepts'.
        3. Convert everything to torch.Tensor.
        4. Return a dict that PyTorch can collate into a batch.
        """
        patient = self.patients[index]
        concepts = torch.tensor(patient.concepts, dtype=torch.long)
        masked_concepts, target = self.masker.mask_patient_concepts(concepts)

        sample = {
            "concept": masked_concepts,
            "target": target,
            "abspos": torch.tensor(patient.abspos, dtype=torch.float),
            "segment": torch.tensor(patient.segments, dtype=torch.long),
            "age": torch.tensor(patient.ages, dtype=torch.float),
        }

        return sample

    def __len__(self):
        return len(self.patients)


class BinaryOutcomeDataset(Dataset):
    """
    outcomes: absolute position when outcome occured for each patient
    outcomes is a list of the outcome timestamps to predict
    """

    def __init__(self, patients: List[PatientData]):
        self.patients = patients

    def __getitem__(self, index: int) -> dict:
        patient = self.patients[index]
        attention_mask = torch.ones(
            len(patient.concepts), dtype=torch.long
        )  # Require attention mask for bi-gru head
        sample = {
            "concept": torch.tensor(patient.concepts, dtype=torch.long),
            "abspos": torch.tensor(patient.abspos, dtype=torch.float),
            "segment": torch.tensor(patient.segments, dtype=torch.long),
            "age": torch.tensor(patient.ages, dtype=torch.float),
            "attention_mask": attention_mask,
            "target": torch.tensor(patient.outcome, dtype=torch.float),
        }
        return sample

    def __len__(self):
        return len(self.patients)
