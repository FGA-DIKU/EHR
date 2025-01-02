import logging
from os.path import join

import dask.dataframe as dd
import pandas as pd

from corebehrt.classes.outcomes import OutcomeHandler
from corebehrt.common.config import Config
from corebehrt.common.loader import FeaturesLoader
from corebehrt.common.saver import Saver

# from corebehrt.common.utils import Data
from corebehrt.data.dataset import MLMDataset
from corebehrt.functional.convert import convert_to_sequences
from corebehrt.functional.data_check import check_max_segment, log_features_in_sequence
from corebehrt.functional.exclude import (
    exclude_pids_from_data,
)
from corebehrt.functional.filter import (
    censor_data,
)
from corebehrt.functional.load import load_pids, load_predefined_pids, load_vocabulary
from corebehrt.functional.save import save_data, save_pids_splits, save_sequence_lengths
from corebehrt.functional.split import load_train_val_split, split_pids_into_train_val
from corebehrt.functional.utils import (
    filter_table_by_pids,
    normalize_segments,
    select_random_subset,
)

logger = logging.getLogger(__name__)  # Get the logger for this module

VOCABULARY_FILE = "vocabulary.pt"


# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loader = FeaturesLoader(cfg)

        self.save_dir = cfg.paths.model
        self.saver = Saver(self.save_dir)

    def prepare_mlm_dataset(self, val_ratio=0.2):
        """Load data, truncate, adapt features, create dataset"""
        predefined_splits = self.cfg.paths.get("predefined_splits", False)
        train_data, val_data = self._prepare_mlm_features(predefined_splits)
        train_dataset = MLMDataset(
            train_data.features, train_data.vocabulary, **self.cfg.data.dataset
        )
        val_dataset = MLMDataset(
            val_data.features, train_data.vocabulary, **self.cfg.data.dataset
        )
        return train_dataset, val_dataset

    def prepare_finetune_data(self):
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # 1. Loading tokenized data
        df = dd.read_parquet(
            join(
                paths_cfg.tokenized,
                "features_finetune",
            )
        ).compute()

        patient_list = dataframe_to_patient_list(df)
        vocab = load_vocabulary(join(paths_cfg.tokenized, VOCABULARY_FILE))
        data = Data(patients=patient_list, vocabulary=vocab)

        # 3. Loading and processing outcomes
        outcomes = pd.read_csv(paths_cfg.outcome)
        exposures = pd.read_csv(paths_cfg.exposure)

        outcomehandler = OutcomeHandler(
            index_date=self.cfg.outcome.get("index_date", None),
        )
        data, index_dates, outcomes = outcomehandler.handle(
            data,
            outcomes=outcomes,
            exposures=exposures,
        )

        # 4. Data censoring
        censor_dates = index_dates + self.cfg.outcome.n_hours_censoring
        data = censor_data(
            data,
            censor_dates,
        )

        # 8. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        background_length = get_background_length(data, vocab)
        data.patients = data.process_in_parallel(
            truncate_patient_namedtuple,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )

        # 9. Normalize segments
        data = normalize_segments(data)

        # Check if max segment is larger than type_vocab_size
        # TODO: pass pt_model_config and perform this check
        # max_segment(data, model_cfg.type_vocab_size)
        # Previously had issue with it

        # save
        save_sequence_lengths(data, self.save_dir, desc="_finetune")
        save_data(data, vocab, self.save_dir, desc="_finetune")
        outcomes.to_csv(join(self.save_dir, "outcomes.csv"), index=False)
        index_dates.to_csv(join(self.save_dir, "index_dates.csv"), index=False)

        # Convert to sequences
        features, pids = convert_to_sequences(data)
        data = Data(features=features, pids=pids, vocabulary=vocab, mode="finetune")
        data.add_outcomes(outcomes)
        data.add_index_dates(index_dates)
        data.check_lengths()

        log_features_in_sequence(data)
        return data

    def _prepare_mlm_features(self, predefined_splits, val_ratio=0.2):
        data_cfg = self.cfg.data
        model_cfg = self.cfg.model
        paths_cfg = self.cfg.paths

        # 1. Load tokenized data + vocab
        df = dd.read_parquet(
            join(
                paths_cfg.tokenized,
                "features_pretrain",
            )
        ).compute()
        patient_list = dataframe_to_patient_list(df)
        vocab = load_vocabulary(join(paths_cfg.tokenized, VOCABULARY_FILE))
        data = Data(patients=patient_list, vocabulary=vocab)

        # 2. Exclude pids
        exclude_pids_path = paths_cfg.get("filter_table_by_exclude_pids", None)
        if exclude_pids_path:
            excluded_pids = load_pids(exclude_pids_path)
            data = exclude_pids_from_data(data, excluded_pids)

        # 3. Select predefined pids, remove the rest
        if predefined_splits:
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            data = filter_table_by_pids(data, load_predefined_pids(predefined_splits))

        # 3. Exclude short sequences
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + get_background_length(data, vocab),
        )

        # 4. Optional: Patient Subset Selection
        if not predefined_splits and data_cfg.get("num_patients"):
            data = select_random_subset(data, data_cfg.num_patients)

        # 5. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        background_length = get_background_length(data, vocab)
        data.patients = data.process_in_parallel(
            truncate_patient_namedtuple,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )
        print(data.patients[0])
        assert False
        # 6. Normalize segments
        data = normalize_segments(data)

        # Check if max segment is larger than type_vocab_size
        check_max_segment(data, model_cfg.type_vocab_size)

        # Save
        save_sequence_lengths(data, self.save_dir, desc="_pretrain")
        save_data(data, vocab, self.save_dir, desc="_pretrain")

        # Splitting data
        if predefined_splits:
            train_data, val_data = load_train_val_split(data, predefined_splits)
        else:
            train_data, val_data = split_pids_into_train_val(data, val_ratio)

        # Save split
        save_pids_splits(train_data, val_data, self.save_dir)

        # Convert to sequences
        train_features, train_pids = convert_to_sequences(train_data)
        train_data = Data(train_features, train_pids, vocabulary=vocab, mode="train")
        val_features, val_pids = convert_to_sequences(val_data)
        val_data = Data(val_features, val_pids, vocabulary=vocab, mode="val")

        return train_data, val_data


from typing import NamedTuple, List


class PatientData(NamedTuple):
    pid: str
    concepts: List[int]  # or List[str], depending on your use
    abspos: List[float]  # or int, depends on your data
    segments: List[int]
    ages: List[float]  # e.g. age at each concept


def dataframe_to_patient_list(df: pd.DataFrame) -> List[PatientData]:
    # Ensure df has at least these columns: pid, concept, abspos, segment, age
    patients_data = []

    # Optional: If you want to preserve the original row order within each patient,
    # you can sort by ['pid', 'abspos'] or your preferred column(s).
    # df = df.sort_values(['pid', 'abspos'])

    grouped = df.groupby("PID", sort=False)
    for pid, group in grouped:
        # Convert each column to a Python list
        concepts_list = group["concept"].tolist()
        abspos_list = group["abspos"].tolist()
        segments_list = group["segment"].tolist()
        ages_list = group["age"].tolist()

        # Create a PatientData instance
        patient = PatientData(
            pid=pid,
            concepts=concepts_list,
            abspos=abspos_list,
            segments=segments_list,
            ages=ages_list,
        )

        patients_data.append(patient)

    return patients_data


class Data:
    def __init__(self, patients: List[PatientData], vocabulary: dict):
        self.patients = patients
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx: int):
        return self.patients[idx]

    def process_in_parallel(self, func, n_jobs=-1, **kwargs):
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs)(
            delayed(func)(p, **kwargs) for p in self.patients
        )


def filter_patients_by_pids(
    patients: List[PatientData], pids: List[str]
) -> List[PatientData]:
    pids_set = set(pids)
    return [p for p in patients if p.pid in pids_set]


def exclude_short_sequences(
    patients: List[PatientData], min_len: int
) -> List[PatientData]:
    return [p for p in patients if len(p.concepts) >= min_len]


def get_background_length(patients: List[PatientData], vocabulary) -> int:
    """Get the length of the background sentence, first SEP token included."""
    background_tokens = set([v for k, v in vocabulary.items() if k.startswith("BG_")])
    example_concepts = patients[0].concepts
    background_length = len(set(example_concepts) & background_tokens)
    if "CLS" in vocabulary:
        return background_length + 2  # +2 for [CLS] and [SEP] tokens
    if "SEP" in vocabulary:
        return background_length + 1  # +1 for [SEP] token
    else:
        return background_length


def truncate_patient_namedtuple(
    patient: PatientData, background_length: int, max_len: int, sep_token: str
) -> PatientData:
    """
    Truncate patient to max_len, keeping background.
    """
    concepts = list(patient.concepts)
    length = len(concepts)

    if length <= max_len:
        return patient  # No truncation needed

    truncation_length = max_len - background_length

    # Check if the boundary item is a SEP token
    if concepts[-truncation_length] == sep_token:
        truncation_length -= 1

    # Create truncated lists for each field in the namedtuple
    truncated_fields = {}
    for field in patient._fields:
        if field == "pid":
            continue
        original_list = getattr(patient, field)
        truncated_fields[field] = (
            original_list[:background_length] + original_list[-truncation_length:]
        )

    # Return a new namedtuple with the truncated lists
    return patient._replace(**truncated_fields)
