import logging
import sys
import time
from os.path import join

import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

from corebehrt.classes.outcomes import OutcomeHandler
from corebehrt.common.config import Config
from corebehrt.common.loader import FeaturesLoader
from corebehrt.common.saver import Saver
from corebehrt.data.dataset import MLMDataset, PatientDataset
from corebehrt.functional.convert import dataframe_to_patient_list
from corebehrt.functional.filter import censor_data, exclude_short_sequences
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.save import save_pids_splits
from corebehrt.functional.split import load_train_val_split, split_pids_into_train_val
from corebehrt.functional.utils import (
    get_background_length,
    normalize_segments_for_patient,
    truncate_patient,
)
import dataclasses
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
            train_data.patients, train_data.vocabulary, **self.cfg.data.dataset
        )
        val_dataset = MLMDataset(
            val_data.patients, val_data.vocabulary, **self.cfg.data.dataset
        )
        return train_dataset, val_dataset

    def prepare_finetune_data(self):
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # 1. Loading tokenized data
        # Enable dask progress bar for reading parquet
        with ProgressBar(dt=10):
            df = dd.read_parquet(
                join(
                    paths_cfg.tokenized,
                    "features_finetune",
                )
            ).compute()
        print("Converting to patient list")
        patient_list = dataframe_to_patient_list(df)
        vocab = load_vocabulary(join(paths_cfg.tokenized, VOCABULARY_FILE))
        data = PatientDataset(patients=patient_list, vocabulary=vocab)

        # 3. Loading and processing outcomes
        outcomes = pd.read_csv(paths_cfg.outcome)
        exposures = pd.read_csv(paths_cfg.exposure)

        outcomehandler = OutcomeHandler(
            index_date=self.cfg.outcome.get("index_date", None),
        )
        index_dates, outcomes = outcomehandler.handle(
            data.get_pids(),
            outcomes=outcomes,
            exposures=exposures,
        )

        # 4. Data censoring
        censor_dates = index_dates + self.cfg.outcome.n_hours_censoring
        data = censor_data(
            data,
            censor_dates,
        )
        background_length = get_background_length(data, vocab)
        # 3. Exclude short sequences
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + background_length,
        )

        # 8. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")

        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )

        # 9. Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)

        # Check if max segment is larger than type_vocab_size
        # TODO: pass pt_model_config and perform this check
        # max_segment(data, model_cfg.type_vocab_size)
        # Previously had issue with it

        # save
        if self.cfg.get("save_processed_data", False):
            data.save(join(self.save_dir, "processed_data"))
            outcomes.to_csv(join(self.save_dir, "outcomes.csv"), index=False)
            index_dates.to_csv(join(self.save_dir, "index_dates.csv"), index=False)

        return data

    def _prepare_mlm_features(self, predefined_splits, val_ratio=0.2):
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # 1. Load tokenized data + vocab
        with ProgressBar(dt=10):
            df = dd.read_parquet(
                join(
                    paths_cfg.tokenized,
                    "features_pretrain",
                )
            ).compute()
        print(
            f"Memory usage of dataframe: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )
        print("Converting to patient list")
        patient_list = dataframe_to_patient_list(df)
        print(f"Number of patients: {len(patient_list)}")
        print(
            f"Memory usage of patient list: {get_recursive_size(patient_list) / 1024**2:.2f} MB"
        )
        vocab = load_vocabulary(join(paths_cfg.tokenized, VOCABULARY_FILE))
        data = PatientDataset(patients=patient_list, vocabulary=vocab)
        print("Excluding short sequences")

        # 3. Exclude short sequences
        start_time = time.time()
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + get_background_length(data, vocab),
        )
        print(
            f"Time to exclude short sequences: {time.time() - start_time:.2f} seconds"
        )
        print("Truncating data")
        # 5. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        background_length = get_background_length(data, vocab)
        start_time = time.time()
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )
        print(f"Time to truncate data: {time.time() - start_time:.2f} seconds")
        # 6. Normalize segments
        start_time = time.time()
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        print(f"Time to normalize segments: {time.time() - start_time:.2f} seconds")
        print(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )
        # Save
        if self.cfg.get("save_processed_data", False):
            data.save(join(self.save_dir, "processed_data"))

        # Splitting data
        if predefined_splits:
            train_data, val_data = load_train_val_split(data, predefined_splits)
        else:
            train_data, val_data = split_pids_into_train_val(data, val_ratio)

        # Save split
        save_pids_splits(train_data, val_data, self.save_dir)

        return train_data, val_data


import sys
def get_recursive_size(obj, visited=None) -> int:
    """
    Recursively compute the approximate memory footprint of a Python object,
    including nested containers and dataclasses.

    Args:
        obj: The object to measure.
        visited: A set of object IDs already visited (to avoid double-counting).
    Returns:
        int: The approximate size of 'obj' in bytes.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    # If already visited this object, return 0 to avoid double-counting.
    if obj_id in visited:
        return 0
    visited.add(obj_id)

    # Shallow size
    size = sys.getsizeof(obj)

    # If it's a dataclass instance, convert to dict for deeper inspection
    # (so we can recurse on its fields).
    if dataclasses.is_dataclass(obj):
        # Convert to a dict, then recurse on that dict.
        obj_dict = dataclasses.asdict(obj)
        size += get_recursive_size(obj_dict, visited)
        return size

    # If it's one of the common container types, recurse on its items.
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += get_recursive_size(k, visited)
            size += get_recursive_size(v, visited)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += get_recursive_size(item, visited)

    return size