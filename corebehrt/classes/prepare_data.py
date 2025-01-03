import logging
from os.path import join

import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

from corebehrt.classes.outcomes import OutcomeHandler
from corebehrt.common.config import Config
from corebehrt.common.loader import FeaturesLoader
from corebehrt.classes.dataset import MLMDataset, PatientDataset
from corebehrt.functional.convert import dataframe_to_patient_list
from corebehrt.functional.filter import censor_patient, exclude_short_sequences
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.save import save_pids_splits
from corebehrt.functional.split import load_train_val_split, split_pids_into_train_val
from corebehrt.functional.truncate import truncate_patient
from corebehrt.functional.utils import (
    get_background_length,
    normalize_segments_for_patient,
    get_non_priority_tokens,
)

logger = logging.getLogger(__name__)  # Get the logger for this module

VOCABULARY_FILE = "vocabulary.pt"


# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loader = FeaturesLoader(cfg)

        self.save_dir = cfg.paths.model

    def prepare_mlm_dataset(self, val_ratio=0.2):
        """Load data, truncate, adapt features, create dataset"""
        predefined_splits = self.cfg.paths.get("predefined_splits", False)
        train_data, val_data = self._prepare_mlm_features(predefined_splits, val_ratio)
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

        logger.info("Converting to patient list")
        patient_list = dataframe_to_patient_list(df)
        logger.info(f"Number of patients: {len(patient_list)}")
        vocab = load_vocabulary(join(paths_cfg.tokenized, VOCABULARY_FILE))
        data = PatientDataset(patients=patient_list, vocabulary=vocab)

        # 3. Loading and processing outcomes
        logger.info("Loading outcomes")
        outcomes = pd.read_csv(paths_cfg.outcome)
        exposures = pd.read_csv(paths_cfg.exposure)

        outcomehandler = OutcomeHandler(
            index_date=self.cfg.outcome.get("index_date", None),
        )
        logger.info("Handling outcomes")
        index_dates, outcomes = outcomehandler.handle(
            data.get_pids(),
            outcomes=outcomes,
            exposures=exposures,
        )

        logger.info("Assigning outcomes")
        data = data.assign_outcomes(outcomes)

        # 4. Data censoring
        censor_dates = index_dates + self.cfg.outcome.n_hours_censoring
        data.patients = data.process_in_parallel(
            censor_patient, censor_dates=censor_dates
        )

        background_length = get_background_length(data, vocab)
        # 3. Exclude short sequences
        logger.info("Excluding short sequences")
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + background_length,
        )

        # 8. Truncation
        non_priority_tokens = get_non_priority_tokens(
            vocab, data_cfg.get("low_priority_prefixes", None)
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
            non_priority_tokens=non_priority_tokens,
        )

        # 9. Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        # Check if max segment is larger than type_vocab b_size
        # TODO: pass pt_model_config and perform this check
        # max_segment(data, model_cfg.type_vocab_size)
        # Previously had issue with it

        # save
        if self.cfg.get("save_processed_data", False):
            processed_dir = join(self.save_dir, "processed_data")
            data.save(processed_dir)
            outcomes.to_csv(join(processed_dir, "outcomes.csv"), index=False)
            index_dates.to_csv(join(processed_dir, "index_dates.csv"), index=False)

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
        print("Converting to patient list")
        patient_list = dataframe_to_patient_list(df)
        logger.info(f"Number of patients: {len(patient_list)}")
        vocab = load_vocabulary(join(paths_cfg.tokenized, VOCABULARY_FILE))
        data = PatientDataset(patients=patient_list, vocabulary=vocab)
        logger.info("Excluding short sequences")
        # 3. Exclude short sequences
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + get_background_length(data, vocab),
        )
        background_length = get_background_length(data, vocab)
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )
        # 6. Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        logger.info(
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
