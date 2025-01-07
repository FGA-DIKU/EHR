import logging
import os
from os.path import join
from typing import Tuple

import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

from corebehrt.classes.dataset import PatientDataset
from corebehrt.classes.outcomes import OutcomeHandler
from corebehrt.common.config import Config
from corebehrt.common.setup import INDEX_DATES_FILE, OUTCOMES_FILE, PROCESSED_DATA_DIR
from corebehrt.functional.convert import dataframe_to_patient_list
from corebehrt.functional.filter import censor_patient, exclude_short_sequences
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.save import save_vocabulary
from corebehrt.functional.truncate import truncate_patient
from corebehrt.functional.utils import (
    get_background_length,
    get_non_priority_tokens,
    normalize_segments_for_patient,
)

logger = logging.getLogger(__name__)  # Get the logger for this module


# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.save_dir = cfg.paths.model
        self.processed_dir = join(self.save_dir, PROCESSED_DATA_DIR)

    def prepare_finetune_data(self) -> PatientDataset:
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # Load tokenized data
        with ProgressBar(dt=10):
            df = dd.read_parquet(
                join(
                    paths_cfg.tokenized,
                    "features_finetune",
                )
            ).compute()

        patient_list = dataframe_to_patient_list(df)
        logger.info(f"Number of patients: {len(patient_list)}")
        vocab = load_vocabulary(self.processed_dir)
        data = PatientDataset(patients=patient_list)

        # Loading and processing outcomes
        outcomes = pd.read_csv(paths_cfg.outcome)
        exposures = pd.read_csv(paths_cfg.exposure)

        logger.info("Handling outcomes")
        outcomehandler = OutcomeHandler(
            index_date=self.cfg.outcome.get("index_date", None),
        )

        index_dates, outcomes = outcomehandler.handle(
            data.get_pids(),
            outcomes=outcomes,
            exposures=exposures,
        )

        logger.info("Assigning outcomes")
        data = data.assign_outcomes(outcomes)

        # Data censoring
        censor_dates = index_dates + self.cfg.outcome.n_hours_censoring
        data.patients = data.process_in_parallel(
            censor_patient, censor_dates=censor_dates
        )

        background_length = get_background_length(data, vocab)
        # Exclude short sequences
        logger.info("Excluding short sequences")
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + background_length,
        )

        # Truncation
        non_priority_tokens = (
            None
            if data_cfg.get("low_priority_prefixes", None) is None
            else get_non_priority_tokens(vocab, data_cfg.low_priority_prefixes)
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
            non_priority_tokens=non_priority_tokens,
        )

        # Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        # Check if max segment is larger than type_vocab b_size
        # TODO: pass pt_model_config and perform this check
        # max_segment(data, model_cfg.type_vocab_size)
        # Previously had issue with it
        logger.info(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )
        # save
        os.makedirs(self.processed_dir, exist_ok=True)
        save_vocabulary(vocab, self.processed_dir)
        if self.cfg.get("save_processed_data", False):
            data.save(self.processed_dir)
            outcomes.to_csv(join(self.processed_dir, OUTCOMES_FILE), index=False)
            index_dates.to_csv(join(self.processed_dir, INDEX_DATES_FILE), index=False)

        return data

    def prepare_pretrain_data(self) -> Tuple[PatientDataset, dict]:
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # Load tokenized data + vocab
        with ProgressBar(dt=10):
            df = dd.read_parquet(
                join(
                    paths_cfg.tokenized,
                    "features_pretrain",
                )
            ).compute()

        patient_list = dataframe_to_patient_list(df)
        logger.info(f"Number of patients: {len(patient_list)}")
        vocab = load_vocabulary(paths_cfg.tokenized)
        data = PatientDataset(patients=patient_list)

        # Excluding short sequences
        logger.info("Excluding short sequences")
        background_length = get_background_length(data, vocab)
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 1) + background_length,
        )

        # Truncation
        non_priority_tokens = (
            None
            if data_cfg.get("low_priority_prefixes", None) is None
            else get_non_priority_tokens(vocab, data_cfg.low_priority_prefixes)
        )
        data.patients = data.process_in_parallel(
            truncate_patient,
            max_len=data_cfg.truncation_len,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
            non_priority_tokens=non_priority_tokens,
        )

        # Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        logger.info(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )

        # Save
        os.makedirs(self.processed_dir, exist_ok=True)
        save_vocabulary(vocab, self.processed_dir)
        if self.cfg.get("save_processed_data", False):
            data.save(self.processed_dir)
        return data
