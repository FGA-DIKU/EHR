import logging
import os
from datetime import datetime
from os.path import join
from typing import Tuple

import dask.dataframe as dd
import pandas as pd
import torch
from dask.diagnostics import ProgressBar

from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.setup.config import Config, load_config
from corebehrt.common.constants import ABSPOS_COL, PID_COL, TIMESTAMP_COL


from corebehrt.modules.setup.directory import (
    INDEX_DATES_FILE,
    OUTCOMES_FILE,
    PID_FILE,
    PROCESSED_DATA_DIR,
)

from corebehrt.functional.convert import dataframe_to_patient_list
from corebehrt.functional.filter import censor_patient, exclude_short_sequences
from corebehrt.functional.load import load_vocabulary
from corebehrt.functional.outcomes import get_binary_outcomes
from corebehrt.functional.save import save_vocabulary
from corebehrt.functional.truncate import truncate_patient
from corebehrt.functional.utils import (
    get_abspos_from_origin_point,
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

        pids = self.load_cohort(paths_cfg)

        # Load index dates and convert to abspos
        index_dates = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        origin_point = load_config(
            join(paths_cfg.features, "data_config.yaml")
        ).features.origin_point
        index_dates[ABSPOS_COL] = get_abspos_from_origin_point(
            index_dates[TIMESTAMP_COL], datetime(**origin_point)
        )

        # Load tokenized data
        with ProgressBar(dt=1):
            df = dd.read_parquet(
                join(
                    paths_cfg.tokenized,
                    "features_finetune",
                )
            )
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            # !TODO: if index date is the same for all patients, then we can censor here.
            df = df.compute()

        patient_list = dataframe_to_patient_list(df)
        logger.info(f"Number of patients: {len(patient_list)}")
        vocab = load_vocabulary(paths_cfg.tokenized)
        data = PatientDataset(patients=patient_list)

        # Loading and processing outcomes
        outcomes = pd.read_csv(paths_cfg.outcome)
        outcomes = filter_df_by_pids(outcomes, data.get_pids())
        logger.info("Handling outcomes")
        # Outcome Handler now only needs to do 1 thing: if outcome is in follow up window 1 else 0
        binary_outcomes = get_binary_outcomes(
            index_dates,
            outcomes,
            data_cfg.get("n_hours_start_follow_up", 0),
            data_cfg.get("n_hours_end_follow_up", None),
        )

        logger.info("Assigning outcomes")
        data = data.assign_outcomes(binary_outcomes)

        censor_dates = pd.Series(
            index_dates[ABSPOS_COL] + self.cfg.outcome.n_hours_censoring,
            index=index_dates[PID_COL],
        )

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

        pids = self.load_cohort(paths_cfg)
        # Load tokenized data + vocab
        with ProgressBar(dt=10):
            df = dd.read_parquet(
                join(
                    paths_cfg.tokenized,
                    "features_pretrain",
                )
            )
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            df = df.compute()

        patient_list = dataframe_to_patient_list(df)
        logger.info(f"Number of patients: {len(patient_list)}")
        vocab = load_vocabulary(paths_cfg.tokenized)
        data = PatientDataset(patients=patient_list)

        # Excluding short sequences
        logger.info("Excluding short sequences")
        background_length = get_background_length(data, vocab)
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 0) + background_length,
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

    @staticmethod
    def load_cohort(paths_cfg):
        pids = None
        if paths_cfg.get("cohort"):
            pids = set(torch.load(join(paths_cfg.cohort, PID_FILE)))
        return pids
