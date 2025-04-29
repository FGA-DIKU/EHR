import logging
import os
from datetime import datetime
from os.path import join
from typing import Tuple, List

import pandas as pd
import torch
from tqdm import tqdm

from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.constants.paths import INDEX_DATES_FILE, OUTCOMES_FILE, PID_FILE
from corebehrt.functional.cohort_handling.outcomes import get_binary_outcomes
from corebehrt.functional.features.normalize import normalize_segments_for_patient
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.io_operations.save import save_vocabulary
from corebehrt.functional.preparation.convert import dataframe_to_patient_list
from corebehrt.functional.preparation.filter import (
    censor_patient,
    exclude_short_sequences,
    censor_patient_with_delays,
)
from corebehrt.functional.preparation.truncate import (
    truncate_patient,
    truncate_patient_df,
)
from corebehrt.functional.preparation.utils import (
    get_background_length,
    get_background_length_pd,
    get_non_priority_tokens,
    get_concept_id_to_delay,
)
from corebehrt.functional.utils.time import get_hours_since_epoch
from corebehrt.modules.cohort_handling.patient_filter import filter_df_by_pids
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.monitoring.logger import TqdmToLogger
from corebehrt.modules.preparation.dataset import PatientDataset, PatientData
from corebehrt.modules.setup.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module


# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.processed_dir = cfg.paths.prepared_data

    def prepare_finetune_data(self, mode="tuning") -> PatientDataset:
        outcome_cfg = self.cfg.outcome
        paths_cfg = self.cfg.paths
        data_cfg = self.cfg.data

        pids = self.load_cohort(paths_cfg)

        # Load index dates and convert to abspos
        index_dates = pd.read_csv(
            join(paths_cfg.cohort, INDEX_DATES_FILE), parse_dates=[TIMESTAMP_COL]
        )
        index_dates[PID_COL] = index_dates[PID_COL].astype(int)
        index_dates[ABSPOS_COL] = get_hours_since_epoch(index_dates[TIMESTAMP_COL])

        # Load tokenized data
        loader = ShardLoader(
            data_dir=paths_cfg.tokenized,
            splits=[f"features_{mode}"],
            patient_info_path=None,
        )
        patient_list = []
        for df, _ in tqdm(
            loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
        ):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            if data_cfg.get("cutoff_date"):
                df = self._cutoff_data(df, data_cfg.cutoff_date)
            # !TODO: if index date is the same for all patients, then we can censor here.
            self._check_sorted(df)
            batch_patient_list = dataframe_to_patient_list(df)
            patient_list.extend(batch_patient_list)
        logger.info(f"Number of patients: {len(patient_list)}")
        data = PatientDataset(patients=patient_list)
        vocab = load_vocabulary(paths_cfg.tokenized)

        # Loading and processing outcomes
        outcomes = pd.read_csv(paths_cfg.outcome)
        outcomes[PID_COL] = outcomes[PID_COL].astype(int)
        outcomes = filter_df_by_pids(outcomes, data.get_pids())
        logger.info("Handling outcomes")
        # Outcome Handler now only needs to do 1 thing: if outcome is in follow up window 1 else 0
        binary_outcomes = get_binary_outcomes(
            index_dates,
            outcomes,
            outcome_cfg.get("n_hours_start_follow_up", 0),
            outcome_cfg.get("n_hours_end_follow_up", None),
        )

        logger.info("Assigning outcomes")
        data = data.assign_outcomes(binary_outcomes)

        censor_dates = (
            index_dates.set_index(PID_COL)[ABSPOS_COL]
            + self.cfg.outcome.n_hours_censoring
        )
        self._validate_censoring(data.patients, censor_dates, logger)
        if "concept_pattern_hours_delay" in self.cfg:
            concept_id_to_delay = get_concept_id_to_delay(
                self.cfg.concept_pattern_hours_delay, vocab
            )
            data.patients = data.process_in_parallel(
                censor_patient_with_delays,
                censor_dates=censor_dates,
                concept_id_to_delay=concept_id_to_delay,
            )
        else:
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
        data.save(self.processed_dir)
        outcomes.to_csv(join(self.processed_dir, OUTCOMES_FILE), index=False)
        index_dates.to_csv(join(self.processed_dir, INDEX_DATES_FILE), index=False)

        return data

    def prepare_pretrain_data(self, save_data=False) -> Tuple[PatientDataset, dict]:
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        pids = self.load_cohort(paths_cfg)
        # Load tokenized data + vocab
        vocab = load_vocabulary(paths_cfg.tokenized)
        loader = ShardLoader(
            data_dir=paths_cfg.tokenized,
            splits=["features_train"],
            patient_info_path=None,
        )
        patient_list = []
        for df, _ in tqdm(
            loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
        ):
            if pids is not None:
                df = filter_df_by_pids(df, pids)
            df = df.set_index(PID_COL, drop=True)

            if data_cfg.get("cutoff_date"):
                df = self._cutoff_data(df, data_cfg.cutoff_date)
            df = self._truncate(df, vocab, data_cfg.truncation_len)
            df = df.reset_index(drop=False)
            self._check_sorted(df)
            batch_patient_list = dataframe_to_patient_list(df)
            patient_list.extend(batch_patient_list)

        logger.info(f"Number of patients: {len(patient_list)}")
        data = PatientDataset(patients=patient_list)

        logger.info("Excluding short sequences")
        background_length = get_background_length(data, vocab)
        data.patients = exclude_short_sequences(
            data.patients,
            data_cfg.get("min_len", 0) + background_length,
        )

        # Normalize segments
        data.patients = data.process_in_parallel(normalize_segments_for_patient)
        logger.info(
            f"Max segment length: {max(max(patient.segments) for patient in data.patients)}"
        )

        # Save
        os.makedirs(self.processed_dir, exist_ok=True)
        save_vocabulary(vocab, self.processed_dir)
        if save_data:
            data.save(self.processed_dir)
        return data

    @staticmethod
    def _truncate(
        df: pd.DataFrame, vocab: dict, truncation_length: int
    ) -> pd.DataFrame:
        """
        Truncate the dataframe to the truncation length.
        """
        background_length = get_background_length_pd(df, vocab)

        df = df.groupby(PID_COL, group_keys=False).apply(
            truncate_patient_df,
            max_len=truncation_length,
            background_length=background_length,
            sep_token=vocab["[SEP]"],
        )
        return df

    @staticmethod
    def load_cohort(paths_cfg):
        pids = None
        if paths_cfg.get("cohort"):
            pids = torch.load(join(paths_cfg.cohort, PID_FILE))
        return pids

    @staticmethod
    def _check_sorted(df: pd.DataFrame, n_patients: int = 10):
        """Verify abspos sorting within each sampled patient"""
        sample_patients = df[PID_COL].unique()[:n_patients]
        for pid in sample_patients:
            patient_df = df[df[PID_COL] == pid]
            if not patient_df[ABSPOS_COL].is_monotonic_increasing:
                raise ValueError(f"Patient {pid} has unsorted abspos values")

    def _cutoff_data(self, df: pd.DataFrame, cutoff_date: dict) -> pd.DataFrame:
        """Cutoff data after a given date."""
        cutoff_abspos = get_hours_since_epoch(datetime(**cutoff_date))
        df = df[df[ABSPOS_COL] <= cutoff_abspos]
        return df

    @staticmethod
    def _validate_censoring(
        patients: List["PatientData"], censor_dates: pd.Series, logger: logging.Logger
    ) -> None:
        """Validate censoring dates and log basic statistics.

        Args:
            patients: List of patient data objects
            censor_dates: Series with censoring dates indexed by patient ID
            logger: Logger instance
        """
        patient_pids = set(p.pid for p in patients)
        censor_pids = set(censor_dates.index)

        missing_censor_dates = patient_pids - censor_pids
        if missing_censor_dates:
            logger.error(
                f"Missing censor dates for {len(missing_censor_dates)} patients"
            )
            raise ValueError("Some patients are missing censor dates")

        logger.info(f"Censoring validated for {len(patient_pids)} patients")

        # Check for NaN values in censor dates
        nan_censor_dates = censor_dates.isna().sum()
        if nan_censor_dates > 0:
            logger.error(f"Found {nan_censor_dates} NaN values in censor dates")
            raise ValueError("NaN values detected in censor dates")

        logger.info(f"Censoring validated for {len(patient_pids)} patients")
