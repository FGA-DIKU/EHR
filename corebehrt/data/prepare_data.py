import logging
import os
from os.path import join

import dask.dataframe as dd
import pandas as pd
import torch

from corebehrt.classes.outcomes import OutcomeHandler
from corebehrt.common.config import Config, load_config
from corebehrt.common.loader import FeaturesLoader
from corebehrt.common.saver import Saver
from corebehrt.common.utils import Data
from corebehrt.data.dataset import MLMDataset
from corebehrt.functional.convert import convert_to_sequences
from corebehrt.functional.data_check import check_max_segment, log_features_in_sequence
from corebehrt.functional.exclude import (
    exclude_short_sequences_dask,
    filter_patients_by_gender,
    exclude_pids_from_data,
)
from corebehrt.functional.filter import (
    censor_data,
    filter_patients_by_age_at_last_event,
)
from corebehrt.functional.load import load_pids, load_predefined_pids
from corebehrt.functional.save import save_data, save_pids_splits, save_sequence_lengths
from corebehrt.functional.split import load_train_val_split, split_pids_into_train_val
from corebehrt.functional.utils import (
    filter_table_by_pids,
    get_background_length_dd,
    normalize_segments,
    select_random_subset,
    truncate_data,
    truncate_patient,
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

    def prepare_finetune_data(self) -> Data:
        data_cfg = self.cfg.data
        paths_cfg = self.cfg.paths

        # 1. Loading tokenized data
        data = dd.read_csv(
            join(
                paths_cfg.tokenized,
                "features_finetune",
                "*.csv",
            )
        )
        vocab = torch.load(join(paths_cfg.tokenized, VOCABULARY_FILE))
        if paths_cfg.get("exclude_pids", None):
            pids_to_exclude = load_pids(paths_cfg.exclude_pids)
            data = exclude_pids_from_data(data, pids_to_exclude)

        predefined_splits = paths_cfg.get("predefined_splits", False)
        if predefined_splits:
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            logger.warning("Use original censoring time. Overwrite n_hours parameter.")
            if not os.path.exists(predefined_splits):
                raise ValueError(
                    f"Predefined splits folder {predefined_splits} does not exist."
                )
            if os.path.exists(join(predefined_splits, "finetune_config.yaml")):
                original_config = load_config(
                    join(predefined_splits, "finetune_config.yaml")
                )
            else:
                if "model_path" not in paths_cfg:
                    raise ValueError(
                        "Model path must be provided if no finetune_config in predefined splits folder."
                    )
                original_config = load_config(
                    join(paths_cfg.model_path, "finetune_config.yaml")
                )
            self.cfg.outcome.n_hours_censoring = (
                original_config.outcome.n_hours_censoring
            )
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            data = filter_table_by_pids(data, load_predefined_pids(predefined_splits))
            outcomes = pd.read_csv(join(predefined_splits, "outcomes.csv"))
            exposures = pd.read_csv(join(predefined_splits, "censor_outcomes.csv"))

        else:
            # 2. Optional: Select gender group
            data = filter_patients_by_gender(
                data, vocab, self.cfg.data.get("gender", None)
            )

            # 3. Loading and processing outcomes
            outcomes = pd.read_csv(paths_cfg.outcome)
            exposures = pd.read_csv(paths_cfg.exposure)

            outcomehandler = OutcomeHandler(
                index_date=self.cfg.outcome.get("index_date", None),
                select_patient_group=data_cfg.get(
                    "select_patient_group", None
                ),  # exposed/unexposed
                exclude_pre_followup_outcome_patients=self.cfg.outcome.get(
                    "first_time_outcomes_only", False
                ),
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

        if not predefined_splits:
            # 5. Optional: Select Patients By Age
            if data_cfg.get("min_age") or data_cfg.get("max_age"):
                data = filter_patients_by_age_at_last_event(
                    data, data_cfg.min_age, data_cfg.max_age
                )

        # 6. Exclude short sequences
        data = exclude_short_sequences_dask(
            data, data_cfg.get("min_len", 1), get_background_length_dd(data, vocab)
        )

        # 7. Optional: Patient Subset Selection
        if not predefined_splits and data_cfg.get("num_patients"):
            data = select_random_subset(data, data_cfg.num_patients)

        # 8. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        data = truncate_data(data, data_cfg.truncation_len, vocab, truncate_patient)

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

    def _prepare_mlm_features(self, predefined_splits, val_ratio=0.2) -> Data:
        data_cfg = self.cfg.data
        model_cfg = self.cfg.model
        paths_cfg = self.cfg.paths

        # 1. Load tokenized data + vocab
        data = dd.read_csv(
            join(
                paths_cfg.tokenized,
                "features_pretrain",
                "*.csv",
            )
        )  # self.loader.load_tokenized_data(mode='pretrain')
        vocab = torch.load(join(paths_cfg.tokenized, VOCABULARY_FILE))

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
        data = exclude_short_sequences_dask(
            data, data_cfg.get("min_len", 1), get_background_length_dd(data, vocab)
        )

        # 4. Optional: Patient Subset Selection
        if not predefined_splits and data_cfg.get("num_patients"):
            data = select_random_subset(data, data_cfg.num_patients)

        # 5. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        data = truncate_data(data, data_cfg.truncation_len, vocab, truncate_patient)

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
