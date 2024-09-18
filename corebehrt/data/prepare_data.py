import logging
import os
from os.path import join
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch

from corebehrt.common.config import Config, instantiate, load_config
from corebehrt.common.loader import (
    FeaturesLoader,
    get_pids_file,
    load_and_select_splits,
    load_exclude_pids,
)
from corebehrt.common.saver import Saver
from corebehrt.common.utils import Data
from corebehrt.data.dataset import MLMDataset
from corebehrt.data.filter import CodeTypeFilter, PatientFilter
from corebehrt.data.utils import Utilities
from corebehrt.data_fixes.truncate import Truncator
from corebehrt.classes.outcomes import OutcomeHandler

# New stuff
import dask.dataframe as dd
from corebehrt.functional.exclude import (
    exclude_short_sequences,
    filter_table_by_exclude_pids,
)
from corebehrt.functional.split import split_pids_into_train_val, load_train_val_split
from corebehrt.functional.convert import convert_to_sequences
from corebehrt.functional.load import load_pids, load_predefined_pids
from corebehrt.functional.utils import (
    filter_table_by_pids,
    select_random_subset,
    truncate_data,
    truncate_patient,
    normalize_segments,
    get_background_length_dd,
)
from corebehrt.functional.data_check import check_max_segment, log_features_in_sequence
from corebehrt.functional.save import save_sequence_lengths, save_data, save_pids_splits

from corebehrt.functional.filter import censor_data, filter_patients_by_age_at_last_event
from corebehrt.functional.exclude import filter_patients_by_gender

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = "PID"
VOCABULARY_FILE = "vocabulary.pt"
DEFAULT_CENSORER = "corebehrt.data_fixes.censor.Censorer"


# TODO: Add option to load test set only!
class DatasetPreparer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loader = FeaturesLoader(cfg)

        run_folder = join(self.cfg.paths.output_path, self.cfg.paths.run_name)
        self.saver = Saver(run_folder)

        self.patient_filter = PatientFilter(cfg)
        self.code_type_filter = CodeTypeFilter(cfg)
        self.data_modifier = DataModifier(cfg)

    def prepare_mlm_dataset(self, val_ratio=0.2):
        """Load data, truncate, adapt features, create dataset"""
        predefined_splits = self.cfg.paths.get("predefined_pids", False)
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
                paths_cfg.data_path,
                paths_cfg.tokenized_dir,
                "features_finetune",
                "*.csv",
            )
        )  
        vocab = torch.load(
            join(paths_cfg.data_path, paths_cfg.tokenized_dir, VOCABULARY_FILE)
        )

        initial_pids = data['PID'].unique().compute().tolist()
        data = filter_table_by_exclude_pids(data, paths_cfg.get("filter_table_by_exclude_pids", None))
        
        predefined_splits = paths_cfg.get("predefined_splits", False)
        if predefined_splits:
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            logger.warning("Use original censoring time. Overwrite n_hours parameter.")
            if not os.path.exists(predefined_splits):
                raise ValueError(
                    f"Predefined splits folder {predefined_splits} does not exist."
                )
            if os.path.exists(
                join(predefined_splits, "finetune_config.yaml")
            ):
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
            self.cfg.outcome.n_hours = original_config.outcome.n_hours
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            data = filter_table_by_pids(data, load_predefined_pids(predefined_splits))
            outcomes = pd.read_csv(join(predefined_splits, "outcomes.csv"))
            exposures = pd.read_csv(join(predefined_splits, "censor_outcomes.csv"))

        else:
            # 2. Optional: Select gender group
            data = filter_patients_by_gender(data, vocab, self.cfg.data.get('gender', None))

            # 3. Loading and processing outcomes
            outcomes = pd.read_csv(paths_cfg.outcome)
            exposures = pd.read_csv(paths_cfg.exposure)
            
            outcomehandler = OutcomeHandler(
                index_date=self.cfg.outcome.get("index_date", None),
                select_patient_group=data_cfg.get("select_patient_group", None), # exposed/unexposed
                exclude_pre_followup_outcome_patients=self.cfg.outcome.get("first_time_outcomes_only", False),
            )
            data, index_dates, outcomes = outcomehandler.handle(
                data,
                outcomes=outcomes, 
                exposures=exposures, 
                )

        # 4. Data censoring
        censor_dates = index_dates + self.cfg.outcome.n_hours
        data = censor_data(
            data,
            censor_dates, 
        )
        
        if not predefined_splits:
            # 5. Optional: Select Patients By Age
            if data_cfg.get("min_age") or data_cfg.get("max_age"):
                data = filter_patients_by_age_at_last_event(data, data_cfg.min_age, data_cfg.max_age)

        # 6. Exclude short sequences
        data = exclude_short_sequences(
            data, data_cfg.get("min_len", 1), get_background_length_dd(data, vocab)
        )

        # 7. Optional: Patient Subset Selection
        if not predefined_splits and data_cfg.get("num_patients"):
            data = select_random_subset(data, data_cfg.num_patients)

        # 8. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        data = truncate_data(data, data_cfg.truncation_len, vocab, truncate_patient)
        
        features, pids = convert_to_sequences(data)
        data = Data(features=features, 
                    pids=pids, vocabulary=vocab, mode="finetune")
        data.add_outcomes(outcomes)
        data.add_index_dates(index_dates)
        data.check_lengths()

        # 13. Normalize segments
        data = Utilities.process_data(data, self.data_modifier.normalize_segments)

        # Verify and save
        data.check_lengths()
        data = Utilities.process_data(data, self.saver.save_sequence_lengths)

        excluded_pids = list(set(initial_pids).difference(set(data.pids)))
        self.saver.save_list(excluded_pids, "excluded_pids.pt")

        self.saver.save_data(data)
        self._log_features(data)
        return data

    def _prepare_mlm_features(self, predefined_splits, val_ratio=0.2) -> Data:
        """
        1. Load tokenized data
        2. Optional: Remove background tokens
        3. Exclude short sequences
        4. Optional: Select subset of patients
        5. Truncation
        6. Normalize segments
        """
        data_cfg = self.cfg.data
        model_cfg = self.cfg.model
        paths_cfg = self.cfg.paths

        # 1. Load tokenized data + vocab
        data = dd.read_csv(
            join(
                paths_cfg.data_path,
                paths_cfg.tokenized_dir,
                "features_pretrain",
                "*.csv",
            )
        )  # self.loader.load_tokenized_data(mode='pretrain')
        vocab = torch.load(
            join(paths_cfg.data_path, paths_cfg.tokenized_dir, VOCABULARY_FILE)
        )

        # 2. Exclude pids
        data = filter_table_by_exclude_pids(data, paths_cfg.get("exclude_pids", None))

        # 3. Select predefined pids, remove the rest
        predefined_pids = self.cfg.paths.get("predefined_pids", False)
        if predefined_pids:
            logger.warning("Using predefined splits. Ignoring test_split parameter")
            data = filter_table_by_pids(data, load_predefined_pids(predefined_pids))

        # 3. Exclude short sequences
        data = exclude_short_sequences(
            data, data_cfg.get("min_len", 1), get_background_length_dd(data, vocab)
        )

        # 4. Optional: Patient Subset Selection
        if not predefined_pids and data_cfg.get("num_patients"):
            data = select_random_subset(data, data_cfg.num_patients)

        # 5. Truncation
        logger.info(f"Truncating data to {data_cfg.truncation_len} tokens")
        data = truncate_data(data, data_cfg.truncation_len, vocab, truncate_patient)

        # 6. Normalize segments
        data = normalize_segments(data)

        # Check if max segment is larger than type_vocab_size
        check_max_segment(data, model_cfg.type_vocab_size)

        # Save
        save_dir = join(self.cfg.paths.output_path, self.cfg.paths.run_name)
        save_sequence_lengths(data, save_dir, desc="_pretrain")
        save_data(data, vocab, save_dir, desc="_pretrain")

        # Splitting data
        if predefined_splits:
            train_data, val_data = load_train_val_split(data, predefined_splits)
        else:
            train_data, val_data = split_pids_into_train_val(data, val_ratio)

        # Save split
        save_pids_splits(train_data, val_data, save_dir)

        # Convert to sequences
        train_features, train_pids = convert_to_sequences(train_data)
        train_data = Data(train_features, train_pids, vocabulary=vocab, mode="train")
        val_features, val_pids = convert_to_sequences(val_data)
        val_data = Data(val_features, val_pids, vocabulary=vocab, mode="val")

        log_features_in_sequence(train_data)

        return train_data, val_data

    def _retrieve_and_assign_outcomes(
        self, data: Data, outcomes: Dict, censor_outcomes: Dict
    ) -> Data:
        """Retrieve outcomes and assign them to the data instance"""
        data.outcomes = Utilities.select_and_order_outcomes_for_patients(
            outcomes, data.pids, self.cfg.outcome.type
        )
        if self.cfg.outcome.get("censor_type") is not None:
            data.censor_outcomes = Utilities.select_and_order_outcomes_for_patients(
                censor_outcomes, data.pids, self.cfg.outcome.censor_type
            )
        else:
            data.censor_outcomes = [None] * len(outcomes)
        return data

    @staticmethod
    def _get_predefined_pids(predefined_splits_path) -> List:
        """Return pids from predefined splits"""
        if os.path.exists(join(predefined_splits_path, "pids.pt")):
            return torch.load(join(predefined_splits_path, "pids.pt"))
        else:
            train_pids = torch.load(get_pids_file(predefined_splits_path, "train"))
            val_pids = torch.load(get_pids_file(predefined_splits_path, "val"))
            return train_pids + val_pids

    def _select_predefined_pids(self, data: Data):
        """Validate predefined splits as subset of data."""
        predefined_splits_path = self.cfg.paths.predefined_splits
        predefined_pids = self._get_predefined_pids(predefined_splits_path)
        if not set(predefined_pids).issubset(set(data.pids)):
            raise ValueError(
                f"Pids in the predefined splits must be a subset of data.pids. There are {len(set(predefined_pids).difference(set(data.pids)))} pids in the data that are not in the predefined splits"
            )
        data = data.select_data_subset_by_pids(predefined_pids, mode=data.mode)
        return data

    def _load_outcomes_to_data(self, data: Data) -> None:
        """Load outcomes and censor outcomes to data."""
        for outcome_type in ["outcomes", "censor_outcomes"]:
            setattr(
                data,
                outcome_type,
                torch.load(
                    join(self.cfg.paths.predefined_splits, f"{outcome_type}.pt")
                ),
            )

    def _log_features(self, data: Data) -> None:
        logger.info(f"Final features: {data.features.keys()}")
        logger.info("Example features: ")
        for k, v in data.features.items():
            logger.info(f"{k}: {v[0]}")


class DataModifier:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @staticmethod
    def truncate(data: Data, truncation_len: int) -> Data:
        truncator = Truncator(max_len=truncation_len, vocabulary=data.vocabulary)
        data.features = truncator(data.features)
        return data

    def censor_data(self, data: Data, n_hours: float) -> Data:
        """Censors data n_hours after censor_outcome."""
        censorer_cfg = self.cfg.data.get("censorer", {"_target_": DEFAULT_CENSORER})
        censorer = instantiate(
            censorer_cfg, vocabulary=data.vocabulary, n_hours=n_hours
        )
        logger.info(
            f"Censoring data {n_hours} hours after outcome with {censorer.__class__.__name__}"
        )
        data.features = censorer(data.features, data.censor_outcomes)
        return data

    @staticmethod
    def normalize_segments(data: Data) -> Data:
        """Normalize segments after truncation to start with 1 and increase by 1 then normalize those."""
        data.features = normalize_segments(data.features)
        return data


def retrieve_outcomes(
    all_outcomes: Dict, all_censor_outcomes: Dict, cfg: Config
) -> Union[List, List]:
    """From the configuration, load the outcomes and censor outcomes."""
    pids = all_outcomes[PID_KEY]
    outcomes = all_outcomes.get(cfg.outcome.type, [None] * len(all_outcomes[PID_KEY]))
    censor_outcomes = all_censor_outcomes.get(
        cfg.outcome.get("censor_type"), [None] * len(outcomes)
    )
    return outcomes, censor_outcomes, pids


def select_positives(
    outcomes: List, censor_outcomes: List, pids: List
) -> Tuple[List, List, List]:
    """Select only positive outcomes."""
    logger.info("Selecting only positive outcomes")
    select_indices = set([i for i, outcome in enumerate(outcomes) if pd.notna(outcome)])
    outcomes = [outcomes[i] for i in select_indices]
    censor_outcomes = [censor_outcomes[i] for i in select_indices]
    pids = [pids[i] for i in select_indices]
    return outcomes, censor_outcomes, pids
