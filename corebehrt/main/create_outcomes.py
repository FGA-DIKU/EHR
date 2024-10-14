"""Create tokenized features from formatted data. config template: data.yaml"""

from collections import defaultdict
from os.path import join

import pandas as pd
from tqdm import tqdm

from corebehrt.classes.outcomes import OutcomeMaker
from corebehrt.common.config import load_config
from corebehrt.common.logger import TqdmToLogger
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.common.utils import check_patient_counts
from corebehrt.data.concept_loader import ConceptLoaderLarge

CONFIG_PATH = "./corebehrt/configs/outcomes_test.yaml"


def process_data(loader, cfg, features_cfg, logger) -> dict:
    all_outcomes = defaultdict(list)
    for concept_batch, patient_batch in tqdm(
        loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
    ):
        check_patient_counts(concept_batch, patient_batch, logger)
        pids = concept_batch.PID.unique()
        outcome_tables = OutcomeMaker(cfg.outcomes, features_cfg.features.origin_point)(
            concept_batch, patient_batch, pids
        )
        # Concatenate the tables for each key
        for key, df in outcome_tables.items():
            if key in all_outcomes:
                all_outcomes[key] = pd.concat([all_outcomes[key], df])
            else:
                all_outcomes[key] = df
    return all_outcomes


def main_data(config_path):
    cfg = load_config(config_path)
    if "outcome_dir" not in cfg.paths:
        cfg.paths.outcome_dir = join(cfg.features_dir, "outcomes", cfg.outcomes_name)

    logger = DirectoryPreparer(config_path).prepare_directory_outcomes(
        cfg.paths.outcome_dir, cfg.outcomes_name
    )
    logger.info("Mount Dataset")
    logger.info("Starting outcomes creation")
    features_cfg = load_config(join(cfg.features_dir, "data_config.yaml"))
    outcome_tables = process_data(
        ConceptLoaderLarge(**cfg.loader), cfg, features_cfg, logger
    )

    for key, df in outcome_tables.items():
        df.to_csv(join(cfg.paths.outcome_dir, f"{key}.csv"), index=False)

    logger.info("Finish outcomes creation")
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
