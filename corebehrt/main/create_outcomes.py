"""Create tokenized features from formatted data. config template: data.yaml"""

from collections import defaultdict
from os.path import join
import logging

import pandas as pd
from tqdm import tqdm

from corebehrt.classes.outcomes import OutcomeMaker
from corebehrt.common.config import load_config
from corebehrt.common.logger import TqdmToLogger
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.data.concept_loader import ConceptLoaderLarge

CONFIG_PATH = "./corebehrt/configs/outcomes_test.yaml"


def process_data(loader, cfg, features_cfg, logger) -> dict:
    all_outcomes = defaultdict(list)
    for concept_batch, patient_batch in tqdm(
        loader(), desc="Batch Process Data", file=TqdmToLogger(logger)
    ):
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

    prepper = DirectoryPreparer(cfg)
    prepper.setup_create_outcomes()

    logger = logging.getLogger("create_outcomes")
    logger.info("Starting outcomes creation")
    features_cfg = prepper.get_config("features")
    outcome_tables = process_data(
        ConceptLoaderLarge(data_dir=cfg.paths.data, **cfg.loader),
        cfg,
        features_cfg,
        logger,
    )

    for key, df in outcome_tables.items():
        df.to_csv(join(cfg.paths.outcomes, f"{key}.csv"), index=False)

    logger.info("Finish outcomes creation")
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
