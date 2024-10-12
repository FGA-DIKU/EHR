"""Create tokenized features from formatted data. config template: data.yaml"""

from collections import defaultdict
from os.path import join

import pandas as pd
from tqdm import tqdm

from corebehrt.classes.outcomes import OutcomeMaker
from corebehrt.common.azure import AzurePathContext, save_to_blobstore
from corebehrt.common.config import load_config
from corebehrt.common.logger import TqdmToLogger
from corebehrt.common.setup import DirectoryPreparer, get_args
from corebehrt.data.concept_loader import ConceptLoaderLarge

BLOBSTORE = "PHAIR"
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
    cfg.paths.outcome_dir = join(cfg.features_dir, "outcomes", cfg.outcomes_name)

    cfg, _, mount_context = AzurePathContext(
        cfg, dataset_name=BLOBSTORE
    ).azure_outcomes_setup()

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

    if cfg.env == "azure":
        save_to_blobstore(
            local_path="outcomes",
            remote_path=join(BLOBSTORE, "outcomes", cfg.paths.run_name),
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
