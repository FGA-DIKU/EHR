"""Create tokenized features from formatted data. config template: data.yaml"""

import logging
from os.path import join

from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.features.loader import ConceptLoaderLarge
from corebehrt.main.helper.create_outcomes import process_data

CONFIG_PATH = "./corebehrt/configs/outcomes_test.yaml"


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
