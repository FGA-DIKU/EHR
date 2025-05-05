"""Create tokenized features from formatted data. config template: data.yaml"""

import logging
from os.path import join

from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.create_outcomes import process_data
from corebehrt.modules.features.loader import ShardLoader
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer

CONFIG_PATH = "./corebehrt/configs/outcomes_test.yaml"


def main_data(config_path):
    cfg = load_config(config_path)

    prepper = DirectoryPreparer(cfg)
    prepper.setup_create_outcomes()

    logger = logging.getLogger("create_outcomes")
    logger.info("Starting outcomes creation")
    outcome_tables = process_data(
        ShardLoader(
            data_dir=cfg.paths.data,
            splits=cfg.paths.get("splits", None),
            patient_info_path=join(cfg.paths.features, "patient_info.parquet"),
        ),
        cfg,
        logger,
    )

    for key, df in outcome_tables.items():
        if df.empty:
            logger.warning(f"Outcomes table for {key} is empty")
        df.to_csv(join(cfg.paths.outcomes, f"{key}.csv"), index=False)

    logger.info("Finish outcomes creation")
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_data(args.config_path)
