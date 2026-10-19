import logging
from pathlib import Path

import hydra
import polars as pl
from dotenv import load_dotenv
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

from bonsai.functional.outcomes import (
    get_censor_dates,
    get_cohort,
    get_index_dates,
    get_outcome_dates,
)
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from bonsai.paths import get_config_path

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="example_outcome1",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting create_outcome_cohort for `{save_path.stem}`")

    all_outcomes = []
    for i, split in enumerate(cfg.splits):
        shards = sorted((input_dir / split).glob("*.parquet"))
        assert len(shards) > 0
        for j, shard in enumerate(shards):
            refs = {}
            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])
            df = df.drop_nulls(["subject_id", "code"])

            refs["index_dates"] = get_index_dates(df, cfg.index)
            refs["outcome_dates"] = get_outcome_dates(df, cfg.outcome, refs)
            refs["censor_dates"] = get_censor_dates(df, cfg.censor, refs)

            shard_cohort = get_cohort(df, cfg.cohort, refs)

            all_outcomes.append(
                shard_cohort.select("subject_id")
                .unique()
                .join(refs["outcome_dates"], on="subject_id", how="left")
                .join(refs["index_dates"], on="subject_id", how="inner")
                .join(refs["censor_dates"], on="subject_id", how="inner")
                .with_columns(split=pl.lit(split))
            )

    all_outcomes = pl.concat(all_outcomes).sort("subject_id", maintain_order=True)
    logging.info(
        f"Total number of subjects: {len(all_outcomes):_} ({all_outcomes['outcome_date'].is_not_null().sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.write_parquet(save_path)


if __name__ == "__main__":
    main()
