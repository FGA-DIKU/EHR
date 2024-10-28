import logging

import dask.dataframe as dd

logger = logging.getLogger(__name__)


def check_max_segment(data: dd.DataFrame, vocab_size: int) -> None:
    """Check max segment. If it's larger or equal to the model type vocab size, change accordingly."""
    max_segment = data["segment"].max().compute()
    if max_segment >= vocab_size:
        raise ValueError(
            f"You've set vocab too low. Max segment {max_segment} >= type_vocab_size {vocab_size}\
                         Change it to {max_segment+1}."
        )


def log_features_in_sequence(data: dict) -> None:
    logger.info(f"Final features: {data.features.keys()}")
    logger.info("Example features: ")
    for k, v in data.features.items():
        logger.info(f"{k}: {v[0]}")
