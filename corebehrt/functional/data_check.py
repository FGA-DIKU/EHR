import dask.dataframe as dd
import logging

logger = logging.getLogger(__name__)


def check_max_segment(data: dd.DataFrame, vocab_size: int) -> None:
    """Check max segment. If it's larger or equal to the model type vocab size, change accordingly."""
    max_segment = data["segment"].max().compute()
    if max_segment >= vocab_size:
        logger.warning(
            f"You've set vocab too low. Max segment {max_segment} >= type_vocab_size {vocab_size}\
                         Change it to {max_segment+1}."
        )
