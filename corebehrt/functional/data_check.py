import logging

import dask.dataframe as dd
import pandas as pd

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

def check_patient_counts(concepts: pd.DataFrame, patients_info: pd.DataFrame, logger)->None:
    """
    Check that the number of patients in concepts and patients_info match.
    Issue a warning if they don't.
    """
    if concepts.PID.nunique() != patients_info.PID.nunique():
            logger.warning(f"patients info contains {patients_info.PID.nunique()} patients != \
                        {concepts.PID.nunique()} unique patients in concepts")