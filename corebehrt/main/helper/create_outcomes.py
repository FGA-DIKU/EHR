from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from corebehrt.modules.cohort_handling.outcomes import OutcomeMaker
from corebehrt.modules.monitoring.logger import TqdmToLogger


def process_data(loader, cfg, features_cfg, logger) -> dict:
    """Process batches of concept and patient data to create outcome tables.

    Args:
        loader: A callable that yields tuples of (concept_batch, patient_batch) DataFrames.
        cfg: Configuration object containing outcome settings.
        features_cfg: Configuration object containing feature settings, including origin point.
        logger: Logger object for tracking progress.

    Returns:
        dict: A dictionary mapping outcome names to their corresponding DataFrame tables.
            Each DataFrame contains the processed outcome data for all patients.

    Note:
        The function processes data in batches to handle large datasets efficiently.
        It uses the OutcomeMaker class to generate outcome tables for each batch,
        then concatenates the results across all batches.
    """
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
