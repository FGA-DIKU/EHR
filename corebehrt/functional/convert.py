import dask.dataframe as dd
from typing import Tuple, Dict, List


def convert_to_sequences(features: dd.DataFrame) -> Tuple[Dict[str, List], List[str]]:
    """Convert the DataFrame to a dictionary of features and a list of PIDs."""
    features = features.compute()  # Compute Dask DataFrame to pandas DataFrame
    pids = features["PID"].unique().tolist()  # List of unique PIDs

    grouped = features.groupby("PID")
    features_dict = {
        col: grouped[col].apply(list).to_dict()
        for col in features.columns
        if col != "PID"
    }

    for col in features_dict:
        # Convert dictionary of lists from grouped structure to normal lists
        features_dict[col] = list(features_dict[col].values())

    return features_dict, pids
