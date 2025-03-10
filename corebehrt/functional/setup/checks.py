import pandas as pd
from typing import Set
from corebehrt.constants.data import CONCEPT_COL, TIMESTAMP_COL, PID_COL, VALUE_COL


def check_categories(categories: dict) -> None:
    for col, rules in categories.items():
        # Check that we don't have both 'include' and 'exclude' in one category

        if "include" in rules and "exclude" in rules:
            raise ValueError(
                f"Category '{col}' has both 'include' and 'exclude' rules defined. "
                "Please specify only one."
            )
        if "include" not in rules and "exclude" not in rules:
            raise ValueError(
                f"Category '{col}' has no 'include' or 'exclude' rules defined. "
                "Please specify at least one."
            )


def check_concepts_columns(df: pd.DataFrame) -> None:
    """Check if required columns are present in concepts."""
    required_columns = {PID_COL, TIMESTAMP_COL, CONCEPT_COL, VALUE_COL}
    check_required_columns(df, required_columns, "concepts")


def check_features_columns(df: pd.DataFrame) -> None:
    """Check if required columns are present in features."""
    required_columns = {PID_COL, TIMESTAMP_COL, CONCEPT_COL}
    check_required_columns(df, required_columns, "features")


def check_required_columns(
    df: pd.DataFrame, required_columns: Set[str], type_: str
) -> None:
    if not required_columns.issubset(set(df.columns)):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in {type_}: {missing_columns}")
