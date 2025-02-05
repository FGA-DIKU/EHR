import dask.dataframe as dd
from typing import Set


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


def check_concepts_columns(df: dd.DataFrame) -> None:
    """Check if required columns are present in concepts."""
    required_columns = {"PID", "CONCEPT", "TIMESTAMP", "ADMISSION_ID"}
    check_required_columns(df, required_columns, "concepts")


def check_patients_info_columns(
    df: dd.DataFrame, background_vars: Set[str] = set()
) -> None:
    """Check if required columns are present in patients_info."""
    required_columns = {"PID", "BIRTHDATE", "DEATHDATE"}.union(set(background_vars))
    check_required_columns(df, required_columns, "patients_info")


def check_required_columns(
    df: dd.DataFrame, required_columns: Set[str], type_: str
) -> None:
    if not required_columns.issubset(set(df.columns)):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in {type_}: {missing_columns}")
