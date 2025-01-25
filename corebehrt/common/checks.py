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
