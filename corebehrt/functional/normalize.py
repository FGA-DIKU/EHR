

def min_max_normalize(value, min_value, max_value) -> float:
    """
    Performs min-max normalisation on a single value.
    """
    return (value - min_value) / (max_value - min_value)