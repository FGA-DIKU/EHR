def parse_pair_args(pair_args: list) -> dict:
    """
    Parses the append arguments.
    Each argument is expected to be of the form <key>=<value>

    :param args: List of arguments to be parsed.

    :return: a dict/mapping from key to value
    """
    pairs = [p.split("=") for p in pair_args]
    assert all(len(p) == 2 for p in pairs), "Invalid paired arg..."
    return dict(pairs)
