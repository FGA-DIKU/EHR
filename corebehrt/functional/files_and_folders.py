from pathlib import Path
from typing import List


def split_path(path_str: str) -> List[str]:
    """Split path into its components."""
    return list(Path(path_str).parts)
