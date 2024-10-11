import os


def split_path(path_str: str) -> list:
    """Split path into its components."""
    directories = []
    while path_str:
        path_str, directory = os.path.split(path_str)
        # If we've reached the root directory
        if directory:
            directories.append(directory)
        elif path_str:
            break
    return directories[::-1]  # Reverse the list to get original order
