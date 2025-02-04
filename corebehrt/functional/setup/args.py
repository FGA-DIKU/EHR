import argparse


def get_args(default_config_name, default_run_name=None):
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=default_config_name)
    parser.add_argument(
        "--run_name",
        type=str,
        default=(
            default_run_name if default_run_name else default_config_name.split(".")[0]
        ),
    )
    return parser.parse_args()
