import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments common to all pipeline types.

    Args:
        parser: The parser to add arguments to
    """
    parser.add_argument(
        "COMPUTE",
        type=str,
        default=None,
        nargs="?",
        help="Default compute target to use. If not set, compute targets must be specified for all components.",
    )
    parser.add_argument(
        "CONFIG_DIR",
        type=str,
        default=None,
        nargs="?",
        help="Path to folder with configs. If not set, configs must be specified for all components.",
    )
    parser.add_argument(
        "-cp",
        "--compute",
        type=str,
        action="append",
        default=[],
        help="Compute target to use for a specific pipeline step. Use: '-cp <job_name>=<compute>'.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        default=[],
        help="Path to configuration file for each component. Use: '-c <job_name>=<path-to-cfg>'. If not set, default is file from repo.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="corebehrt_runs",
        help="Experiment to run the job in.",
    )
    parser.add_argument(
        "-o",
        "--register_output",
        type=str,
        action="append",
        default=[],
        help="If an output from any step in the pipeline should be registered, provide a name for the Azure asset using the format '--register_output <job_name>.<input>=<name>'.",
    )
    parser.add_argument(
        "-lsm",
        "--log_system_metrics",
        action="store_true",
        default=False,
        help="If set, system metrics such as CPU, GPU and memory usage are logged in Azure.",
    )
