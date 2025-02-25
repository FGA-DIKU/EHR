"""
Command-line utility for starting CoreBEHRT jobs on Azure clusters.
Requires installation of azure-ml-ai python package and a valid Azure workspace.
"""

import sys
import argparse
import yaml
from . import environment, util


def parse_register_output(register_output_args: list) -> dict:
    """
    Parses the register output append argument.
    Each argument is expected to be of the form <output_id>=<asset_name>

    :param args: List of arguments to be parsed.

    :return: a dict/mapping from output_id to asset_name
    """
    register_output = [o.split("=") for o in register_output_args]
    assert all(len(o) == 2 for o in register_output), "Invalid arg for register_output"
    return dict(register_output)


def create_and_run_job(args) -> None:
    """
    Run the job from the given arguments.
    """

    cfg_path = args.config or f"./corebehrt/configs/{args.JOB}.yaml"
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    register_output = parse_register_output(args.register_output)

    job = util.create_job(
        args.JOB,
        cfg,
        compute=args.COMPUTE,
        register_output=register_output,
        log_system_metrics=args.log_system_metrics,
    )

    # Start job
    util.run_job(job, args.experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="corebehrt.azure", description="Run corebehrt jobs and pipelines in Azure"
    )

    # Sub-parsers
    subparsers = parser.add_subparsers(dest="call_type")
    # Environment parser
    env_parser = subparsers.add_parser(
        "build_env", help="Build the CoreBEHRT environment"
    )
    # Job parser
    job_parser = subparsers.add_parser("job", help="Run a single job.")
    job_parser.add_argument(
        "JOB",
        type=str,
        choices={
            "create_data",
            "pretrain",
            "create_outcomes",
            "select_cohort",
            "finetune_cv",
        },
        help="Job to run.",
    )
    job_parser.add_argument(
        "COMPUTE",
        type=str,
        help="Compute target to use.",
    )
    job_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file. Default is file from repo.",
    )
    job_parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="corebehrt_runs",
        help="Experiment to run the job in.",
    )
    job_parser.add_argument(
        "-o",
        "--register_output",
        type=str,
        action="append",
        default=[],
        help="If an output should be registered, provide a name for the Azure asset using the format '--register_output <input>=<name>'.",
    )
    job_parser.add_argument(
        "-lsm",
        "--log_system_metrics",
        action="store_true",
        default=False,
        help="If set, system metrics such as CPU, GPU and memory usage are logged in Azure.",
    )

    # Parse args
    args = parser.parse_args()

    if args.call_type == "build_env":
        # Build environment
        environment.build()
    elif args.call_type == "job":
        # Handle job
        create_and_run_job(args)
    else:
        parser.print_help()
        sys.exit(1)
